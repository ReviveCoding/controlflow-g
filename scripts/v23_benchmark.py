from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.checkpoint import git_state
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v23.integrity import load_frozen_approval_issuer
from controlflow.v23.latency import bootstrap_quantile_ci, parse_prometheus, percentile_summary
from controlflow.v23.telemetry import NvidiaSampler, PrometheusSampler
from controlflow.v23.vllm_client import AttributedVllmClient

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)


def _maximum_overlap(diagnostics: list[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for row in diagnostics:
        events.extend(((float(row["request_start"]), 1), (float(row["request_end"]), -1)))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _scrape(endpoint_root: str) -> dict[str, Any]:
    response = httpx.get(f"{endpoint_root}/metrics", timeout=20)
    response.raise_for_status()
    samples = [item for item in parse_prometheus(response.text) if item["name"].startswith("vllm:")]
    return {"captured_at": utc_now(), "samples": samples}


def _metric_sum(snapshot: dict[str, Any], name: str) -> float:
    return sum(float(item["value"]) for item in snapshot["samples"] if item["name"] == name)


def _assert_cold_server(snapshot: dict[str, Any]) -> None:
    if _metric_sum(snapshot, "vllm:request_success_total") != 0.0:
        raise RuntimeError("LIVE_V23_SERVER_NOT_FRESH: prior completed requests exist")
    if _metric_sum(snapshot, "vllm:prefix_cache_queries_total") != 0.0:
        raise RuntimeError("LIVE_V23_SERVER_NOT_FRESH: prefix cache has prior queries")


def _server_metrics(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("server_metrics") or {}
    aliases: dict[str, tuple[tuple[str, float], ...]] = {
        "queue_time_seconds": (
            ("queue_time", 1.0),
            ("time_in_queue", 1.0),
            ("request_queue_time", 1.0),
            ("queue_time_ms", 0.001),
        ),
        "ttft_seconds": (("time_to_first_token", 1.0), ("ttft", 1.0), ("time_to_first_token_ms", 0.001)),
        "prefill_time_seconds": (("prefill_time", 1.0), ("request_prefill_time", 1.0)),
        "decode_time_seconds": (
            ("decode_time", 1.0),
            ("request_decode_time", 1.0),
            ("generation_time_ms", 0.001),
        ),
        "inter_token_latency_seconds": (
            ("inter_token_latency", 1.0),
            ("itl", 1.0),
            ("tpot", 1.0),
            ("mean_itl_ms", 0.001),
        ),
    }
    output: dict[str, Any] = {"server_metrics_raw": raw}
    for target, candidates in aliases.items():
        output[target] = next(
            (float(raw[name]) * scale for name, scale in candidates if raw.get(name) is not None), None
        )
    return output


def _verify_server(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    endpoint_root = f"http://{config['host']}:{config['port']}"
    models = httpx.get(f"{endpoint_root}/v1/models", timeout=20)
    models.raise_for_status()
    served = {str(item["id"]) for item in models.json()["data"]}
    process_output = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            f"pgrep -af 'vllm serve {config['model']}'",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process_lines = [line for line in process_output.splitlines() if "vllm serve" in line]
    if len(process_lines) != 1:
        raise RuntimeError(f"LIVE_V23_SERVER_PROCESS_COUNT_MISMATCH:{len(process_lines)}")
    command = process_lines[0]
    pid = int(command.split(maxsplit=1)[0])
    expected = [
        config["revision"],
        f"--served-model-name {config['served_model_name']}",
        "--dtype bfloat16",
        "--gpu-memory-utilization 0.72",
        "--max-model-len 4096",
        "--max-num-seqs 2",
        "--structured-outputs-config.backend xgrammar",
    ]
    if args.server_profile == "v23":
        prefix_flag = "--enable-prefix-caching" if args.prefix_cache else "--no-enable-prefix-caching"
        expected.extend(
            (
                "--enable-chunked-prefill",
                "--enable-per-request-metrics",
                f"--performance-mode {args.performance_mode}",
                f"--optimization-level {args.optimization_level}",
                prefix_flag,
            )
        )
        if args.batched_tokens:
            expected.append(f"--max-num-batched-tokens {args.batched_tokens}")
        elif "--max-num-batched-tokens" in command:
            raise RuntimeError("LIVE_V23_SERVER_ARGUMENT_MISMATCH: expected installed default batched-token budget")
    verified = config["served_model_name"] in served and all(item in command for item in expected)
    if not verified:
        raise RuntimeError("LIVE_V23_SERVER_ARGUMENT_MISMATCH")
    version = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            '"$HOME/.venvs/controlflow-g-v2/bin/python" -c "import vllm; print(vllm.__version__)"',
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    environment = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            f"tr '\\0' '\\n' < /proc/{pid}/environ | grep -E '^(V23_RUN_ROLE|VLLM_USE_V2_MODEL_RUNNER)='",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    expected_role = getattr(args, "run_role", None)
    if version != config["vllm_version"] or "VLLM_USE_V2_MODEL_RUNNER=0" not in environment:
        raise RuntimeError("LIVE_V23_SERVER_ENVIRONMENT_MISMATCH")
    if expected_role is not None and f"V23_RUN_ROLE={expected_role}" not in environment:
        raise RuntimeError("LIVE_V23_SERVER_ROLE_MISMATCH")
    started_at = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-22.04", "--", "ps", "-p", str(pid), "-o", "lstart="],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "endpoint_root": endpoint_root,
        "served_models": sorted(served),
        "process_command": command,
        "required_arguments": expected,
        "verified": True,
        "pid": pid,
        "process_started_at": started_at,
        "vllm_version": version,
        "environment": sorted(environment),
        "run_role": expected_role or "development",
    }


def _warmup(client: AttributedVllmClient, count: int) -> None:
    for index in range(count):
        client(
            {
                "case_id": f"V23WARM-{index:03d}",
                "deterministic_summary": "Warm-up only; simulated development input.",
                "severity": "LOW",
                "root_cause": "PROCESS",
                "evidence_ids": ["WARM-EVIDENCE-A", "WARM-EVIDENCE-B"],
                "policy_id": "warmup-policy",
                "policy_decision": "ALLOW",
            }
        )
    client.clear_diagnostics()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--server-profile", choices=("v22-baseline", "v23"), default="v23")
    parser.add_argument("--development-namespace", default=None)
    parser.add_argument("--performance-mode", choices=("balanced", "interactivity", "throughput"), required=True)
    parser.add_argument(
        "--batched-tokens",
        type=int,
        choices=(0, 1024, 2048, 4096),
        required=True,
        help="Use 0 to retain the installed vLLM default.",
    )
    parser.add_argument("--contract", choices=("current", "minimal"), required=True)
    parser.add_argument("--max-tokens", type=int, choices=(96, 128, 160), required=True)
    parser.add_argument("--optimization-level", type=int, choices=(2, 3), default=2)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.requests < 2:
        raise RuntimeError("at least two requests are required to establish concurrency")
    output_dir = ROOT / "results/v23" / args.config_id
    artifact_dir = ROOT / "artifacts/v23" / args.config_id
    if output_dir.exists() or artifact_dir.exists():
        raise RuntimeError(f"immutable benchmark namespace already exists: {args.config_id}")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    if args.server_profile == "v22-baseline":
        config.update({"served_model_name": "controlflow-g-v22-qwen3-4b", "port": 8022})
        if (args.performance_mode, args.contract, args.max_tokens) != ("balanced", "current", 160):
            raise RuntimeError("V2.2 baseline requires balanced/current/160")
    tournament = yaml.safe_load((ROOT / "configs/v23/tournament.yaml").read_text(encoding="utf-8"))
    live = _verify_server(config, args)
    if args.contract == "minimal":
        schema_path = ROOT / "configs/v23/explanation_schema_minimal.json"
        prompt_path = ROOT / "configs/v23/explanation_prompt_minimal.txt"
    else:
        schema_path = ROOT / "configs/v22/explanation_schema.json"
        prompt_path = ROOT / "configs/v22/explanation_prompt.txt"
    client = AttributedVllmClient(
        endpoint=f"{live['endpoint_root']}/v1/chat/completions",
        model=config["served_model_name"],
        schema=json.loads(schema_path.read_text(encoding="utf-8")),
        prompt_template=prompt_path.read_text(encoding="utf-8"),
        max_tokens=args.max_tokens,
        contract=args.contract,
        diagnostics_path=artifact_dir / "request_diagnostics.partial.jsonl",
    )
    metrics_cold = _scrape(live["endpoint_root"])
    warmup_count = int(tournament["warmup_case_count"]) if args.warm else 0
    if warmup_count:
        _warmup(client, warmup_count)
    metrics_start = _scrape(live["endpoint_root"])
    namespace = args.development_namespace or str(tournament["development_namespace"])
    runtime_path = ROOT / f"data/v22/{namespace}/validation/runtime_cases.parquet"
    truth_path = ROOT / f"data/v22/{namespace}/validation/evaluator_truth.parquet"
    evidence_path = ROOT / f"data/v22/{namespace}/validation/evidence_corpus.parquet"
    authorization_path = ROOT / f"data/v22/{namespace}/validation/authorization_state.parquet"
    runtime = pd.read_parquet(runtime_path).head(args.requests)
    issuer = load_frozen_approval_issuer(ROOT)
    ledger = artifact_dir / "development.sqlite"
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=evidence_path,
        authorization_path=authorization_path,
        ledger_path=ledger,
        explanation_client=client,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-v23-development",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    telemetry_path = output_dir / "nvidia_telemetry.json"
    result_path = output_dir / "candidate_results.parquet"
    git_commit, dirty_hash = git_state(ROOT)
    checkpoint_fields = {
        "git_commit": git_commit,
        "dirty_state_hash": dirty_hash,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(runtime_path),
        "evidence_corpus_hash": sha256_file(evidence_path),
        "authorization_state_hash": sha256_file(authorization_path),
        "candidate_bundle_hash": sha256_file(ROOT / "state/v22_model_bundle.json"),
        "model_hashes": {
            name: bundle.payload[name]["sha256"]
            for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
        },
        "critical_threshold": bundle.threshold,
        "qwen_revision": config["revision"],
        "vllm_version": config["vllm_version"],
        "structured_backend": config["structured_output_backend"],
        "prompt_schema_hashes": {"prompt": sha256_file(prompt_path), "schema": sha256_file(schema_path)},
        "retrieval_config_hash": bundle.payload["retrieval_config"]["sha256"],
        "temporal_config_hash": bundle.payload["temporal_config"]["sha256"],
        "pdp_action_registry_hashes": {
            "pdp": bundle.payload["policy_config"]["sha256"],
            "actions": bundle.payload["action_registry"]["sha256"],
        },
        "evaluator_protocol_hash": evaluator_protocol_hash(ROOT),
        "gate_config_hash": sha256_file(ROOT / "configs/v23/qualification_gates.yaml"),
        "qualification_gate_freeze_hash": sha256_file(ROOT / "configs/v23/qualification_gates.yaml"),
        "seed": namespace,
        "concurrency": 2,
    }
    periodic_metrics_path = output_dir / "vllm_metrics_periodic.json"
    with (
        NvidiaSampler(telemetry_path, interval_seconds=1.0),
        PrometheusSampler(live["endpoint_root"], periodic_metrics_path, interval_seconds=5.0),
    ):
        results = runner.run_dataset(
            runtime,
            output_path=result_path,
            partial_path=artifact_dir / "development.partial.jsonl",
            checkpoint_path=artifact_dir / "development.checkpoint.json",
            checkpoint_fields=checkpoint_fields,
            concurrency=2,
        )
    metrics_end = _scrape(live["endpoint_root"])
    frame = pd.DataFrame([item.model_dump(mode="json") for item in results])
    frame.to_parquet(result_path, index=False)
    truth_subset_path = output_dir / "evaluator_truth_subset.parquet"
    truth = pd.read_parquet(truth_path)
    truth_subset = truth[truth.case_id.astype(str).isin(frame.case_id.astype(str))]
    if len(truth_subset) != len(frame):
        raise RuntimeError("DEVELOPMENT_TRUTH_SUBSET_CARDINALITY_MISMATCH")
    truth_subset.to_parquet(truth_subset_path, index=False)
    diagnostics = {str(item["case_id"]): item for item in client.diagnostics}
    request_rows = []
    for row in frame.to_dict(orient="records"):
        diagnostic = diagnostics[str(row["case_id"])]
        request_rows.append(
            {
                **diagnostic,
                **_server_metrics(diagnostic),
                "non_llm_pipeline_latency_seconds": float(row["non_llm_latency_seconds"]),
                "total_latency_seconds": float(row["total_latency_seconds"]),
                "structured_output_status": bool(row["structured_output_valid"]),
                "severity": row["severity"],
                "disposition": row["disposition"],
                "evidence_count": len(row["evidence_ids"]),
                "critical": row["severity"] == "CRITICAL",
            }
        )
    request_path = output_dir / "request_attribution.json"
    atomic_write_json(request_path, {"schema_version": 1, "requests": request_rows})
    metric_snapshots_path = output_dir / "vllm_metrics_snapshots.json"
    atomic_write_json(
        metric_snapshots_path,
        {
            "schema_version": 1,
            "cold_pre_warmup": metrics_cold,
            "measurement_start": metrics_start,
            "measurement_end": metrics_end,
        },
    )
    quality_path = output_dir / "quality_metrics.json"
    quality = evaluate(
        result_path,
        truth_subset_path,
        ledger,
        quality_path,
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="V23_DEVELOPMENT_VALIDATION",
    )
    actual_concurrency = _maximum_overlap(client.diagnostics)
    if actual_concurrency != 2:
        raise RuntimeError(f"ACTUAL_CONCURRENCY_NOT_TWO: observed {actual_concurrency}")
    failure_fields = (
        "http_failure",
        "json_parse_failure",
        "schema_failure",
        "truncation_failure",
        "semantic_reference_failure",
    )
    failures = {field: sum(bool(row[field]) for row in request_rows) for field in failure_fields}
    completion = [row["completion_tokens"] for row in request_rows if row.get("completion_tokens") is not None]
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "role": "DEVELOPMENT_ONLY",
        "config_id": args.config_id,
        "requests": len(frame),
        "concurrency_requested": 2,
        "actual_maximum_llm_concurrency": actual_concurrency,
        "warmup": {
            "enabled": args.warm,
            "fixed_development_only_requests": warmup_count,
            "excluded_from_measurement": True,
        },
        "configuration": {
            "server_profile": args.server_profile,
            "development_namespace": namespace,
            "performance_mode": args.performance_mode,
            "max_num_batched_tokens": args.batched_tokens or "installed_default",
            "contract": args.contract,
            "max_tokens": args.max_tokens,
            "optimization_level": args.optimization_level,
            "prefix_cache": args.prefix_cache,
        },
        "live_server": live,
        "latency_seconds": {
            "llm": percentile_summary(frame.llm_latency_seconds),
            "non_llm": percentile_summary(frame.non_llm_latency_seconds),
            "total": percentile_summary(frame.total_latency_seconds),
            "total_p95_bootstrap_ci": bootstrap_quantile_ci(frame.total_latency_seconds),
        },
        "completion_tokens": percentile_summary(completion),
        "structured_failures": failures,
        "structured_failure_count": sum(any(bool(row[field]) for field in failure_fields) for row in request_rows),
        "structured_failure_rate": sum(any(bool(row[field]) for field in failure_fields) for row in request_rows)
        / len(request_rows),
        "quality_metrics": quality,
        "artifacts": {
            "candidate_results": {"path": result_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(result_path)},
            "request_attribution": {
                "path": request_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(request_path),
            },
            "telemetry": {"path": telemetry_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(telemetry_path)},
            "periodic_metrics": {
                "path": periodic_metrics_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(periodic_metrics_path),
            },
            "durable_request_diagnostics": {
                "path": client.diagnostics_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(client.diagnostics_path),
            },
            "metrics_snapshots": {
                "path": metric_snapshots_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(metric_snapshots_path),
            },
            "quality": {"path": quality_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(quality_path)},
        },
        "provenance": {
            "runtime": {"path": runtime_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(runtime_path)},
            "truth_source": {"path": truth_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(truth_path)},
            "truth_subset": {
                "path": truth_subset_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(truth_subset_path),
            },
            "typed_core": {
                "path": "state/v23_typed_core_manifest.json",
                "sha256": sha256_file(ROOT / "state/v23_typed_core_manifest.json"),
            },
            "schema": {"path": schema_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(schema_path)},
            "prompt": {"path": prompt_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(prompt_path)},
        },
    }
    atomic_write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
