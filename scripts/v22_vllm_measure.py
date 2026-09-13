from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateModelBundle
from controlflow.v22.evaluation import bootstrap_quantile_ci
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v22.vllm_client import StructuredVllmClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument("--no-semantic-verifier", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    serving = yaml.safe_load((ROOT / "configs/v22/serving.yaml").read_text(encoding="utf-8"))
    namespace = config["development"]["namespace"]
    runtime_path = ROOT / f"data/v22/{namespace}/validation/runtime_cases.parquet"
    evidence_path = ROOT / f"data/v22/{namespace}/validation/evidence_corpus.parquet"
    runtime = pd.read_parquet(runtime_path).head(args.requests)
    public_key = ROOT / "artifacts/v22/approval_public_key.pem"
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public_key)
    slug = "no_semantic" if args.no_semantic_verifier else "unconstrained" if args.unconstrained else "structured"
    ledger = ROOT / f"artifacts/v22/{namespace}/vllm_{slug}.sqlite"
    if ledger.exists():
        raise RuntimeError(f"serving evidence namespace already exists: {ledger}")
    bundle = CandidateModelBundle(ROOT / "state/v22_model_bundle.json", ROOT)
    endpoint_root = f"http://{serving['host']}:{serving['port']}"
    health = httpx.get(f"{endpoint_root}/v1/models", timeout=10)
    health.raise_for_status()
    served_models = {str(item["id"]) for item in health.json()["data"]}
    live_model_identity_verified = "controlflow-g-v22-qwen3-4b" in served_models
    process_probe = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            "pgrep -af 'vllm serve Qwen/Qwen3-4B-Instruct-2507'",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required_arguments = (
        str(serving["revision"]),
        "--dtype bfloat16",
        f"--gpu-memory-utilization {serving['gpu_memory_utilization']}",
        f"--max-model-len {serving['max_model_len']}",
        "--max-num-seqs 2",
        "--structured-outputs-config.backend xgrammar",
    )
    live_command_line_verified = all(item in process_probe for item in required_arguments)
    if not live_model_identity_verified or not live_command_line_verified:
        raise RuntimeError("LIVE_VLLM_IDENTITY_OR_ARGUMENT_MISMATCH")
    client = StructuredVllmClient(
        endpoint=f"{endpoint_root}/v1/chat/completions",
        model="controlflow-g-v22-qwen3-4b",
        schema=json.loads(bundle.artifact_path("schema").read_text(encoding="utf-8")),
        prompt_template=bundle.artifact_path("prompt").read_text(encoding="utf-8"),
        constrained=not args.unconstrained,
        semantic_verify=not args.no_semantic_verifier,
    )
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=evidence_path,
        authorization_path=ROOT / f"data/v22/{namespace}/validation/authorization_state.parquet",
        ledger_path=ledger,
        explanation_client=client,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-serving",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    rows = runtime.to_dict(orient="records")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(runner.run_case, rows))
    frame = pd.DataFrame([item.model_dump(mode="json") for item in results])
    output = ROOT / f"results/v22/{namespace}_vllm_{slug}_c2.parquet"
    frame.to_parquet(output, index=False)
    diagnostics = client.diagnostics
    failure_fields = ("http_failure", "json_parse_failure", "schema_failure", "semantic_reference_failure")
    failures = {name: sum(bool(item[name]) for item in diagnostics) for name in failure_fields}
    total_failures = sum(any(bool(item[name]) for name in failure_fields) for item in diagnostics)
    if len(diagnostics) != len(frame) or failures["http_failure"] == len(frame):
        raise RuntimeError("LIVE_VLLM_MEASUREMENT_INVALID")
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "role": "DEVELOPMENT_VALIDATION",
        "actual_live_vllm": True,
        "live_model_identity_verified": live_model_identity_verified,
        "live_command_line_verified": live_command_line_verified,
        "live_served_models": sorted(served_models),
        "live_process_command": process_probe,
        "model": serving["model"],
        "model_revision": serving["revision"],
        "vllm_version": json.loads(
            next(
                line
                for line in (ROOT / "state/v22_environment_evidence/wsl_probe.txt").read_text().splitlines()
                if '"vllm"' in line
            )
        )["vllm"],
        "structured_backend": None if args.unconstrained else serving["structured_output_backend"],
        "server_args": {
            "dtype": serving["dtype"],
            "gpu_memory_utilization": serving["gpu_memory_utilization"],
            "max_model_len": serving["max_model_len"],
            "max_num_seqs": 2,
            "port": serving["port"],
        },
        "concurrency": 2,
        "requests": len(frame),
        "failures": failures,
        "total_structured_failures": total_failures,
        "structured_failure_rate": total_failures / len(frame),
        "llm_p95_seconds": float(frame.llm_latency_seconds.quantile(0.95)),
        "non_llm_p95_seconds": float(frame.non_llm_latency_seconds.quantile(0.95)),
        "total_p95_seconds": float(frame.total_latency_seconds.quantile(0.95)),
        "latency_p95_confidence_intervals": {
            "llm": bootstrap_quantile_ci(frame.llm_latency_seconds),
            "non_llm": bootstrap_quantile_ci(frame.non_llm_latency_seconds),
            "total": bootstrap_quantile_ci(frame.total_latency_seconds),
        },
        "response_artifact": {"path": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output)},
        "diagnostics": diagnostics,
    }
    atomic_write_json(ROOT / f"state/v22_vllm_{slug}_c2.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
