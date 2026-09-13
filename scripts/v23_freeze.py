from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def _development_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    quality = summary["quality_metrics"]
    security = quality["security"]
    checks = {
        "requests_below_600": summary["requests"] < 600,
        "actual_concurrency_not_2": summary["actual_maximum_llm_concurrency"] != 2,
        "p95_above_13": summary["latency_seconds"]["total"]["p95"] > 13.0,
        "bootstrap_upper_not_below_15": summary["latency_seconds"]["total_p95_bootstrap_ci"]["upper"] >= 15.0,
        "critical_recall_below_0_95": quality["binary_critical_recall"]["estimate"] < 0.95,
        "core_stc_below_0_80": quality["core_stc"]["estimate"] < 0.80,
        "structured_failure_not_below_0_005": summary["structured_failure_rate"] >= 0.005,
        "temporal_accuracy_below_0_98": quality["temporal_policy_accuracy"]["estimate"] < 0.98,
        "approval_bypass_nonzero": security["approval_bypass_commits"] != 0,
        "unauthorized_commit_nonzero": security["unauthorized_committed_actions"] != 0,
        "duplicate_commit_nonzero": security["duplicate_commits"] != 0,
    }
    failures.extend(name for name, failed in checks.items() if failed)
    return not failures, failures


def _binding(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--independent", required=True)
    args = parser.parse_args()
    qualification = json.loads((ROOT / "state/v23_qualification_manifest.json").read_text(encoding="utf-8"))
    if qualification.get("one_shot_opened") or qualification.get("qualification_executed"):
        raise RuntimeError("cannot freeze after V23QUAL has opened")
    reviews = json.loads((ROOT / "state/v23_review_findings.json").read_text(encoding="utf-8"))
    if reviews.get("status") != "PRE_QUALIFICATION_CLEAR" or reviews.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("pre-qualification reviews are not clear")
    prequalification_review_path = ROOT / "state/v23_prequalification_review_findings.json"
    if prequalification_review_path.exists():
        if json.loads(prequalification_review_path.read_text(encoding="utf-8")) != reviews:
            raise RuntimeError("immutable pre-qualification review record already differs")
    else:
        atomic_write_json(prequalification_review_path, reviews)
    summaries = []
    for config_id in (args.primary, args.independent):
        path = ROOT / "results/v23" / config_id / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        passed, failures = _development_pass(summary)
        if not passed:
            raise RuntimeError(f"development eligibility failed for {config_id}: {failures}")
        summaries.append((path, summary))
    primary = summaries[0][1]
    independent = summaries[1][1]
    first_config = primary["configuration"]
    second_config = independent["configuration"]
    comparable = (
        "performance_mode",
        "max_num_batched_tokens",
        "contract",
        "max_tokens",
        "optimization_level",
        "prefix_cache",
    )
    if any(first_config[key] != second_config[key] for key in comparable):
        raise RuntimeError("independent soak did not use the selected serving configuration")
    serving_path = ROOT / "configs/v23/serving.yaml"
    serving = yaml.safe_load(serving_path.read_text(encoding="utf-8"))
    expected = {
        "performance_mode": serving["performance_mode"],
        "max_num_batched_tokens": serving["max_num_batched_tokens"],
        "optimization_level": serving["optimization_level"],
        "prefix_cache": serving["enable_prefix_caching"],
    }
    if any(first_config[key] != value for key, value in expected.items()):
        raise RuntimeError("selected soak and frozen serving.yaml differ")
    gate_path = ROOT / "configs/v23/qualification_gates.yaml"
    gates = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if not gates.get("frozen_before_qualification"):
        raise RuntimeError("set frozen_before_qualification before creating freeze evidence")
    bindings = [
        _binding(ROOT / "state/v23_typed_core_manifest.json"),
        _binding(ROOT / "state/v23_integrity.json"),
        _binding(prequalification_review_path),
        _binding(ROOT / "state/v23_historical_boundary.json"),
        _binding(ROOT / "state/v23_interrupted_namespace_binding.json"),
        _binding(ROOT / "results/v23/v23_tests.xml"),
        _binding(ROOT / "results/v23/tail_analysis.json"),
        _binding(ROOT / "results/v23/ablations.json"),
        _binding(serving_path),
        _binding(ROOT / "configs/v23/explanation_schema_minimal.json"),
        _binding(ROOT / "configs/v23/explanation_prompt_minimal.txt"),
        _binding(gate_path),
        _binding(ROOT / "configs/v22/runtime.yaml"),
        _binding(ROOT / "configs/v22/temporal_policies.yaml"),
        _binding(ROOT / "configs/v22/policy.yaml"),
        _binding(ROOT / "configs/v22/action_registry.yaml"),
        _binding(ROOT / "src/controlflow/v22/approval.py"),
        _binding(ROOT / "src/controlflow/v22/executor.py"),
        _binding(ROOT / "src/controlflow/v22/evaluation.py"),
        _binding(ROOT / "artifacts/v22/approval_public_key.pem"),
        _binding(ROOT / "src/controlflow/v23/vllm_client.py"),
        _binding(ROOT / "src/controlflow/v23/integrity.py"),
        _binding(ROOT / "src/controlflow/v23/telemetry.py"),
        _binding(ROOT / "src/controlflow/v23/dgp.py"),
        _binding(ROOT / "scripts/v23_server.sh"),
        _binding(ROOT / "scripts/v23_benchmark.py"),
        _binding(ROOT / "scripts/v23_freeze.py"),
        _binding(ROOT / "scripts/v23_qualify.py"),
        _binding(ROOT / "scripts/v23_prepare_final.py"),
        _binding(ROOT / "scripts/v23_final.py"),
        _binding(ROOT / "scripts/v23_record_integrity.py"),
        _binding(ROOT / "configs/v23/temporal_truth_fixtures.yaml"),
    ]
    typed = json.loads((ROOT / "state/v23_typed_core_manifest.json").read_text(encoding="utf-8"))
    bindings.extend(_binding(ROOT / item["path"]) for item in typed["bindings"].values())
    bindings.append(_binding(ROOT / typed["source_bundle"]["path"]))
    interrupted = json.loads((ROOT / "state/v23_interrupted_namespace_binding.json").read_text(encoding="utf-8"))
    bindings.extend(_binding(ROOT / item["path"]) for item in interrupted["bindings"])
    integrity = json.loads((ROOT / "state/v23_integrity.json").read_text(encoding="utf-8"))
    for cohort in integrity["development_cohorts"]:
        bindings.extend(
            [
                cohort["summary"],
                cohort["checkpoint_binding"],
                cohort["ledger"]["binding"],
                *cohort["request_diagnostics"]["bindings"],
                *cohort["summary_artifact_bindings"],
                *cohort["durable_sampler_bindings"],
            ]
        )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    freeze_body = {
        "schema_version": 1,
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "source_commit": git_commit,
        "dependencies": _binding(ROOT / "uv.lock"),
        "qwen": {"model": serving["model"], "revision": serving["revision"]},
        "serving": {
            "vllm_version": serving["vllm_version"],
            "dtype": serving["dtype"],
            "performance_mode": serving["performance_mode"],
            "optimization_level": serving["optimization_level"],
            "structured_backend": serving["structured_output_backend"],
            "max_num_batched_tokens": serving["max_num_batched_tokens"],
            "max_num_seqs": serving["max_num_seqs"],
            "max_model_len": serving["max_model_len"],
            "gpu_memory_utilization": serving["gpu_memory_utilization"],
            "enable_chunked_prefill": serving["enable_chunked_prefill"],
            "enable_prefix_caching": serving["enable_prefix_caching"],
            "speculative_decoding": None,
            "schema_path": "configs/v23/explanation_schema_minimal.json",
            "prompt_path": "configs/v23/explanation_prompt_minimal.txt",
            "max_tokens": first_config["max_tokens"],
        },
        "warmup_protocol": {
            "request_count": 8,
            "development_only_case_prefix": "V23WARM-",
            "excluded_from_measurement": True,
            "qualification_or_final_examples_used": False,
        },
        "development_evidence": [
            {**_binding(path), "p95_seconds": summary["latency_seconds"]["total"]["p95"]} for path, summary in summaries
        ],
        "bindings": bindings,
        "qualification_seed": 23907,
        "qualification_count": 600,
        "final_not_generated": True,
    }
    freeze_body["freeze_hash"] = hashlib.sha256(canonical_json(freeze_body)).hexdigest()
    freeze_body["created_at"] = utc_now()
    atomic_write_json(ROOT / "state/v23_freeze_manifest.json", freeze_body)
    atomic_write_json(
        ROOT / "configs/v23/qualification_gate_freeze.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "FROZEN_BEFORE_QUALIFICATION",
            "source_commit": git_commit,
            "gate_config_sha256": sha256_file(gate_path),
            "qualification_results_present_at_freeze": False,
            "qualification_manifest_status": qualification.get("status"),
            "protocol_freeze_hash": freeze_body["freeze_hash"],
        },
    )


if __name__ == "__main__":
    main()
