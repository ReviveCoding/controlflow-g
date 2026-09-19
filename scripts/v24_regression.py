"""Small fresh-development regression of the frozen V2.3 candidate."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.evaluation import evaluate
from controlflow.v22.executor import verify_ledger
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v23.integrity import load_frozen_approval_issuer
from controlflow.v23.vllm_client import AttributedVllmClient
from controlflow.v24.sqlite_lifecycle import assert_all_closed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    legacy = importlib.import_module("v23_benchmark")
    source = ROOT / "data/v24/development/seed_24101_v2"
    output = ROOT / "results/v24/development_regression"
    artifacts = ROOT / "artifacts/v24/development_regression"
    if output.exists() or artifacts.exists():
        raise RuntimeError("V24_DEVELOPMENT_REGRESSION_ALREADY_EXISTS")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    args = argparse.Namespace(
        server_profile="v23",
        prefix_cache=True,
        performance_mode="interactivity",
        optimization_level=2,
        batched_tokens=2048,
        run_role="development",
    )
    live = legacy._verify_server(serving, args)
    cold = legacy._scrape(live["endpoint_root"])
    legacy._assert_cold_server(cold)
    atomic_write_json(output / "server_preflight.json", {"server_provenance": live, "cold_metrics": cold})
    client = AttributedVllmClient(
        endpoint=f"{live['endpoint_root']}/v1/chat/completions",
        model=serving["served_model_name"],
        schema=json.loads((ROOT / "configs/v23/explanation_schema_minimal.json").read_text(encoding="utf-8")),
        prompt_template=(ROOT / "configs/v23/explanation_prompt_minimal.txt").read_text(encoding="utf-8"),
        max_tokens=128,
        contract="minimal",
        diagnostics_path=artifacts / "request_diagnostics.partial.jsonl",
    )
    legacy._warmup(client, 8)
    runtime = pd.read_parquet(source / "runtime_cases.parquet").iloc[:120].copy()
    truth = pd.read_parquet(source / "evaluator_truth.parquet")
    truth = truth[truth.case_id.isin(set(runtime.case_id))].copy()
    runtime.to_parquet(output / "runtime_subset.parquet", index=False)
    truth.to_parquet(output / "truth_subset.parquet", index=False)
    issuer = load_frozen_approval_issuer(ROOT)
    ledger_path = artifacts / "development.sqlite"
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=source / "evidence_corpus.parquet",
        authorization_path=source / "authorization_state.parquet",
        ledger_path=ledger_path,
        explanation_client=client,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action, policy=prepared.proposal, reviewer_id="v24-development-reviewer", approve=True
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    checkpoint_fields = {
        "git_commit": "V24_DEVELOPMENT_ONLY",
        "dirty_state_hash": "V24_DEVELOPMENT_ONLY",
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(output / "runtime_subset.parquet"),
        "evidence_corpus_hash": sha256_file(source / "evidence_corpus.parquet"),
        "authorization_state_hash": sha256_file(source / "authorization_state.parquet"),
        "candidate_bundle_hash": bundle.bundle_hash,
        "model_hashes": {
            name: bundle.payload[name]["sha256"]
            for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
        },
        "critical_threshold": bundle.threshold,
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "structured_backend": serving["structured_output_backend"],
        "prompt_schema_hashes": {
            "prompt": sha256_file(ROOT / "configs/v23/explanation_prompt_minimal.txt"),
            "schema": sha256_file(ROOT / "configs/v23/explanation_schema_minimal.json"),
        },
        "retrieval_config_hash": bundle.payload["retrieval_config"]["sha256"],
        "temporal_config_hash": bundle.payload["temporal_config"]["sha256"],
        "pdp_action_registry_hashes": {
            "pdp": bundle.payload["policy_config"]["sha256"],
            "actions": bundle.payload["action_registry"]["sha256"],
        },
        "evaluator_protocol_hash": "V24_DEVELOPMENT_ONLY",
        "gate_config_hash": sha256_file(ROOT / "configs/v24/qualification_gates.yaml"),
        "qualification_gate_freeze_hash": "V24_DEVELOPMENT_ONLY",
        "seed": 24101,
        "concurrency": 2,
    }
    runner.run_dataset(
        runtime,
        output_path=output / "candidate_results.parquet",
        partial_path=artifacts / "development.partial.jsonl",
        checkpoint_path=artifacts / "development.checkpoint.json",
        checkpoint_fields=checkpoint_fields,
        concurrency=2,
    )
    actual_concurrency = legacy._maximum_overlap(client.diagnostics)
    metrics = evaluate(
        output / "candidate_results.parquet",
        output / "truth_subset.parquet",
        ledger_path,
        output / "metrics.json",
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="DEVELOPMENT_REGRESSION",
    )
    ledger = verify_ledger(ledger_path)
    assert_all_closed(ledger_path)
    summary = {
        "schema_version": 1,
        "role": "DEVELOPMENT_ONLY",
        "created_at": utc_now(),
        "case_count": len(runtime),
        "actual_maximum_llm_concurrency": actual_concurrency,
        "binary_critical_recall": metrics["binary_critical_recall"],
        "core_stc": metrics["core_stc"],
        "structured_output_failure_rate": metrics["structured_output_failure_rate"],
        "temporal_policy_accuracy": metrics["temporal_policy_accuracy"],
        "evidence_completeness": metrics["evidence_completeness"],
        "approval_bypass_commits": metrics["security"]["approval_bypass_commits"],
        "unauthorized_committed_actions": metrics["security"]["unauthorized_committed_actions"],
        "p95_seconds": metrics["latency"]["total_p95_seconds"],
        "ledger_valid": ledger["valid"],
        "candidate_results_sha256": sha256_file(output / "candidate_results.parquet"),
        "metrics_sha256": sha256_file(output / "metrics.json"),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
