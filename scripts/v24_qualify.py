"""One-shot V24QUAL execution, closure, independent verification, and gates."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.checkpoint import git_state, records_hash, validate_checkpoint
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.executor import verify_ledger
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v23.integrity import load_frozen_approval_issuer, verify_request_diagnostics
from controlflow.v23.telemetry import NvidiaSampler, PrometheusSampler
from controlflow.v23.vllm_client import AttributedVllmClient
from controlflow.v24.admission import structural_admission
from controlflow.v24.artifact_closure import make_graph, scan_to_report
from controlflow.v24.closure_receipt import create_receipt
from controlflow.v24.denominators import recompute
from controlflow.v24.freeze_validation import verify_freeze
from controlflow.v24.gates import evaluate_gates
from controlflow.v24.gpu_preflight import verify_gpu_ownership
from controlflow.v24.ledger_snapshot import snapshot
from controlflow.v24.sqlite_lifecycle import assert_all_closed
from controlflow.v24.terminal_decision import create_anchor

_legacy_server = importlib.import_module("v23_benchmark")
_assert_cold_server = _legacy_server._assert_cold_server
_maximum_overlap = _legacy_server._maximum_overlap
_scrape = _legacy_server._scrape
_server_metrics = _legacy_server._server_metrics
_verify_server = _legacy_server._verify_server
_warmup = _legacy_server._warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v24/qualification/V24QUAL"
RESULTS = ROOT / "results/v24/qualification"
ARTIFACTS = ROOT / "artifacts/v24/qualification"
MANIFEST = ROOT / "state/v24_qualification_manifest.json"


def _paths() -> dict[str, Path]:
    return {
        "runtime_cases": DATA / "runtime_cases.parquet",
        "evaluator_truth": DATA / "evaluator_truth.parquet",
        "evidence_corpus": DATA / "evidence_corpus.parquet",
        "authorization_state": DATA / "authorization_state.parquet",
        "dataset_manifest": DATA / "manifest.json",
        "stratum_manifest": DATA / "stratum_manifest.json",
        "denominator_contract": ROOT / "configs/v24/denominator_contract.yaml",
        "structural_admission": RESULTS / "structural_admission.json",
        "contamination_report": RESULTS / "contamination.json",
        "candidate_output": RESULTS / "candidate_results.parquet",
        "per_case_metrics": RESULTS / "aggregate_metrics.per_case.parquet",
        "aggregate_metrics": RESULTS / "aggregate_metrics.json",
        "denominator_report": RESULTS / "denominators.json",
        "request_attribution": RESULTS / "request_attribution.json",
        "request_diagnostics": ARTIFACTS / "request_diagnostics.partial.jsonl",
        "checkpoint": ARTIFACTS / "qualification.checkpoint.json",
        "durable_candidate_stream": ARTIFACTS / "qualification.partial.jsonl",
        "server_preflight": RESULTS / "server_preflight.json",
        "vllm_metrics": RESULTS / "vllm_metrics.json",
        "vllm_metrics_periodic": RESULTS / "vllm_metrics_periodic.json",
        "vllm_metrics_stream": RESULTS / "vllm_metrics_periodic.json.partial.jsonl",
        "nvidia_telemetry": RESULTS / "nvidia_telemetry.json",
        "nvidia_telemetry_stream": RESULTS / "nvidia_telemetry.json.partial.jsonl",
        "qualification_sqlite": ARTIFACTS / "qualification.sqlite",
        "sqlite_finalization": RESULTS / "sqlite_finalization.json",
        "sqlite_stability": RESULTS / "sqlite_stability.json",
        "ledger_verification": RESULTS / "ledger_verification.json",
        "ledger_snapshot": RESULTS / "ledger_snapshot.json",
        "environment_manifest": ROOT / "state/v24_environment_manifest.json",
        "wsl_dependency_inventory": ROOT / "state/v24_wsl_pip_freeze.txt",
        "gate_config": ROOT / "configs/v24/qualification_gates.yaml",
        "freeze_manifest": ROOT / "state/v24_freeze_manifest.json",
        "artifact_rules": ROOT / "configs/v24/artifact_rules.yaml",
    }


def _preconditions() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze = verify_freeze(ROOT)
    admission = json.loads((RESULTS / "structural_admission.json").read_text(encoding="utf-8"))
    reviews = json.loads((ROOT / "state/v24_prequalification_review_findings.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "ADMITTED_NOT_EXECUTED"
        or admission.get("status") != "ADMITTED"
        or manifest.get("freeze_hash") != freeze.get("freeze_hash")
        or reviews.get("status") != "PRE_QUALIFICATION_CLEAR"
        or reviews.get("counts", {}).get("unresolved_BLOCKER") != 0
        or reviews.get("counts", {}).get("unresolved_HIGH") != 0
    ):
        raise RuntimeError("V24_QUALIFICATION_PRECONDITION_INVALID")
    contamination = json.loads((RESULTS / "contamination.json").read_text(encoding="utf-8"))
    fresh_admission = structural_admission(DATA, ROOT, contamination)
    if fresh_admission != admission:
        raise RuntimeError("V24_QUALIFICATION_ADMISSION_DRIFT")
    return manifest, freeze


def _checkpoint_fields(
    paths: dict[str, Path], bundle: Any, serving: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    commit, dirty = git_state(ROOT)
    return {
        "git_commit": commit,
        "dirty_state_hash": dirty,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(paths["runtime_cases"]),
        "evidence_corpus_hash": sha256_file(paths["evidence_corpus"]),
        "authorization_state_hash": sha256_file(paths["authorization_state"]),
        "candidate_bundle_hash": freeze["freeze_hash"],
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
        "evaluator_protocol_hash": evaluator_protocol_hash(ROOT),
        "gate_config_hash": sha256_file(paths["gate_config"]),
        "qualification_gate_freeze_hash": sha256_file(paths["freeze_manifest"]),
        "seed": 24041,
        "concurrency": 2,
    }


def _run_subprocess(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)


def run() -> None:
    manifest, freeze = _preconditions()
    paths = _paths()
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    args = argparse.Namespace(
        server_profile="v23",
        prefix_cache=True,
        performance_mode="interactivity",
        optimization_level=2,
        batched_tokens=2048,
        run_role="qualification",
    )
    live = _verify_server(serving, args)
    gpu = verify_gpu_ownership(ROOT, int(live["pid"]))
    cold = _scrape(live["endpoint_root"])
    _assert_cold_server(cold)
    atomic_write_json(
        paths["server_preflight"],
        {"schema_version": 1, "server_provenance": live, "cold_metrics": cold, "gpu_ownership": gpu},
    )
    manifest.update(
        {"status": "QUALIFICATION_EXECUTION_STARTED", "qualification_executed": True, "started_at": utc_now()}
    )
    atomic_write_json(MANIFEST, manifest)
    client = AttributedVllmClient(
        endpoint=f"{live['endpoint_root']}/v1/chat/completions",
        model=serving["served_model_name"],
        schema=json.loads((ROOT / "configs/v23/explanation_schema_minimal.json").read_text(encoding="utf-8")),
        prompt_template=(ROOT / "configs/v23/explanation_prompt_minimal.txt").read_text(encoding="utf-8"),
        max_tokens=128,
        contract="minimal",
        diagnostics_path=paths["request_diagnostics"],
    )
    _warmup(client, 8)
    metric_start = _scrape(live["endpoint_root"])
    runtime = pd.read_parquet(paths["runtime_cases"])
    issuer = load_frozen_approval_issuer(ROOT)
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=paths["evidence_corpus"],
        authorization_path=paths["authorization_state"],
        ledger_path=paths["qualification_sqlite"],
        explanation_client=client,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-v24-qualification",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    fields = _checkpoint_fields(paths, bundle, serving, freeze)
    with (
        NvidiaSampler(paths["nvidia_telemetry"], interval_seconds=1.0),
        PrometheusSampler(live["endpoint_root"], paths["vllm_metrics_periodic"], interval_seconds=5.0),
    ):
        runner.run_dataset(
            runtime,
            output_path=paths["candidate_output"],
            partial_path=paths["durable_candidate_stream"],
            checkpoint_path=paths["checkpoint"],
            checkpoint_fields=fields,
            concurrency=2,
        )
    metric_end = _scrape(live["endpoint_root"])
    actual_concurrency = _maximum_overlap(client.diagnostics)
    if actual_concurrency != 2:
        raise RuntimeError(f"V24_CONCURRENCY_INVALID:{actual_concurrency}")
    atomic_write_json(
        paths["request_attribution"],
        {"schema_version": 1, "requests": [{**row, **_server_metrics(row)} for row in client.diagnostics]},
    )
    atomic_write_json(
        paths["vllm_metrics"],
        {
            "schema_version": 1,
            "cold_pre_warmup": cold,
            "measurement_start": metric_start,
            "measurement_end": metric_end,
        },
    )
    diagnostics = verify_request_diagnostics(
        ROOT,
        paths["request_diagnostics"],
        paths["durable_candidate_stream"],
        paths["candidate_output"],
        paths["request_attribution"],
        600,
    )
    if not diagnostics["valid"]:
        raise RuntimeError(f"V24_REQUEST_DIAGNOSTICS_INVALID:{diagnostics['violations']}")
    # Evaluator-only truth is opened only after all candidate output is closed.
    dataset_manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    if sha256_file(paths["evaluator_truth"]) != dataset_manifest["truth_sha256"]:
        raise RuntimeError("V24_EVALUATOR_TRUTH_MUTATED")
    metrics = evaluate(
        paths["candidate_output"],
        paths["evaluator_truth"],
        paths["qualification_sqlite"],
        paths["aggregate_metrics"],
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="QUALIFICATION",
    )
    checkpoint = validate_checkpoint(paths["checkpoint"], fields)
    records = [
        json.loads(line) for line in paths["durable_candidate_stream"].read_text(encoding="utf-8").splitlines() if line
    ]
    checkpoint_violations = int(
        checkpoint["completed_records_sha256"] != records_hash(records)
        or checkpoint["completed_case_ids"] != [str(row["case_id"]) for row in records]
    )
    ledger = verify_ledger(paths["qualification_sqlite"])
    atomic_write_json(paths["ledger_verification"], ledger)
    snapshot(paths["qualification_sqlite"], paths["ledger_snapshot"])
    assert_all_closed(paths["qualification_sqlite"])
    _run_subprocess(
        "controlflow.v24.sqlite_finalization",
        str(paths["qualification_sqlite"]),
        str(paths["sqlite_finalization"]),
        "--expected-events",
        "600",
    )
    strata = yaml.safe_load((ROOT / "configs/v24/strata.yaml").read_text(encoding="utf-8"))
    _run_subprocess(
        "controlflow.v24.hash_stability",
        str(paths["qualification_sqlite"]),
        str(paths["sqlite_stability"]),
        "--interval-seconds",
        str(strata["stability_interval_seconds"]),
    )
    denominators = recompute(
        ROOT,
        DATA,
        paths["candidate_output"],
        paths["per_case_metrics"],
        paths["aggregate_metrics"],
        paths["ledger_snapshot"],
        paths["denominator_report"],
        critical_threshold=bundle.threshold,
    )
    graph_path = ROOT / "state/v24_qualification_bindings.json"
    make_graph(ROOT, paths, graph_path, role="qualification")
    unbound = scan_to_report(ROOT, graph_path, ROOT / "state/v24_unbound_artifacts.json")
    if unbound["status"] != "CLEAN":
        raise RuntimeError("UNBOUND_SUBSTANTIVE_ARTIFACT")
    postclose_path = ROOT / "state/v24_postclose_verification.json"
    _run_subprocess("controlflow.v24.postclose", str(ROOT), str(graph_path), str(postclose_path))
    postclose = json.loads(postclose_path.read_text(encoding="utf-8"))
    gates_config = yaml.safe_load(paths["gate_config"].read_text(encoding="utf-8"))
    review = json.loads((ROOT / "state/v24_prequalification_review_findings.json").read_text(encoding="utf-8"))
    gates = evaluate_gates(
        gates_config,
        metrics,
        denominators,
        leakage_findings=0,
        checkpoint_violations=checkpoint_violations,
        artifact_binding_violations=len(postclose["failures"]),
        ledger_failures=len(ledger["failures"]),
        unresolved_blocker=review["counts"]["unresolved_BLOCKER"],
        unresolved_high=review["counts"]["unresolved_HIGH"],
    )
    decision_path = RESULTS / "gate_decision.json"
    atomic_write_json(decision_path, gates)
    receipt_path = ROOT / "state/v24_qualification_closure_receipt.json"
    create_receipt(ROOT, graph_path, receipt_path)
    _run_subprocess("controlflow.v24.closure_receipt", str(ROOT), str(graph_path), str(receipt_path))
    status = "COMPLETE" if gates["all_passed"] and denominators["status"] == "VALID" else "V24_DEVELOPMENT_NO_GO"
    proposed = ROOT / "state/v24_qualification_manifest.proposed.json"
    terminal_anchor = ROOT / "state/v24_qualification_terminal_anchor.json"
    terminal_manifest = {
        **manifest,
        "status": status,
        "completed_at": utc_now(),
        "actual_maximum_llm_concurrency": actual_concurrency,
        "artifact_graph_sha256": sha256_file(graph_path),
        "postclose_verification_sha256": sha256_file(postclose_path),
        "closure_receipt_sha256": sha256_file(receipt_path),
        "gate_decision_sha256": sha256_file(decision_path),
        "gates": gates,
        "all_passed": status == "COMPLETE",
    }
    atomic_write_json(proposed, terminal_manifest)
    create_anchor(ROOT, receipt_path, proposed, terminal_anchor)
    terminal_args = (str(ROOT), str(graph_path), str(receipt_path), str(proposed), str(terminal_anchor))
    _run_subprocess("controlflow.v24.terminal_decision", *terminal_args)
    atomic_write_json(MANIFEST, terminal_manifest)
    _run_subprocess("controlflow.v24.terminal_decision", *terminal_args, "--after")
    if status != "COMPLETE":
        raise RuntimeError("V24_DEVELOPMENT_NO_GO")


def main() -> None:
    try:
        run()
    except BaseException as exc:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("qualification_executed") and manifest.get("status") != "V24_DEVELOPMENT_NO_GO":
            manifest["status"] = "V24_DEVELOPMENT_NO_GO"
            manifest["failure_type"] = type(exc).__name__
            manifest["failure_message"] = str(exc)
            atomic_write_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()
