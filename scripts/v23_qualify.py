from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from v23_benchmark import _assert_cold_server, _maximum_overlap, _scrape, _server_metrics, _verify_server, _warmup

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.checkpoint import git_state, records_hash, validate_checkpoint
from controlflow.v22.dgp import contamination_against_prior, resolve_prior_runtime_paths
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.executor import verify_ledger
from controlflow.v22.gates import apply_gates
from controlflow.v22.run_lock import ExecutionLock
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v23.dgp import generate_v23_split
from controlflow.v23.integrity import (
    file_binding,
    verify_file_bindings,
    verify_hash_bindings,
    verify_historical_boundary,
    verify_request_diagnostics,
)
from controlflow.v23.telemetry import PERIODIC_METRICS, NvidiaSampler, PrometheusSampler
from controlflow.v23.vllm_client import AttributedVllmClient

ROOT = Path(__file__).resolve().parents[1]


def _committed_exact(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    tracked = (
        subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0
    )
    if not tracked:
        return False
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True).stdout
    return committed == path.read_bytes()


def _verified_preconditions() -> tuple[dict, dict, dict, dict]:
    qualification_path = ROOT / "state/v23_qualification_manifest.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("qualification_executed") or qualification.get("one_shot_opened"):
        raise RuntimeError("V23 qualification one-shot has already been opened")
    reviews = json.loads((ROOT / "state/v23_prequalification_review_findings.json").read_text(encoding="utf-8"))
    if reviews.get("status") != "PRE_QUALIFICATION_CLEAR" or reviews.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("QUALIFICATION_PROHIBITED: independent reviews have not cleared")
    integrity = json.loads((ROOT / "state/v23_integrity.json").read_text(encoding="utf-8"))
    unsigned_integrity = {key: value for key, value in integrity.items() if key != "integrity_sha256"}
    integrity_valid = (
        integrity.get("integrity_sha256") == hashlib.sha256(canonical_json(unsigned_integrity)).hexdigest()
    )
    if not integrity.get("qualification_eligible") or not integrity_valid:
        raise RuntimeError("QUALIFICATION_PROHIBITED: integrity is not eligible")
    gates = yaml.safe_load((ROOT / "configs/v23/qualification_gates.yaml").read_text(encoding="utf-8"))
    gate_freeze_path = ROOT / "configs/v23/qualification_gate_freeze.json"
    gate_freeze = json.loads(gate_freeze_path.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "state/v23_freeze_manifest.json").read_text(encoding="utf-8"))
    unsigned_freeze = {key: value for key, value in freeze.items() if key not in {"freeze_hash", "created_at"}}
    freeze_hash_valid = freeze.get("freeze_hash") == hashlib.sha256(canonical_json(unsigned_freeze)).hexdigest()
    binding_failures = verify_file_bindings(ROOT, [freeze["dependencies"], *freeze.get("bindings", [])])
    typed = json.loads((ROOT / "state/v23_typed_core_manifest.json").read_text(encoding="utf-8"))
    typed_failures = verify_hash_bindings(ROOT, [typed["source_bundle"], *typed["bindings"].values()])
    historical = verify_historical_boundary(ROOT, ROOT / "state/v23_historical_boundary.json")
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    serving_mismatch = serving["model"] != freeze["qwen"]["model"] or serving["revision"] != freeze["qwen"]["revision"]
    source_is_ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD"], cwd=ROOT).returncode
        == 0
    )
    executable_bindings = [
        item
        for item in freeze.get("bindings", [])
        if str(item["path"]).startswith(("configs/", "scripts/", "src/", "state/"))
    ]
    executable_committed = all(_committed_exact(ROOT / item["path"]) for item in executable_bindings)
    if not gates.get("frozen_before_qualification"):
        raise RuntimeError("QUALIFICATION_PROHIBITED: gates are not frozen")
    if (
        gate_freeze.get("status") != "FROZEN_BEFORE_QUALIFICATION"
        or gate_freeze.get("gate_config_sha256") != sha256_file(ROOT / "configs/v23/qualification_gates.yaml")
        or gate_freeze.get("qualification_results_present_at_freeze") is not False
        or not _committed_exact(gate_freeze_path)
        or not _committed_exact(ROOT / "state/v23_prequalification_review_findings.json")
        or not _committed_exact(ROOT / "state/v23_integrity.json")
        or not _committed_exact(ROOT / "state/v23_freeze_manifest.json")
        or freeze.get("status") != "QUALIFICATION_PROTOCOL_FROZEN"
        or not freeze_hash_valid
        or binding_failures
        or typed_failures
        or not historical["valid"]
        or serving_mismatch
        or not source_is_ancestor
        or not executable_committed
        or gate_freeze.get("protocol_freeze_hash") != freeze.get("freeze_hash")
    ):
        raise RuntimeError("QUALIFICATION_PROHIBITED: freeze evidence mismatch")
    return reviews, integrity, gates, freeze


def main() -> None:
    reviews, integrity, gates, freeze = _verified_preconditions()
    qualification_path = ROOT / "state/v23_qualification_manifest.json"
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    selected = freeze["serving"]
    namespace = argparse.Namespace(
        server_profile="v23",
        prefix_cache=bool(selected["enable_prefix_caching"]),
        performance_mode=selected["performance_mode"],
        optimization_level=int(selected["optimization_level"]),
        batched_tokens=int(selected["max_num_batched_tokens"]),
        run_role="qualification",
    )
    live = _verify_server(serving, namespace)
    metrics_cold = _scrape(live["endpoint_root"])
    _assert_cold_server(metrics_cold)
    preflight_path = ROOT / "results/v23/qualification_server_preflight.json"
    atomic_write_json(
        preflight_path,
        {"schema_version": 1, "created_at": utc_now(), "server_provenance": live, "cold_metrics": metrics_cold},
    )
    directory = ROOT / "data/v23/qualification/V23QUAL"
    prior_paths = resolve_prior_runtime_paths(
        ROOT,
        ROOT / "configs/v23/prior_runtime_manifest.yaml",
        excluded_directory=directory,
    )
    atomic_write_json(
        qualification_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "QUALIFICATION_EXECUTION_STARTED",
            "one_shot_opened": True,
            "qualification_executed": False,
            "partial_aggregate_inspection_prohibited": True,
            "server_preflight": file_binding(ROOT, preflight_path),
        },
    )
    manifest = generate_v23_split(directory, role="QUALIFICATION", count=600, seed=23907, prefix="V23QUAL", root=ROOT)
    runtime_path = directory / "runtime_cases.parquet"
    truth_path = directory / "evaluator_truth.parquet"
    evidence_path = directory / "evidence_corpus.parquet"
    authorization_path = directory / "authorization_state.parquet"
    for path, field in (
        (runtime_path, "runtime_sha256"),
        (evidence_path, "evidence_sha256"),
        (authorization_path, "authorization_sha256"),
    ):
        if sha256_file(path) != manifest[field]:
            raise RuntimeError("QUALIFICATION_INVALID: generated dataset hash mismatch")
    contamination = contamination_against_prior(runtime_path, prior_paths)
    contamination_path = ROOT / "results/v23/qualification_contamination.json"
    atomic_write_json(contamination_path, contamination)
    if contamination["leakage_findings"]:
        atomic_write_json(
            qualification_path,
            {
                "schema_version": 1,
                "status": "INVALID_CONTAMINATED_NOT_EXECUTED",
                "one_shot_opened": True,
                "qualification_executed": False,
                "dataset": manifest,
                "contamination": contamination,
            },
        )
        raise RuntimeError("QUALIFICATION_PROHIBITED: contamination detected")

    client = AttributedVllmClient(
        endpoint=f"{live['endpoint_root']}/v1/chat/completions",
        model=serving["served_model_name"],
        schema=json.loads((ROOT / selected["schema_path"]).read_text(encoding="utf-8")),
        prompt_template=(ROOT / selected["prompt_path"]).read_text(encoding="utf-8"),
        max_tokens=int(selected["max_tokens"]),
        contract="minimal",
        diagnostics_path=ROOT / "artifacts/v23/qualification_request_diagnostics.partial.jsonl",
    )
    _warmup(client, int(freeze["warmup_protocol"]["request_count"]))
    metrics_start = _scrape(live["endpoint_root"])
    runtime = pd.read_parquet(runtime_path)
    private_key = serialization.load_pem_private_key(
        (ROOT / "artifacts/v22/local_keys/approval_ed25519.private.pem").read_bytes(), password=None
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("frozen approval private key is not Ed25519")
    issuer = ApprovalIssuer(private_key, key_id="v22-dev-reviewer")
    ledger = ROOT / "artifacts/v23/qualification.sqlite"
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
                reviewer_id="external-reviewer-v23-qualification",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    git_commit, dirty_hash = git_state(ROOT)
    gate_freeze_path = ROOT / "configs/v23/qualification_gate_freeze.json"
    checkpoint_fields = {
        "git_commit": git_commit,
        "dirty_state_hash": dirty_hash,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(runtime_path),
        "evidence_corpus_hash": sha256_file(evidence_path),
        "authorization_state_hash": sha256_file(authorization_path),
        "candidate_bundle_hash": sha256_file(ROOT / "state/v23_freeze_manifest.json"),
        "model_hashes": {
            name: bundle.payload[name]["sha256"]
            for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
        },
        "critical_threshold": bundle.threshold,
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "structured_backend": serving["structured_output_backend"],
        "prompt_schema_hashes": {
            "prompt": sha256_file(ROOT / selected["prompt_path"]),
            "schema": sha256_file(ROOT / selected["schema_path"]),
        },
        "retrieval_config_hash": bundle.payload["retrieval_config"]["sha256"],
        "temporal_config_hash": bundle.payload["temporal_config"]["sha256"],
        "pdp_action_registry_hashes": {
            "pdp": bundle.payload["policy_config"]["sha256"],
            "actions": bundle.payload["action_registry"]["sha256"],
        },
        "evaluator_protocol_hash": evaluator_protocol_hash(ROOT),
        "gate_config_hash": sha256_file(ROOT / "configs/v23/qualification_gates.yaml"),
        "qualification_gate_freeze_hash": sha256_file(gate_freeze_path),
        "seed": 23907,
        "concurrency": 2,
    }
    output_path = ROOT / "results/v23/qualification_candidate_results.parquet"
    telemetry_path = ROOT / "results/v23/qualification_nvidia_telemetry.json"
    periodic_metrics_path = ROOT / "results/v23/qualification_vllm_metrics_periodic.json"
    with (
        NvidiaSampler(telemetry_path, interval_seconds=1.0),
        PrometheusSampler(live["endpoint_root"], periodic_metrics_path, interval_seconds=5.0),
    ):
        runner.run_dataset(
            runtime,
            output_path=output_path,
            partial_path=ROOT / "artifacts/v23/qualification.partial.jsonl",
            checkpoint_path=ROOT / "artifacts/v23/qualification.checkpoint.json",
            checkpoint_fields=checkpoint_fields,
            concurrency=2,
        )
    metrics_end = _scrape(live["endpoint_root"])
    actual_concurrency = _maximum_overlap(client.diagnostics)
    if actual_concurrency != 2:
        raise RuntimeError(f"QUALIFICATION_INVALID: observed maximum LLM concurrency {actual_concurrency}")
    attribution_path = ROOT / "results/v23/qualification_request_attribution.json"
    atomic_write_json(
        attribution_path,
        {
            "schema_version": 1,
            "requests": [{**row, **_server_metrics(row)} for row in client.diagnostics],
        },
    )
    metrics_snapshots_path = ROOT / "results/v23/qualification_vllm_metrics.json"
    atomic_write_json(
        metrics_snapshots_path,
        {
            "schema_version": 1,
            "cold_pre_warmup": metrics_cold,
            "measurement_start": metrics_start,
            "measurement_end": metrics_end,
        },
    )
    # Evaluator truth remains unopened until candidate output has durably closed.
    if sha256_file(truth_path) != manifest["truth_sha256"]:
        raise RuntimeError("QUALIFICATION_INVALID: evaluator truth hash mismatch after candidate closure")
    metrics_path = ROOT / "results/v23/qualification_metrics.json"
    metrics = evaluate(
        output_path,
        truth_path,
        ledger,
        metrics_path,
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="QUALIFICATION",
    )
    partial_path = ROOT / "artifacts/v23/qualification.partial.jsonl"
    checkpoint_path = ROOT / "artifacts/v23/qualification.checkpoint.json"
    checkpoint = validate_checkpoint(checkpoint_path, checkpoint_fields)
    records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
    checkpoint_violations = int(
        checkpoint["completed_records_sha256"] != records_hash(records)
        or checkpoint["completed_case_ids"] != [str(row["case_id"]) for row in records]
    )
    ledger_integrity = verify_ledger(ledger)
    diagnostic_evidence = verify_request_diagnostics(
        ROOT,
        ROOT / "artifacts/v23/qualification_request_diagnostics.partial.jsonl",
        partial_path,
        output_path,
        attribution_path,
        600,
    )
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    periodic_metrics = json.loads(periodic_metrics_path.read_text(encoding="utf-8"))
    diagnostic_rows = [
        json.loads(line)
        for line in (ROOT / "artifacts/v23/qualification_request_diagnostics.partial.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    request_start = min(float(row["request_start"]) for row in diagnostic_rows)
    request_end = max(float(row["request_end"]) for row in diagnostic_rows)
    if not diagnostic_evidence["valid"]:
        raise RuntimeError(f"QUALIFICATION_INVALID: request diagnostics {diagnostic_evidence['violations']}")
    if telemetry["errors"] or len(telemetry["samples"]) < 2:
        raise RuntimeError("QUALIFICATION_INVALID: NVIDIA telemetry coverage/errors")
    telemetry_epochs = [datetime.fromisoformat(row["sampled_at_utc"]).timestamp() for row in telemetry["samples"]]
    if min(telemetry_epochs) > request_start + 2 or max(telemetry_epochs) < request_end - 2:
        raise RuntimeError("QUALIFICATION_INVALID: NVIDIA telemetry measurement window gap")
    if periodic_metrics["errors"] or len(periodic_metrics["samples"]) < 2:
        raise RuntimeError("QUALIFICATION_INVALID: periodic vLLM metrics coverage/errors")
    observed_periodic_names = {
        str(sample["name"]) for snapshot in periodic_metrics["samples"] for sample in snapshot.get("samples", [])
    }
    if PERIODIC_METRICS - observed_periodic_names:
        raise RuntimeError("QUALIFICATION_INVALID: required periodic vLLM metrics missing")
    periodic_epochs = [datetime.fromisoformat(row["captured_at"]).timestamp() for row in periodic_metrics["samples"]]
    if min(periodic_epochs) > request_start + 6 or max(periodic_epochs) < request_end - 6:
        raise RuntimeError("QUALIFICATION_INVALID: periodic vLLM metrics measurement window gap")
    qualification_integrity = {
        **integrity,
        "leakage_findings": contamination["leakage_findings"],
        "checkpoint_violations": checkpoint_violations,
        "ledger_tamper_verification_failures": len(ledger_integrity["failures"]),
    }
    gate_result = apply_gates(
        metrics=metrics,
        serving={"actual_live_vllm": True, "concurrency": 2},
        integrity=qualification_integrity,
        reviews=reviews,
        gate_config=gates,
    )
    status = "PASS_POST_REVIEW_REQUIRED" if gate_result["all_passed"] else "V23_DEVELOPMENT_NO_GO"
    atomic_write_json(
        qualification_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": status,
            "one_shot": True,
            "one_shot_opened": True,
            "qualification_executed": True,
            "actual_maximum_llm_concurrency": actual_concurrency,
            "dataset": manifest,
            "contamination": contamination,
            "candidate_output_sha256": sha256_file(output_path),
            "metrics_sha256": sha256_file(metrics_path),
            "server_provenance": live,
            "diagnostic_evidence": diagnostic_evidence,
            "artifact_bindings": [
                file_binding(ROOT, path)
                for path in (
                    output_path,
                    preflight_path,
                    metrics_path,
                    attribution_path,
                    ROOT / "artifacts/v23/qualification_request_diagnostics.partial.jsonl",
                    metrics_snapshots_path,
                    periodic_metrics_path,
                    periodic_metrics_path.with_suffix(periodic_metrics_path.suffix + ".partial.jsonl"),
                    telemetry_path,
                    telemetry_path.with_suffix(telemetry_path.suffix + ".partial.jsonl"),
                    partial_path,
                    checkpoint_path,
                    ledger,
                )
            ],
            "ledger_integrity": ledger_integrity,
            "telemetry_sample_count": len(telemetry["samples"]),
            "periodic_vllm_metric_sample_count": len(periodic_metrics["samples"]),
            "frozen_gates_sha256": sha256_file(ROOT / "configs/v23/qualification_gates.yaml"),
            **gate_result,
        },
    )


if __name__ == "__main__":
    with ExecutionLock(ROOT / "state/v23_qualification_execution.lock"):
        main()
