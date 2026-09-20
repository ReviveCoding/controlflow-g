"""V2.6 development rehearsal and one-shot sealed run through one closure path."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.checkpoint import git_state
from controlflow.v22.executor import verify_ledger
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v23.integrity import load_frozen_approval_issuer, verify_request_diagnostics
from controlflow.v23.telemetry import NvidiaSampler, PrometheusSampler
from controlflow.v23.vllm_client import AttributedVllmClient
from controlflow.v24.gates import evaluate_gates
from controlflow.v24.gpu_preflight import verify_gpu_ownership
from controlflow.v24.ledger_snapshot import snapshot
from controlflow.v24.sqlite_lifecycle import assert_all_closed
from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v25.checkpoint_integrity import expectations_from_evidence, verify_checkpoint
from controlflow.v25.evaluation import evaluate
from controlflow.v26.admission import structural_admission
from controlflow.v26.artifact_closure import make_graph, scan_to_report
from controlflow.v26.closure_receipt import create_receipt, verify_receipt
from controlflow.v26.denominators import recompute
from controlflow.v26.freeze_validation import verify_freeze
from controlflow.v26.terminal_decision import create_anchor, verify_terminal
from controlflow.v26.transition import verify_transition

_legacy_server = importlib.import_module("v23_benchmark")
_assert_cold_server = _legacy_server._assert_cold_server
_maximum_overlap = _legacy_server._maximum_overlap
_scrape = _legacy_server._scrape
_server_metrics = _legacy_server._server_metrics
_verify_server = _legacy_server._verify_server
_warmup = _legacy_server._warmup

ROOT = Path(__file__).resolve().parents[1]


def _locations(role: str, development_seed: int | None = None) -> tuple[Path, Path, Path, Path]:
    if role == "development":
        if development_seed is None:
            raise ValueError("development seed required")
        return (
            ROOT / f"data/v26/development/rehearsal_{development_seed}",
            ROOT / f"results/v26/development/rehearsal_{development_seed}",
            ROOT / f"artifacts/v26/development/rehearsal_{development_seed}",
            ROOT / f"state/v26_development_rehearsal_{development_seed}_manifest.json",
        )
    if role == "qualification":
        return (
            ROOT / "data/v26/qualification/V26QUAL",
            ROOT / "results/v26/qualification",
            ROOT / "artifacts/v26/qualification",
            ROOT / "state/v26_qualification_manifest.json",
        )
    if role == "final":
        return (
            ROOT / "data/v26/final/V26FINAL",
            ROOT / "results/v26/final",
            ROOT / "artifacts/v26/final",
            ROOT / "state/v26_final_manifest.json",
        )
    raise ValueError("unsupported run role")


def _paths(role: str, development_seed: int | None = None) -> dict[str, Path]:
    data, results, artifacts, _ = _locations(role, development_seed)
    prefix = {"development": "rehearsal", "qualification": "qualification", "final": "final"}[role]
    identity = f"development_rehearsal_{development_seed}" if role == "development" else role
    return {
        "runtime_cases": data / "runtime_cases.parquet",
        "evaluator_truth": data / "evaluator_truth.parquet",
        "evidence_corpus": data / "evidence_corpus.parquet",
        "authorization_state": data / "authorization_state.parquet",
        "dataset_manifest": data / "manifest.json",
        "stratum_manifest": data / "stratum_manifest.json",
        "denominator_contract": ROOT
        / "configs/v26"
        / ("rehearsal_denominator_contract.yaml" if role == "development" else "denominator_contract.yaml"),
        "structural_admission": results / "structural_admission.json",
        "contamination_report": results / "contamination.json",
        "contamination_recheck": results / "contamination_recheck.json",
        "transition_receipt": results / "transition_receipt.json",
        "prior_inventory": results / "prior_inventory.json",
        "candidate_output": results / "candidate_results.parquet",
        "per_case_metrics": results / "aggregate_metrics.per_case.parquet",
        "aggregate_metrics": results / "aggregate_metrics.json",
        "denominator_report": results / "denominators.json",
        "request_attribution": results / "request_attribution.json",
        "request_diagnostics": artifacts / "request_diagnostics.partial.jsonl",
        "checkpoint": artifacts / f"{prefix}.checkpoint.json",
        "durable_candidate_stream": artifacts / f"{prefix}.partial.jsonl",
        "server_preflight": results / "server_preflight.json",
        "vllm_metrics": results / "vllm_metrics.json",
        "vllm_metrics_periodic": results / "vllm_metrics_periodic.json",
        "vllm_metrics_stream": results / "vllm_metrics_periodic.json.partial.jsonl",
        "nvidia_telemetry": results / "nvidia_telemetry.json",
        "nvidia_telemetry_stream": results / "nvidia_telemetry.json.partial.jsonl",
        "qualification_sqlite": artifacts / f"{prefix}.sqlite",
        "sqlite_finalization": results / "sqlite_finalization.json",
        "sqlite_stability": results / "sqlite_stability.json",
        "ledger_verification": results / "ledger_verification.json",
        "ledger_snapshot": results / "ledger_snapshot.json",
        "environment_manifest": ROOT / f"state/v26_{identity}_environment_manifest.json",
        "candidate_manifest": ROOT / "state/v24_candidate_manifest.json",
        "gate_config": ROOT
        / "configs/v26"
        / ("development_gates.yaml" if role == "development" else "qualification_gates.yaml"),
        "freeze_manifest": ROOT
        / (
            f"state/v26_development_rehearsal_{development_seed}_protocol.json"
            if role == "development"
            else "state/v26_final_freeze_manifest.json"
            if role == "final"
            else "state/v26_freeze_manifest.json"
        ),
        "artifact_rules": ROOT / "configs/v26/artifact_rules.yaml",
    }


def _preconditions(
    role: str, paths: dict[str, Path], development_seed: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, _, _, manifest_path = _locations(role, development_seed)
    freeze = verify_freeze(ROOT, role=role, development_seed=development_seed)
    admission = json.loads(paths["structural_admission"].read_text(encoding="utf-8"))
    contamination = json.loads(paths["contamination_report"].read_text(encoding="utf-8"))
    fresh_admission = structural_admission(data, ROOT, contamination, role=role.upper())
    if fresh_admission != admission:
        raise RuntimeError("V26_QUALIFICATION_ADMISSION_DRIFT")
    if admission.get("status") != "ADMITTED":
        raise RuntimeError("V26_ADMISSION_INVALID")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_manifest_sha256") != sha256_file(paths["dataset_manifest"]):
        raise RuntimeError("V26_DATASET_GENERATION_BINDING_INVALID")
    if sha256_file(paths["contamination_recheck"]) != manifest.get("fresh_contamination_sha256") or sha256_file(
        paths["transition_receipt"]
    ) != manifest.get("transition_receipt_sha256"):
        raise RuntimeError("V26_TRANSITION_EVIDENCE_MUTATED")
    transition = verify_transition(
        ROOT,
        candidate_path=paths["runtime_cases"],
        config_path=ROOT / "configs/v26/prior_runtime_manifest.yaml",
        inventory_path=paths["prior_inventory"],
        stored_path=paths["contamination_report"],
        fresh_path=paths["contamination_recheck"],
        manifest_path=manifest_path,
        receipt_path=paths["transition_receipt"],
        persist=False,
    )
    recorded_transition = json.loads(paths["transition_receipt"].read_text(encoding="utf-8"))
    if transition["status"] != "PASS" or transition != recorded_transition:
        raise RuntimeError(f"V26_HOLDOUT_GENERATION_BINDING_INVALID:{transition['failures']}")
    if role == "development":
        if manifest.get("status") != "TRANSITION_PASS":
            raise RuntimeError("V26_DEVELOPMENT_TRANSITION_NOT_ADMITTED")
    else:
        reviews_path = ROOT / (
            "state/v26_postqualification_review_findings.json"
            if role == "final"
            else "state/v26_prequalification_review_findings.json"
        )
        reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
        expected_review_status = "POST_QUALIFICATION_CLEAR" if role == "final" else "PRE_QUALIFICATION_CLEAR"
        if (
            manifest.get("status") != "ADMITTED_NOT_EXECUTED"
            or manifest.get("freeze_hash") != freeze.get("freeze_hash")
            or reviews.get("status") != expected_review_status
            or reviews.get("counts", {}).get("unresolved_BLOCKER") != 0
            or reviews.get("counts", {}).get("unresolved_HIGH") != 0
        ):
            raise RuntimeError("V26_QUALIFICATION_PRECONDITION_INVALID")
        if role == "final":
            qualification = json.loads((ROOT / "state/v26_qualification_manifest.json").read_text(encoding="utf-8"))
            qualification_path = ROOT / "state/v26_qualification_manifest.json"
            graph = ROOT / "state/v26_qualification_bindings.json"
            receipt = ROOT / "state/v26_qualification_closure_receipt.json"
            proposed = ROOT / "state/v26_qualification_manifest.proposed.json"
            anchor = ROOT / "state/v26_qualification_terminal_anchor.json"
            postclose = json.loads((ROOT / "state/v26_postclose_verification.json").read_text(encoding="utf-8"))
            decision = json.loads((ROOT / "results/v26/qualification/gate_decision.json").read_text(encoding="utf-8"))
            verify_freeze(ROOT, role="qualification")
            if (
                qualification.get("status") != "COMPLETE"
                or qualification.get("all_passed") is not True
                or decision.get("all_passed") is not True
                or postclose.get("status") != "PASS"
                or reviews.get("qualification_manifest_sha256") != sha256_file(qualification_path)
                or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
                or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
            ):
                raise RuntimeError("V26_FINAL_REQUIRES_SEALED_ADMISSIBLE_QUALIFICATION")
    if manifest.get("execution_started") or any(
        paths[name].exists()
        for name in (
            "candidate_output",
            "checkpoint",
            "durable_candidate_stream",
            "qualification_sqlite",
            "request_diagnostics",
            "server_preflight",
        )
    ):
        raise RuntimeError("V26_ONE_SHOT_CANDIDATE_ALREADY_OPENED")
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
        "evaluator_protocol_hash": sha256_file(ROOT / "src/controlflow/v25/evaluation.py"),
        "gate_config_hash": sha256_file(paths["gate_config"]),
        "qualification_gate_freeze_hash": sha256_file(paths["freeze_manifest"]),
        "seed": json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))["seed"],
        "concurrency": 2,
    }


def _run_subprocess(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)


def run(role: str, development_seed: int | None = None) -> None:
    data, results, artifacts, manifest_path = _locations(role, development_seed)
    paths = _paths(role, development_seed)
    manifest, freeze = _preconditions(role, paths, development_seed)
    artifacts.mkdir(parents=True, exist_ok=True)
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    args = argparse.Namespace(
        server_profile="v23",
        prefix_cache=True,
        performance_mode="interactivity",
        optimization_level=2,
        batched_tokens=2048,
        run_role=role,
    )
    live = _verify_server(serving, args)
    gpu = verify_gpu_ownership(ROOT, int(live["pid"]))
    cold = _scrape(live["endpoint_root"])
    _assert_cold_server(cold)
    atomic_write_json(
        paths["server_preflight"],
        {"schema_version": 1, "server_provenance": live, "cold_metrics": cold, "gpu_ownership": gpu},
    )
    atomic_write_json(
        paths["environment_manifest"],
        {"schema_version": 1, "role": role, "python": sys.version, "platform": platform.platform(), "server": live},
    )
    manifest.update(
        {"status": "EXECUTION_STARTED", "one_shot_opened": True, "execution_started": True, "started_at": utc_now()}
    )
    atomic_write_json(manifest_path, manifest)
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
                reviewer_id=f"external-reviewer-v26-{role}",
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
            checkpoint_contract=True,
        )
    metric_end = _scrape(live["endpoint_root"])
    actual_concurrency = _maximum_overlap(client.diagnostics)
    if actual_concurrency != 2:
        raise RuntimeError(f"V26_CONCURRENCY_INVALID:{actual_concurrency}")
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
        len(runtime),
    )
    if not diagnostics["valid"]:
        raise RuntimeError(f"V26_REQUEST_DIAGNOSTICS_INVALID:{diagnostics['violations']}")
    # Evaluator-only truth is opened only after all candidate output is closed.
    dataset_manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    if sha256_file(paths["evaluator_truth"]) != dataset_manifest["truth_sha256"]:
        raise RuntimeError("V26_EVALUATOR_TRUTH_MUTATED")
    metrics = evaluate(
        paths["candidate_output"],
        paths["evaluator_truth"],
        paths["qualification_sqlite"],
        paths["aggregate_metrics"],
        critical_threshold=bundle.threshold,
        concurrency=2,
        role=role.upper(),
    )
    checkpoint_view = CheckpointEnvelope.load(paths["checkpoint"])
    checkpoint_expected = expectations_from_evidence(
        candidate_threshold=float(
            json.loads(paths["candidate_manifest"].read_text(encoding="utf-8"))["critical_threshold"]
        ),
        candidate_bundle_hash=str(freeze["freeze_hash"]),
        gate_config_path=paths["gate_config"],
        freeze_path=paths["freeze_manifest"],
        runtime_path=paths["runtime_cases"],
        evidence_path=paths["evidence_corpus"],
        authorization_path=paths["authorization_state"],
        qwen_revision=str(freeze["qwen_revision"]),
        vllm_version=str(freeze["vllm_version"]),
        seed=int(dataset_manifest["seed"]),
        sample_count=len(runtime),
    )
    checkpoint_violations = len(
        verify_checkpoint(paths["checkpoint"], paths["durable_candidate_stream"], checkpoint_expected)["failures"]
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
        str(len(runtime)),
    )
    strata_name = "rehearsal_strata.yaml" if role == "development" else "strata.yaml"
    strata = yaml.safe_load((ROOT / "configs/v26" / strata_name).read_text(encoding="utf-8"))
    _run_subprocess(
        "controlflow.v24.hash_stability",
        str(paths["qualification_sqlite"]),
        str(paths["sqlite_stability"]),
        "--interval-seconds",
        str(strata["stability_interval_seconds"]),
    )
    denominators = recompute(
        ROOT,
        data,
        paths["candidate_output"],
        paths["per_case_metrics"],
        paths["aggregate_metrics"],
        paths["ledger_snapshot"],
        paths["denominator_report"],
        critical_threshold=checkpoint_view.critical_threshold,
        role=role.upper(),
    )
    identity = f"development_rehearsal_{development_seed}" if role == "development" else role
    graph_path = ROOT / f"state/v26_{identity}_bindings.json"
    make_graph(ROOT, paths, graph_path, role=role, identity=identity)
    unbound_path = ROOT / (
        "state/v26_unbound_artifacts.json"
        if role == "qualification"
        else f"state/v26_{identity}_unbound_artifacts.json"
    )
    unbound = scan_to_report(ROOT, graph_path, unbound_path)
    if unbound["status"] != "CLEAN":
        raise RuntimeError("UNBOUND_SUBSTANTIVE_ARTIFACT")
    postclose_path = ROOT / (
        "state/v26_postclose_verification.json"
        if role == "qualification"
        else f"state/v26_{identity}_postclose_verification.json"
    )
    _run_subprocess("controlflow.v26.postclose", str(ROOT), str(graph_path), str(postclose_path))
    postclose = json.loads(postclose_path.read_text(encoding="utf-8"))
    if role == "development":
        checks = {
            "postclose": postclose["status"] == "PASS",
            "denominators": denominators["status"] == "VALID",
            "checkpoint": checkpoint_violations == 0,
            "ledger": not ledger["failures"],
            "artifact_bindings": not postclose["failures"],
            "actual_concurrency": actual_concurrency == 2,
        }
        gates = {
            "role": "DEVELOPMENT_ONLY",
            "gates": {name: {"passed": passed} for name, passed in checks.items()},
            "all_passed": all(checks.values()),
            "performance_diagnostic_only": metrics,
        }
    else:
        gates_config = yaml.safe_load(paths["gate_config"].read_text(encoding="utf-8"))
        review_path = ROOT / (
            "state/v26_postqualification_review_findings.json"
            if role == "final"
            else "state/v26_prequalification_review_findings.json"
        )
        review = json.loads(review_path.read_text(encoding="utf-8"))
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
    decision_path = results / "gate_decision.json"
    atomic_write_json(decision_path, gates)
    receipt_path = ROOT / f"state/v26_{identity}_closure_receipt.json"
    create_receipt(ROOT, graph_path, receipt_path)
    _run_subprocess("controlflow.v26.closure_receipt", str(ROOT), str(graph_path), str(receipt_path))
    status = (
        "DEVELOPMENT_REHEARSAL_PASS"
        if role == "development" and gates["all_passed"]
        else "COMPLETE"
        if gates["all_passed"] and denominators["status"] == "VALID"
        else "V26_DEVELOPMENT_NO_GO"
        if role == "development"
        else "NO_PROMOTE"
    )
    proposed = ROOT / f"state/v26_{identity}_manifest.proposed.json"
    terminal_anchor = ROOT / f"state/v26_{identity}_terminal_anchor.json"
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
        "all_passed": status in {"COMPLETE", "DEVELOPMENT_REHEARSAL_PASS"},
    }
    atomic_write_json(proposed, terminal_manifest)
    create_anchor(ROOT, receipt_path, proposed, terminal_anchor)
    terminal_args = (str(ROOT), str(graph_path), str(receipt_path), str(proposed), str(terminal_anchor))
    _run_subprocess("controlflow.v26.terminal_decision", *terminal_args)
    atomic_write_json(manifest_path, terminal_manifest)
    _run_subprocess("controlflow.v26.terminal_decision", *terminal_args, "--after")
    if status not in {"COMPLETE", "DEVELOPMENT_REHEARSAL_PASS"}:
        raise RuntimeError("V26_DEVELOPMENT_NO_GO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("development", "qualification", "final"), required=True)
    parser.add_argument("--development-seed", type=int)
    args = parser.parse_args()
    if (args.role == "development") != (args.development_seed is not None):
        parser.error("--development-seed is required only for development")
    _, _, _, manifest_path = _locations(args.role, args.development_seed)
    try:
        run(args.role, args.development_seed)
    except BaseException as exc:
        if not manifest_path.is_file():
            raise
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        terminal_failure = "V26_DEVELOPMENT_NO_GO" if args.role == "development" else "NO_PROMOTE"
        if (args.role != "development" or manifest.get("execution_started")) and manifest.get(
            "status"
        ) != terminal_failure:
            manifest["status"] = terminal_failure
            manifest["failure_type"] = type(exc).__name__
            manifest["failure_message"] = str(exc)
            atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
