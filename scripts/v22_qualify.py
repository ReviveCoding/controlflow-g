from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateModelBundle
from controlflow.v22.checkpoint import git_state
from controlflow.v22.dgp import contamination_against_prior, generate_split
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.gates import apply_gates
from controlflow.v22.integrity import verify_integrity_report
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v22.vllm_client import StructuredVllmClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    qualification_manifest_path = ROOT / "state/v22_qualification_manifest.json"
    prior_manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    if prior_manifest.get("qualification_executed"):
        raise RuntimeError("V22 qualification is one-shot and has already been executed")
    execution = json.loads((ROOT / "state/v22_execution_state.json").read_text(encoding="utf-8"))
    validation_ledger = ROOT / execution["validation_ledger"]["path"]
    integrity = verify_integrity_report(ROOT / "state/v22_integrity.json", root=ROOT, ledger_path=validation_ledger)
    reviews = json.loads((ROOT / "state/v22_review_findings.json").read_text(encoding="utf-8"))
    if not integrity.get("qualification_eligible") or reviews["counts"] != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("QUALIFICATION_PROHIBITED: integrity or independent review has not cleared")
    gates_config = yaml.safe_load((ROOT / "configs/v22/qualification_gates.yaml").read_text(encoding="utf-8"))
    if not gates_config.get("frozen_before_qualification"):
        raise RuntimeError("QUALIFICATION_PROHIBITED: gates are not frozen")
    gate_freeze_path = ROOT / "configs/v22/qualification_gate_freeze.json"
    if not gate_freeze_path.is_file():
        raise RuntimeError("QUALIFICATION_PROHIBITED: committed gate freeze evidence missing")
    gate_freeze = json.loads(gate_freeze_path.read_text(encoding="utf-8"))
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if (
        gate_freeze.get("status") != "FROZEN_BEFORE_QUALIFICATION"
        or gate_freeze.get("gate_config_sha256") != sha256_file(ROOT / "configs/v22/qualification_gates.yaml")
        or gate_freeze.get("qualification_results_present_at_freeze") is not False
        or subprocess.run(
            ["git", "ls-files", "--error-unmatch", "configs/v22/qualification_gate_freeze.json"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        != 0
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(gate_freeze.get("source_commit")), current_commit],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        != 0
        or subprocess.run(
            ["git", "show", "HEAD:configs/v22/qualification_gate_freeze.json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        != gate_freeze_path.read_bytes()
    ):
        raise RuntimeError("QUALIFICATION_PROHIBITED: gate freeze mismatch")
    serving = json.loads((ROOT / "state/v22_vllm_structured_c2.json").read_text(encoding="utf-8"))
    if not serving.get("actual_live_vllm") or serving.get("concurrency") != 2:
        raise RuntimeError("QUALIFICATION_PROHIBITED: live concurrency-2 serving evidence missing")
    directory = ROOT / "data/v22/qualification/V22QUAL"
    dataset_manifest_path = directory / "manifest.json"
    if prior_manifest.get("one_shot_opened"):
        if not dataset_manifest_path.is_file():
            raise RuntimeError("QUALIFICATION_INVALID: interrupted dataset generation")
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    else:
        atomic_write_json(
            qualification_manifest_path,
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "status": "QUALIFICATION_EXECUTION_STARTED",
                "one_shot_opened": True,
                "qualification_executed": False,
            },
        )
        manifest = generate_split(directory, role="QUALIFICATION", count=600, seed=22901, prefix="V22QUAL")
    runtime_path = directory / "runtime_cases.parquet"
    truth_path = directory / "evaluator_truth.parquet"
    evidence_path = directory / "evidence_corpus.parquet"
    for path, field in (
        (runtime_path, "runtime_sha256"),
        (truth_path, "truth_sha256"),
        (evidence_path, "evidence_sha256"),
        (directory / "authorization_state.parquet", "authorization_sha256"),
    ):
        if sha256_file(path) != manifest[field]:
            raise RuntimeError("QUALIFICATION_INVALID: generated dataset hash mismatch")
    prior_paths = []
    for path in (ROOT / "data").rglob("*.parquet"):
        if path.resolve() == runtime_path.resolve():
            continue
        columns = set(pd.read_parquet(path, columns=None).columns)
        if {"case_id", "narrative"} <= columns:
            prior_paths.append(path)
    contamination = contamination_against_prior(runtime_path, prior_paths)
    contamination_path = ROOT / "results/v22/qualification_contamination.json"
    atomic_write_json(contamination_path, contamination)
    if contamination["leakage_findings"]:
        atomic_write_json(
            qualification_manifest_path,
            {
                "schema_version": 1,
                "status": "INVALID_CONTAMINATED_NOT_EXECUTED",
                "qualification_executed": False,
                "dataset": manifest,
                "contamination": contamination,
            },
        )
        raise RuntimeError("QUALIFICATION_PROHIBITED: contamination detected")
    runtime = pd.read_parquet(runtime_path)
    serving_config = yaml.safe_load((ROOT / "configs/v22/serving.yaml").read_text(encoding="utf-8"))
    bundle = CandidateModelBundle(ROOT / "state/v22_model_bundle.json", ROOT)
    public_key = ROOT / "artifacts/v22/approval_public_key.pem"
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public_key)
    ledger = ROOT / "artifacts/v22/qualification.sqlite"
    explanation = StructuredVllmClient(
        endpoint=f"http://{serving_config['host']}:{serving_config['port']}/v1/chat/completions",
        model="controlflow-g-v22-qwen3-4b",
        schema=json.loads(bundle.artifact_path("schema").read_text(encoding="utf-8")),
        prompt_template=bundle.artifact_path("prompt").read_text(encoding="utf-8"),
    )
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=evidence_path,
        authorization_path=directory / "authorization_state.parquet",
        ledger_path=ledger,
        explanation_client=explanation,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-qualification",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    git_commit, dirty_hash = git_state(ROOT)
    checkpoint_fields = {
        "git_commit": git_commit,
        "dirty_state_hash": dirty_hash,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(runtime_path),
        "evidence_corpus_hash": sha256_file(evidence_path),
        "authorization_state_hash": sha256_file(directory / "authorization_state.parquet"),
        "candidate_bundle_hash": sha256_file(ROOT / "state/v22_model_bundle.json"),
        "model_hashes": {
            name: bundle.payload[name]["sha256"]
            for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
        },
        "critical_threshold": bundle.threshold,
        "qwen_revision": bundle.payload["qwen_revision"],
        "vllm_version": bundle.payload["vllm_version"],
        "structured_backend": bundle.payload["structured_output_backend"],
        "prompt_schema_hashes": {
            "prompt": bundle.payload["prompt"]["sha256"],
            "schema": bundle.payload["schema"]["sha256"],
        },
        "retrieval_config_hash": bundle.payload["retrieval_config"]["sha256"],
        "temporal_config_hash": bundle.payload["temporal_config"]["sha256"],
        "pdp_action_registry_hashes": {
            "pdp": bundle.payload["policy_config"]["sha256"],
            "actions": bundle.payload["action_registry"]["sha256"],
        },
        "evaluator_protocol_hash": evaluator_protocol_hash(ROOT),
        "gate_config_hash": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
        "qualification_gate_freeze_hash": sha256_file(gate_freeze_path),
        "seed": 22901,
        "concurrency": 2,
    }
    output_path = ROOT / "results/v22/qualification_candidate_results.parquet"
    runner.run_dataset(
        runtime,
        output_path=output_path,
        partial_path=ROOT / "artifacts/v22/qualification.partial.jsonl",
        checkpoint_path=ROOT / "artifacts/v22/qualification.checkpoint.json",
        checkpoint_fields=checkpoint_fields,
        concurrency=2,
    )
    # Evaluator truth is opened only after candidate output has closed.
    metrics_path = ROOT / "results/v22/qualification_metrics.json"
    metrics = evaluate(
        output_path,
        truth_path,
        ledger,
        metrics_path,
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="QUALIFICATION",
    )
    gate_result = apply_gates(
        metrics=metrics,
        serving=serving,
        integrity=integrity,
        reviews=reviews,
        gate_config=gates_config,
    )
    status = "PASS_POST_REVIEW_REQUIRED" if gate_result["all_passed"] else "V22_DEVELOPMENT_NO_GO"
    atomic_write_json(
        qualification_manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": status,
            "qualification_executed": True,
            "one_shot": True,
            "one_shot_opened": True,
            "dataset": manifest,
            "contamination": contamination,
            "candidate_output_sha256": sha256_file(output_path),
            "metrics_sha256": sha256_file(metrics_path),
            "frozen_gates_sha256": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
            **gate_result,
        },
    )


if __name__ == "__main__":
    lock = ROOT / "state/v22_qualification_execution.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("QUALIFICATION_EXECUTION_ALREADY_ACTIVE") from exc
    try:
        main()
    finally:
        lock.rmdir()
