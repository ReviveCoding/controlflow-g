from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateModelBundle
from controlflow.v22.checkpoint import git_state
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.gates import apply_gates
from controlflow.v22.run_lock import ExecutionLock
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision
from controlflow.v22.serving import verify_live_bundle_server
from controlflow.v22.vllm_client import StructuredVllmClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    final_manifest_path = ROOT / "state/v22_final_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if final_manifest.get("one_shot_executed") or final_manifest.get("status") == "INVALID":
        raise RuntimeError("V22 final is one-shot and has already been executed")
    if final_manifest.get("status") not in {"GENERATED_SEALED_NOT_RUN", "FINAL_EXECUTION_STARTED"}:
        raise RuntimeError("FINAL_EXECUTION_PROHIBITED: final data is not sealed")
    if final_manifest.get("status") == "GENERATED_SEALED_NOT_RUN":
        final_manifest.update({"status": "FINAL_EXECUTION_STARTED", "one_shot_opened": True, "opened_at": utc_now()})
        atomic_write_json(final_manifest_path, final_manifest)
    freeze_path = ROOT / "state/v22_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    claimed_hash = freeze.get("freeze_hash")
    unsigned = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if claimed_hash != hashlib.sha256(canonical_json(unsigned)).hexdigest():
        raise RuntimeError("FINAL_FREEZE_MISMATCH: manifest hash")
    commit, dirty_hash = git_state(ROOT)
    if commit != freeze["bindings"]["git_commit"] or dirty_hash != freeze["bindings"]["dirty_state_hash"]:
        raise RuntimeError("FINAL_FREEZE_MISMATCH: source state")
    for name, relative in (
        ("dependency_lock_sha256", "uv.lock"),
        ("candidate_bundle_sha256", "state/v22_model_bundle.json"),
        ("qualification_gates_sha256", "configs/v22/qualification_gates.yaml"),
        ("integrity_sha256", "state/v22_integrity.json"),
        ("postqualification_review_sha256", "state/v22_postqualification_review.json"),
    ):
        if freeze["bindings"][name] != sha256_file(ROOT / relative):
            raise RuntimeError(f"FINAL_FREEZE_MISMATCH: {name}")
    directory = ROOT / "data/v22/final/V22FINAL"
    paths = {
        "runtime": directory / "runtime_cases.parquet",
        "truth": directory / "evaluator_truth.parquet",
        "evidence": directory / "evidence_corpus.parquet",
        "authorization": directory / "authorization_state.parquet",
    }
    # The candidate may authenticate only runtime-visible inputs before its
    # output is closed.  Sealed evaluator truth is verified below, after the
    # candidate run, so even hashing cannot expose truth bytes to execution.
    for key, binding in (
        ("runtime", "final_runtime_sha256"),
        ("evidence", "final_evidence_sha256"),
        ("authorization", "final_authorization_sha256"),
    ):
        if sha256_file(paths[key]) != freeze["bindings"][binding]:
            raise RuntimeError(f"FINAL_FREEZE_MISMATCH: {key}")
    bundle = CandidateModelBundle(ROOT / "state/v22_model_bundle.json", ROOT)
    live = verify_live_bundle_server(ROOT, bundle)
    endpoint_root = str(live["endpoint_root"])
    client = StructuredVllmClient(
        endpoint=f"{endpoint_root}/v1/chat/completions",
        model=str(bundle.payload["qwen_served_model"]),
        schema=json.loads(bundle.artifact_path("schema").read_text(encoding="utf-8")),
        prompt_template=bundle.artifact_path("prompt").read_text(encoding="utf-8"),
    )
    public_key = bundle.artifact_path("approval_public_key")
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public_key)
    ledger = ROOT / "artifacts/v22/final.sqlite"
    candidate, _executor, bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=paths["evidence"],
        authorization_path=paths["authorization"],
        ledger_path=ledger,
        explanation_client=client,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-final",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    checkpoint_fields = {
        "git_commit": commit,
        "dirty_state_hash": dirty_hash,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(paths["runtime"]),
        "evidence_corpus_hash": sha256_file(paths["evidence"]),
        "authorization_state_hash": sha256_file(paths["authorization"]),
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
        "qualification_gate_freeze_hash": sha256_file(ROOT / "configs/v22/qualification_gate_freeze.json"),
        "seed": 23901,
        "concurrency": 2,
    }
    output = ROOT / "results/v22/final_candidate_results.parquet"
    runner.run_dataset(
        pd.read_parquet(paths["runtime"]),
        output_path=output,
        partial_path=ROOT / "artifacts/v22/final.partial.jsonl",
        checkpoint_path=ROOT / "artifacts/v22/final.checkpoint.json",
        checkpoint_fields=checkpoint_fields,
        concurrency=2,
    )
    if sha256_file(paths["truth"]) != freeze["bindings"]["final_truth_sha256"]:
        raise RuntimeError("FINAL_FREEZE_MISMATCH: truth")
    metrics_path = ROOT / "results/v22/final_metrics.json"
    metrics = evaluate(
        output,
        paths["truth"],
        ledger,
        metrics_path,
        critical_threshold=bundle.threshold,
        concurrency=2,
        role="FINAL",
    )
    integrity = json.loads((ROOT / "state/v22_integrity.json").read_text(encoding="utf-8"))
    reviews = json.loads((ROOT / "state/v22_postqualification_review.json").read_text(encoding="utf-8"))
    gate_result = apply_gates(
        metrics=metrics,
        serving=json.loads((ROOT / "state/v22_vllm_structured_c2.json").read_text(encoding="utf-8")),
        integrity=integrity,
        reviews=reviews,
        gate_config=yaml.safe_load((ROOT / "configs/v22/qualification_gates.yaml").read_text(encoding="utf-8")),
    )
    decision = "PROMOTE" if gate_result["all_passed"] else "NO_PROMOTE"
    final_manifest.update(
        {
            "status": "FINAL_COMPLETE",
            "one_shot_executed": True,
            "executed_at": utc_now(),
            "freeze_hash": claimed_hash,
            "candidate_output_sha256": sha256_file(output),
            "metrics_sha256": sha256_file(metrics_path),
            **gate_result,
            "release_decision": decision,
        }
    )
    atomic_write_json(final_manifest_path, final_manifest)
    (ROOT / "reports/v22/15_release_decision.md").write_text(f"# V2.2 Release Decision\n\n{decision}\n")
    print(decision)


if __name__ == "__main__":
    with ExecutionLock(ROOT / "state/v22_final_execution.lock"):
        main()
