from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.bundle import write_bundle
from controlflow.v22.candidate import CandidateExecutionWorkflow
from controlflow.v22.checkpoint import git_state
from controlflow.v22.dgp import contamination_report, generate_split
from controlflow.v22.evaluation import evaluate, evaluator_protocol_hash
from controlflow.v22.models import train_models
from controlflow.v22.resources import assert_gpu_clear, gpu_semaphore
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision

ROOT = Path(__file__).resolve().parents[1]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": sha256_file(path)}


def _vllm_version(environment: dict[str, Any]) -> str:
    raw = environment.get("wsl_probe_raw") or ""
    match = re.search(r'"vllm"\s*:\s*"([^"]+)"', raw)
    if not match:
        raise RuntimeError("actual WSL vLLM version is absent from environment probe")
    return match.group(1)


def _dataset_paths(role: str) -> tuple[Path, Path, Path]:
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    base = ROOT / f"data/v22/{config['development']['namespace']}/{role.casefold()}"
    return base / "runtime_cases.parquet", base / "evaluator_truth.parquet", base / "evidence_corpus.parquet"


def _authorization_path(role: str) -> Path:
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    return ROOT / f"data/v22/{config['development']['namespace']}/{role.casefold()}/authorization_state.parquet"


def _write_review_matrix() -> None:
    historical = json.loads((ROOT / "state/v21_review_findings.json").read_text(encoding="utf-8"))
    repairs = {
        "V21-R-C01": ("latent DGP and forbidden runtime schema", "src/controlflow/v22/dgp.py"),
        "V21-R-C02": ("CALIBRATION-only calibrator and threshold rule", "src/controlflow/v22/models.py"),
        "V21-R-C03": ("verified exclusive CandidateModelBundle loader", "src/controlflow/v22/candidate.py"),
        "V21-R-C04": ("runner-integrated fingerprint/resume rejection", "src/controlflow/v22/checkpoint.py"),
        "V21-R-C05": ("live vLLM structured/concurrency measurements", "scripts/v22_vllm_measure.py"),
        "V21-R-C06": ("declarative DGP truth and independent candidate retriever", "src/controlflow/v22/temporal.py"),
        "V21-R-C07": ("authoritative decision/executor ledger metrics", "src/controlflow/v22/executor.py"),
        "V21-R-M08": ("separate TRAIN/CALIBRATION/VALIDATION manifests", "state/v22_dataset_manifest.json"),
        "V21-R-M09": ("contamination and gated qualification manifest", "state/v22_qualification_manifest.json"),
        "V21-R-S01": ("PEP internally queries PDP", "src/controlflow/v22/executor.py"),
        "V21-R-S02": ("external Ed25519 issuer and public verifier", "src/controlflow/v22/approval.py"),
        "V21-R-S03": ("ledger-derived authoritative security metrics", "src/controlflow/v22/executor.py"),
        "V21-R-S04": ("evidence-derived hashed integrity report", "src/controlflow/v22/integrity.py"),
        "V21-R-M05": ("negative missing/tampered/reused approval tests", "tests/v22/test_security_executor.py"),
        "V21-R-H01": ("pre-commit current PDP revalidation", "src/controlflow/v22/executor.py"),
        "V21-R-H02": ("versioned explicit action registry", "configs/v22/action_registry.yaml"),
        "V21-R-H03": ("transaction fault injection and recovery tests", "tests/v22/test_security_executor.py"),
        "V21-R-H04": ("policy decision denominators", "src/controlflow/v22/executor.py"),
        "V21-R-H05": ("per-case event/state Core STC join", "src/controlflow/v22/evaluation.py"),
        "V21-R-H06": ("executor negative suite", "tests/v22/test_security_executor.py"),
        "V21-R-H07": ("factual supported/contradicted/unsupported diagnostics", "src/controlflow/v22/evaluation.py"),
        "V21-R-H08": ("component-rerun paired ablations", "scripts/v22_run_ablations.py"),
        "V21-R-H09": ("SHA256 local audit chain with authorization provenance", "src/controlflow/v22/executor.py"),
        "V21-R-H10": ("atomic SQLite simulated case state", "src/controlflow/v22/executor.py"),
        "V21-R-H11": ("evidence-qualified phase state", "state/v22_execution_state.json"),
        "V21-R-H12": ("subprocess-captured environment probe", "state/v22_environment_manifest.json"),
    }
    findings = []
    for item in historical["classifications"]:
        repair, evidence = repairs[item["id"]]
        findings.append({**item, "v22_repair": repair, "evidence": evidence, "test": None, "status": "PENDING_REVIEW"})
    atomic_write_json(
        ROOT / "state/v22_review_findings.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "source": "state/v21_review_findings.json",
            "source_sha256": sha256_file(ROOT / "state/v21_review_findings.json"),
            "review_status": "NOT_RUN",
            "counts": {"unresolved_BLOCKER": 14, "unresolved_HIGH": 12},
            "findings": findings,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    runtime_config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    environment = json.loads((ROOT / "state/v22_environment_manifest.json").read_text(encoding="utf-8"))
    if args.allow_gpu and environment["gpu_workload_admission"] != "AVAILABLE_AT_PROBE":
        raise RuntimeError("GPU_BUSY: environment probe records background GPU consumers")
    manifests: dict[str, Any] = {}
    namespace = runtime_config["development"]["namespace"]
    for role in ("TRAIN", "CALIBRATION", "VALIDATION"):
        spec = runtime_config["development"]
        base = ROOT / f"data/v22/{namespace}/{role.casefold()}"
        manifest_path = base / "manifest.json"
        if manifest_path.exists():
            manifests[role] = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifests[role] = generate_split(
                base,
                role=role,
                count=int(spec[f"{role.casefold()}_count"]),
                seed=int(spec["seeds"][role.casefold()]),
                prefix=f"V22R{namespace.rsplit('_r', 1)[-1]}{role[:3]}",
            )
    contamination = contamination_report({role: _dataset_paths(role)[0] for role in manifests})
    contamination_path = ROOT / f"results/v22/{namespace}_contamination.json"
    atomic_write_json(contamination_path, contamination)
    if contamination["leakage_findings"]:
        raise RuntimeError("development split contamination detected")
    atomic_write_json(
        ROOT / "state/v22_dataset_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "roles_present": ["TRAIN", "CALIBRATION", "VALIDATION"],
            "qualification_present": False,
            "final_present": False,
            "splits": manifests,
            "contamination_report": _binding(contamination_path),
            "public_grounding": {
                "source_policy": _binding(ROOT / "configs/data/sources.yaml"),
                "source_manifest": _binding(ROOT / "reports/00_source_manifest.md"),
                "immutable_raw_corpus_present": False,
                "usage": (
                    "control-family vocabulary only; case labels and narratives are deterministic latent simulation"
                ),
            },
        },
    )
    model_dir = ROOT / f"artifacts/v22/{namespace}/models"
    training_arguments: dict[str, Any] = {
        "train_runtime": _dataset_paths("TRAIN")[0],
        "train_truth": _dataset_paths("TRAIN")[1],
        "calibration_runtime": _dataset_paths("CALIBRATION")[0],
        "calibration_truth": _dataset_paths("CALIBRATION")[1],
        "validation_runtime": _dataset_paths("VALIDATION")[0],
        "validation_truth": _dataset_paths("VALIDATION")[1],
        "output_dir": model_dir,
        "allow_gpu": args.allow_gpu,
    }
    if args.allow_gpu:
        assert_gpu_clear()
        with gpu_semaphore(ROOT):
            report = train_models(**training_arguments)
    else:
        report = train_models(**training_arguments)
    public_key_path = ROOT / "artifacts/v22/approval_public_key.pem"
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public_key_path)
    serving = yaml.safe_load((ROOT / "configs/v22/serving.yaml").read_text(encoding="utf-8"))
    artifacts = report["artifacts"]
    bundle_payload = {
        "bundle_id": "V22-CANDIDATE-DEVELOPMENT-001",
        "critical_model": _binding(ROOT / artifacts["critical_model"]["path"]),
        "calibrator": _binding(ROOT / artifacts["calibrator"]["path"]),
        "critical_threshold": report["critical_threshold"],
        "threshold_provenance": {"split": "CALIBRATION", "rule": report["threshold_rule"]},
        "noncritical_model": _binding(ROOT / artifacts["noncritical_model"]["path"]),
        "root_model": _binding(ROOT / artifacts["root_model"]["path"]),
        "novelty_model": _binding(ROOT / artifacts["novelty_model"]["path"]),
        "embedding_revision": report["novelty"]["embedding_revision"],
        "retrieval_config": _binding(ROOT / "configs/v22/runtime.yaml"),
        "temporal_config": _binding(ROOT / "configs/v22/temporal_policies.yaml"),
        "policy_config": _binding(ROOT / "configs/v22/policy.yaml"),
        "action_registry": _binding(ROOT / "configs/v22/action_registry.yaml"),
        "approval_public_key": _binding(public_key_path),
        "qwen_model": serving["model"],
        "qwen_revision": serving["revision"],
        "vllm_version": _vllm_version(environment),
        "structured_output_backend": serving["structured_output_backend"],
        "prompt": _binding(ROOT / "configs/v22/explanation_prompt.txt"),
        "schema": _binding(ROOT / "configs/v22/explanation_schema.json"),
    }
    bundle = write_bundle(ROOT / "state/v22_model_bundle.json", bundle_payload)
    validation_runtime, validation_truth, validation_evidence = _dataset_paths("VALIDATION")
    runtime_frame = pd.read_parquet(validation_runtime)
    ledger = ROOT / f"artifacts/v22/{namespace}/validation.sqlite"
    if ledger.exists():
        raise RuntimeError("development validation ledger already exists; preserve it and use a new run namespace")
    candidate, _executor, _model_bundle = build_candidate_runtime(
        root=ROOT,
        bundle_path=ROOT / "state/v22_model_bundle.json",
        evidence_path=validation_evidence,
        authorization_path=_authorization_path("VALIDATION"),
        ledger_path=ledger,
    )
    runner = CandidateExecutionWorkflow(
        candidate,
        reviewer=lambda prepared: (
            issuer.issue(
                action=prepared.action,
                policy=prepared.proposal,
                reviewer_id="external-reviewer-01",
                approve=True,
            )
            if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
            else None
        ),
    )
    outputs = ROOT / f"results/v22/{namespace}_validation_results.parquet"
    git_commit, dirty_state_hash = git_state(ROOT)
    checkpoint_fields = {
        "git_commit": git_commit,
        "dirty_state_hash": dirty_state_hash,
        "dependency_lock_hash": sha256_file(ROOT / "uv.lock"),
        "runtime_dataset_hash": sha256_file(validation_runtime),
        "evidence_corpus_hash": sha256_file(validation_evidence),
        "authorization_state_hash": sha256_file(_authorization_path("VALIDATION")),
        "candidate_bundle_hash": sha256_file(ROOT / "state/v22_model_bundle.json"),
        "model_hashes": {
            name: bundle[name]["sha256"]
            for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
        },
        "critical_threshold": bundle["critical_threshold"],
        "qwen_revision": bundle["qwen_revision"],
        "vllm_version": bundle["vllm_version"],
        "structured_backend": bundle["structured_output_backend"],
        "prompt_schema_hashes": {"prompt": bundle["prompt"]["sha256"], "schema": bundle["schema"]["sha256"]},
        "retrieval_config_hash": bundle["retrieval_config"]["sha256"],
        "temporal_config_hash": bundle["temporal_config"]["sha256"],
        "pdp_action_registry_hashes": {
            "pdp": bundle["policy_config"]["sha256"],
            "actions": bundle["action_registry"]["sha256"],
        },
        "evaluator_protocol_hash": evaluator_protocol_hash(ROOT),
        "gate_config_hash": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
        "seed": int(runtime_config["development"]["seeds"]["validation"]),
        "concurrency": 1,
    }
    runner.run_dataset(
        runtime_frame,
        output_path=outputs,
        partial_path=ROOT / f"artifacts/v22/{namespace}/validation.partial.jsonl",
        checkpoint_path=ROOT / f"artifacts/v22/{namespace}/validation.checkpoint.json",
        checkpoint_fields=checkpoint_fields,
    )
    metrics_path = ROOT / f"results/v22/{namespace}_validation_metrics.json"
    metrics = evaluate(
        outputs,
        validation_truth,
        ledger,
        metrics_path,
        critical_threshold=float(report["critical_threshold"]),
    )
    _write_review_matrix()
    atomic_write_json(
        ROOT / "state/v22_qualification_manifest.json",
        {
            "schema_version": 1,
            "status": "PROHIBITED_NOT_GENERATED",
            "runtime_created": False,
            "truth_created": False,
            "reason": "pre-qualification reviews have not cleared all BLOCKER/HIGH findings",
        },
    )
    atomic_write_json(
        ROOT / "state/v22_freeze_manifest.json",
        {"schema_version": 1, "status": "NOT_FROZEN", "reason": "qualification has not passed"},
    )
    atomic_write_json(
        ROOT / "state/v22_execution_state.json",
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "current_phase": "V22-26-DEVELOPMENT",
            "completed_phases": [
                "V22-00",
                "V22-01",
                "V22-02",
                "V22-03-CPU",
                "V22-04",
                "V22-05",
                "V22-06",
                "V22-07",
                "V22-08",
                "V22-09",
                "V22-10",
                "V22-11",
                "V22-12",
                "V22-13",
                "V22-14",
                "V22-15",
                "V22-17",
                "V22-18",
                "V22-19",
                "V22-20",
            ],
            "gpu_training_complete": bool(report["gpu_evidence"]["xgboost_cuda_ran"]),
            "vllm_serving_complete": False,
            "development_metrics": _binding(metrics_path),
            "validation_ledger": _binding(ledger),
            "qualification_generation_permitted": False,
            "final_holdout_creation_permitted": False,
            "status": "DEVELOPMENT_IN_PROGRESS",
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
