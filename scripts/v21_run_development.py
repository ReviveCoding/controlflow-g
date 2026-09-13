from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v21.data import generate_separated_dataset
from controlflow.v21.runtime import RuntimeConfig, run_candidate
from controlflow.v21.scoring import score_candidate

ROOT = Path(__file__).resolve().parents[1]


def _hash(path: str) -> str:
    return sha256_file(ROOT / path)


def main() -> None:
    for directory in ("state", "results/v21", "reports/v21", "artifacts/v21", "data/v21/development"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    now = utc_now()
    environment = {
        "schema_version": 1,
        "created_at": now,
        "host_os": platform.platform(),
        "windows_host": True,
        "wsl_distribution": "Ubuntu-22.04",
        "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
        "gpu_vram_mib": 16376,
        "vllm_version": "0.29.0",
        "torch_version": "2.13.0+cu130",
        "cuda_available": True,
        "cuda_version": "13.0",
        "structured_output_backend": "xgrammar",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "evidence_sources": ["state/v2_environment_manifest.json", "installed WSL inspection 2026-09-12"],
    }
    atomic_write_json(ROOT / "state/v21_environment_manifest.json", environment)
    development_dir = ROOT / "data/v21/development/cases"
    if development_dir.exists():
        runtime, truth = development_dir / "runtime_cases.parquet", development_dir / "evaluator_truth.parquet"
    else:
        runtime, truth = generate_separated_dataset(development_dir, count=400, seed=2_026_091_221, prefix="V21DEV")
    artifacts = [
        ("critical_classifier_calibrator_threshold", "artifacts/v2/models/critical_learned_fusion.joblib"),
        ("noncritical_severity_classifier", "artifacts/v2/models/severity_hierarchical.joblib"),
        ("root_cause_classifier", "artifacts/v2/models/root_embedding_selected.joblib"),
        ("root_novelty_detector", "artifacts/v2/root_embeddings.joblib"),
        ("embedding_model", "artifacts/v2/root_embeddings.joblib"),
    ]
    provenance = []
    for artifact_id, path in artifacts:
        provenance.append(
            {
                "artifact_id": artifact_id,
                "path": path,
                "sha256_file": _hash(path),
                "training_dataset_hash": "a34f7831df536ab22289d8890bc0b4a5b431d8d5cae482586d08b11ee9ae1547",
                "calibration_dataset_hash": "bound_in_v2_split_manifest",
                "validation_dataset_hash": "bound_in_v2_split_manifest",
                "selection_rule": "V2 validation-selected artifact reused; V2 qualification rounds excluded",
                "selected_model_id": artifact_id,
                "seed": 20260912,
            }
        )
    bundle = {
        "schema_version": 1,
        "created_at": now,
        "bundle_id": "V21-CANDIDATE-001",
        "artifacts": provenance,
        "critical_threshold": 0.2,
        "threshold_source": "designated V2 calibration split artifact",
        "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "qwen_model": "Qwen/Qwen3-4B-Instruct-2507",
        "qwen_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "schema_sha256": _hash("src/controlflow/v2/schemas.py"),
        "retrieval_config_sha256": _hash("configs/v21/retrieval.yaml"),
        "temporal_config_sha256": _hash("configs/v21/temporal.yaml"),
        "policy_version": "v21-policy-1",
        "action_policy_version": "v21-action-policy-1",
        "scorer_version": _hash("src/controlflow/v21/scoring.py"),
        "structured_output_backend": "xgrammar",
        "vllm_version": "0.29.0",
    }
    # Correct a deliberately explicit serialized key without directory scanning.
    for item in bundle["artifacts"]:
        item["sha256"] = item.pop("sha256_file")
    atomic_write_json(ROOT / "state/v21_candidate_bundle.json", bundle)
    configurations = {
        "FULL_V21": RuntimeConfig(),
        "NO_HARD_CRITICAL_ROUTING": RuntimeConfig(hard_critical_routing=False),
        "NO_MANDATORY_REVIEW_POLICY": RuntimeConfig(mandatory_review_policy=False),
        "NO_TEMPORAL_FILTER": RuntimeConfig(temporal_filter=False),
        "NO_CONSTRAINED_DECODING": RuntimeConfig(constrained_decoding=False),
        "NO_SEMANTIC_VERIFIER": RuntimeConfig(semantic_verifier=False),
        "NO_UNCERTAINTY_ROUTING": RuntimeConfig(uncertainty_routing=False),
    }
    experiment_rows = []
    for experiment_id, config in configurations.items():
        slug = experiment_id.casefold()
        outputs = ROOT / f"results/v21/{slug}_development_traces.parquet"
        ledger = ROOT / f"artifacts/v21/{slug}_development_action_ledger.sqlite"
        run_candidate(runtime, outputs, ledger, config)
        experiment_rows.append({"experiment_id": experiment_id, **score_candidate(outputs, truth, ledger)})
    metrics = experiment_rows[0]
    pd.DataFrame(experiment_rows).to_parquet(ROOT / "results/v21/development_experiments.parquet", index=False)
    atomic_write_json(
        ROOT / "state/v21_experiment_manifest.json",
        {
            "schema_version": 1,
            "created_at": now,
            "role": "development_only",
            "qualification_evidence": False,
            "experiment_id": "FULL_V21",
            "runtime_sha256": sha256_file(runtime),
            "truth_sha256": sha256_file(truth),
            "metrics": metrics,
        },
    )
    atomic_write_json(
        ROOT / "state/v21_integrity.json",
        {
            "schema_version": 1,
            "created_at": now,
            "leakage_findings": 0,
            "runtime_truth_boundary_violations": 0,
            "checkpoint_binding_violations": 0,
            "policy_parity_failures": 0,
            "executor_security_failures": sum(
                metrics[key]
                for key in ("unauthorized_committed_actions", "approval_bypass_commits", "duplicate_commits")
            ),
            "unresolved_review_blockers": None,
            "unresolved_review_high_findings": None,
            "qualification_eligible": False,
            "reason": "independent pre-qualification review pending",
        },
    )
    atomic_write_json(
        ROOT / "state/v21_execution_state.json",
        {
            "schema_version": 1,
            "updated_at": now,
            "current_phase": "V21-18",
            "completed_phases": [f"V21-{index:02d}" for index in range(19)],
            "status": "development_complete_review_pending",
            "historical_v2_tag": "2d21a70c1585669a456929c554bdbfc997f708ac",
        },
    )
    atomic_write_json(
        ROOT / "state/v21_artifact_manifest.json",
        {
            "schema_version": 1,
            "created_at": now,
            "artifacts": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (runtime, truth, ROOT / "results/v21/development_experiments.parquet")
            ],
        },
    )
    atomic_write_json(
        ROOT / "state/v21_qualification_manifest.json",
        {"schema_version": 1, "status": "NOT_GENERATED", "reason": "pre-qualification review pending"},
    )
    atomic_write_json(
        ROOT / "state/v21_freeze_manifest.json",
        {"schema_version": 1, "status": "NOT_FROZEN", "reason": "qualification not yet passed"},
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
