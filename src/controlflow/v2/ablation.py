from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split


def run_ablation_study(dataset: Path) -> Path:
    paths = ProjectPaths.discover()
    results = paths.root / "results/v2"
    validation = pd.read_parquet(dataset)
    validation = validation[validation["case_id"].isin(set(load_v2_split("validation")))].copy()
    tournament = pd.read_parquet(results / "model_tournament.parquet").set_index("architecture")
    candidate = pd.read_parquet(results / "selected_candidate_validation.parquet").iloc[0]
    critical = pd.read_parquet(results / "critical_classifier.parquet").set_index("model")
    fusion = pd.read_parquet(results / "fusion.parquet").set_index("model")
    selected = joblib.load(paths.root / "artifacts/v2/models/critical_learned_fusion.joblib")
    raw_probability = selected["model"].predict_proba(validation)[:, 1]
    truth = validation["severity"].eq("CRITICAL").to_numpy()
    raw_prediction = raw_probability >= 0.5
    stale = validation["case_type"].astype(str).eq("stale_policy")
    latest_only_mismatch = stale & ~validation["policy_version"].astype(str).eq("policy-v2")

    rows: list[dict[str, Any]] = [
        {
            "ablation": "full_v2_no_qlora",
            "safe_task_completion": float(candidate["safe_task_completion"]),
            "critical_recall": float(candidate["critical_recall"]),
            "structured_output_failure_rate": float(candidate["structured_output_failure_rate"]),
            "source": "selected_candidate_validation.parquet",
        },
        {
            "ablation": "no_constrained_decoding",
            "safe_task_completion": float(tournament.loc["V2-A1_qwen3_unconstrained", "safe_task_completion"]),
            "critical_recall": float(tournament.loc["V2-A1_qwen3_unconstrained", "critical_recall"]),
            "structured_output_failure_rate": float(
                tournament.loc["V2-A1_qwen3_unconstrained", "structured_output_failure_rate"]
            ),
            "source": "model_tournament.parquet",
        },
        {
            "ablation": "flat_no_dedicated_critical_classifier",
            "safe_task_completion": float(tournament.loc["V2-A3_decomposed_flat", "safe_task_completion"]),
            "critical_recall": float(tournament.loc["V2-A3_decomposed_flat", "critical_recall"]),
            "source": "model_tournament.parquet",
        },
        {
            "ablation": "hierarchical_critical_classifier",
            "safe_task_completion": float(tournament.loc["V2-A4_decomposed_hierarchical", "safe_task_completion"]),
            "critical_recall": float(tournament.loc["V2-A4_decomposed_hierarchical", "critical_recall"]),
            "source": "model_tournament.parquet",
        },
        {
            "ablation": "no_text_features_tabular_xgboost",
            "critical_recall": float(critical.loc["weighted_xgboost_cuda", "critical_recall"]),
            "review_rate": float(critical.loc["weighted_xgboost_cuda", "review_rate"]),
            "source": "critical_classifier.parquet",
        },
        {
            "ablation": "no_fusion_text_embedding_xgboost",
            "critical_recall": float(fusion.loc["text_embedding_xgboost_cuda", "critical_recall"]),
            "review_rate": float(fusion.loc["text_embedding_xgboost_cuda", "review_rate"]),
            "source": "fusion.parquet",
        },
        {
            "ablation": "no_calibration_raw_probability_0.5",
            "critical_recall": float((raw_prediction & truth).sum() / max(1, truth.sum())),
            "review_rate": float(raw_prediction.mean()),
            "source": "critical_learned_fusion.joblib",
        },
        {
            "ablation": "no_semantic_verifier",
            "unsafe_semantic_acceptance_count": int(
                tournament.loc["V2-A2_qwen3_schema", "semantic_failure_rate"]
                * tournament.loc["V2-A2_qwen3_schema", "cases"]
            ),
            "source": "model_tournament.parquet",
        },
        {
            "ablation": "no_temporal_retrieval_latest_only",
            "stale_policy_error_rate": float(latest_only_mismatch.mean()),
            "source": "development validation counterfactual",
        },
        {
            "ablation": "no_authorization",
            "authorization_policy_violations": int(validation["authorization_outcome"].astype(str).eq("DENY").sum()),
            "source": "development validation counterfactual",
        },
        {
            "ablation": "no_hitl",
            "approval_bypasses": int(validation["review_requirement"].astype(bool).sum()),
            "source": "development validation counterfactual",
        },
    ]
    target = results / "ablation.parquet"
    table = pd.DataFrame(rows)
    table.to_parquet(target, index=False)
    atomic_write_json(
        paths.state / "v2_ablation_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "ablation_started_after_non_floor_stc": True,
            "full_candidate_stc": float(candidate["safe_task_completion"]),
            "results": json.loads(table.to_json(orient="records")),
        },
    )
    return target
