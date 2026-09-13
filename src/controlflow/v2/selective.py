from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split
from controlflow.v2.decomposed import NONCRITICAL, _ordered_probability


def run_selective_prediction_study(dataset: Path) -> Path:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    validation = frame[frame["case_id"].isin(set(load_v2_split("validation")))].copy()

    selected = joblib.load(paths.root / "artifacts/v2/models/critical_learned_fusion.joblib")
    selected_probability = selected["calibrator"].predict(selected["model"].predict_proba(validation)[:, 1])
    selected_threshold = float(selected["operating_point"]["threshold"])
    hierarchical = joblib.load(paths.root / "artifacts/v2/models/severity_hierarchical.joblib")
    hierarchy_probability = hierarchical["critical"].predict_proba(validation)[:, 1]
    noncritical_probability = _ordered_probability(hierarchical["noncritical"], validation, NONCRITICAL)
    maximum_severity_probability = np.maximum(
        hierarchy_probability,
        noncritical_probability.max(axis=1) * (1 - hierarchy_probability),
    )

    outcomes = np.full(len(validation), "AUTO", dtype=object)
    missing = validation["evidence_status"].astype(str).eq("MISSING").to_numpy()
    denied = validation["authorization_outcome"].astype(str).eq("DENY").to_numpy()
    disagreement = np.abs(selected_probability - hierarchy_probability) >= 0.15
    uncertain = maximum_severity_probability < 0.75
    risk_review = (selected_probability >= selected_threshold) | (hierarchy_probability >= 0.20)
    outcomes[disagreement | uncertain | risk_review] = "REVIEW_REQUIRED"
    outcomes[missing] = "INSUFFICIENT_EVIDENCE"
    outcomes[denied] = "DENY"

    truth_critical = validation["severity"].astype(str).eq("CRITICAL").to_numpy()
    referred = outcomes != "AUTO"
    critical_caught = referred & truth_critical
    review_only = outcomes == "REVIEW_REQUIRED"
    record: dict[str, Any] = {
        "cases": len(validation),
        "auto_coverage": float((outcomes == "AUTO").mean()),
        "review_rate": float(review_only.mean()),
        "non_auto_rate": float(referred.mean()),
        "critical_recall_before_automation": float(critical_caught.sum() / max(1, truth_critical.sum())),
        "critical_residual_risk_rate": float((truth_critical & ~referred).sum() / max(1, truth_critical.sum())),
        "review_precision": float((truth_critical & review_only).sum() / max(1, review_only.sum())),
        "denied_count": int((outcomes == "DENY").sum()),
        "insufficient_evidence_count": int((outcomes == "INSUFFICIENT_EVIDENCE").sum()),
        "model_disagreement_rate": float(disagreement.mean()),
        "uncertainty_rate": float(uncertain.mean()),
        "selected_critical_threshold": selected_threshold,
        "hierarchical_review_threshold": 0.20,
        "maximum_severity_probability_floor": 0.75,
    }
    target = paths.root / "results/v2/selective_prediction.parquet"
    pd.DataFrame([record]).to_parquet(target, index=False)
    atomic_write_json(
        paths.state / "v2_selective_prediction_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "split": "validation",
            "outcome_counts": {str(key): int(value) for key, value in pd.Series(outcomes).value_counts().items()},
            "metrics": json.loads(json.dumps(record)),
            "uses_truth_or_generator_only_features": False,
        },
    )
    return target
