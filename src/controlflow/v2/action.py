from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.schemas import AuthorizationOutcome, Severity
from controlflow.v2.data import load_v2_split
from controlflow.v2.decomposed import NONCRITICAL, SEVERITIES, _features, _ordered_probability
from controlflow.v2.policy import deterministic_policy


def _safety_metrics(frame: pd.DataFrame, prediction: np.ndarray[Any, Any]) -> dict[str, Any]:
    unauthorized = (frame["authorization_outcome"].astype(str).eq("DENY") & (prediction == "AUTO")).sum()
    bypass = (frame["review_requirement"].astype(bool) & (prediction == "AUTO")).sum()
    return {
        "accuracy": float(accuracy_score(frame["expected_disposition"], prediction)),
        "unauthorized_irreversible_actions": int(unauthorized),
        "approval_bypasses": int(bypass),
        "review_rate": float(np.isin(prediction, ["REVIEW_REQUIRED", "DENY", "INSUFFICIENT_EVIDENCE"]).mean()),
    }


def run_action_policy_study(dataset: Path, seed: int = 20_260_912) -> Path:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    train = frame[frame["case_id"].isin(set(load_v2_split("train")))].copy()
    validation = frame[frame["case_id"].isin(set(load_v2_split("validation")))].copy()

    learned = Pipeline(
        [
            ("features", _features()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    learned.fit(train, train["expected_disposition"])
    learned_prediction = learned.predict(validation)

    hierarchical = joblib.load(paths.root / "artifacts/v2/models/severity_hierarchical.joblib")
    critical_probability = hierarchical["critical"].predict_proba(validation)[:, 1]
    noncritical_probability = _ordered_probability(hierarchical["noncritical"], validation, NONCRITICAL)
    probability = np.column_stack(
        [
            noncritical_probability[:, 0] * (1 - critical_probability),
            noncritical_probability[:, 1] * (1 - critical_probability),
            noncritical_probability[:, 2] * (1 - critical_probability),
            critical_probability,
        ]
    )
    severity = np.asarray(SEVERITIES)[probability.argmax(axis=1)]
    deterministic: list[str] = []
    for position, (_, row) in enumerate(validation.iterrows()):
        evidence_sufficient = str(row["evidence_status"]) != "MISSING" and len(row["required_evidence"]) > 0
        disposition, _ = deterministic_policy(
            severity=Severity(str(severity[position])),
            critical_probability=float(critical_probability[position]),
            evidence_sufficient=evidence_sufficient,
            authorization=AuthorizationOutcome(str(row["authorization_outcome"])),
            review_threshold=0.2,
        )
        deterministic.append(disposition.value)
    deterministic_prediction = np.asarray(deterministic)

    results = pd.DataFrame(
        [
            {"approach": "learned_disposition_classifier", **_safety_metrics(validation, learned_prediction)},
            {"approach": "deterministic_policy_mapping", **_safety_metrics(validation, deterministic_prediction)},
        ]
    )
    target = paths.root / "results/v2/action_policy.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(target, index=False)
    joblib.dump(learned, paths.root / "artifacts/v2/models/action_learned.joblib")
    atomic_write_json(
        paths.state / "v2_action_policy_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "split": "validation",
            "results": json.loads(results.to_json(orient="records")),
            "selection_rule": "prefer deterministic mapping when it preserves zero authorization and approval failures",
            "selected": "deterministic_policy_mapping"
            if int(results.loc[1, "unauthorized_irreversible_actions"]) == 0
            and int(results.loc[1, "approval_bypasses"]) == 0
            else "none",
        },
    )
    return target
