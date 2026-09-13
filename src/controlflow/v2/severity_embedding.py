from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split
from controlflow.v2.decomposed import SEVERITIES
from controlflow.v2.root_embedding import predict_root_causes_runtime

NUMERIC = ["log_amount", "repeat_count", "historical_failures", "data_sensitivity"]
CATEGORICAL = ["root_cause_candidate"]


def _features(frame: pd.DataFrame, root_prediction: np.ndarray[Any, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["log_amount"] = np.log1p(result["amount"].astype(float))
    result["root_cause_candidate"] = root_prediction
    return result[NUMERIC + CATEGORICAL]


def run_severity_embedding_study(dataset: Path) -> Path:
    """Fit severity from point-in-time tabular data plus the text-derived root candidate."""
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    root_prediction = predict_root_causes_runtime(frame)
    values = _features(frame, root_prediction)
    train_mask = frame["case_id"].isin(set(load_v2_split("train")))
    validation_mask = frame["case_id"].isin(set(load_v2_split("validation")))
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("root", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline(
        [
            ("features", transformer),
            ("model", LogisticRegression(max_iter=5_000, class_weight="balanced", C=3_000, random_state=20_260_912)),
        ]
    )
    model.fit(values.loc[train_mask], frame.loc[train_mask, "severity"])
    prediction = model.predict(values.loc[validation_mask])
    table = pd.DataFrame(
        [
            {
                "model": "text_root_candidate_tabular_linear_fusion",
                "validation_accuracy": float(accuracy_score(frame.loc[validation_mask, "severity"], prediction)),
                "validation_macro_f1": float(
                    f1_score(
                        frame.loc[validation_mask, "severity"],
                        prediction,
                        labels=SEVERITIES,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        ]
    )
    target = paths.root / "results/v2/severity_embedding_fusion.parquet"
    table.to_parquet(target, index=False)
    artifact_path = paths.root / "artifacts/v2/models/severity_embedding_fusion.joblib"
    joblib.dump({"model": model}, artifact_path)
    atomic_write_json(
        paths.state / "v2_severity_embedding_manifest.json",
        {
            "schema_version": 2,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "training_split": "train",
            "selection_split": "validation",
            "features": ["text_derived_root_candidate", *NUMERIC],
            "results": json.loads(table.to_json(orient="records")),
        },
    )
    return target


def predict_severity_runtime(frame: pd.DataFrame, root_prediction: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    paths = ProjectPaths.discover()
    artifact = joblib.load(paths.root / "artifacts/v2/models/severity_embedding_fusion.joblib")
    return np.asarray(artifact["model"].predict(_features(frame, root_prediction)), dtype=object)


def predict_severity_probabilities_runtime(
    frame: pd.DataFrame, root_prediction: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    paths = ProjectPaths.discover()
    artifact = joblib.load(paths.root / "artifacts/v2/models/severity_embedding_fusion.joblib")
    model = artifact["model"]
    raw = model.predict_proba(_features(frame, root_prediction))
    classes = [str(value) for value in model.classes_]
    output = np.zeros((len(frame), len(SEVERITIES)))
    for index, label in enumerate(SEVERITIES):
        output[:, index] = raw[:, classes.index(label)]
    return output
