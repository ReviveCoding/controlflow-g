from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.critical import CATEGORICAL, NUMERIC, _ece
from controlflow.v2.data import load_v2_split
from controlflow.v2.resources import CrossPlatformGpuSemaphore

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
NONCRITICAL = ["LOW", "MEDIUM", "HIGH"]
ROOT_CAUSES = [
    "ROUTINE_VARIANCE",
    "OWNERSHIP_AMBIGUITY",
    "MATERIAL_CONTROL_BREAKDOWN",
    "NOVEL_THIRD_PARTY_FAILURE",
    "SOURCE_EVIDENCE_MISSING",
    "AUTHORITATIVE_SOURCE_CONFLICT",
    "TEMPORAL_POLICY_MISMATCH",
    "AUTHORIZATION_SCOPE_VIOLATION",
    "PROMPT_INJECTION_ATTEMPT",
]


def predict_root_causes(model: Any, frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Combine the supervised taxonomy model with a deterministic injection guard."""
    prediction = np.asarray(model.predict(frame), dtype=object)
    injection = (
        frame["narrative"]
        .astype(str)
        .str.contains(
            re.compile(
                r"(?is)(?:ignore|disregard|override|forget|system\s+prompt).{0,100}(?:policy|approval|authorization|tool|instruction|unrestricted)"
                r"|(?:policy|approval|authorization|tool|instruction|unrestricted).{0,100}(?:ignore|disregard|override|forget)"
            ),
            regex=True,
        )
    )
    prediction[injection.to_numpy()] = "PROMPT_INJECTION_ATTEMPT"
    return prediction


def _features(*, sparse: bool = True) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12_000), "narrative"),
        ],
        sparse_threshold=1.0 if sparse else 0.0,
    )


def _metric_record(y: np.ndarray[Any, Any], probability: np.ndarray[Any, Any], labels: list[str]) -> dict[str, float]:
    prediction = np.asarray(labels)[probability.argmax(axis=1)]
    critical = y == "CRITICAL"
    critical_probability = probability[:, labels.index("CRITICAL")] if "CRITICAL" in labels else np.zeros(len(y))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "critical_recall": float(recall_score(critical, prediction == "CRITICAL", zero_division=0))
        if "CRITICAL" in labels
        else 0.0,
        "critical_fnr": 1.0 - float(recall_score(critical, prediction == "CRITICAL", zero_division=0))
        if "CRITICAL" in labels
        else 0.0,
        "critical_auprc": float(average_precision_score(critical.astype(int), critical_probability))
        if "CRITICAL" in labels
        else 0.0,
        "critical_brier": float(brier_score_loss(critical.astype(int), critical_probability))
        if "CRITICAL" in labels
        else 0.0,
        "critical_ece": _ece(critical.astype(int), critical_probability) if "CRITICAL" in labels else 0.0,
        "review_coverage": float((prediction == "CRITICAL").mean()) if "CRITICAL" in labels else 0.0,
    }


def _root_models(seed: int) -> dict[str, Pipeline]:
    return {
        "root_linear": Pipeline(
            [
                ("features", _features()),
                ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "root_small_neural": Pipeline(
            [
                ("features", _features(sparse=False)),
                (
                    "model",
                    MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed),
                ),
            ]
        ),
    }


def run_decomposed_models(dataset: Path, seed: int = 20_260_912) -> tuple[Path, Path]:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    train = frame[frame["case_id"].isin(set(load_v2_split("train")))].copy()
    validation = frame[frame["case_id"].isin(set(load_v2_split("validation")))].copy()
    model_dir = paths.root / "artifacts/v2/models"
    model_dir.mkdir(parents=True, exist_ok=True)

    severity_records: list[dict[str, Any]] = []
    flat = Pipeline(
        [
            ("features", _features()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    started = time.perf_counter()
    flat.fit(train, train["severity"])
    flat_probability = _ordered_probability(flat, validation, SEVERITIES)
    severity_records.append(
        {
            "model": "flat_text_tabular",
            **_metric_record(validation["severity"].to_numpy(), flat_probability, SEVERITIES),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    joblib.dump(flat, model_dir / "severity_flat.joblib")

    stage1 = Pipeline(
        [
            ("features", _features()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    stage2 = Pipeline(
        [
            ("features", _features()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    started = time.perf_counter()
    stage1.fit(train, train["severity"].eq("CRITICAL"))
    noncritical_train = train[~train["severity"].eq("CRITICAL")]
    stage2.fit(noncritical_train, noncritical_train["severity"])
    critical_probability = stage1.predict_proba(validation)[:, list(stage1.classes_).index(True)]
    noncritical_probability = _ordered_probability(stage2, validation, NONCRITICAL)
    hierarchical_probability = np.column_stack(
        [
            noncritical_probability[:, 0] * (1 - critical_probability),
            noncritical_probability[:, 1] * (1 - critical_probability),
            noncritical_probability[:, 2] * (1 - critical_probability),
            critical_probability,
        ]
    )
    severity_records.append(
        {
            "model": "hierarchical_text_tabular",
            **_metric_record(validation["severity"].to_numpy(), hierarchical_probability, SEVERITIES),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    joblib.dump({"critical": stage1, "noncritical": stage2}, model_dir / "severity_hierarchical.joblib")

    selected_path = model_dir / "critical_learned_fusion.joblib"
    if selected_path.exists():
        selected = joblib.load(selected_path)
        raw_critical = selected["model"].predict_proba(validation)[:, 1]
        selected_critical = np.asarray(selected["calibrator"].predict(raw_critical))
        selected_threshold = float(selected["operating_point"]["threshold"])
        selected_prediction = np.asarray(NONCRITICAL)[noncritical_probability.argmax(axis=1)].astype(object)
        selected_prediction[selected_critical >= selected_threshold] = "CRITICAL"
        truth = validation["severity"].astype(str).to_numpy()
        critical_truth = truth == "CRITICAL"
        severity_records.append(
            {
                "model": "hierarchical_selected_calibrated_critical",
                "accuracy": float(accuracy_score(truth, selected_prediction)),
                "macro_f1": float(
                    f1_score(truth, selected_prediction, labels=SEVERITIES, average="macro", zero_division=0)
                ),
                "critical_recall": float(
                    recall_score(critical_truth, selected_prediction == "CRITICAL", zero_division=0)
                ),
                "critical_fnr": 1.0
                - float(recall_score(critical_truth, selected_prediction == "CRITICAL", zero_division=0)),
                "critical_auprc": float(average_precision_score(critical_truth.astype(int), selected_critical)),
                "critical_brier": float(brier_score_loss(critical_truth.astype(int), selected_critical)),
                "critical_ece": _ece(critical_truth.astype(int), selected_critical),
                "review_coverage": float((selected_prediction == "CRITICAL").mean()),
                "selected_threshold": selected_threshold,
                "runtime_seconds": 0.0,
            }
        )

    severity_target = paths.root / "results/v2/hierarchical_severity.parquet"
    severity_target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(severity_records).to_parquet(severity_target, index=False)

    root_records: list[dict[str, Any]] = []
    root_models = _root_models(seed)
    for name, model in root_models.items():
        started = time.perf_counter()
        model.fit(train, train["root_cause_code"])
        probability = _ordered_probability(model, validation, ROOT_CAUSES)
        prediction = np.asarray(ROOT_CAUSES)[probability.argmax(axis=1)]
        root_records.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(validation["root_cause_code"], prediction)),
                "macro_f1": float(
                    f1_score(
                        validation["root_cause_code"], prediction, labels=ROOT_CAUSES, average="macro", zero_division=0
                    )
                ),
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        joblib.dump(model, model_dir / f"{name}.joblib")
    from xgboost import XGBClassifier

    root_encoder = _features(sparse=False)
    root_train_x = root_encoder.fit_transform(train)
    root_validation_x = root_encoder.transform(validation)
    root_train_y = pd.Categorical(train["root_cause_code"], categories=ROOT_CAUSES).codes
    started = time.perf_counter()
    root_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(ROOT_CAUSES),
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=1,
    )
    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        root_xgb.fit(root_train_x, root_train_y)
    if '"device":"cuda' not in root_xgb.get_booster().save_config():
        raise RuntimeError("root-cause XGBoost CPU fallback refused")
    root_xgb_probability = root_xgb.predict_proba(root_validation_x)
    root_xgb_prediction = np.asarray(ROOT_CAUSES)[root_xgb_probability.argmax(axis=1)]
    root_records.append(
        {
            "model": "root_xgboost_cuda",
            "accuracy": float(accuracy_score(validation["root_cause_code"], root_xgb_prediction)),
            "macro_f1": float(
                f1_score(
                    validation["root_cause_code"],
                    root_xgb_prediction,
                    labels=ROOT_CAUSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    joblib.dump({"encoder": root_encoder, "model": root_xgb}, model_dir / "root_xgboost_cuda.joblib")
    root_target = paths.root / "results/v2/root_cause_classifier.parquet"
    pd.DataFrame(root_records).to_parquet(root_target, index=False)
    atomic_write_json(
        paths.state / "v2_decomposed_models.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "severity_results": json.loads(pd.DataFrame(severity_records).to_json(orient="records")),
            "root_cause_results": root_records,
            "typed_fields_from_llm": False,
        },
    )
    return severity_target, root_target


def _ordered_probability(model: Pipeline, frame: pd.DataFrame, labels: list[str]) -> np.ndarray[Any, Any]:
    raw = model.predict_proba(frame)
    classes = [str(value) for value in model.classes_]
    output = np.zeros((len(frame), len(labels)))
    for index, label in enumerate(labels):
        if label in classes:
            output[:, index] = raw[:, classes.index(label)]
    return output
