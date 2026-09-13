from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

NUMERIC = [
    "control_test_count",
    "control_test_failures",
    "historical_incidents",
    "transaction_count",
    "anomaly_count",
    "privileged_event_count",
    "repeat_exception_ratio",
    "customer_impact_signal",
    "policy_risk_signal",
    "affected_customers",
    "amount_variance",
    "scope_difference",
    "clearance",
    "data_classification",
]
CATEGORICAL = ["control_family", "business_unit", "region", "role", "requested_scope"]


def _features() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3500), "narrative"),
        ]
    )


def _target(runtime_path: Path, truth_path: Path, column: str) -> tuple[pd.DataFrame, pd.Series]:
    runtime = pd.read_parquet(runtime_path)
    truth = pd.read_parquet(truth_path)[["case_id", column]]
    merged = runtime.merge(truth, on="case_id", validate="one_to_one")
    return merged.drop(columns=[column]), merged[column]


def _critical_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    eligible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        recall = float(recall_score(y, prediction, zero_division=0))
        negatives = y == 0
        fpr = float(np.mean(prediction[negatives])) if negatives.any() else 0.0
        if recall >= 0.95:
            residual_critical_risk = 1.0 - recall
            operational_burden = fpr + 4.0 * residual_critical_risk
            eligible.append((operational_burden, fpr, -float(threshold), recall))
    if not eligible:
        raise RuntimeError("no calibration-only threshold meets critical recall >= 0.95")
    operational_burden, fpr, negative_threshold, recall = min(eligible)
    return -negative_threshold, {
        "critical_recall": recall,
        "false_positive_rate": fpr,
        "residual_critical_risk": 1.0 - recall,
        "operational_burden": operational_burden,
    }


def _try_xgboost(train_x: pd.DataFrame, train_y: pd.Series, validation_x: pd.DataFrame) -> tuple[Any, np.ndarray]:
    from xgboost import XGBClassifier

    preprocessor = _features()
    x_train = preprocessor.fit_transform(train_x)
    x_validation = preprocessor.transform(validation_x)
    positive = max(1, int(train_y.sum()))
    scale = max(1.0, float((len(train_y) - positive) / positive))
    model = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=22001,
        device="cuda",
        tree_method="hist",
        scale_pos_weight=scale,
    )
    model.fit(x_train, train_y)
    return {"preprocessor": preprocessor, "classifier": model, "model_type": "xgboost_cuda"}, model.predict_proba(
        x_validation
    )[:, 1]


class EncodedXGBClassifier:
    def __init__(self, preprocessor: ColumnTransformer, classifier: Any, classes: np.ndarray) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.classes_ = classes

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(self.preprocessor.transform(frame)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = np.asarray(self.classifier.predict(self.preprocessor.transform(frame)), dtype=int)
        return cast(np.ndarray, self.classes_[encoded])


def _try_noncritical_xgboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
) -> tuple[EncodedXGBClassifier, np.ndarray]:
    from xgboost import XGBClassifier

    classes = np.asarray(sorted(train_y.unique()))
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = train_y.map(class_to_index).to_numpy(dtype=int)
    preprocessor = _features()
    transformed = preprocessor.fit_transform(train_x)
    classifier = XGBClassifier(
        n_estimators=220,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=22003,
        device="cuda",
        tree_method="hist",
    )
    classifier.fit(transformed, encoded)
    wrapped = EncodedXGBClassifier(preprocessor, classifier, classes)
    return wrapped, wrapped.predict(validation_x)


def _probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        return np.asarray(model["classifier"].predict_proba(model["preprocessor"].transform(frame))[:, 1])
    return np.asarray(model.predict_proba(frame)[:, 1])


def _fit_calibrator(uncalibrated: np.ndarray, truth: pd.Series) -> tuple[str, Any, dict[str, float], np.ndarray]:
    selection_mask = np.arange(len(truth)) % 3 == 0
    fit_mask = ~selection_mask
    platt_select = LogisticRegression(random_state=22002).fit(
        uncalibrated[fit_mask].reshape(-1, 1), truth.iloc[np.flatnonzero(fit_mask)]
    )
    from sklearn.isotonic import IsotonicRegression

    isotonic_select = IsotonicRegression(out_of_bounds="clip").fit(
        uncalibrated[fit_mask], truth.iloc[np.flatnonzero(fit_mask)]
    )
    selection_candidates = {
        "platt": platt_select.predict_proba(uncalibrated[selection_mask].reshape(-1, 1))[:, 1],
        "isotonic": isotonic_select.predict(uncalibrated[selection_mask]),
    }
    scores = {
        name: float(brier_score_loss(truth.iloc[np.flatnonzero(selection_mask)], values))
        for name, values in selection_candidates.items()
    }
    method = min(scores, key=lambda name: scores[name])
    platt = LogisticRegression(random_state=22002).fit(uncalibrated.reshape(-1, 1), truth)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(uncalibrated, truth)
    calibrator = platt if method == "platt" else isotonic
    calibrated = (
        platt.predict_proba(uncalibrated.reshape(-1, 1))[:, 1] if method == "platt" else isotonic.predict(uncalibrated)
    )
    return method, calibrator, scores, np.asarray(calibrated)


def _apply_calibrator(method: str, calibrator: Any, probabilities: np.ndarray) -> np.ndarray:
    if method == "platt":
        return np.asarray(calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1])
    return np.asarray(calibrator.predict(probabilities))


def train_models(
    *,
    train_runtime: Path,
    train_truth: Path,
    calibration_runtime: Path,
    calibration_truth: Path,
    validation_runtime: Path,
    validation_truth: Path,
    output_dir: Path,
    allow_gpu: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_critical = _target(train_runtime, train_truth, "truth_critical")
    calibration_x, calibration_critical = _target(calibration_runtime, calibration_truth, "truth_critical")
    validation_x, validation_critical = _target(validation_runtime, validation_truth, "truth_critical")
    logistic = Pipeline(
        [
            ("features", _features()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=22001)),
        ]
    )
    logistic.fit(train_x, train_critical)
    candidates: dict[str, tuple[Any, np.ndarray]] = {
        "weighted_logistic": (logistic, logistic.predict_proba(validation_x)[:, 1])
    }
    gpu_evidence: dict[str, Any] = {"requested": allow_gpu, "xgboost_cuda_ran": False, "device": None}
    if allow_gpu:
        xgb_model, xgb_probabilities = _try_xgboost(train_x, train_critical, validation_x)
        booster_config = xgb_model["classifier"].get_booster().save_config()
        if '"device":"cuda' not in booster_config:
            raise RuntimeError("XGBOOST_REQUIRED_CUDA_NOT_CONFIRMED")
        candidates["xgboost_cuda"] = (xgb_model, xgb_probabilities)
        gpu_evidence.update({"xgboost_cuda_ran": True, "device": "cuda", "booster_config": json.loads(booster_config)})
    comparisons: dict[str, Any] = {}
    candidate_calibrations: dict[str, tuple[str, Any, dict[str, float], float, dict[str, float]]] = {}
    validation_truth_array = validation_critical.to_numpy(dtype=bool)
    for name, (model, raw_validation_probabilities) in candidates.items():
        method, calibrator, scores, calibrated_on_calibration = _fit_calibrator(
            _probability(model, calibration_x), calibration_critical
        )
        threshold, threshold_metrics = _critical_threshold(
            calibration_critical.to_numpy(dtype=int), calibrated_on_calibration
        )
        calibrated_on_validation = _apply_calibrator(method, calibrator, raw_validation_probabilities)
        validation_prediction = calibrated_on_validation >= threshold
        validation_negatives = ~validation_truth_array
        comparisons[name] = {
            "validation_brier": float(brier_score_loss(validation_critical, calibrated_on_validation)),
            "validation_recall_at_calibrated_threshold": float(
                recall_score(validation_critical, validation_prediction, zero_division=0)
            ),
            "validation_fpr_at_calibrated_threshold": float(
                np.mean(validation_prediction[validation_negatives]) if validation_negatives.any() else 0.0
            ),
            "calibration_threshold": threshold,
            "calibration_threshold_metrics": threshold_metrics,
        }
        candidate_calibrations[name] = (method, calibrator, scores, threshold, threshold_metrics)
    recall_eligible = [
        name for name, metrics in comparisons.items() if metrics["validation_recall_at_calibrated_threshold"] >= 0.95
    ]
    if recall_eligible:
        selected_name = min(
            recall_eligible,
            key=lambda item: (
                comparisons[item]["validation_fpr_at_calibrated_threshold"],
                comparisons[item]["validation_brier"],
            ),
        )
    else:
        selected_name = min(
            comparisons,
            key=lambda item: (
                -comparisons[item]["validation_recall_at_calibrated_threshold"],
                comparisons[item]["validation_fpr_at_calibrated_threshold"],
            ),
        )
    selected_model = candidates[selected_name][0]
    calibrator_name, calibrator, calibration_scores, threshold, threshold_metrics = candidate_calibrations[
        selected_name
    ]
    critical_artifact = output_dir / "critical_model.joblib"
    calibrator_artifact = output_dir / "critical_calibrator.joblib"
    joblib.dump(selected_model, critical_artifact)
    joblib.dump({"method": calibrator_name, "model": calibrator}, calibrator_artifact)

    train_severity_x, train_severity = _target(train_runtime, train_truth, "truth_severity")
    mask = train_severity != "CRITICAL"
    logistic_noncritical = Pipeline(
        [
            ("features", _features()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=22003)),
        ]
    ).fit(train_severity_x.loc[mask], train_severity.loc[mask])
    validation_severity_x, validation_severity = _target(validation_runtime, validation_truth, "truth_severity")
    validation_noncritical_mask = validation_severity != "CRITICAL"
    noncritical_candidates: dict[str, tuple[Any, np.ndarray]] = {
        "weighted_logistic": (
            logistic_noncritical,
            logistic_noncritical.predict(validation_severity_x.loc[validation_noncritical_mask]),
        )
    }
    if allow_gpu:
        xgb_noncritical, xgb_predictions = _try_noncritical_xgboost(
            train_severity_x.loc[mask],
            train_severity.loc[mask],
            validation_severity_x.loc[validation_noncritical_mask],
        )
        noncritical_candidates["xgboost_cuda"] = (xgb_noncritical, xgb_predictions)
    noncritical_comparisons = {
        name: float(np.mean(predictions == validation_severity.loc[validation_noncritical_mask].to_numpy()))
        for name, (_, predictions) in noncritical_candidates.items()
    }
    selected_noncritical_name = max(noncritical_comparisons, key=lambda name: noncritical_comparisons[name])
    noncritical = noncritical_candidates[selected_noncritical_name][0]
    noncritical_artifact = output_dir / "noncritical_severity_model.joblib"
    joblib.dump(noncritical, noncritical_artifact)

    train_root_x, train_root = _target(train_runtime, train_truth, "truth_root_cause")
    root = Pipeline(
        [
            ("features", _features()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=22004)),
        ]
    ).fit(train_root_x, train_root)
    root_artifact = output_dir / "root_cause_model.joblib"
    joblib.dump(root, root_artifact)

    novelty_features = Pipeline([("features", _features())])
    transformed_train = novelty_features.fit_transform(train_x)
    isolation = IsolationForest(n_estimators=160, contamination="auto", random_state=22005).fit(transformed_train)
    transformed_calibration = novelty_features.transform(calibration_x)
    novelty_scores = -isolation.score_samples(transformed_calibration)
    novelty_threshold = float(np.quantile(novelty_scores, 0.95))
    novelty_artifact = output_dir / "novelty_model.joblib"
    joblib.dump(
        {
            "preprocessor": novelty_features,
            "model": isolation,
            "threshold": novelty_threshold,
            "embedding_revision": "tfidf-v22-1",
        },
        novelty_artifact,
    )
    validation_root_x, validation_root = _target(validation_runtime, validation_truth, "truth_root_cause")
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "split_roles": {
            "fitting": "TRAIN",
            "calibration_threshold": "CALIBRATION",
            "architecture_selection": "VALIDATION",
        },
        "critical_candidates": comparisons,
        "selected_critical_model": selected_name,
        "critical_model_selection_rule": (
            "evaluate each deployable CALIBRATION-calibrated/thresholded pipeline on VALIDATION; retain recall "
            ">= 0.95 then minimize FPR and Brier, otherwise maximize recall then minimize FPR"
        ),
        "noncritical_candidates": noncritical_comparisons,
        "selected_noncritical_model": selected_noncritical_name,
        "calibration_candidates": calibration_scores,
        "calibration_method_selection": "deterministic internal CALIBRATION holdback; refit on full CALIBRATION",
        "selected_calibrator": calibrator_name,
        "critical_threshold": threshold,
        "threshold_rule": (
            "CALIBRATION recall >= 0.95 then minimize FPR + 4*residual-critical-risk; break ties by highest threshold"
        ),
        "threshold_metrics": threshold_metrics,
        "gpu_evidence": gpu_evidence,
        "validation_noncritical_report": classification_report(
            validation_severity[validation_severity != "CRITICAL"],
            noncritical.predict(validation_severity_x.loc[validation_severity != "CRITICAL"]),
            output_dict=True,
            zero_division=0,
        ),
        "validation_root_report": classification_report(
            validation_root, root.predict(validation_root_x), output_dict=True, zero_division=0
        ),
        "novelty": {
            "model_type": "IsolationForest",
            "fit_split": "TRAIN",
            "threshold_split": "CALIBRATION",
            "threshold_quantile": 0.95,
            "threshold": novelty_threshold,
            "embedding_revision": "tfidf-v22-1",
        },
        "artifacts": {
            "critical_model": {"path": critical_artifact.as_posix(), "sha256": sha256_file(critical_artifact)},
            "calibrator": {"path": calibrator_artifact.as_posix(), "sha256": sha256_file(calibrator_artifact)},
            "noncritical_model": {"path": noncritical_artifact.as_posix(), "sha256": sha256_file(noncritical_artifact)},
            "root_model": {"path": root_artifact.as_posix(), "sha256": sha256_file(root_artifact)},
            "novelty_model": {"path": novelty_artifact.as_posix(), "sha256": sha256_file(novelty_artifact)},
        },
    }
    atomic_write_json(output_dir / "training_report.json", report)
    return report


def calibrated_probability(model: Any, calibrator: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    raw = _probability(model, frame)
    if calibrator["method"] == "platt":
        return np.asarray(calibrator["model"].predict_proba(raw.reshape(-1, 1))[:, 1])
    return np.asarray(calibrator["model"].predict(raw), dtype=float)
