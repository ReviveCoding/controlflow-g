from __future__ import annotations

import hashlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from controlflow.agents.workflow import RiskService
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.eval.metrics import classification_metrics
from controlflow.modeling.calibration import fit_calibrator, risk_coverage
from controlflow.modeling.supervised import _preprocessor, _xy


def _temperature(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    fitted = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    temperature = float(np.exp(fitted.x))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    return temperature, np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data" / "silver" / "synthetic_cases_development.parquet"
    with PhaseRun("P12", paths) as phase:
        frame = pd.read_parquet(source)
        train = frame[frame.case_id.isin(set(load_split("train", phase="P12")))]
        validation = frame[frame.case_id.isin(set(load_split("validation", phase="P12")))].sort_values(
            "event_timestamp"
        )
        calibration_set, selection_set, threshold_set = np.array_split(validation, 3)
        train_x, train_y = _xy(train)
        calibration_x, calibration_y = _xy(calibration_set)
        selection_x, selection_y = _xy(selection_set)
        threshold_x, threshold_y = _xy(threshold_set)
        model = Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17),
                ),
            ]
        )
        model.fit(train_x, train_y)
        fit_probability = model.predict_proba(calibration_x)
        selection_probability = model.predict_proba(selection_x)
        threshold_probability = model.predict_proba(threshold_x)
        rows: list[dict[str, Any]] = [
            {"method": "uncalibrated", **classification_metrics(selection_y, selection_probability)}
        ]
        calibrated_predictions: dict[str, np.ndarray] = {"uncalibrated": threshold_probability}
        calibrators: dict[str, object | None] = {"uncalibrated": None}
        for method in ("platt", "isotonic"):
            calibrator = fit_calibrator(fit_probability, calibration_y, method)
            calibrators[method] = calibrator
            selection_prediction = calibrator.predict(selection_probability)
            calibrated_predictions[method] = calibrator.predict(threshold_probability)
            rows.append({"method": method, **classification_metrics(selection_y, selection_prediction)})
        temperature, _ = _temperature(fit_probability, calibration_y)
        logits = np.log(np.clip(selection_probability, 1e-8, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        temp_probability = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        threshold_logits = np.log(np.clip(threshold_probability, 1e-8, 1.0)) / temperature
        threshold_logits -= threshold_logits.max(axis=1, keepdims=True)
        calibrated_predictions["temperature"] = np.exp(threshold_logits) / np.exp(threshold_logits).sum(
            axis=1, keepdims=True
        )
        calibrators["temperature"] = None
        rows.append(
            {
                "method": "temperature",
                "temperature": temperature,
                **classification_metrics(selection_y, temp_probability),
            }
        )
        result = pd.DataFrame(rows)
        result["experiment_id"] = result.method.map(lambda value: f"calibration-{value}-s17")
        result["config_hash"] = result.method.map(
            lambda value: hashlib.sha256(canonical_json({"method": value, "seed": 17})).hexdigest()
        )
        result["dataset_hash"] = sha256_file(source)
        result["split_identifier"] = "validation_calibration_fit/selection"
        result["seed"] = 17
        result["hardware_runtime"] = "CPU"
        result["timestamp"] = utc_now()
        result["status"] = "ok"
        target = paths.root / "results" / "calibration.parquet"
        result.to_parquet(target, index=False)
        selected = str(min(rows, key=lambda row: float(row["brier_score"]))["method"])
        bin_rows: list[dict[str, Any]] = []
        for method, probabilities in calibrated_predictions.items():
            confidence = probabilities.max(axis=1)
            correct = probabilities.argmax(axis=1) == threshold_y
            assignments = np.minimum((confidence * 10).astype(int), 9)
            for bin_index in range(10):
                mask = assignments == bin_index
                if not mask.any():
                    continue
                bin_rows.append(
                    {
                        "experiment_id": f"calibration-bin-{method}-{bin_index}",
                        "config_hash": hashlib.sha256(
                            canonical_json({"method": method, "bin": bin_index, "bins": 10})
                        ).hexdigest(),
                        "dataset_hash": sha256_file(source),
                        "split_identifier": "validation_threshold_selection",
                        "seed": 17,
                        "hardware_runtime": "CPU",
                        "timestamp": utc_now(),
                        "status": "ok",
                        "method": method,
                        "bin": bin_index,
                        "count": int(mask.sum()),
                        "mean_confidence": float(confidence[mask].mean()),
                        "observed_accuracy": float(correct[mask].mean()),
                    }
                )
        calibration_bins = paths.root / "results" / "calibration_bins.parquet"
        pd.DataFrame(bin_rows).to_parquet(calibration_bins, index=False)
        coverage = risk_coverage(calibrated_predictions[selected], threshold_y)
        coverage["calibration_method"] = selected
        coverage["evaluation_count"] = len(threshold_y)
        coverage["experiment_id"] = coverage.threshold.map(lambda value: f"risk-coverage-{value:.2f}")
        coverage["config_hash"] = hashlib.sha256(
            canonical_json({"calibration_method": selected, "seed": 17})
        ).hexdigest()
        coverage["dataset_hash"] = sha256_file(source)
        coverage["split_identifier"] = "validation_threshold_selection"
        coverage["seed"] = 17
        coverage["hardware_runtime"] = "CPU"
        coverage["timestamp"] = utc_now()
        coverage["status"] = "ok"
        coverage.to_parquet(paths.root / "results" / "risk_coverage.parquet", index=False)
        eligible = coverage[
            (coverage["residual_critical_error_ci95_high"] <= 0.05) & (coverage["critical_capture_ci95_low"] >= 0.9)
        ]
        # Fail closed when the threshold subset is too small to establish the
        # predeclared safety bounds: no case is automatically actioned.
        review_threshold = float(eligible.threshold.min()) if len(eligible) else 1.01
        risk_service = RiskService(train)
        risk_service.calibrator = calibrators[selected]
        risk_service.review_threshold = review_threshold
        risk_artifact = paths.root / "artifacts/calibrated_risk_service.joblib"
        risk_artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(risk_service, risk_artifact)
        phase.register(target, "result_table")
        phase.register(calibration_bins, "result_table")
        phase.register(paths.root / "results" / "risk_coverage.parquet", "result_table")
        phase.register(risk_artifact, "calibrated_model")
    return str(target)


if __name__ == "__main__":
    print(run())
