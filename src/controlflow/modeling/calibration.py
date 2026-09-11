from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from controlflow.eval.metrics import classification_metrics


@dataclass
class OneVsRestCalibrator:
    method: str
    models: list[Any]

    def predict(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        columns: list[NDArray[np.float64]] = []
        for index, model in enumerate(self.models):
            if self.method == "platt":
                columns.append(model.predict_proba(probabilities[:, [index]])[:, 1])
            else:
                columns.append(model.predict(probabilities[:, index]))
        calibrated = np.column_stack(columns)
        denominator = calibrated.sum(axis=1, keepdims=True)
        return cast(NDArray[np.float64], calibrated / np.maximum(denominator, 1e-12))


def fit_calibrator(probabilities: NDArray[np.float64], labels: NDArray[np.int_], method: str) -> OneVsRestCalibrator:
    models: list[Any] = []
    for index in range(probabilities.shape[1]):
        binary = (labels == index).astype(int)
        if method == "platt":
            model = LogisticRegression().fit(probabilities[:, [index]], binary)
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip").fit(probabilities[:, index], binary)
        else:
            raise ValueError(f"unsupported calibration method {method}")
        models.append(model)
    return OneVsRestCalibrator(method, models)


def compare_calibration(probabilities: NDArray[np.float64], labels: NDArray[np.int_]) -> pd.DataFrame:
    """Validation-only comparison; callers freeze the selected method before final evaluation."""
    rows = [{"method": "uncalibrated", **classification_metrics(labels, probabilities)}]
    for method in ("platt", "isotonic"):
        calibrated = fit_calibrator(probabilities, labels, method).predict(probabilities)
        rows.append({"method": method, **classification_metrics(labels, calibrated)})
    return pd.DataFrame(rows)


def risk_coverage(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int_],
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows: list[dict[str, float]] = []
    z = 1.959963984540054

    def wilson(successes: int, trials: int) -> tuple[float, float]:
        if trials == 0:
            return 0.0, 1.0
        proportion = successes / trials
        denominator = 1 + z**2 / trials
        center = (proportion + z**2 / (2 * trials)) / denominator
        radius = z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
        return float(max(0.0, center - radius)), float(min(1.0, center + radius))

    for threshold in thresholds:
        automatic = confidence >= threshold
        critical = labels == 3
        errors = predicted != labels
        critical_count = int(critical.sum())
        capture_count = int((~automatic & critical).sum())
        residual_count = int((automatic & critical & errors).sum())
        capture_low, capture_high = wilson(capture_count, critical_count)
        residual_low, residual_high = wilson(residual_count, critical_count)
        rows.append(
            {
                "threshold": threshold,
                "automation_coverage": float(automatic.mean()),
                "review_rate": float((~automatic).mean()),
                "critical_capture": float((~automatic & critical).sum() / max(1, critical.sum())),
                "residual_critical_error": float((automatic & critical & errors).sum() / max(1, critical.sum())),
                "review_precision": float((~automatic & errors).sum() / max(1, (~automatic).sum())),
                "critical_count": float(critical_count),
                "critical_capture_ci95_low": capture_low,
                "critical_capture_ci95_high": capture_high,
                "residual_critical_error_ci95_low": residual_low,
                "residual_critical_error_ci95_high": residual_high,
            }
        )
    return pd.DataFrame(rows)
