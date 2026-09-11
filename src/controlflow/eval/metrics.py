from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: NDArray[np.int_], probabilities: NDArray[np.float64], bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in pairwise(edges):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def classification_metrics(
    y_true: NDArray[np.int_], probabilities: NDArray[np.float64], critical_class: int = 3
) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    one_hot = label_binarize(y_true, classes=classes)
    critical_true = y_true == critical_class
    critical_recall = recall_score(critical_true, predicted == critical_class, zero_division=0)
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    metrics = {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "critical_recall": float(critical_recall),
        "critical_fnr": float(1.0 - critical_recall),
        "brier_score": multiclass_brier,
        "ece": expected_calibration_error(y_true, probabilities),
    }
    try:
        metrics["auroc_macro_ovr"] = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
        metrics["auprc_macro"] = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        metrics["auroc_macro_ovr"] = float("nan")
        metrics["auprc_macro"] = float("nan")
    return metrics


def paired_bootstrap(
    values_a: NDArray[np.float64],
    values_b: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    *,
    repeats: int = 2000,
    seed: int = 1729,
) -> dict[str, float]:
    if values_a.shape != values_b.shape:
        raise ValueError("paired arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    deltas = values_a - values_b
    indices = rng.integers(0, len(deltas), size=(repeats, len(deltas)))
    samples = np.asarray([statistic(deltas[index]) for index in indices])
    return {
        "effect": float(statistic(deltas)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()
