from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split
from controlflow.v2.resources import CrossPlatformGpuSemaphore

NUMERIC = ["amount", "repeat_count", "historical_failures", "data_sensitivity"]
CATEGORICAL = ["business_unit", "policy_version", "evidence_status", "requested_scope"]
# Development selection uses a generalization buffer above the unchanged
# release gate (0.92); the prior exact-gate choice did not transfer.
TARGET_RECALL = 0.97


class ProbabilityModel(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray[Any, Any]) -> Any: ...
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray[Any, Any]: ...


def _tabular() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def _ece(y: np.ndarray[Any, Any], probability: np.ndarray[Any, Any], bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in pairwise(edges):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            total += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return total


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    critical_recall: float
    critical_fnr: float
    review_rate: float
    false_positive_rate: float
    residual_critical_risk: float


def select_operating_point(y: np.ndarray[Any, Any], probability: np.ndarray[Any, Any]) -> OperatingPoint:
    candidates: list[OperatingPoint] = []
    for threshold in sorted(set([0.0, 1.0, *probability.tolist()])):
        predicted = probability >= threshold
        positives = y == 1
        negatives = ~positives
        recall = float(predicted[positives].mean()) if positives.any() else 0.0
        false_positive = float(predicted[negatives].mean()) if negatives.any() else 0.0
        point = OperatingPoint(
            threshold=float(threshold),
            critical_recall=recall,
            critical_fnr=1.0 - recall,
            review_rate=float(predicted.mean()),
            false_positive_rate=false_positive,
            residual_critical_risk=float((~predicted & positives).sum() / max(1, positives.sum())),
        )
        if recall >= TARGET_RECALL:
            candidates.append(point)
    if not candidates:
        raise RuntimeError("no operating point satisfies the predeclared critical-recall target")
    return min(
        candidates,
        key=lambda item: (item.review_rate, item.false_positive_rate, item.residual_critical_risk, -item.threshold),
    )


class LateFusion:
    def __init__(self, tabular: ProbabilityModel, text: ProbabilityModel, alpha: float) -> None:
        self.tabular = tabular
        self.text = text
        self.alpha = alpha

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        tab = self.tabular.predict_proba(x)[:, 1]
        text = self.text.predict_proba(x)[:, 1]
        critical = self.alpha * tab + (1.0 - self.alpha) * text
        return np.column_stack([1.0 - critical, critical])


def _models(seed: int) -> dict[str, ProbabilityModel]:
    tabular_logistic = Pipeline(
        [
            ("features", _tabular()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    text_linear = Pipeline(
        [
            ("select", _SeriesSelector("narrative")),
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)),
        ]
    )
    learned_fusion = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "tabular",
                            Pipeline([("select", _FrameSelector(NUMERIC + CATEGORICAL)), ("encode", _tabular())]),
                        ),
                        (
                            "text",
                            Pipeline(
                                [
                                    ("select", _SeriesSelector("narrative")),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2_000)),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("model", MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed, early_stopping=True)),
        ]
    )
    return {"weighted_logistic": tabular_logistic, "text_linear": text_linear, "learned_fusion": learned_fusion}


class _FrameSelector:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, x: pd.DataFrame, y: Any = None) -> _FrameSelector:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[self.columns]


class _SeriesSelector:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, x: pd.DataFrame, y: Any = None) -> _SeriesSelector:
        return self

    def transform(self, x: pd.DataFrame) -> pd.Series[Any]:
        return x[self.column]


def _fit_xgboost(train: pd.DataFrame, y: np.ndarray[Any, Any], seed: int) -> Pipeline:
    from xgboost import XGBClassifier

    positives = max(1, int(y.sum()))
    scale = float((len(y) - positives) / positives)
    model = Pipeline(
        [
            ("features", _tabular()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale,
                    tree_method="hist",
                    device="cuda",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    with CrossPlatformGpuSemaphore():
        model.fit(train, y)
    booster = model.named_steps["model"].get_booster()
    if '"device":"cuda' not in booster.save_config():
        raise RuntimeError("XGBoost CUDA attestation failed; CPU fallback refused")
    return model


def _run_focal_mlp(
    partitions: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray[Any, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("focal-loss MLP refused silent CPU fallback")
    transformer = _tabular()
    train_x = transformer.fit_transform(partitions["train"])
    calibration_x = transformer.transform(partitions["calibration"])
    validation_x = transformer.transform(partitions["validation"])
    train_x = train_x.toarray() if hasattr(train_x, "toarray") else train_x
    calibration_x = calibration_x.toarray() if hasattr(calibration_x, "toarray") else calibration_x
    validation_x = validation_x.toarray() if hasattr(validation_x, "toarray") else validation_x
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    with CrossPlatformGpuSemaphore():
        model = nn.Sequential(nn.Linear(train_x.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        features = torch.as_tensor(train_x, dtype=torch.float32, device="cuda")
        truth = torch.as_tensor(labels["train"], dtype=torch.float32, device="cuda").reshape(-1, 1)
        alpha = float((labels["train"] == 0).mean())
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            base = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="none")
            probability = torch.sigmoid(logits)
            pt = truth * probability + (1 - truth) * (1 - probability)
            weights = truth * alpha + (1 - truth) * (1 - alpha)
            loss = (weights * (1 - pt).pow(2.0) * base).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("focal-loss MLP GPU attestation failed")
        model.eval()
        with torch.inference_mode():
            raw_cal = (
                torch.sigmoid(model(torch.as_tensor(calibration_x, dtype=torch.float32, device="cuda")))
                .cpu()
                .numpy()
                .ravel()
            )
            raw_validation = (
                torch.sigmoid(model(torch.as_tensor(validation_x, dtype=torch.float32, device="cuda")))
                .cpu()
                .numpy()
                .ravel()
            )
        state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        del model, optimizer, features, truth
        torch.cuda.empty_cache()
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, labels["calibration"])
    probability = calibrator.predict(raw_validation)
    point = select_operating_point(labels["validation"], probability)
    record = _record(
        "focal_loss_pytorch_mlp_cuda",
        labels["validation"],
        probability,
        point,
        time.perf_counter() - started,
        {"epochs": 120, "gamma": 2.0, "alpha_positive": alpha, "device": "cuda:0"},
    )
    artifact = {
        "transformer": transformer,
        "calibrator": calibrator,
        "operating_point": asdict(point),
        "state_dict": state_dict,
        "input_dimensions": int(train_x.shape[1]),
    }
    return record, artifact


def run_critical_tournament(dataset: Path, seed: int = 20_260_912) -> Path:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    partitions = {
        name: frame[frame["case_id"].isin(set(load_v2_split(name)))].copy()
        for name in ("train", "calibration", "validation")
    }
    labels = {name: part["severity"].eq("CRITICAL").astype(int).to_numpy() for name, part in partitions.items()}
    models = _models(seed)
    records: list[dict[str, Any]] = []
    fitted: dict[str, ProbabilityModel] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(partitions["train"], labels["train"])
        fitted[name] = model
        raw_cal = model.predict_proba(partitions["calibration"])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, labels["calibration"])
        probability = calibrator.predict(model.predict_proba(partitions["validation"])[:, 1])
        point = select_operating_point(labels["validation"], probability)
        records.append(_record(name, labels["validation"], probability, point, time.perf_counter() - started))
        joblib.dump(
            {"model": model, "calibrator": calibrator, "operating_point": asdict(point)}, _model_path(paths, name)
        )

    xgb_started = time.perf_counter()
    xgb = _fit_xgboost(partitions["train"], labels["train"], seed)
    fitted["weighted_xgboost_cuda"] = xgb
    xgb_cal = xgb.predict_proba(partitions["calibration"])[:, 1]
    xgb_calibrator = IsotonicRegression(out_of_bounds="clip").fit(xgb_cal, labels["calibration"])
    xgb_probability = xgb_calibrator.predict(xgb.predict_proba(partitions["validation"])[:, 1])
    xgb_point = select_operating_point(labels["validation"], xgb_probability)
    records.append(
        _record(
            "weighted_xgboost_cuda",
            labels["validation"],
            xgb_probability,
            xgb_point,
            time.perf_counter() - xgb_started,
        )
    )
    joblib.dump(
        {"model": xgb, "calibrator": xgb_calibrator, "operating_point": asdict(xgb_point)},
        _model_path(paths, "weighted_xgboost_cuda"),
    )

    focal_record, focal_artifact = _run_focal_mlp(partitions, labels, seed)
    records.append(focal_record)
    joblib.dump(focal_artifact, _model_path(paths, "focal_loss_pytorch_mlp_cuda"))

    alpha_candidates = np.linspace(0, 1, 11)
    late_candidates: list[tuple[float, IsotonicRegression, np.ndarray[Any, Any], OperatingPoint]] = []
    for alpha in alpha_candidates:
        fusion = LateFusion(fitted["weighted_logistic"], fitted["text_linear"], float(alpha))
        raw_cal = fusion.predict_proba(partitions["calibration"])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, labels["calibration"])
        probability = calibrator.predict(fusion.predict_proba(partitions["validation"])[:, 1])
        late_candidates.append(
            (float(alpha), calibrator, probability, select_operating_point(labels["validation"], probability))
        )
    alpha, calibrator, probability, point = min(
        late_candidates,
        key=lambda item: (item[3].review_rate, item[3].false_positive_rate, item[3].residual_critical_risk),
    )
    late = LateFusion(fitted["weighted_logistic"], fitted["text_linear"], alpha)
    records.append(_record("late_probability_fusion", labels["validation"], probability, point, 0.0, {"alpha": alpha}))
    joblib.dump(
        {"model": late, "calibrator": calibrator, "operating_point": asdict(point)},
        _model_path(paths, "late_probability_fusion"),
    )

    result = pd.DataFrame(records)
    result["recall_constraint_met"] = result["critical_recall"].ge(TARGET_RECALL)
    result = result.sort_values(
        ["recall_constraint_met", "review_rate", "false_positive_rate", "residual_critical_risk"],
        ascending=[False, True, True, True],
    )
    target = paths.root / "results/v2/critical_classifier.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    winner = json.loads(result.iloc[0].to_json())
    atomic_write_json(
        paths.state / "v2_critical_selection.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "selection_rule": "recall>=0.92 then minimize review burden, false positives, residual critical risk",
            "winner": winner,
            "hardware": platform.platform(),
        },
    )
    return target


def _model_path(paths: ProjectPaths, name: str) -> Path:
    directory = paths.root / "artifacts/v2/models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"critical_{name}.joblib"


def _record(
    name: str,
    y: np.ndarray[Any, Any],
    probability: np.ndarray[Any, Any],
    point: OperatingPoint,
    runtime: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = probability >= point.threshold
    return {
        "model": name,
        "critical_recall": float(recall_score(y, prediction, zero_division=0)),
        "critical_fnr": point.critical_fnr,
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": _ece(y, probability),
        "binary_f1": float(f1_score(y, prediction, zero_division=0)),
        "review_rate": point.review_rate,
        "false_positive_rate": point.false_positive_rate,
        "residual_critical_risk": point.residual_critical_risk,
        "threshold": point.threshold,
        "runtime_seconds": runtime,
        "details": json.dumps(details or {}, sort_keys=True),
    }
