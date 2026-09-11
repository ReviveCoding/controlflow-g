from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.eval.metrics import classification_metrics

LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
NUMERIC = ["amount", "repeat_count", "historical_failures", "data_sensitivity"]
# `case_type` is deliberately excluded: it names the synthetic scenario class
# and would create a label shortcut rather than a deployable feature.
CATEGORICAL = ["business_unit", "policy_version"]


class ProbabilisticClassifier(Protocol):
    def fit(self, x: Any, y: np.ndarray[Any, Any]) -> Any: ...
    def predict_proba(self, x: Any) -> np.ndarray[Any, Any]: ...


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    return frame[NUMERIC + CATEGORICAL], pd.Categorical(frame["severity"], categories=LABELS).codes


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


class RuleModel:
    def fit(self, x: pd.DataFrame, y: np.ndarray[Any, Any]) -> RuleModel:
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        score = (
            0.35 * np.log1p(x["amount"].to_numpy())
            + 0.65 * x["repeat_count"].to_numpy()
            + 1.1 * x["historical_failures"].to_numpy()
            + 1.4 * x["data_sensitivity"].to_numpy()
        )
        predicted = np.digitize(score, [2.8, 4.4, 6.4])
        probabilities = np.full((len(x), 4), 0.05)
        probabilities[np.arange(len(x)), predicted] = 0.85
        return probabilities


def _models(seed: int) -> dict[str, ProbabilisticClassifier]:
    from lightgbm import LGBMClassifier

    return {
        "M0_rules": RuleModel(),
        "M1_logistic": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        # A single process is intentional on managed Windows: joblib process
        # creation may be denied, while the estimator itself remains threaded C.
        "M2_random_forest": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "M4_lightgbm_cpu": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def _record(
    model: str,
    seed: int,
    dataset_hash: str,
    split: str,
    elapsed: float,
    metrics: dict[str, float],
    status: str = "ok",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {"model": model, "seed": seed, "features": NUMERIC + CATEGORICAL}
    return {
        "experiment_id": f"risk-{model}-s{seed}",
        "config_hash": hashlib.sha256(canonical_json(config)).hexdigest(),
        "dataset_hash": dataset_hash,
        "split_identifier": split,
        "seed": seed,
        "hardware_runtime": f"{platform.system()}-{platform.machine()}",
        "timestamp": utc_now(),
        "status": status,
        "runtime_seconds": elapsed,
        "metrics": json.dumps(metrics, sort_keys=True),
        "details": json.dumps(details or {}, sort_keys=True),
    }


def run_tabular_validation(cases_path: Path, seeds: tuple[int, ...] = (17, 29)) -> pd.DataFrame:
    frame = pd.read_parquet(cases_path)
    train_ids = set(load_split("train", phase="P10"))
    validation_ids = set(load_split("validation", phase="P10"))
    train = frame[frame["case_id"].isin(train_ids)]
    validation = frame[frame["case_id"].isin(validation_ids)]
    train_x, train_y = _xy(train)
    val_x, val_y = _xy(validation)
    records: list[dict[str, Any]] = []
    dataset_hash = sha256_file(cases_path)
    for seed in seeds:
        for name, model in _models(seed).items():
            started = time.perf_counter()
            model.fit(train_x, train_y)
            probability = model.predict_proba(val_x)
            records.append(
                _record(
                    name,
                    seed,
                    dataset_hash,
                    "validation",
                    time.perf_counter() - started,
                    classification_metrics(val_y, probability),
                )
            )
        records.append(_run_xgboost(train_x, train_y, val_x, val_y, seed, dataset_hash))
    result = pd.DataFrame(records)
    target = ProjectPaths.discover().root / "results" / "ml_models.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    return result


def _run_xgboost(
    train_x: pd.DataFrame,
    train_y: np.ndarray[Any, Any],
    val_x: pd.DataFrame,
    val_y: np.ndarray[Any, Any],
    seed: int,
    dataset_hash: str,
) -> dict[str, Any]:
    import xgboost as xgb
    from xgboost import XGBClassifier

    preprocessor = _preprocessor()
    transformed_train = preprocessor.fit_transform(train_x)
    transformed_val = preprocessor.transform(val_x)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        device="cuda",
        random_state=seed,
    )
    started = time.perf_counter()
    with GpuSemaphore():
        model.fit(transformed_train, train_y)
        # Booster+DMatrix prediction avoids sklearn's CPU inplace-predict path
        # and its explicit mismatched-device fallback.
        probability = model.get_booster().predict(xgb.DMatrix(transformed_val))
    config = json.loads(model.get_booster().save_config())
    device = config["learner"]["generic_param"]["device"]
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"XGBoost silently fell back to {device}")
    return _record(
        "M3_xgboost_cuda",
        seed,
        dataset_hash,
        "validation",
        time.perf_counter() - started,
        classification_metrics(val_y, probability),
        details={"booster_device": device, "xgboost_config": config},
    )


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data" / "silver" / "synthetic_cases_development.parquet"
    with PhaseRun("P10", paths) as phase:
        run_tabular_validation(source)
        target = paths.root / "results" / "ml_models.parquet"
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
