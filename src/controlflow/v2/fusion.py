from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.critical import _record, select_operating_point
from controlflow.v2.data import load_v2_split
from controlflow.v2.resources import CrossPlatformGpuSemaphore

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _tabular_arrays(
    train: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    numeric = ["amount", "repeat_count", "historical_failures", "data_sensitivity"]
    categorical = ["business_unit", "policy_version", "evidence_status", "requested_scope"]
    scaler = StandardScaler().fit(train[numeric])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[categorical])

    def transform(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
        return np.column_stack([scaler.transform(frame[numeric]), encoder.transform(frame[categorical])]).astype(
            "float32"
        )

    return transform(train), transform(calibration), transform(validation), {"scaler": scaler, "encoder": encoder}


def run_fusion_study(dataset: Path, seed: int = 20_260_912) -> Path:
    from sentence_transformers import SentenceTransformer
    from xgboost import XGBClassifier

    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    parts = {
        name: frame[frame["case_id"].isin(set(load_v2_split(name)))].copy()
        for name in ("train", "calibration", "validation")
    }
    labels = {name: part["severity"].eq("CRITICAL").astype(int).to_numpy() for name, part in parts.items()}
    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        encoder_model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device="cuda",
            trust_remote_code=False,
        )
        embeddings = {
            name: encoder_model.encode(
                part["narrative"].astype(str).tolist(),
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for name, part in parts.items()
        }
        device = str(encoder_model.device)
    if not device.startswith("cuda"):
        raise RuntimeError(f"text embedding CPU fallback refused: {device}")
    tab_train, tab_cal, tab_val, tab_transformers = _tabular_arrays(
        parts["train"], parts["calibration"], parts["validation"]
    )
    fused = {
        "train": np.column_stack([embeddings["train"], tab_train]),
        "calibration": np.column_stack([embeddings["calibration"], tab_cal]),
        "validation": np.column_stack([embeddings["validation"], tab_val]),
    }
    models: dict[str, Any] = {
        "text_embedding_linear": LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed),
        "learned_mlp_fusion": MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=seed
        ),
    }
    inputs = {"text_embedding_linear": embeddings, "learned_mlp_fusion": fused}
    results: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    model_dir = paths.root / "artifacts/v2/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        started = time.perf_counter()
        values = inputs[name]
        model.fit(values["train"], labels["train"])
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            model.predict_proba(values["calibration"])[:, 1], labels["calibration"]
        )
        probability = calibrator.predict(model.predict_proba(values["validation"])[:, 1])
        point = select_operating_point(labels["validation"], probability)
        results.append(_record(name, labels["validation"], probability, point, time.perf_counter() - started))
        trained[name] = model
        joblib.dump(
            {"model": model, "calibrator": calibrator, "tabular": tab_transformers}, model_dir / f"{name}.joblib"
        )

    positives = max(1, int(labels["train"].sum()))
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=(len(labels["train"]) - positives) / positives,
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=1,
    )
    started = time.perf_counter()
    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        xgb.fit(fused["train"], labels["train"])
    if '"device":"cuda' not in xgb.get_booster().save_config():
        raise RuntimeError("embedding XGBoost CPU fallback refused")
    xgb_cal = IsotonicRegression(out_of_bounds="clip").fit(
        xgb.predict_proba(fused["calibration"])[:, 1], labels["calibration"]
    )
    probability = xgb_cal.predict(xgb.predict_proba(fused["validation"])[:, 1])
    point = select_operating_point(labels["validation"], probability)
    results.append(
        _record("text_embedding_xgboost_cuda", labels["validation"], probability, point, time.perf_counter() - started)
    )
    joblib.dump(
        {"model": xgb, "calibrator": xgb_cal, "tabular": tab_transformers},
        model_dir / "text_embedding_xgboost_cuda.joblib",
    )

    text = trained["text_embedding_linear"]
    tab = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed).fit(tab_train, labels["train"])
    best: tuple[float, IsotonicRegression, np.ndarray[Any, Any], Any] | None = None
    for alpha in np.linspace(0, 1, 11):
        raw_cal = (
            alpha * text.predict_proba(embeddings["calibration"])[:, 1] + (1 - alpha) * tab.predict_proba(tab_cal)[:, 1]
        )
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, labels["calibration"])
        raw_val = (
            alpha * text.predict_proba(embeddings["validation"])[:, 1] + (1 - alpha) * tab.predict_proba(tab_val)[:, 1]
        )
        probability = calibrator.predict(raw_val)
        point = select_operating_point(labels["validation"], probability)
        candidate = (float(alpha), calibrator, probability, point)
        if best is None or (point.review_rate, point.false_positive_rate) < (
            best[3].review_rate,
            best[3].false_positive_rate,
        ):
            best = candidate
    assert best is not None
    alpha, calibrator, probability, point = best
    results.append(
        _record("late_embedding_tabular_fusion", labels["validation"], probability, point, 0.0, {"alpha": alpha})
    )
    joblib.dump(
        {"text": text, "tabular_model": tab, "calibrator": calibrator, "alpha": alpha, "tabular": tab_transformers},
        model_dir / "late_embedding_tabular_fusion.joblib",
    )

    critical_table = paths.root / "results/v2/critical_classifier.parquet"
    if critical_table.exists():
        tabular_row = pd.read_parquet(critical_table)
        tabular_row = tabular_row.loc[tabular_row["model"].eq("weighted_xgboost_cuda")].iloc[0].to_dict()
        tabular_row["model"] = "tabular_xgboost_cuda"
        results.append(tabular_row)

    target = paths.root / "results/v2/fusion.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(target, index=False)
    atomic_write_json(
        paths.state / "v2_fusion_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "embedding_device": device,
            "results": results,
        },
    )
    return target
