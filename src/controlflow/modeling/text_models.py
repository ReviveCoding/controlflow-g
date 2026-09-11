from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.eval.metrics import classification_metrics
from controlflow.modeling.supervised import CATEGORICAL, LABELS, NUMERIC
from controlflow.modeling.torch_models import run_mlp_validation

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def run_text_validation(cases_path: Path) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("embedding experiment refused CPU fallback")
    frame = pd.read_parquet(cases_path)
    train = frame[frame["case_id"].isin(set(load_split("train", phase="P11")))]
    validation = frame[frame["case_id"].isin(set(load_split("validation", phase="P11")))]
    started = time.perf_counter()
    with GpuSemaphore():
        encoder = SentenceTransformer(MODEL, revision=REVISION, device="cuda")
        train_embeddings = encoder.encode(
            train["narrative"].tolist(),
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        val_embeddings = encoder.encode(
            validation["narrative"].tolist(),
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_device = str(next(encoder.parameters()).device)
        del encoder
        torch.cuda.empty_cache()
    if embedding_device != "cuda:0":
        raise RuntimeError(f"embedding device attestation failed: {embedding_device}")
    train_y = pd.Categorical(train["severity"], categories=LABELS).codes
    val_y = pd.Categorical(validation["severity"], categories=LABELS).codes
    structured = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    train_structured = structured.fit_transform(train)
    val_structured = structured.transform(validation)
    records = []
    for name, train_x, val_x in (
        ("M6_frozen_transformer_text", train_embeddings, val_embeddings),
        (
            "M7_text_structured_fusion",
            np.column_stack([train_embeddings, train_structured]),
            np.column_stack([val_embeddings, val_structured]),
        ),
    ):
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17)
        model.fit(train_x, train_y)
        metrics = classification_metrics(val_y, model.predict_proba(val_x))
        config = {"model": name, "encoder": MODEL, "revision": REVISION, "seed": 17}
        records.append(
            {
                "experiment_id": f"risk-{name}-s17",
                "config_hash": hashlib.sha256(canonical_json(config)).hexdigest(),
                "dataset_hash": sha256_file(cases_path),
                "split_identifier": "validation",
                "seed": 17,
                "hardware_runtime": f"{platform.system()}-{platform.machine()}-cuda:{torch.cuda.get_device_name(0)}",
                "timestamp": utc_now(),
                "status": "ok",
                "runtime_seconds": time.perf_counter() - started,
                "metrics": json.dumps(metrics, sort_keys=True),
                "details": json.dumps(
                    {
                        "encoder": MODEL,
                        "revision": REVISION,
                        "embedding_device": embedding_device,
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(records)


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data" / "silver" / "synthetic_cases_development.parquet"
    with PhaseRun("P11", paths) as phase:
        mlp = run_mlp_validation(source)
        text = run_text_validation(source)
        combined = pd.concat([mlp, text], ignore_index=True)
        target = paths.root / "results" / "ml_models_advanced.parquet"
        combined.to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
