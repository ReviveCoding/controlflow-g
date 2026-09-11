from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from controlflow.core.state import ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.eval.metrics import classification_metrics
from controlflow.modeling.supervised import LABELS, NUMERIC


def run_label_efficiency(cases_path: Path, seed: int = 17) -> pd.DataFrame:
    frame = pd.read_parquet(cases_path)
    train = frame[frame["case_id"].isin(set(load_split("train", phase="P13")))].copy()
    validation = frame[frame["case_id"].isin(set(load_split("validation", phase="P13")))].copy()
    scaler = StandardScaler().fit(train[NUMERIC])
    x = scaler.transform(train[NUMERIC])
    y = pd.Categorical(train["severity"], categories=LABELS).codes
    validation_x = scaler.transform(validation[NUMERIC])
    validation_y = pd.Categorical(validation["severity"], categories=LABELS).codes
    rng = np.random.default_rng(seed)
    # Nested, class-stratified subsets ensure the 1% experiment is estimable
    # and all methods at a fraction receive exactly the same labeled rows.
    class_orders = [rng.permutation(np.flatnonzero(y == label)) for label in range(len(LABELS))]
    rows: list[dict[str, Any]] = []
    for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
        target_count = max(8, int(len(train) * fraction))
        per_class = max(2, target_count // len(LABELS))
        labeled = np.unique(np.concatenate([indices[: min(per_class, len(indices))] for indices in class_orders]))
        count = len(labeled)
        unlabeled_mask = np.ones(len(train), dtype=bool)
        unlabeled_mask[labeled] = False
        unlabeled = np.flatnonzero(unlabeled_mask)
        for method in ("supervised_only", "pseudo_label", "iterative_self_training"):
            model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)
            model.fit(x[labeled], y[labeled])
            if method != "supervised_only" and len(unlabeled):
                rounds = 1 if method == "pseudo_label" else 3
                selected = labeled.copy()
                pseudo_y = y[labeled].copy()
                remaining = unlabeled.copy()
                for _ in range(rounds):
                    probability = model.predict_proba(x[remaining])
                    keep = probability.max(axis=1) >= 0.90
                    if not keep.any():
                        break
                    selected = np.concatenate([selected, remaining[keep]])
                    pseudo_y = np.concatenate([pseudo_y, probability[keep].argmax(axis=1)])
                    remaining = remaining[~keep]
                    model.fit(x[selected], pseudo_y)
            metrics = classification_metrics(validation_y, model.predict_proba(validation_x))
            config = {
                "method": method,
                "fraction": fraction,
                "seed": seed,
                "subset_hash": hashlib.sha256(labeled.tobytes()).hexdigest(),
            }
            rows.append(
                {
                    "experiment_id": f"semi-{method}-{fraction:.2f}-s{seed}",
                    "config_hash": hashlib.sha256(canonical_json(config)).hexdigest(),
                    "dataset_hash": sha256_file(cases_path),
                    "split_identifier": "validation",
                    "seed": seed,
                    "hardware_runtime": f"{platform.system()}-{platform.machine()}",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "labeled_fraction": fraction,
                    "labeled_count": count,
                    "labeled_subset_hash": config["subset_hash"],
                    "method": method,
                    "metrics": json.dumps(metrics, sort_keys=True),
                }
            )
    result = pd.DataFrame(rows)
    target = ProjectPaths.discover().root / "results" / "semi_supervised.parquet"
    result.to_parquet(target, index=False)
    return result
