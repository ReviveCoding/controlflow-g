from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now
from controlflow.features.point_in_time import (
    build_case_features,
    build_intentionally_leaky_features,
)


def generate(seed: int = 1729, entities: int = 200, cases_per_entity: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1, tzinfo=UTC)
    cases: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for entity_index in range(entities):
        entity = f"ENTITY-{entity_index:05d}"
        failures = int(rng.integers(0, 3))
        for ordinal in range(cases_per_entity):
            timestamp = start + timedelta(days=ordinal * 14, minutes=entity_index)
            amount = float(rng.lognormal(5.5, 0.8))
            score = 0.8 * failures + 0.003 * amount + rng.normal(0, 1.3)
            label = int(score > 2.4)
            cases.append(
                {
                    "case_id": f"PIT-{entity_index:05d}-{ordinal:03d}",
                    "entity_id": entity,
                    "event_timestamp": timestamp,
                    "amount": amount,
                    "label": label,
                }
            )
            events.append(
                {
                    "entity_id": entity,
                    "event_timestamp": timestamp - timedelta(seconds=1),
                    "system_known_at": timestamp - timedelta(seconds=1),
                    "known_failures": failures,
                    "future_outcome": 0,
                }
            )
            failures += label
        events.append(
            {
                "entity_id": entity,
                "event_timestamp": start + timedelta(days=cases_per_entity * 14 + 1),
                "system_known_at": start + timedelta(days=cases_per_entity * 14 + 1),
                "known_failures": failures,
                "future_outcome": int(sum(int(case["label"]) for case in cases[-cases_per_entity:]) > 0),
            }
        )
    return pd.DataFrame(cases), pd.DataFrame(events)


def run() -> str:
    paths = ProjectPaths.discover()
    phase = PhaseRun("P07", paths)
    phase.__enter__()
    cases, events = generate()
    correct = build_case_features(cases, events)
    leaky = build_intentionally_leaky_features(cases, events)
    if (correct["matched_event_timestamp"] > correct["event_timestamp"]).any() or (
        correct["matched_system_known_at"] > correct["prediction_system_time"]
    ).any():
        raise RuntimeError("point-in-time join used unavailable evidence")
    cutoff = cases["event_timestamp"].quantile(0.7)
    train_ids = set(cases.loc[cases["event_timestamp"] <= cutoff, "case_id"])
    validation_ids = set(cases.loc[cases["event_timestamp"] > cutoff, "case_id"])
    rows: list[dict[str, object]] = []
    for name, frame, features in [
        ("CORRECT", correct, ["amount", "known_failures", "future_outcome"]),
        ("LEAKY", leaky, ["amount", "known_failures", "future_outcome"]),
    ]:
        train = frame[frame["case_id"].isin(train_ids)]
        validation = frame[frame["case_id"].isin(validation_ids)]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=17))
        model.fit(train[features].fillna(0), train["label"])
        probability = model.predict_proba(validation[features].fillna(0))[:, 1]
        rows.append(
            {
                "experiment_id": f"P07-{name.casefold()}-features",
                "config_hash": hashlib.sha256(
                    canonical_json({"pipeline": name, "features": features, "seed": 17})
                ).hexdigest(),
                "dataset_hash": hashlib.sha256(
                    canonical_json({"generator": "pit-v1", "seed": 1729, "rows": len(cases)})
                ).hexdigest(),
                "split_identifier": "temporal_validation_70_30",
                "seed": 17,
                "hardware_runtime": "CPU",
                "timestamp": utc_now(),
                "status": "ok",
                "pipeline": name,
                "sample_size": len(validation),
                "macro_f1": float(f1_score(validation["label"], probability >= 0.5, average="macro")),
                "auroc": float(roc_auc_score(validation["label"], probability)),
            }
        )
    result = pd.DataFrame(rows)
    correct_row = result[result["pipeline"] == "CORRECT"].iloc[0]
    leaky_row = result[result["pipeline"] == "LEAKY"].iloc[0]
    result["apparent_macro_f1_inflation"] = float(leaky_row["macro_f1"] - correct_row["macro_f1"])
    result["apparent_auroc_inflation"] = float(leaky_row["auroc"] - correct_row["auroc"])
    target = paths.root / "results" / "leakage.parquet"
    result.to_parquet(target, index=False)
    phase.register(target, "result_table")
    phase.__exit__(None, None, None)
    return str(target)


if __name__ == "__main__":
    print(run())
