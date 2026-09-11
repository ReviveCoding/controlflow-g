from __future__ import annotations

import pandas as pd

PIT_RAW_COLUMNS = frozenset(
    {
        "case_id",
        "entity_id",
        "event_timestamp",
        "feature_event_timestamp",
        "feature_system_known_at",
        "historical_failures",
    }
)


def build_case_features(cases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Bitemporal as-of join using business and system-known cutoffs."""
    left = cases.copy()
    right = events.copy()
    left["prediction_system_time"] = left.get("prediction_system_time", left["event_timestamp"])
    left["case_sequence"] = left.get("case_sequence", 2**31 - 1)
    right["system_known_at"] = right.get("system_known_at", right["event_timestamp"])
    right["event_sequence"] = right.get("event_sequence", 0)
    feature_columns = [column for column in right.columns if column not in {"entity_id", "event_timestamp"}]
    rows: list[dict[str, object]] = []
    for case in left.sort_values(["event_timestamp", "entity_id"]).to_dict("records"):
        candidates = right[
            (right["entity_id"] == case["entity_id"])
            & (right["event_timestamp"] <= case["event_timestamp"])
            & (right["system_known_at"] <= case["prediction_system_time"])
            & ((right["event_timestamp"] < case["event_timestamp"]) | (right["event_sequence"] < case["case_sequence"]))
        ].sort_values(["event_timestamp", "system_known_at", "event_sequence"])
        output = dict(case)
        if not candidates.empty:
            match = candidates.iloc[-1]
            output["matched_event_timestamp"] = match["event_timestamp"]
            output["matched_system_known_at"] = match["system_known_at"]
            for column in feature_columns:
                output[column if column not in output else f"{column}_history"] = match[column]
        rows.append(output)
    return pd.DataFrame(rows)


def build_intentionally_leaky_features(cases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Negative-control baseline: deliberately selects each entity's latest future state."""
    latest = events.sort_values("event_timestamp").groupby("entity_id", as_index=False).tail(1)
    return cases.merge(
        latest.drop(columns=["event_timestamp"]),
        on="entity_id",
        how="left",
        suffixes=("", "_future"),
    )


def attach_synthetic_pit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same frozen bitemporal transform to development and holdout cases."""
    missing = PIT_RAW_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"point-in-time input columns missing: {sorted(missing)}")
    development = frame.copy()
    events = development[
        ["case_id", "entity_id", "feature_event_timestamp", "feature_system_known_at", "historical_failures"]
    ].rename(
        columns={
            "case_id": "feature_source_id",
            "feature_event_timestamp": "event_timestamp",
            "feature_system_known_at": "system_known_at",
            "historical_failures": "pit_historical_failures",
        }
    )
    pit = build_case_features(development[["case_id", "entity_id", "event_timestamp"]], events)
    lineage = pit[
        [
            "case_id",
            "matched_event_timestamp",
            "matched_system_known_at",
            "feature_source_id",
            "pit_historical_failures",
        ]
    ]
    development = development.drop(columns=["feature_event_timestamp", "feature_system_known_at"]).merge(
        lineage, on="case_id", how="left", validate="one_to_one"
    )
    development = development.rename(
        columns={
            "matched_event_timestamp": "feature_event_timestamp",
            "matched_system_known_at": "feature_system_known_at",
        }
    )
    if development[["feature_source_id", "pit_historical_failures"]].isna().any().any():
        raise ValueError("point-in-time feature lineage is incomplete")
    development["historical_failures"] = development["pit_historical_failures"]
    return development
