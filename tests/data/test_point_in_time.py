from __future__ import annotations

import pandas as pd

from controlflow.features.point_in_time import build_case_features


def test_asof_join_never_uses_future_event() -> None:
    cases = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "entity_id": ["e", "e"],
            "event_timestamp": pd.to_datetime(["2025-01-02", "2025-01-04"], utc=True),
        }
    )
    events = pd.DataFrame(
        {
            "entity_id": ["e", "e"],
            "event_timestamp": pd.to_datetime(["2025-01-01", "2025-01-03"], utc=True),
            "value": [1, 2],
        }
    )
    result = build_case_features(cases, events)
    assert result["value"].tolist() == [1, 2]


def test_asof_join_rejects_late_known_correction() -> None:
    cases = pd.DataFrame(
        {
            "case_id": ["c"],
            "entity_id": ["e"],
            "event_timestamp": pd.to_datetime(["2025-01-04"], utc=True),
            "prediction_system_time": pd.to_datetime(["2025-01-04"], utc=True),
        }
    )
    events = pd.DataFrame(
        {
            "entity_id": ["e", "e"],
            "event_timestamp": pd.to_datetime(["2025-01-01", "2025-01-03"], utc=True),
            "system_known_at": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True),
            "value": [1, 99],
        }
    )
    assert build_case_features(cases, events).iloc[0]["value"] == 1
