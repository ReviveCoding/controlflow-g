import pandas as pd
import pytest

from controlflow.features.point_in_time import build_case_features


@pytest.mark.parametrize("reverse", [False, True])
def test_grouped_pit_keeps_dual_cutoffs_ties_and_unmatched_cases(reverse):
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    cases = pd.DataFrame(
        {
            "case_id": ["earlier", "later", "unknown", "missing"],
            "entity_id": ["a", "a", "z", None],
            "event_timestamp": [origin + pd.Timedelta(hours=2)] * 4,
            "prediction_system_time": [origin + pd.Timedelta(hours=3)] * 4,
            "case_sequence": [2, 4, 2, 2],
        }
    )
    events = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", None],
            "event_timestamp": [origin + pd.Timedelta(hours=2)] * 5,
            "system_known_at": [origin, origin, origin + pd.Timedelta(days=1), origin, origin],
            "event_sequence": [1, 3, 0, 1, 1],
            "value": [10, 30, 999, 888, 777],
        }
    )
    if reverse:
        events = events.iloc[::-1]
    original_events = events.copy(deep=True)
    result = build_case_features(cases, events).set_index("case_id")
    assert result.loc["earlier", "value"] == 10
    assert result.loc["later", "value"] == 30
    assert pd.isna(result.loc["unknown", "value"])
    assert pd.isna(result.loc["missing", "value"])
    pd.testing.assert_frame_equal(events, original_events)
