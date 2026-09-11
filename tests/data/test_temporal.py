from __future__ import annotations

from datetime import datetime

from controlflow.data.temporal import apply_bitemporal_event, point_in_time_lookup


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_late_arrival_preserves_prior_system_knowledge() -> None:
    versions = apply_bitemporal_event(
        [],
        entity_id="policy",
        value={"v": 2},
        business_valid_from=dt("2024-01-01"),
        system_known_from=dt("2024-01-02"),
    )
    versions = apply_bitemporal_event(
        versions,
        entity_id="policy",
        value={"v": 1},
        business_valid_from=dt("2023-01-01"),
        system_known_from=dt("2024-03-01"),
    )
    before_correction = point_in_time_lookup(versions, "policy", dt("2023-06-01"), dt("2024-02-01"))
    after_correction = point_in_time_lookup(versions, "policy", dt("2023-06-01"), dt("2024-04-01"))
    current_after_correction = point_in_time_lookup(versions, "policy", dt("2024-06-01"), dt("2024-04-01"))
    assert before_correction is None
    assert after_correction is not None and after_correction.value["v"] == 1
    assert current_after_correction is not None and current_after_correction.value["v"] == 2
