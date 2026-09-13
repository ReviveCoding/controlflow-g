from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

from pydantic import BaseModel, ConfigDict


class PolicyVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    business_valid_from: datetime
    business_valid_to: datetime | None
    system_known_from: datetime
    system_known_to: datetime | None = None
    correction_rank: int = 0

    def visible(self, event_time: datetime, system_time: datetime) -> bool:
        return (
            self.business_valid_from <= event_time
            and (self.business_valid_to is None or event_time < self.business_valid_to)
            and self.system_known_from <= system_time
            and (self.system_known_to is None or system_time < self.system_known_to)
        )


POLICIES = (
    PolicyVersion(
        policy_id="policy-v1",
        business_valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        business_valid_to=datetime(2024, 1, 1, tzinfo=UTC),
        system_known_from=datetime(2019, 12, 1, tzinfo=UTC),
    ),
    PolicyVersion(
        policy_id="policy-v2",
        business_valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        business_valid_to=datetime(2026, 1, 1, tzinfo=UTC),
        system_known_from=datetime(2023, 12, 1, tzinfo=UTC),
    ),
    PolicyVersion(
        policy_id="policy-v3",
        business_valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        business_valid_to=None,
        system_known_from=datetime(2025, 12, 1, tzinfo=UTC),
    ),
)


def validate_versions(versions: tuple[PolicyVersion, ...]) -> None:
    ordered = sorted(versions, key=lambda item: (item.business_valid_from, item.correction_rank))
    for left, right in pairwise(ordered):
        if left.correction_rank == right.correction_rank and (
            left.business_valid_to is None or right.business_valid_from < left.business_valid_to
        ):
            raise ValueError("invalid overlapping policy versions")


def select_policy(event_time: datetime, system_time: datetime, versions: tuple[PolicyVersion, ...] = POLICIES) -> str:
    visible = [item for item in versions if item.visible(event_time, system_time)]
    if not visible:
        raise LookupError("no policy version visible at both business and system time")
    return max(visible, key=lambda item: (item.correction_rank, item.system_known_from)).policy_id
