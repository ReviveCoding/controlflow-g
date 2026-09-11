from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

OPEN_END = datetime.max.replace(tzinfo=None)


@dataclass(frozen=True)
class BitemporalVersion:
    entity_id: str
    value: dict[str, Any]
    business_valid_from: datetime
    business_valid_to: datetime
    system_known_from: datetime
    system_known_to: datetime

    def visible_at(self, business_time: datetime, system_time: datetime) -> bool:
        return (
            self.business_valid_from <= business_time < self.business_valid_to
            and self.system_known_from <= system_time < self.system_known_to
        )


def apply_bitemporal_event(
    versions: list[BitemporalVersion],
    *,
    entity_id: str,
    value: dict[str, Any],
    business_valid_from: datetime,
    system_known_from: datetime,
) -> list[BitemporalVersion]:
    """Append a correction while retaining prior system-time knowledge."""
    output: list[BitemporalVersion] = []
    active = [version for version in versions if version.entity_id == entity_id and version.system_known_to == OPEN_END]
    for version in versions:
        if version in active:
            output.append(
                BitemporalVersion(
                    entity_id=version.entity_id,
                    value=version.value,
                    business_valid_from=version.business_valid_from,
                    business_valid_to=version.business_valid_to,
                    system_known_from=version.system_known_from,
                    system_known_to=system_known_from,
                )
            )
        else:
            output.append(version)
    points = {version.business_valid_from: version.value for version in active}
    points[business_valid_from] = value
    timeline = sorted(points.items())
    for index, (valid_from, point_value) in enumerate(timeline):
        end = timeline[index + 1][0] if index + 1 < len(timeline) else OPEN_END
        output.append(BitemporalVersion(entity_id, point_value, valid_from, end, system_known_from, OPEN_END))
    return output


def point_in_time_lookup(
    versions: list[BitemporalVersion],
    entity_id: str,
    business_time: datetime,
    system_time: datetime,
) -> BitemporalVersion | None:
    visible = [
        version
        for version in versions
        if version.entity_id == entity_id and version.visible_at(business_time, system_time)
    ]
    if not visible:
        return None
    return max(visible, key=lambda item: (item.system_known_from, item.business_valid_from))
