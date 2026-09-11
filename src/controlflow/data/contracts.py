from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SourceClass(StrEnum):
    STRICT = "STRICT"
    EVOLVING = "EVOLVING"
    QUARANTINE = "QUARANTINE"


class QualityAction(StrEnum):
    WARN = "WARN"
    DROP = "DROP"
    FAIL = "FAIL"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class ColumnRule:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = False


@dataclass(frozen=True)
class ContractResult:
    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    metrics: dict[str, int]


class DataContractError(ValueError):
    pass


def validate_frame(
    frame: pd.DataFrame,
    rules: tuple[ColumnRule, ...],
    *,
    source_class: SourceClass,
    invalid_action: QualityAction,
    primary_key: str | None = None,
) -> ContractResult:
    expected = {rule.name for rule in rules}
    actual = set(frame.columns)
    missing = {rule.name for rule in rules if rule.required} - actual
    unexpected = actual - expected
    if missing:
        raise DataContractError(f"missing required columns: {sorted(missing)}")
    if unexpected and source_class is SourceClass.STRICT:
        raise DataContractError(f"unexpected columns: {sorted(unexpected)}")

    invalid = pd.Series(False, index=frame.index)
    type_failures = 0
    for rule in rules:
        if rule.name not in frame:
            continue
        series = frame[rule.name]
        if not rule.nullable:
            invalid |= series.isna()
        if rule.dtype.startswith("datetime"):
            converted = pd.to_datetime(series, errors="coerce", utc=True)
        elif rule.dtype.startswith(("int", "float")):
            converted = pd.to_numeric(series, errors="coerce")
        else:
            try:
                converted = series.astype(rule.dtype, errors="raise")
            except (TypeError, ValueError):
                converted = pd.Series(pd.NA, index=series.index)
        conversion_failed = series.notna() & pd.isna(converted)
        type_failures += int(conversion_failed.sum())
        invalid |= conversion_failed
    duplicate_count = 0
    if primary_key is not None:
        duplicates = frame[primary_key].duplicated(keep="first") | frame[primary_key].isna()
        duplicate_count = int(duplicates.sum())
        invalid |= duplicates

    invalid_count = int(invalid.sum())
    if invalid_count and invalid_action is QualityAction.FAIL:
        raise DataContractError(f"{invalid_count} invalid rows")
    quarantined = frame.loc[invalid].copy() if invalid_action is QualityAction.QUARANTINE else frame.iloc[0:0].copy()
    accepted = (
        frame.loc[~invalid].copy() if invalid_action in {QualityAction.DROP, QualityAction.QUARANTINE} else frame.copy()
    )
    return ContractResult(
        accepted=accepted,
        quarantined=quarantined,
        metrics={
            "rows_input": len(frame),
            "rows_accepted": len(accepted),
            "rows_quarantined": len(quarantined),
            "invalid_rows": invalid_count,
            "duplicate_keys": duplicate_count,
            "type_failures": type_failures,
            "unexpected_columns": len(unexpected),
            "missing_columns": len(missing),
        },
    )
