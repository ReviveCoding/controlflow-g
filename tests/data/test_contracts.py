from __future__ import annotations

import pandas as pd
import pytest

from controlflow.data.contracts import (
    ColumnRule,
    DataContractError,
    QualityAction,
    SourceClass,
    validate_frame,
)

RULES = (ColumnRule("id", "string"), ColumnRule("timestamp", "datetime64[ns, UTC]"))


def test_strict_contract_rejects_new_column() -> None:
    frame = pd.DataFrame({"id": ["a"], "timestamp": ["2025-01-01"], "new": [1]})
    with pytest.raises(DataContractError, match="unexpected"):
        validate_frame(frame, RULES, source_class=SourceClass.STRICT, invalid_action=QualityAction.FAIL)


def test_quarantines_duplicate_and_bad_timestamp() -> None:
    frame = pd.DataFrame({"id": ["a", "a", "b"], "timestamp": ["2025-01-01", "2025-01-02", "bad"]})
    result = validate_frame(
        frame,
        RULES,
        source_class=SourceClass.EVOLVING,
        invalid_action=QualityAction.QUARANTINE,
        primary_key="id",
    )
    assert result.metrics["duplicate_keys"] == 1
    assert result.metrics["invalid_rows"] == 2
    assert len(result.accepted) == 1
