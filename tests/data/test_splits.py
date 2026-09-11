from __future__ import annotations

import pytest

from controlflow.data.splits import SealedTestAccessError, load_split


def test_final_split_is_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTROLFLOW_UNLOCK_FINAL", raising=False)
    with pytest.raises(SealedTestAccessError):
        load_split("locked_final_test", phase="P10")
