from __future__ import annotations

import pytest

from controlflow.data.splits import SealedTestAccessError, load_split


def test_final_ids_are_not_available_to_optimization() -> None:
    with pytest.raises(SealedTestAccessError):
        load_split("locked_final_test", phase="P10")
