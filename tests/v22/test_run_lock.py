from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from controlflow.v22.run_lock import ExecutionLock


def test_execution_lock_rejects_live_owner_and_recovers_dead_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    with (
        ExecutionLock(lock_path),
        pytest.raises(RuntimeError, match="EXECUTION_ALREADY_ACTIVE"),
        ExecutionLock(lock_path),
    ):
        pass
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(json.dumps({"pid": 99999999, "process_created_at": 0.0}), encoding="utf-8")
    with ExecutionLock(lock_path):
        assert lock_path.is_dir()
    assert not lock_path.exists()


def test_concurrent_dead_owner_recovery_has_exactly_one_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(json.dumps({"pid": 99999999, "process_created_at": 0.0}), encoding="utf-8")
    start = threading.Barrier(2)
    release = threading.Event()

    def contender() -> str:
        start.wait()
        try:
            with ExecutionLock(lock_path):
                release.wait(timeout=1)
                return "OWNER"
        except RuntimeError as exc:
            assert str(exc) == "EXECUTION_ALREADY_ACTIVE"
            release.set()
            return "REJECTED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: contender(), range(2)))
    assert sorted(results) == ["OWNER", "REJECTED"]
