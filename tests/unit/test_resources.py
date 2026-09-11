from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import ProjectPaths


def test_gpu_semaphore_is_exclusive(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    paths = ProjectPaths(tmp_path)
    with GpuSemaphore(paths), pytest.raises(RuntimeError, match="locked"):
        GpuSemaphore(paths).__enter__()
    assert not (tmp_path / "state/gpu.lock").exists()


def test_gpu_semaphore_recovers_stale_lock(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    lock = tmp_path / "state/gpu.lock"
    lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    with GpuSemaphore(ProjectPaths(tmp_path)):
        assert lock.exists()
    assert not lock.exists()
