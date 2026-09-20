from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import ProjectPaths


def test_gpu_semaphore_is_exclusive(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    paths = ProjectPaths(tmp_path)
    owner_path = tmp_path / "state/gpu.owner.json"
    with GpuSemaphore(paths) as first:
        first_owner = json.loads(owner_path.read_text(encoding="utf-8"))
        assert first_owner["owner_token"] == first.owner_token
        with pytest.raises(RuntimeError, match="locked"):
            GpuSemaphore(paths).__enter__()
        assert json.loads(owner_path.read_text(encoding="utf-8")) == first_owner
    assert not owner_path.exists()
    with GpuSemaphore(paths) as second:
        assert json.loads(owner_path.read_text(encoding="utf-8"))["owner_token"] == second.owner_token
        assert second.owner_token != first.owner_token
    assert not owner_path.exists()


def test_gpu_semaphore_recovers_stale_lock(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    lock = tmp_path / "state/gpu.lock"
    lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    owner_path = tmp_path / "state/gpu.owner.json"
    owner_path.write_text(json.dumps({"pid": 999_999_999, "owner_token": "stale"}), encoding="utf-8")
    with GpuSemaphore(ProjectPaths(tmp_path)) as first:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        assert owner["owner_token"] == first.owner_token != "stale"
        assert lock.exists()
    assert not owner_path.exists()
    with GpuSemaphore(ProjectPaths(tmp_path)) as second:
        assert json.loads(owner_path.read_text(encoding="utf-8"))["owner_token"] == second.owner_token
    assert not owner_path.exists()
