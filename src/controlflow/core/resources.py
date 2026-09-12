from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Condition
from typing import Literal

import psutil
from filelock import FileLock, Timeout

from controlflow.core.state import ProjectPaths, atomic_write_json, utc_now

_TOOL_WORKER_CONDITION = Condition()
_ACTIVE_TOOL_WORKERS = 0
_GPU_SCOPE_CLOSING = False


def register_tool_worker() -> None:
    """Register work that may retain CUDA state beyond its caller deadline."""
    global _ACTIVE_TOOL_WORKERS
    if not _TOOL_WORKER_CONDITION.acquire(blocking=False):
        raise RuntimeError("GPU/tool worker admission gate is busy; fail-closed retry required")
    try:
        if _GPU_SCOPE_CLOSING:
            raise RuntimeError("GPU scope is closing; tool worker admission denied")
        _ACTIVE_TOOL_WORKERS += 1
    finally:
        _TOOL_WORKER_CONDITION.release()


def unregister_tool_worker() -> None:
    global _ACTIVE_TOOL_WORKERS
    with _TOOL_WORKER_CONDITION:
        if _ACTIVE_TOOL_WORKERS <= 0:
            raise RuntimeError("tool worker accounting underflow")
        _ACTIVE_TOOL_WORKERS -= 1
        _TOOL_WORKER_CONDITION.notify_all()


def wait_for_tool_workers() -> None:
    """Retain the host-wide GPU lease until every local tool worker exits."""
    with _TOOL_WORKER_CONDITION:
        while _ACTIVE_TOOL_WORKERS:
            _TOOL_WORKER_CONDITION.wait()


class GpuSemaphore(AbstractContextManager["GpuSemaphore"]):
    """Exclusive single-host lock for VRAM-heavy work; stale PID locks are recoverable."""

    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or ProjectPaths.discover()
        self.path = self.paths.root / "state" / "gpu.lock"
        self.owner_path = self.paths.root / "state" / "gpu.owner.json"
        self._lock = FileLock(str(self.path))
        self.owner_token = os.urandom(16).hex()
        self.acquired = False

    def __enter__(self) -> GpuSemaphore:
        try:
            self._lock.acquire(timeout=0)
        except Timeout as error:
            raise RuntimeError("GPU is locked by another process") from error
        atomic_write_json(
            self.owner_path,
            {"pid": os.getpid(), "owner_token": self.owner_token, "acquired_at": utc_now()},
        )
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        global _GPU_SCOPE_CLOSING
        # A timed-out tool runs in a quarantined daemon thread. It can still own
        # CUDA allocations after its caller returns, so do not advertise the
        # physical GPU as available to another process until that worker exits.
        # Admission, drain, and file-lock release form one critical section so
        # a new child worker cannot enter the drain-to-release gap.
        with _TOOL_WORKER_CONDITION:
            _GPU_SCOPE_CLOSING = True
            try:
                while _ACTIVE_TOOL_WORKERS:
                    _TOOL_WORKER_CONDITION.wait()
                if self.acquired and self.owner_path.exists():
                    record = json.loads(self.owner_path.read_text(encoding="utf-8"))
                    if record.get("owner_token") == self.owner_token:
                        self.owner_path.unlink()
                if self.acquired:
                    self._lock.release()
                self.acquired = False
            finally:
                _GPU_SCOPE_CLOSING = False
                _TOOL_WORKER_CONDITION.notify_all()
        return False


def resource_snapshot() -> dict[str, object]:
    root = ProjectPaths.discover().root
    disk = psutil.disk_usage(str(root))
    memory = psutil.virtual_memory()
    result: dict[str, object] = {
        "timestamp": utc_now(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
    }
    try:
        import torch

        result["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(
                gpu_name=torch.cuda.get_device_name(0),
                gpu_free_bytes=free,
                gpu_total_bytes=total,
            )
    except ImportError:
        result["torch_cuda_available"] = False
    return result


def write_resource_snapshot(path: Path) -> None:
    atomic_write_json(path, resource_snapshot())
