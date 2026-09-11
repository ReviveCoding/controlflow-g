from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import psutil

from controlflow.core.state import ProjectPaths, atomic_write_json, utc_now


class GpuSemaphore(AbstractContextManager["GpuSemaphore"]):
    """Exclusive single-host lock for VRAM-heavy work; stale PID locks are recoverable."""

    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or ProjectPaths.discover()
        self.path = self.paths.root / "state" / "gpu.lock"
        self.acquired = False

    def __enter__(self) -> GpuSemaphore:
        if self.path.exists():
            record = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            if psutil.pid_exists(pid):
                raise RuntimeError(f"GPU is locked by live PID {pid}")
            self.path.unlink()
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "acquired_at": utc_now()}, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        if self.acquired and self.path.exists():
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if int(record["pid"]) == os.getpid():
                self.path.unlink()
        self.acquired = False
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
