from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator
from pathlib import Path


def assert_gpu_clear() -> str:
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    if process.stdout.strip():
        raise RuntimeError(f"GPU_BUSY: existing compute consumers:\n{process.stdout.strip()}")
    return process.stdout


@contextlib.contextmanager
def gpu_semaphore(root: Path) -> Iterator[None]:
    lock = root / "state/gpu.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("GPU_BUSY: repository semaphore is held") from exc
    try:
        yield
    finally:
        lock.rmdir()
