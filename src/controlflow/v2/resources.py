from __future__ import annotations

import os
import time
from types import TracebackType

from controlflow.core.state import ProjectPaths


class CrossPlatformGpuSemaphore:
    """Single-GPU admission lock shared by Windows and WSL via atomic mkdir."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.path = ProjectPaths.discover().state / "v2_gpu.lock.d"

    def __enter__(self) -> CrossPlatformGpuSemaphore:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.path.mkdir()
                (self.path / "owner.txt").write_text(
                    f"pid={os.getpid()}\ncreated_unix={time.time()}\n", encoding="utf-8"
                )
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"V2 GPU semaphore busy: {self.path}") from None
                time.sleep(0.25)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        owner = self.path / "owner.txt"
        if owner.exists():
            owner.unlink()
        self.path.rmdir()
