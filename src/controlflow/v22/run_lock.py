from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

import psutil
from filelock import FileLock

from controlflow.core.state import atomic_write_json, utc_now


class ExecutionLock(AbstractContextManager["ExecutionLock"]):
    """Exclusive run lock with owner identity and dead-owner recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner_path = path / "owner.json"
        self._owned = False

    @staticmethod
    def _owner_alive(owner: dict[str, object]) -> bool:
        try:
            pid = owner["pid"]
            created_at = owner["process_created_at"]
            if not isinstance(pid, int) or not isinstance(created_at, int | float):
                return False
            process = psutil.Process(pid)
            return bool(abs(process.create_time() - float(created_at)) < 0.01)
        except (KeyError, TypeError, ValueError, psutil.Error):
            return False

    def __enter__(self) -> ExecutionLock:
        # Serialize the entire inspect/recover/acquire sequence with an OS-level
        # lock. Its kernel lock is released even after hard process termination.
        with FileLock(str(self.path) + ".recovery", timeout=30):
            if self.path.exists():
                try:
                    owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise RuntimeError("EXECUTION_LOCK_CORRUPT_FAIL_CLOSED") from exc
                if self._owner_alive(owner):
                    raise RuntimeError("EXECUTION_ALREADY_ACTIVE") from None
                # The exact lock directory is repository-local and contains only
                # its dead ownership record.
                self.owner_path.unlink()
                self.path.rmdir()
            self.path.mkdir()
            process = psutil.Process(os.getpid())
            atomic_write_json(
                self.owner_path,
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "process_created_at": process.create_time(),
                    "acquired_at": utc_now(),
                },
            )
        self._owned = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._owned:
            self.owner_path.unlink(missing_ok=True)
            self.path.rmdir()
            self._owned = False
