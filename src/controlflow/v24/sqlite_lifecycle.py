"""Track application-owned SQLite connections until finalization."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

_lock = threading.RLock()
_owned: dict[int, Path] = {}


@contextmanager
def owned_connection(
    path: Path,
    *,
    timeout: float = 30,
    isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = None,
) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=timeout, isolation_level=isolation_level)
    key = id(connection)
    with _lock:
        _owned[key] = path.resolve()
    try:
        with connection:
            yield connection
    finally:
        try:
            connection.close()
        finally:
            with _lock:
                _owned.pop(key)


def assert_all_closed(path: Path | None = None) -> None:
    with _lock:
        paths = list(_owned.values())
    if path is not None:
        paths = [item for item in paths if item == path.resolve()]
    if paths:
        raise RuntimeError(f"SQLITE_CONNECTIONS_OPEN: {len(paths)}")
