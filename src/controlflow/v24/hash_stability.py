"""Hash-only SQLite byte stability verifier; deliberately never imports SQLite."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json


def _sample(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def verify_stability(database: Path, report_path: Path, *, interval_seconds: float) -> dict[str, Any]:
    if interval_seconds <= 0 or not database.is_file():
        raise RuntimeError("SQLITE_STABILITY_PRECONDITION_FAILED")
    first = _sample(database)
    time.sleep(interval_seconds)
    second = _sample(database)
    if first != second:
        raise RuntimeError("FINALIZATION_UNSTABLE")
    report = {
        "schema_version": 1,
        "status": "STABLE",
        "pid": os.getpid(),
        "process_identity": "controlflow.v24.hash_stability",
        "verified_at": datetime.now(UTC).isoformat(),
        "interval_seconds": interval_seconds,
        "size_1": first["size"],
        "sha256_1": first["sha256"],
        "size_2": second["size"],
        "sha256_2": second["sha256"],
        "exit_code": 0,
    }
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--interval-seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        verify_stability(args.database, args.report, interval_seconds=args.interval_seconds)
    except Exception as exc:
        print(f"SQLITE_STABILITY_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
