"""SQLite close protocol; this module runs in a dedicated process."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json
from controlflow.v22.executor import verify_ledger


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sidecar_sizes(database: Path) -> tuple[int, int]:
    wal = database.with_name(database.name + "-wal")
    shm = database.with_name(database.name + "-shm")
    return wal.stat().st_size if wal.exists() else 0, shm.stat().st_size if shm.exists() else 0


def finalize(
    database: Path,
    report_path: Path,
    *,
    expected_events: int,
    fault: str | None = None,
) -> dict[str, Any]:
    start = _now()
    if not database.is_file():
        raise RuntimeError("SQLITE_FINALIZER_DB_MISSING")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=rw", uri=True, timeout=0)
    try:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if journal_mode != "wal":
            raise RuntimeError(f"SQLITE_FINALIZER_JOURNAL_MODE:{journal_mode}")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLITE_FINALIZER_INTEGRITY:{integrity}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("SQLITE_FINALIZER_FOREIGN_KEY_INVALID")
        approval_mismatch = int(
            connection.execute(
                """SELECT COUNT(*) FROM action_ledger AS event
                LEFT JOIN approval_consumption AS consumed ON consumed.event_id=event.event_id
                WHERE (event.committed=1 AND event.review_required=1 AND event.approval_valid=1
                    AND (event.approval_token_id IS NULL OR consumed.token_id IS NULL
                        OR consumed.token_id!=event.approval_token_id OR consumed.consumed_at!=event.created_at))
                   OR (NOT (event.committed=1 AND event.review_required=1 AND event.approval_valid=1)
                    AND consumed.token_id IS NOT NULL)"""
            ).fetchone()[0]
        )
        if approval_mismatch:
            raise RuntimeError("SQLITE_FINALIZER_APPROVAL_CONSUMPTION_INVALID")
        approval_count = int(connection.execute("SELECT COUNT(*) FROM approval_consumption").fetchone()[0])
        review_commit_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM action_ledger WHERE committed=1 AND review_required=1 AND approval_valid=1"
            ).fetchone()[0]
        )
        if approval_count != review_commit_count:
            raise RuntimeError("SQLITE_FINALIZER_APPROVAL_COUNT_INVALID")
        ledger = verify_ledger(database)
        if not ledger["valid"] or ledger["event_count"] != expected_events:
            raise RuntimeError("SQLITE_FINALIZER_LEDGER_INVALID")
        if fault == "before_checkpoint":
            raise RuntimeError("INJECTED_FINALIZER_CRASH_BEFORE_CHECKPOINT")
        checkpoint = tuple(int(value) for value in connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
        if len(checkpoint) != 3 or checkpoint[0] != 0 or checkpoint[1] != checkpoint[2]:
            raise RuntimeError(f"SQLITE_FINALIZER_CHECKPOINT_BUSY_OR_INCOMPLETE:{checkpoint}")
        if fault == "after_checkpoint_before_close":
            raise RuntimeError("INJECTED_FINALIZER_CRASH_AFTER_CHECKPOINT")
    finally:
        connection.close()
    wal_bytes, shm_bytes = _sidecar_sizes(database)
    if wal_bytes or shm_bytes:
        raise RuntimeError(f"SQLITE_FINALIZER_SIDECAR_PERSISTS:wal={wal_bytes}:shm={shm_bytes}")
    report = {
        "schema_version": 1,
        "status": "FINALIZED",
        "pid": os.getpid(),
        "process_identity": "controlflow.v24.sqlite_finalization",
        "started_at": start,
        "ended_at": _now(),
        "sqlite_version": sqlite3.sqlite_version,
        "journal_mode": journal_mode,
        "integrity_check": integrity,
        "foreign_key_check": "ok",
        "approval_consumption_count": approval_count,
        "review_commit_count": review_commit_count,
        "checkpoint": list(checkpoint),
        "ledger_event_count": ledger["event_count"],
        "ledger_head": ledger["head"],
        "wal_bytes_after_close": wal_bytes,
        "shm_bytes_after_close": shm_bytes,
        "exit_code": 0,
    }
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-events", type=int, required=True)
    args = parser.parse_args()
    try:
        finalize(args.database, args.report, expected_events=args.expected_events)
    except Exception as exc:
        print(f"SQLITE_FINALIZATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
