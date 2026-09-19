"""Closed, authoritative SQLite evidence for post-finalization scoring."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json
from controlflow.v24.sqlite_lifecycle import owned_connection

TABLES = ("action_ledger", "case_state", "policy_decisions", "ledger_head")


def snapshot(database: Path, output: Path) -> dict[str, Any]:
    if not database.is_file():
        raise RuntimeError("LEDGER_SNAPSHOT_DB_MISSING")
    with owned_connection(database) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            name: [dict(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY rowid")] for name in TABLES
        }
    report = {"schema_version": 1, "tables": tables}
    atomic_write_json(output, report)
    return report
