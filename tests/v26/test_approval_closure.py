"""Independent approval-consumption closure checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from controlflow.v26.postclose import _approval_read_only


def test_approval_consumption_missing_or_mutated_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "approval.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE action_ledger(event_id TEXT PRIMARY KEY, committed INTEGER, review_required INTEGER, "
            "approval_valid INTEGER, approval_token_id TEXT, created_at TEXT)"
        )
        connection.execute(
            "CREATE TABLE approval_consumption(token_id TEXT PRIMARY KEY, event_id TEXT UNIQUE, consumed_at TEXT, "
            "FOREIGN KEY(event_id) REFERENCES action_ledger(event_id))"
        )
        connection.execute("INSERT INTO action_ledger VALUES ('E1',1,1,1,'T1','2026-01-01T00:00:00Z')")
    assert _approval_read_only(database)["valid"] is False
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO approval_consumption VALUES ('T1','E1','2026-01-01T00:00:00Z')")
    assert _approval_read_only(database)["valid"] is True
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE approval_consumption SET token_id='TAMPERED' WHERE event_id='E1'")
    assert _approval_read_only(database)["valid"] is False
