from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controlflow.core.state import canonical_json, utc_now


def action_key(case_id: str, action_type: str, payload: dict[str, Any], workflow_version: str) -> str:
    normalized = {
        "case_id": case_id,
        "action_type": action_type,
        "payload": payload,
        "workflow_version": workflow_version,
    }
    return hashlib.sha256(canonical_json(normalized)).hexdigest()


@dataclass(frozen=True)
class ActionReceipt:
    action_id: int
    idempotency_key: str
    status: str
    executed: bool


class ActionLedger:
    """SQLite transaction makes duplicate/resumed simulated actions exactly-once locally."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS action_ledger (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    normalized_payload TEXT NOT NULL,
                    workflow_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    approved_at TEXT,
                    executed_at TEXT,
                    result_hash TEXT,
                    rollback_id TEXT
                )
                """
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS action_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, action_id INTEGER NOT NULL,
                event_type TEXT NOT NULL, actor_id TEXT NOT NULL, event_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection, *, action_id: int, event_type: str, actor_id: str, payload_hash: str
    ) -> None:
        previous = connection.execute("SELECT event_hash FROM action_events ORDER BY event_id DESC LIMIT 1").fetchone()
        previous_hash = previous[0] if previous else "GENESIS"
        event_at = utc_now()
        event_hash = hashlib.sha256(
            canonical_json(
                {
                    "action_id": action_id,
                    "event_type": event_type,
                    "actor_id": actor_id,
                    "event_at": event_at,
                    "payload_hash": payload_hash,
                    "previous_hash": previous_hash,
                }
            )
        ).hexdigest()
        connection.execute(
            "INSERT INTO action_events "
            "(action_id,event_type,actor_id,event_at,payload_hash,previous_hash,event_hash) "
            "VALUES (?,?,?,?,?,?,?)",
            (action_id, event_type, actor_id, event_at, payload_hash, previous_hash, event_hash),
        )

    def verify_event_chain(self) -> bool:
        with self._connect() as connection:
            previous = "GENESIS"
            for row in connection.execute(
                "SELECT action_id,event_type,actor_id,event_at,payload_hash,previous_hash,event_hash "
                "FROM action_events ORDER BY event_id"
            ):
                body = {
                    "action_id": row[0],
                    "event_type": row[1],
                    "actor_id": row[2],
                    "event_at": row[3],
                    "payload_hash": row[4],
                    "previous_hash": row[5],
                }
                if row[5] != previous or hashlib.sha256(canonical_json(body)).hexdigest() != row[6]:
                    return False
                previous = row[6]
        return True

    def execute_simulated(
        self,
        *,
        case_id: str,
        action_type: str,
        payload: dict[str, Any],
        workflow_version: str,
        authorization_token: str,
        approval_authority: Any,
        policy_version: str = "local-policy-v1",
        evidence_hash: str = "none",
    ) -> ActionReceipt:
        key = action_key(case_id, action_type, payload, workflow_version)
        normalized = canonical_json(payload).decode()
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT action_id, status FROM action_ledger WHERE idempotency_key = ?", (key,)
            ).fetchone()
            approval = approval_authority.verify(
                authorization_token, expected_action_key=key, policy_version=policy_version, evidence_hash=evidence_hash
            )
            reviewer_id = str(approval["reviewer_id"])
            if row is not None:
                if row[1] == "PENDING_REVIEW":
                    result_hash = hashlib.sha256(canonical_json({"simulated": True, "payload": payload})).hexdigest()
                    connection.execute(
                        """UPDATE action_ledger SET status='EXECUTED', approved_at=?, executed_at=?, result_hash=?
                        WHERE action_id=? AND status='PENDING_REVIEW'""",
                        (now, now, result_hash, row[0]),
                    )
                    self._append_event(
                        connection,
                        action_id=row[0],
                        event_type="APPROVED_AND_EXECUTED",
                        actor_id=reviewer_id,
                        payload_hash=result_hash,
                    )
                    connection.execute("COMMIT")
                    return ActionReceipt(row[0], key, "EXECUTED", executed=True)
                connection.execute("COMMIT")
                return ActionReceipt(row[0], key, row[1], executed=False)
            result_hash = hashlib.sha256(canonical_json({"simulated": True, "payload": payload})).hexdigest()
            connection.execute(
                """INSERT INTO action_ledger
                (idempotency_key, case_id, action_type, normalized_payload, workflow_version,
                 status, requested_at, approved_at, executed_at, result_hash)
                 VALUES (?, ?, ?, ?, ?, 'EXECUTED', ?, ?, ?, ?)""",
                (
                    key,
                    case_id,
                    action_type,
                    normalized,
                    workflow_version,
                    now,
                    now,
                    now,
                    result_hash,
                ),
            )
            action_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            self._append_event(
                connection, action_id=action_id, event_type="EXECUTED", actor_id=reviewer_id, payload_hash=result_hash
            )
            connection.execute("COMMIT")
        return ActionReceipt(action_id, key, "EXECUTED", executed=True)

    def request_review(
        self, *, case_id: str, action_type: str, payload: dict[str, Any], workflow_version: str
    ) -> ActionReceipt:
        key = action_key(case_id, action_type, payload, workflow_version)
        normalized = canonical_json(payload).decode()
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT action_id,status FROM action_ledger WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row is not None:
                connection.execute("COMMIT")
                return ActionReceipt(row[0], key, row[1], executed=False)
            connection.execute(
                "INSERT INTO action_ledger "
                "(idempotency_key,case_id,action_type,normalized_payload,workflow_version,status,requested_at) "
                "VALUES (?,?,?,?,?,'PENDING_REVIEW',?)",
                (key, case_id, action_type, normalized, workflow_version, now),
            )
            action_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            self._append_event(
                connection, action_id=action_id, event_type="REVIEW_REQUESTED", actor_id="workflow", payload_hash=key
            )
            connection.execute("COMMIT")
        return ActionReceipt(action_id, key, "PENDING_REVIEW", executed=False)
