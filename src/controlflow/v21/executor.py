from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import canonical_json
from controlflow.schemas import ReviewDecision
from controlflow.v21.schemas import (
    ApprovalToken,
    ExecutionEvent,
    PepDecision,
    PolicyDecision,
    PolicyResult,
    ProposedAction,
)


def action_hash(action: ProposedAction) -> str:
    return hashlib.sha256(canonical_json(action.model_dump(mode="json"))).hexdigest()


class ActionLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, case_id TEXT NOT NULL,
                action_hash TEXT NOT NULL, pdp_decision TEXT NOT NULL, pep_decision TEXT NOT NULL,
                approval_status TEXT NOT NULL, approval_token_id TEXT, idempotency_key TEXT NOT NULL,
                attempted INTEGER NOT NULL, committed INTEGER NOT NULL, denied INTEGER NOT NULL,
                failure_reason TEXT, unauthorized_attempt INTEGER NOT NULL,
                review_required INTEGER NOT NULL, timestamp TEXT NOT NULL)"""
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS one_commit_per_key ON events(idempotency_key) WHERE committed = 1"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if "approval_token_id" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN approval_token_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS one_commit_per_token ON events(approval_token_id) "
                "WHERE committed = 1 AND approval_token_id IS NOT NULL"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def committed(self, idempotency_key: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM events WHERE idempotency_key=? AND committed=1", (idempotency_key,)
                ).fetchone()
                is not None
            )

    def append(self, event: ExecutionEvent) -> None:
        values = event.model_dump(mode="json")
        columns = list(values)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"INSERT INTO events ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(int(v) if isinstance(v, bool) else v for v in values.values()),
            )
            conn.commit()

    def frame(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY timestamp,event_id")]


def validate_approval(
    token: ApprovalToken | None,
    action: ProposedAction,
    *,
    session_id: str,
    now: datetime,
    consumed_token_ids: set[str],
) -> tuple[bool, str]:
    if token is None:
        return False, "missing_approval"
    checks = (
        (token.token_id not in consumed_token_ids, "reused_token"),
        (token.review_decision is ReviewDecision.APPROVE, "not_approved"),
        (token.case_id == action.case_id, "wrong_case"),
        (token.action_hash == action_hash(action), "wrong_action_hash"),
        (token.policy_version == action.policy_version, "wrong_policy_version"),
        (token.workflow_version == action.workflow_version, "wrong_workflow_version"),
        (token.session_id == session_id, "wrong_session"),
        (token.issued_at <= now < token.expires_at, "stale_or_expired_approval"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "valid"


class SimulatedActionExecutor:
    def __init__(self, ledger: ActionLedger) -> None:
        self.ledger = ledger
        self.consumed_token_ids = {
            str(row["approval_token_id"])
            for row in ledger.frame()
            if row.get("committed") and row.get("approval_token_id")
        }

    def execute(
        self,
        action: ProposedAction,
        policy: PolicyResult,
        *,
        approval: ApprovalToken | None,
        session_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> ExecutionEvent:
        timestamp = now or datetime.now(UTC)
        digest = action_hash(action)
        unauthorized = policy.decision is PolicyDecision.DENY
        review_required = policy.decision is PolicyDecision.REQUIRE_REVIEW
        approval_ok, approval_status = validate_approval(
            approval, action, session_id=session_id, now=timestamp, consumed_token_ids=self.consumed_token_ids
        )
        reason: str | None = None
        if policy.policy_version != action.policy_version:
            reason = "policy_version_mismatch"
        elif policy.decision is PolicyDecision.DENY:
            reason = "policy_denied"
        elif policy.decision is PolicyDecision.INSUFFICIENT_EVIDENCE and action.action_type != "REQUEST_EVIDENCE":
            reason = "insufficient_evidence"
        elif action.action_type not in policy.permitted_actions:
            reason = "action_outside_permitted_scope"
        elif review_required and not approval_ok:
            reason = approval_status
        elif self.ledger.committed(idempotency_key):
            reason = "duplicate_idempotency_key"
        committed = reason is None
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            action_id=digest[:24],
            case_id=action.case_id,
            action_hash=digest,
            pdp_decision=policy.decision,
            pep_decision=PepDecision.EXECUTE if committed else PepDecision.BLOCK,
            approval_status=approval_status if review_required else "not_required",
            approval_token_id=approval.token_id if approval is not None else None,
            idempotency_key=idempotency_key,
            attempted=True,
            committed=committed,
            denied=not committed,
            failure_reason=reason,
            unauthorized_attempt=unauthorized,
            review_required=review_required,
            timestamp=timestamp,
        )
        try:
            self.ledger.append(event)
        except sqlite3.IntegrityError:
            event = event.model_copy(
                update={
                    "event_id": str(uuid.uuid4()),
                    "committed": False,
                    "denied": True,
                    "pep_decision": PepDecision.BLOCK,
                    "failure_reason": "concurrent_duplicate",
                }
            )
            self.ledger.append(event)
        if committed and approval is not None:
            self.consumed_token_ids.add(approval.token_id)
        return event


def ledger_security_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "unauthorized_committed_actions": sum(bool(r["committed"]) and bool(r["unauthorized_attempt"]) for r in rows),
        "approval_bypass_commits": sum(
            bool(r["committed"]) and bool(r["review_required"]) and r["approval_status"] != "valid" for r in rows
        ),
        "duplicate_commits": sum(max(0, count - 1) for count in _commit_counts(rows).values()),
    }


def _commit_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["committed"]:
            key = str(row["idempotency_key"])
            counts[key] = counts.get(key, 0) + 1
    return counts
