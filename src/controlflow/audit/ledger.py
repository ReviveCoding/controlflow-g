from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

from filelock import FileLock

from controlflow.core.state import ProjectPaths, atomic_write_json, canonical_json, utc_now
from controlflow.schemas import HumanDecision, ReviewDecision

F = TypeVar("F", bound=Callable[..., Any])


def ledger_locked(function: F) -> F:
    @wraps(function)
    def synchronized(self: ActionLedger, *args: Any, **kwargs: Any) -> Any:
        with self.protocol_lock:
            return function(self, *args, **kwargs)

    return cast(F, synchronized)


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

    def __init__(self, path: Path, *, recovery_authority: Any | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Trust material is deliberately outside the mutable ledger directory.
        default_trust = ProjectPaths.discover().state / "audit_trust"
        trust = Path(os.environ.get("CONTROLFLOW_AUDIT_TRUST_DIR", str(default_trust)))
        trust.mkdir(parents=True, exist_ok=True)
        self.protocol_lock = FileLock(str(path) + ".audit-protocol.lock")
        self.key_path = trust / "root.key"
        ledger_name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        self.head_path = trust / f"{ledger_name}.head.json"
        self.system_head_path = trust / f"{ledger_name}.system.head.json"
        self.recovery_authority = recovery_authority
        with self.protocol_lock:
            if not self.key_path.exists():
                self.key_path.write_bytes(secrets.token_bytes(32))
                self.key_path.chmod(0o400)
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
                """CREATE TABLE IF NOT EXISTS system_audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
                actor_id TEXT NOT NULL, event_at TEXT NOT NULL, detail_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS action_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, action_id INTEGER NOT NULL,
                event_type TEXT NOT NULL, actor_id TEXT NOT NULL, event_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL, detail_hash TEXT NOT NULL DEFAULT '',
                previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(action_events)")}
            if "detail_hash" not in columns:
                connection.execute("ALTER TABLE action_events ADD COLUMN detail_hash TEXT NOT NULL DEFAULT ''")

    def _write_anchor(self, event_hash: str) -> None:
        signature = hmac.new(self.key_path.read_bytes(), event_hash.encode(), hashlib.sha256).hexdigest()
        atomic_write_json(self.head_path, {"event_hash": event_hash, "signature": signature})

    def _write_system_anchor(self, event_hash: str) -> None:
        signature = hmac.new(self.key_path.read_bytes(), event_hash.encode(), hashlib.sha256).hexdigest()
        atomic_write_json(self.system_head_path, {"event_hash": event_hash, "signature": signature})

    @classmethod
    def probe_trust_store(cls) -> Path:
        """Prove atomic durable I/O for the exact configured trust root."""
        trust = Path(os.environ.get("CONTROLFLOW_AUDIT_TRUST_DIR", str(ProjectPaths.discover().state / "audit_trust")))
        trust.mkdir(parents=True, exist_ok=True)
        probe = trust / ".preflight-probe"
        temporary = trust / ".preflight-probe.tmp"
        payload = secrets.token_bytes(32)
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, probe)
        if probe.read_bytes() != payload:
            raise OSError("audit trust-store readback mismatch")
        probe.unlink()
        return trust

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _append_event(
        self, connection: sqlite3.Connection, *, action_id: int, event_type: str, actor_id: str, payload_hash: str
    ) -> None:
        # Bind every event to the complete materialized action state. The
        # caller-supplied detail hash remains in the event hash input as actor
        # context, while payload_hash is the independently reproducible state.
        state_hash = self._action_state_hash(connection, action_id)
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
                    "payload_hash": state_hash,
                    "detail_hash": payload_hash,
                    "previous_hash": previous_hash,
                }
            )
        ).hexdigest()
        connection.execute(
            "INSERT INTO action_events "
            "(action_id,event_type,actor_id,event_at,payload_hash,detail_hash,previous_hash,event_hash) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (action_id, event_type, actor_id, event_at, state_hash, payload_hash, previous_hash, event_hash),
        )

    def _sync_action_anchor(self) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT event_hash FROM action_events ORDER BY event_id DESC LIMIT 1").fetchone()
        if row is not None:
            self._write_anchor(str(row[0]))
        elif self.head_path.exists():
            self.head_path.unlink()

    @staticmethod
    def _action_state_hash(connection: sqlite3.Connection, action_id: int) -> str:
        row = connection.execute(
            "SELECT idempotency_key,case_id,action_type,normalized_payload,workflow_version,status,"
            "requested_at,approved_at,executed_at,result_hash,rollback_id FROM action_ledger WHERE action_id=?",
            (action_id,),
        ).fetchone()
        if row is None:
            raise ValueError("event cannot reference a missing action")
        return hashlib.sha256(canonical_json(list(row))).hexdigest()

    def verify_event_chain(self) -> bool:
        with self._connect() as connection:
            previous = "GENESIS"
            last_event_by_action: dict[int, str] = {}
            allowed_transitions = {
                "REVIEW_REQUESTED": {"REVIEW_APPROVED", "REVIEW_EDITED", "REVIEW_REJECTED"},
                "REVIEW_EDITED": {"REVIEW_APPROVED", "REVIEW_EDITED", "REVIEW_REJECTED"},
                "REVIEW_APPROVED": {"REVIEW_APPROVED", "APPROVED_AND_EXECUTED"},
                "EXECUTED": {"ROLLED_BACK"},
                "APPROVED_AND_EXECUTED": {"ROLLED_BACK"},
                "REVIEW_REJECTED": set(),
                "ROLLED_BACK": set(),
            }
            for row in connection.execute(
                "SELECT action_id,event_type,actor_id,event_at,payload_hash,detail_hash,previous_hash,event_hash "
                "FROM action_events ORDER BY event_id"
            ):
                body = {
                    "action_id": row[0],
                    "event_type": row[1],
                    "actor_id": row[2],
                    "event_at": row[3],
                    "payload_hash": row[4],
                    "detail_hash": row[5],
                    "previous_hash": row[6],
                }
                if row[6] != previous or hashlib.sha256(canonical_json(body)).hexdigest() != row[7]:
                    return False
                action_id = int(row[0])
                event_type = str(row[1])
                prior_type = last_event_by_action.get(action_id)
                if prior_type is None and event_type not in {"REVIEW_REQUESTED", "EXECUTED"}:
                    return False
                if prior_type is not None and event_type not in allowed_transitions.get(prior_type, set()):
                    return False
                last_event_by_action[action_id] = event_type
                previous = row[7]
            for action_id, payload_hash in connection.execute(
                "SELECT e.action_id,e.payload_hash FROM action_events e JOIN "
                "(SELECT action_id,MAX(event_id) AS event_id FROM action_events GROUP BY action_id) latest "
                "ON e.event_id=latest.event_id"
            ):
                if payload_hash != self._action_state_hash(connection, int(action_id)):
                    return False
            orphan = connection.execute(
                "SELECT 1 FROM action_ledger a LEFT JOIN action_events e ON e.action_id=a.action_id "
                "WHERE e.action_id IS NULL LIMIT 1"
            ).fetchone()
            if orphan is not None:
                return False
        if previous == "GENESIS" or not self.head_path.exists():
            return previous == "GENESIS" and not self.head_path.exists()
        anchor = json.loads(self.head_path.read_text(encoding="utf-8"))
        expected = hmac.new(self.key_path.read_bytes(), previous.encode(), hashlib.sha256).hexdigest()
        return anchor.get("event_hash") == previous and hmac.compare_digest(anchor.get("signature", ""), expected)

    @ledger_locked
    def record_system_event(self, event_type: str, actor_id: str, detail: dict[str, Any]) -> None:
        """Persist a separately chained tool/authorization event."""
        detail_json = canonical_json(detail).decode()
        self._require_integrity()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous_row = connection.execute(
                "SELECT event_hash FROM system_audit_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            previous = str(previous_row[0]) if previous_row else "GENESIS"
            event_at = utc_now()
            body = {
                "event_type": event_type,
                "actor_id": actor_id,
                "event_at": event_at,
                "detail_json": detail_json,
                "previous_hash": previous,
            }
            event_hash = hashlib.sha256(canonical_json(body)).hexdigest()
            connection.execute(
                "INSERT INTO system_audit_events "
                "(event_type,actor_id,event_at,detail_json,previous_hash,event_hash) VALUES (?,?,?,?,?,?)",
                (event_type, actor_id, event_at, detail_json, previous, event_hash),
            )
            connection.execute("COMMIT")
        self._write_system_anchor(event_hash)

    @ledger_locked
    def recovery_binding(self, reason: str) -> tuple[dict[str, Any], str]:
        """Return the exact database state an independent recovery approval must bind."""
        with self._connect() as connection:
            action = connection.execute(
                "SELECT event_hash FROM action_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            system = connection.execute(
                "SELECT event_hash FROM system_audit_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
        payload = {
            "ledger_path_sha256": hashlib.sha256(str(self.path.resolve()).encode()).hexdigest(),
            "action_head": str(action[0]) if action else "GENESIS",
            "system_head": str(system[0]) if system else "GENESIS",
            "reason": reason,
        }
        return payload, hashlib.sha256(canonical_json(payload)).hexdigest()

    @ledger_locked
    def reconcile_external_anchors(
        self,
        *,
        authorization_token: str,
        actor_id: str,
        reason: str,
    ) -> None:
        """Recover crash-window anchors only with an independently signed approval."""
        payload, evidence_hash = self.recovery_binding(reason)
        recovery_key = action_key("AUDIT-RECOVERY", "reconcile_audit_anchors", payload, "audit-protocol-v1")
        if self.recovery_authority is None:
            raise PermissionError("no trusted audit-recovery verifier is configured")
        approval = self.recovery_authority.verify(
            authorization_token,
            expected_action_key=recovery_key,
            policy_version="audit-recovery-v1",
            evidence_hash=evidence_hash,
        )
        if approval["decision"] != ReviewDecision.APPROVE.value or approval["reviewer_id"] != actor_id:
            raise PermissionError("audit recovery requires an entitled approval bound to the observed database heads")
        self._sync_action_anchor()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT event_hash FROM system_audit_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
        if row is not None:
            self._write_system_anchor(str(row[0]))
        elif self.system_head_path.exists():
            self.system_head_path.unlink()
        self.record_system_event(
            "AUDIT_ANCHOR_RECOVERY",
            actor_id,
            {
                "reason": reason,
                "binding": payload,
                "approval_sha256": hashlib.sha256(authorization_token.encode()).hexdigest(),
            },
        )

    def _require_integrity(self) -> None:
        if not self.verify_event_chain():
            raise RuntimeError("action audit chain failed closed")
        if not self.verify_system_event_chain():
            raise RuntimeError("system audit chain failed closed")

    def verify_system_event_chain(self) -> bool:
        with self._connect() as connection:
            previous = "GENESIS"
            for row in connection.execute(
                "SELECT event_type,actor_id,event_at,detail_json,previous_hash,event_hash "
                "FROM system_audit_events ORDER BY event_id"
            ):
                body = {
                    "event_type": row[0],
                    "actor_id": row[1],
                    "event_at": row[2],
                    "detail_json": row[3],
                    "previous_hash": row[4],
                }
                if row[4] != previous or hashlib.sha256(canonical_json(body)).hexdigest() != row[5]:
                    return False
                previous = row[5]
        if previous == "GENESIS":
            return not self.system_head_path.exists()
        if not self.system_head_path.exists():
            return False
        anchor = json.loads(self.system_head_path.read_text(encoding="utf-8"))
        expected = hmac.new(self.key_path.read_bytes(), previous.encode(), hashlib.sha256).hexdigest()
        return anchor.get("event_hash") == previous and hmac.compare_digest(anchor.get("signature", ""), expected)

    def execution_event_count(self, action_id: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM action_events WHERE action_id=? "
                "AND event_type IN ('EXECUTED','APPROVED_AND_EXECUTED')",
                (action_id,),
            ).fetchone()
        return int(row[0])

    @ledger_locked
    def rollback(
        self,
        action_id: int,
        *,
        actor_id: str,
        reason: str,
        authorization_token: str,
        approval_authority: Any,
        policy_version: str,
        evidence_hash: str,
    ) -> ActionReceipt:
        self._require_integrity()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT idempotency_key,status,result_hash,case_id,workflow_version "
                "FROM action_ledger WHERE action_id=?",
                (action_id,),
            ).fetchone()
            if row is None or row[1] != "EXECUTED":
                connection.execute("ROLLBACK")
                raise ValueError("only an executed action can be rolled back")
            rollback_payload = {"target_action_id": action_id, "reason": reason}
            rollback_key = action_key(str(row[3]), "rollback", rollback_payload, str(row[4]))
            approval = approval_authority.verify(
                authorization_token,
                expected_action_key=rollback_key,
                policy_version=policy_version,
                evidence_hash=evidence_hash,
            )
            if approval["decision"] != ReviewDecision.APPROVE.value or approval["reviewer_id"] != actor_id:
                connection.execute("ROLLBACK")
                raise PermissionError("rollback requires an entitled reviewer approval bound to this rollback")
            rollback_id = hashlib.sha256(canonical_json(rollback_payload)).hexdigest()
            connection.execute(
                "UPDATE action_ledger SET status='ROLLED_BACK',rollback_id=? WHERE action_id=?",
                (rollback_id, action_id),
            )
            self._append_event(
                connection,
                action_id=action_id,
                event_type="ROLLED_BACK",
                actor_id=actor_id,
                payload_hash=rollback_id,
            )
            connection.execute("COMMIT")
        self._sync_action_anchor()
        return ActionReceipt(action_id, str(row[0]), "ROLLED_BACK", executed=False)

    @ledger_locked
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
        self._require_integrity()
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
            if approval["decision"] not in {ReviewDecision.APPROVE.value, "SYSTEM_AUTO"}:
                connection.execute("ROLLBACK")
                raise PermissionError("only an APPROVE or bounded system decision can execute an action")
            reviewer_id = str(approval["reviewer_id"])
            if row is not None:
                if row[1] == "PENDING_REVIEW":
                    approved = connection.execute(
                        "SELECT 1 FROM action_events WHERE action_id=? AND event_type='REVIEW_APPROVED' "
                        "AND actor_id=? ORDER BY event_id DESC LIMIT 1",
                        (row[0], reviewer_id),
                    ).fetchone()
                    if reviewer_id != "SYSTEM_AUTO" and approved is None:
                        connection.execute("ROLLBACK")
                        raise PermissionError("no persisted entitled approval for this action")
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
                    self._sync_action_anchor()
                    return ActionReceipt(row[0], key, "EXECUTED", executed=True)
                connection.execute("COMMIT")
                return ActionReceipt(row[0], key, row[1], executed=False)
            if reviewer_id != "SYSTEM_AUTO":
                connection.execute("ROLLBACK")
                raise PermissionError("human approval requires a persisted PENDING_REVIEW request")
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
        self._sync_action_anchor()
        return ActionReceipt(action_id, key, "EXECUTED", executed=True)

    @ledger_locked
    def request_review(
        self, *, case_id: str, action_type: str, payload: dict[str, Any], workflow_version: str
    ) -> ActionReceipt:
        self._require_integrity()
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
        self._sync_action_anchor()
        return ActionReceipt(action_id, key, "PENDING_REVIEW", executed=False)

    @ledger_locked
    def record_review(
        self,
        action_id: int,
        decision: HumanDecision,
        *,
        authorization_token: str,
        approval_authority: Any,
        policy_version: str,
        evidence_hash: str,
    ) -> ActionReceipt:
        self._require_integrity()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT idempotency_key,status,case_id,workflow_version FROM action_ledger WHERE action_id=?",
                (action_id,),
            ).fetchone()
            if row is None or row[1] != "PENDING_REVIEW":
                connection.execute("ROLLBACK")
                raise ValueError("review decision requires a pending action")
            if decision.bound_action_hash != row[0]:
                connection.execute("ROLLBACK")
                raise PermissionError("review decision is bound to a different action")
            approval = approval_authority.verify(
                authorization_token,
                expected_action_key=str(row[0]),
                policy_version=policy_version,
                evidence_hash=evidence_hash,
            )
            if (
                approval["reviewer_id"] != decision.reviewer_id
                or approval["reviewer_role"] != decision.reviewer_role
                or approval["reviewer_scope"] != decision.reviewer_scope
                or approval["decision"] != decision.decision.value
            ):
                connection.execute("ROLLBACK")
                raise PermissionError("persisted decision does not match signed reviewer authorization")
            event_type = {
                ReviewDecision.APPROVE: "REVIEW_APPROVED",
                ReviewDecision.EDIT: "REVIEW_EDITED",
                ReviewDecision.REJECT: "REVIEW_REJECTED",
            }[decision.decision]
            next_status = "REJECTED" if decision.decision is ReviewDecision.REJECT else "PENDING_REVIEW"
            resulting_key = str(row[0])
            if decision.decision is ReviewDecision.EDIT:
                if decision.edited_action is None:
                    connection.execute("ROLLBACK")
                    raise ValueError("EDIT requires a typed edited action")
                resulting_key = action_key(
                    str(row[2]),
                    decision.edited_action.action_type,
                    decision.edited_action.payload,
                    str(row[3]),
                )
                connection.execute(
                    "UPDATE action_ledger SET idempotency_key=?,action_type=?,normalized_payload=? WHERE action_id=?",
                    (
                        resulting_key,
                        decision.edited_action.action_type,
                        canonical_json(decision.edited_action.payload).decode(),
                        action_id,
                    ),
                )
            connection.execute("UPDATE action_ledger SET status=? WHERE action_id=?", (next_status, action_id))
            self._append_event(
                connection,
                action_id=action_id,
                event_type=event_type,
                actor_id=decision.reviewer_id,
                payload_hash=hashlib.sha256(canonical_json(decision.model_dump(mode="json"))).hexdigest(),
            )
            connection.execute("COMMIT")
        self._sync_action_anchor()
        return ActionReceipt(action_id, resulting_key, next_status, executed=False)
