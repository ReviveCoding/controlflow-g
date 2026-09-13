from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import canonical_json
from controlflow.v22.approval import ApprovalVerifier, proposed_action_hash
from controlflow.v22.policy import ActionRegistry, PolicyDecisionPoint
from controlflow.v22.schemas import PolicyDecision, PolicyInput, PolicyResult, ProposedAction, SignedApproval


class InjectedCrash(RuntimeError):
    pass


class TransactionalExecutor:
    def __init__(
        self,
        path: Path,
        *,
        pdp: PolicyDecisionPoint,
        registry: ActionRegistry,
        approval_verifier: ApprovalVerifier,
        candidate_bundle_hash: str,
        fault: str | None = None,
    ) -> None:
        self.path = path
        self.pdp = pdp
        self.registry = registry
        self.approval_verifier = approval_verifier
        self.candidate_bundle_hash = candidate_bundle_hash
        self.fault = fault
        self._lock = threading.RLock()
        self._tickets: dict[tuple[str, str], PolicyResult] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS case_state (
                    case_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_decisions (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_decision_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    action_hash TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    action_registry_version TEXT NOT NULL,
                    authorization_context_hash TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_ledger (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    policy_decision_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    action_registry_version TEXT NOT NULL,
                    authorization_context_hash TEXT NOT NULL,
                    approval_token_id TEXT,
                    approval_signature_hash TEXT,
                    candidate_bundle_hash TEXT NOT NULL,
                    action_hash TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    pdp_decision TEXT NOT NULL,
                    review_required INTEGER NOT NULL,
                    approval_valid INTEGER NOT NULL,
                    committed INTEGER NOT NULL,
                    failure_reason TEXT,
                    resulting_state TEXT,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                    ,FOREIGN KEY(policy_decision_id) REFERENCES policy_decisions(policy_decision_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_commit_per_idempotency
                    ON action_ledger(idempotency_key) WHERE committed=1;
                CREATE UNIQUE INDEX IF NOT EXISTS one_event_per_idempotency
                    ON action_ledger(idempotency_key);
                CREATE UNIQUE INDEX IF NOT EXISTS one_logical_action_commit
                    ON action_ledger(case_id,action_hash) WHERE committed=1;
                CREATE TABLE IF NOT EXISTS approval_consumption (
                    token_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    consumed_at TEXT NOT NULL
                    ,FOREIGN KEY(event_id) REFERENCES action_ledger(event_id)
                );
                CREATE TABLE IF NOT EXISTS ledger_head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    event_hash TEXT NOT NULL,
                    sequence_id INTEGER NOT NULL
                );
                """
            )
        if fault == "during_restart":
            # Initialization and recovery inspection complete before the injected
            # process failure, so a subsequent restart can open the same state.
            audit = verify_ledger(path)
            if not audit["valid"]:
                raise RuntimeError("RESTART_LEDGER_RECOVERY_FAILED")
            raise InjectedCrash(fault)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _crash(self, point: str) -> None:
        if self.fault == point:
            raise InjectedCrash(point)

    def prepare(self, action: ProposedAction, policy_input: PolicyInput) -> PolicyResult:
        if action.case_id != policy_input.case_id:
            raise RuntimeError("POLICY_ACTION_CASE_MISMATCH")
        self._crash("before_pdp")
        canonical = self.registry.canonicalize(action.action_name)
        if canonical != action.action_name or canonical != policy_input.proposed_action:
            action = action.model_copy(update={"action_name": canonical})
            policy_input = policy_input.model_copy(update={"proposed_action": canonical})
        result = self.pdp.decide(policy_input)
        self._crash("after_pdp")
        digest = proposed_action_hash(action)
        with self._connect() as conn:
            self._insert_policy_decision(conn, result, action.case_id, digest, "PROPOSAL")
        self._tickets[(action.case_id, digest)] = result
        return result

    @staticmethod
    def _insert_policy_decision(
        conn: sqlite3.Connection, result: PolicyResult, case_id: str, digest: str, stage: str
    ) -> None:
        conn.execute(
            """INSERT INTO policy_decisions
            (policy_decision_id,case_id,action_hash,stage,decision,policy_version,
             action_registry_version,authorization_context_hash,reasons_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                result.policy_decision_id,
                case_id,
                digest,
                stage,
                result.decision.value,
                result.policy_version,
                result.action_registry_version,
                result.authorization_context_hash,
                json.dumps(result.reasons),
                datetime.now(UTC).isoformat(),
            ),
        )

    def commit(
        self,
        action: ProposedAction,
        policy_input: PolicyInput,
        *,
        approval: SignedApproval | None,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if action.case_id != policy_input.case_id:
            raise RuntimeError("POLICY_ACTION_CASE_MISMATCH")
        timestamp = now or datetime.now(UTC)
        canonical = self.registry.canonicalize(action.action_name)
        action = action.model_copy(update={"action_name": canonical})
        policy_input = policy_input.model_copy(update={"proposed_action": canonical})
        digest = proposed_action_hash(action)
        proposal = self._tickets.get((action.case_id, digest))
        if proposal is None:
            proposal = self.prepare(action, policy_input)
        self._crash("before_transaction")
        after_commit_crash = False
        with self.pdp.authorization.locked(), self._lock, self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT * FROM action_ledger WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    if existing["case_id"] != action.case_id or existing["action_hash"] != digest:
                        conn.rollback()
                        raise RuntimeError("IDEMPOTENCY_KEY_BINDING_MISMATCH")
                    conn.rollback()
                    return dict(existing)
                logical_existing = conn.execute(
                    "SELECT event_id FROM action_ledger WHERE case_id=? AND action_hash=? AND committed=1",
                    (action.case_id, digest),
                ).fetchone()
                revalidated = self.pdp.decide(policy_input)
                self._insert_policy_decision(conn, revalidated, action.case_id, digest, "PRE_COMMIT")
                action_definition = self.registry.get(canonical)
                failure: str | None = None
                approval_valid = False
                signature_hash: str | None = None
                approval_token_id: str | None = None
                if logical_existing is not None:
                    failure = "logical_action_already_committed"
                elif (
                    revalidated.decision != proposal.decision
                    or revalidated.policy_version != proposal.policy_version
                    or revalidated.action_registry_version != proposal.action_registry_version
                    or revalidated.authorization_context_hash != proposal.authorization_context_hash
                ):
                    failure = "authorization_changed_since_proposal"
                elif revalidated.decision is PolicyDecision.DENY:
                    failure = "policy_denied"
                elif revalidated.decision is PolicyDecision.INSUFFICIENT_EVIDENCE and canonical != "REQUEST_EVIDENCE":
                    failure = "insufficient_evidence"
                elif action_definition is None:
                    failure = "unknown_action"
                review_required = revalidated.decision is PolicyDecision.REQUIRE_REVIEW
                if approval is not None:
                    approval_token_id = approval.payload.token_id
                    signature_hash = self.approval_verifier.signature_hash(approval)
                if failure is None and review_required:
                    if approval is None:
                        failure = "missing_approval"
                    elif not self.approval_verifier.verify(approval):
                        failure = "invalid_approval_signature"
                    else:
                        payload = approval.payload
                        approval_valid = all(
                            (
                                payload.review_decision == "APPROVE",
                                payload.case_id == action.case_id,
                                payload.action_hash == digest,
                                payload.policy_decision_id == proposal.policy_decision_id,
                                payload.authorization_snapshot_hash == proposal.authorization_context_hash,
                                payload.policy_version == proposal.policy_version,
                                payload.action_registry_version == proposal.action_registry_version,
                                payload.workflow_version == action.workflow_version,
                                payload.issued_at <= timestamp < payload.expires_at,
                            )
                        )
                        if not approval_valid:
                            failure = "approval_binding_or_expiry_invalid"
                        elif conn.execute(
                            "SELECT 1 FROM approval_consumption WHERE token_id=?", (payload.token_id,)
                        ).fetchone():
                            approval_valid = False
                            failure = "approval_already_consumed"
                self._crash("after_approval_verification")
                committed = failure is None
                resulting_state: str | None = None
                if committed:
                    assert action_definition is not None
                    resulting_state = action_definition.state_transition
                    conn.execute(
                        """INSERT INTO case_state(case_id,state,version,updated_at) VALUES(?,?,1,?)
                        ON CONFLICT(case_id) DO UPDATE SET state=excluded.state,
                        version=case_state.version+1,updated_at=excluded.updated_at""",
                        (action.case_id, resulting_state, timestamp.isoformat()),
                    )
                    self._crash("after_state_mutation_before_commit")
                head = conn.execute("SELECT event_hash,sequence_id FROM ledger_head WHERE singleton=1").fetchone()
                previous_hash = "GENESIS" if head is None else str(head["event_hash"])
                event_id = str(uuid.uuid4())
                event_core = {
                    "event_id": event_id,
                    "case_id": action.case_id,
                    "policy_decision_id": revalidated.policy_decision_id,
                    "policy_version": revalidated.policy_version,
                    "action_registry_version": revalidated.action_registry_version,
                    "authorization_context_hash": revalidated.authorization_context_hash,
                    "approval_token_id": approval_token_id,
                    "approval_signature_hash": signature_hash,
                    "candidate_bundle_hash": self.candidate_bundle_hash,
                    "action_hash": digest,
                    "action_name": canonical,
                    "idempotency_key": idempotency_key,
                    "pdp_decision": revalidated.decision.value,
                    "review_required": review_required,
                    "approval_valid": approval_valid,
                    "committed": committed,
                    "failure_reason": failure,
                    "resulting_state": resulting_state,
                    "previous_event_hash": previous_hash,
                    "created_at": timestamp.isoformat(),
                }
                event_hash = hashlib.sha256(canonical_json(event_core)).hexdigest()
                self._crash("during_ledger_write")
                columns = [*list(event_core), "event_hash"]
                values = [int(value) if isinstance(value, bool) else value for value in event_core.values()] + [
                    event_hash
                ]
                conn.execute(
                    f"INSERT INTO action_ledger({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                    values,
                )
                sequence_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                conn.execute(
                    """INSERT INTO ledger_head(singleton,event_hash,sequence_id) VALUES(1,?,?)
                    ON CONFLICT(singleton) DO UPDATE SET
                    event_hash=excluded.event_hash,sequence_id=excluded.sequence_id""",
                    (event_hash, sequence_id),
                )
                if committed and review_required and approval_valid and approval is not None:
                    conn.execute(
                        "INSERT INTO approval_consumption(token_id,event_id,consumed_at) VALUES(?,?,?)",
                        (approval.payload.token_id, event_id, timestamp.isoformat()),
                    )
                conn.commit()
                after_commit_crash = self.fault == "immediately_after_commit"
                row = conn.execute("SELECT * FROM action_ledger WHERE event_id=?", (event_id,)).fetchone()
                result = dict(row)
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
        if after_commit_crash:
            raise InjectedCrash("immediately_after_commit")
        return result

    def rows(self, table: str) -> list[dict[str, Any]]:
        if table not in {"case_state", "policy_decisions", "action_ledger", "approval_consumption"}:
            raise ValueError("unknown table")
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def verify_ledger(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM action_ledger ORDER BY sequence_id")]
        head = conn.execute("SELECT event_hash,sequence_id FROM ledger_head WHERE singleton=1").fetchone()
    failures: list[str] = []
    previous = "GENESIS"
    for expected_sequence, row in enumerate(rows, start=1):
        if row["sequence_id"] != expected_sequence:
            failures.append(f"sequence_gap:{expected_sequence}")
        if row["previous_event_hash"] != previous:
            failures.append(f"broken_predecessor:{row['sequence_id']}")
        core = {key: value for key, value in row.items() if key not in {"sequence_id", "event_hash"}}
        for key in ("review_required", "approval_valid", "committed"):
            core[key] = bool(core[key])
        calculated = hashlib.sha256(canonical_json(core)).hexdigest()
        if calculated != row["event_hash"]:
            failures.append(f"modified_event:{row['sequence_id']}")
        previous = row["event_hash"]
    if rows and (head is None or head["event_hash"] != previous or head["sequence_id"] != rows[-1]["sequence_id"]):
        failures.append("head_mismatch")
    if not rows and head is not None:
        failures.append("head_without_events")
    return {
        "valid": not failures,
        "failures": failures,
        "event_count": len(rows),
        "head": None if head is None else dict(head),
    }


def ledger_security_metrics(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        review_denominator = int(
            conn.execute("SELECT COUNT(*) FROM action_ledger WHERE review_required=1").fetchone()[0]
        )
        deny_denominator = int(
            conn.execute("SELECT COUNT(*) FROM action_ledger WHERE pdp_decision='DENY'").fetchone()[0]
        )
        bypass = int(
            conn.execute(
                "SELECT COUNT(*) FROM action_ledger WHERE committed=1 AND review_required=1 AND approval_valid=0"
            ).fetchone()[0]
        )
        unauthorized = int(
            conn.execute("SELECT COUNT(*) FROM action_ledger WHERE committed=1 AND pdp_decision='DENY'").fetchone()[0]
        )
        duplicates = int(
            conn.execute(
                """SELECT COALESCE(SUM(n-1),0) FROM (
                SELECT COUNT(*) n FROM action_ledger WHERE committed=1
                GROUP BY case_id,action_hash HAVING n>1)"""
            ).fetchone()[0]
        )
    return {
        "approval_bypass_commits": bypass,
        "require_review_opportunities": review_denominator,
        "unauthorized_committed_actions": unauthorized,
        "deny_action_attempts": deny_denominator,
        "duplicate_commits": duplicates,
    }
