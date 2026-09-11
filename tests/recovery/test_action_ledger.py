from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from controlflow.audit.ledger import ActionLedger
from controlflow.hitl.approval import ApprovalAuthority, InvalidApproval
from controlflow.schemas import HumanDecision, ProposedAction, ReviewDecision


def test_duplicate_action_executes_once(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    kwargs = {
        "case_id": "case-1",
        "action_type": "propose_case_update",
        "payload": {"status": "investigated"},
        "workflow_version": "v1",
    }
    authority = ApprovalAuthority(b"test-secret", reviewer_entitlements={"reviewer-1": ("Risk Manager", "enterprise")})
    kwargs["authorization_token"] = authority.issue_system(
        **{key: kwargs[key] for key in ("case_id", "action_type", "payload", "workflow_version")},
        policy_version="local-policy-v1",
        evidence_hash="none",
        risk_tier=1,
        authorization_outcome="ALLOW",
    )
    kwargs["approval_authority"] = authority
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _: ledger.execute_simulated(**kwargs), range(20)))
    assert sum(receipt.executed for receipt in receipts) == 1
    assert len({receipt.action_id for receipt in receipts}) == 1


def test_unapproved_action_never_executes(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    receipt = ledger.request_review(
        case_id="case-2",
        action_type="propose_case_update",
        payload={"status": "closed"},
        workflow_version="v1",
    )
    assert receipt.status == "PENDING_REVIEW"
    assert not receipt.executed


def test_pending_action_can_be_approved_exactly_once(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    arguments = dict(
        case_id="case-2",
        action_type="propose_case_update",
        payload={"status": "closed"},
        workflow_version="v1",
    )
    authority = ApprovalAuthority(b"test-secret", reviewer_entitlements={"reviewer-1": ("Risk Manager", "enterprise")})
    pending = ledger.request_review(**arguments)
    token = authority.issue(
        **arguments,
        reviewer_id="reviewer-1",
        reviewer_role="Risk Manager",
        reviewer_scope="enterprise",
        decision=ReviewDecision.APPROVE,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    ledger.record_review(
        pending.action_id,
        HumanDecision(
            reviewer_id="reviewer-1",
            reviewer_role="Risk Manager",
            reviewer_scope="enterprise",
            decision=ReviewDecision.APPROVE,
            decided_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            bound_action_hash=pending.idempotency_key,
        ),
        authorization_token=token,
        approval_authority=authority,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    approved = ledger.execute_simulated(
        **arguments, authorization_token=token, approval_authority=authority, evidence_hash="ev"
    )
    replay = ledger.execute_simulated(
        **arguments, authorization_token=token, approval_authority=authority, evidence_hash="ev"
    )
    assert pending.status == "PENDING_REVIEW"
    assert approved.executed is True and approved.status == "EXECUTED"
    assert replay.executed is False and replay.action_id == approved.action_id
    assert ledger.verify_event_chain()
    rollback_payload = {"target_action_id": approved.action_id, "reason": "validation rollback"}
    rollback_token = authority.issue(
        case_id="case-2",
        action_type="rollback",
        payload=rollback_payload,
        workflow_version="v1",
        reviewer_id="reviewer-1",
        reviewer_role="Risk Manager",
        reviewer_scope="enterprise",
        decision=ReviewDecision.APPROVE,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    rolled_back = ledger.rollback(
        approved.action_id,
        actor_id="reviewer-1",
        reason="validation rollback",
        authorization_token=rollback_token,
        approval_authority=authority,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    assert rolled_back.status == "ROLLED_BACK"
    assert ledger.verify_event_chain()


def test_audit_anchor_detects_truncation(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    ledger.request_review(
        case_id="case-anchor", action_type="case_update", payload={"status": "pending"}, workflow_version="v1"
    )
    with ledger._connect() as connection:
        connection.execute("DELETE FROM action_events")
    assert not ledger.verify_event_chain()


def test_audit_chain_detects_materialized_action_mutation(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    ledger.request_review(
        case_id="case-mutation", action_type="case_update", payload={"status": "pending"}, workflow_version="v1"
    )
    assert ledger.verify_event_chain()
    with ledger._connect() as connection:
        connection.execute("UPDATE action_ledger SET normalized_payload='{}'")
    assert not ledger.verify_event_chain()


def test_audit_chain_detects_orphan_action(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "orphan.sqlite")
    with ledger._connect() as connection:
        connection.execute(
            "INSERT INTO action_ledger "
            "(idempotency_key,case_id,action_type,normalized_payload,workflow_version,status,requested_at) "
            "VALUES ('orphan','case','case_update','{}','v1','EXECUTED','now')"
        )
    assert not ledger.verify_event_chain()


def test_durable_system_audit_chain_detects_tampering(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "system.sqlite")
    ledger.record_system_event("TOOL_CALL", "analyst", {"tool": "search_controls", "status": "success"})
    assert ledger.verify_system_event_chain()
    with ledger._connect() as connection:
        connection.execute("UPDATE system_audit_events SET detail_json='{}'")
    assert not ledger.verify_system_event_chain()


def test_corrupt_external_anchor_fails_closed_until_explicit_reconciliation(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "fail-closed.sqlite")
    ledger.record_system_event("TOOL_CALL", "analyst", {"tool": "search_controls"})
    ledger.system_head_path.write_text('{"event_hash":"bad","signature":"bad"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit chain failed closed"):
        ledger.record_system_event("TOOL_CALL", "analyst", {"tool": "search_cases"})
    ledger.reconcile_external_anchors()
    assert ledger.verify_system_event_chain()


def test_concurrent_system_events_preserve_external_anchor_order(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "concurrent-audit.sqlite")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: ledger.record_system_event("TOOL_CALL", "analyst", {"index": index}), range(32)))
    assert ledger.verify_system_event_chain()


def test_unprovisioned_reviewer_cannot_self_assert_entitlement() -> None:
    authority = ApprovalAuthority(b"test-secret")
    with pytest.raises(InvalidApproval, match="not provisioned"):
        authority.issue(
            case_id="case-spoof",
            action_type="case_update",
            payload={"status": "closed"},
            workflow_version="v1",
            reviewer_id="attacker",
            reviewer_role="Risk Manager",
            reviewer_scope="enterprise",
            decision=ReviewDecision.APPROVE,
            policy_version="local-policy-v1",
            evidence_hash="ev",
        )


def test_edited_review_requires_fresh_approval_for_edited_action(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "edited.sqlite")
    authority = ApprovalAuthority(b"test-secret", reviewer_entitlements={"reviewer-1": ("Risk Manager", "enterprise")})
    original = dict(case_id="case-edit", action_type="case_update", payload={"status": "closed"}, workflow_version="v1")
    pending = ledger.request_review(**original)
    edit_token = authority.issue(
        **original,
        reviewer_id="reviewer-1",
        reviewer_role="Risk Manager",
        reviewer_scope="enterprise",
        decision=ReviewDecision.EDIT,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    edited_action = ProposedAction(
        action_type="case_update", payload={"status": "investigated"}, risk_tier=2, rollback_available=True
    )
    edited = ledger.record_review(
        pending.action_id,
        HumanDecision(
            reviewer_id="reviewer-1",
            reviewer_role="Risk Manager",
            reviewer_scope="enterprise",
            decision=ReviewDecision.EDIT,
            decided_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            bound_action_hash=pending.idempotency_key,
            edited_action=edited_action,
        ),
        authorization_token=edit_token,
        approval_authority=authority,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    assert edited.status == "PENDING_REVIEW" and edited.idempotency_key != pending.idempotency_key
    edited_arguments = {**original, "payload": edited_action.payload}
    approve_token = authority.issue(
        **edited_arguments,
        reviewer_id="reviewer-1",
        reviewer_role="Risk Manager",
        reviewer_scope="enterprise",
        decision=ReviewDecision.APPROVE,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    ledger.record_review(
        pending.action_id,
        HumanDecision(
            reviewer_id="reviewer-1",
            reviewer_role="Risk Manager",
            reviewer_scope="enterprise",
            decision=ReviewDecision.APPROVE,
            decided_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            bound_action_hash=edited.idempotency_key,
        ),
        authorization_token=approve_token,
        approval_authority=authority,
        policy_version="local-policy-v1",
        evidence_hash="ev",
    )
    executed = ledger.execute_simulated(
        **edited_arguments,
        authorization_token=approve_token,
        approval_authority=authority,
        evidence_hash="ev",
    )
    assert executed.executed and ledger.verify_event_chain()
