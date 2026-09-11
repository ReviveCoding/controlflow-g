from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from controlflow.audit.ledger import ActionLedger
from controlflow.hitl.approval import ApprovalAuthority


def test_duplicate_action_executes_once(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "ledger.sqlite")
    kwargs = {
        "case_id": "case-1",
        "action_type": "propose_case_update",
        "payload": {"status": "investigated"},
        "workflow_version": "v1",
    }
    authority = ApprovalAuthority(b"test-secret")
    kwargs["authorization_token"] = authority.issue(
        **{key: kwargs[key] for key in ("case_id", "action_type", "payload", "workflow_version")},
        reviewer_id="SYSTEM_AUTO",
        policy_version="local-policy-v1",
        evidence_hash="none",
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
    authority = ApprovalAuthority(b"test-secret")
    pending = ledger.request_review(**arguments)
    token = authority.issue(**arguments, reviewer_id="reviewer-1", policy_version="local-policy-v1", evidence_hash="ev")
    approved = ledger.execute_simulated(
        **arguments, authorization_token=token, approval_authority=authority, evidence_hash="ev"
    )
    replay = ledger.execute_simulated(
        **arguments, authorization_token=token, approval_authority=authority, evidence_hash="ev"
    )
    assert pending.status == "PENDING_REVIEW"
    assert approved.executed is True and approved.status == "EXECUTED"
    assert replay.executed is False and replay.action_id == approved.action_id
