from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from controlflow.audit.ledger import ActionLedger
from controlflow.core.state import atomic_write_json
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import HumanDecision, ReviewDecision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["checkpoint", "review", "approval", "action"])
    parser.add_argument("path")
    parser.add_argument("case_id")
    parser.add_argument("--secret", default="")
    parser.add_argument("--workflow-version", default="v3")
    args = parser.parse_args()
    if args.mode == "checkpoint":
        atomic_write_json(Path(args.path), {"step": "before_fault", "case_id": args.case_id})
    else:
        ledger = ActionLedger(Path(args.path))
        action = {
            "case_id": args.case_id,
            "action_type": "case_update",
            "payload": {"state": "investigated"},
            "workflow_version": args.workflow_version,
        }
        if args.mode == "review":
            ledger.request_review(**action)
        elif args.mode == "approval":
            authority = ApprovalAuthority(
                bytes.fromhex(args.secret), reviewer_entitlements={"reviewer": ("Risk Manager", "enterprise")}
            )
            pending = ledger.request_review(**action)
            if pending.status == "PENDING_REVIEW":
                token = authority.issue(
                    **action,
                    reviewer_id="reviewer",
                    reviewer_role="Risk Manager",
                    reviewer_scope="enterprise",
                    decision=ReviewDecision.APPROVE,
                    policy_version="local-policy-v1",
                    evidence_hash="fault-evidence",
                )
                ledger.record_review(
                    pending.action_id,
                    HumanDecision(
                        reviewer_id="reviewer",
                        reviewer_role="Risk Manager",
                        reviewer_scope="enterprise",
                        decision=ReviewDecision.APPROVE,
                        decided_at=datetime.now(UTC),
                        bound_action_hash=pending.idempotency_key,
                    ),
                    authorization_token=token,
                    approval_authority=authority,
                    policy_version="local-policy-v1",
                    evidence_hash="fault-evidence",
                )
        else:
            authority = ApprovalAuthority(bytes.fromhex(args.secret))
            token = authority.issue_system(
                **action,
                policy_version="local-policy-v1",
                evidence_hash="fault-evidence",
                risk_tier=1,
                authorization_outcome="ALLOW",
            )
            ledger.execute_simulated(
                **action,
                authorization_token=token,
                approval_authority=authority,
                evidence_hash="fault-evidence",
            )
    os._exit(91)


if __name__ == "__main__":
    main()
