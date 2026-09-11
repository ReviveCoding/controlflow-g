from __future__ import annotations

import argparse
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from controlflow.agents.durable import DurableWorkflowRunner, InjectedWorkflowCrash
from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig
from controlflow.audit.ledger import ActionLedger
from controlflow.core.state import ProjectPaths, atomic_write_json
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import HumanDecision, ReviewDecision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["checkpoint", "review", "approval", "action", "durable"])
    parser.add_argument("path")
    parser.add_argument("case_id")
    parser.add_argument("--secret", default="")
    parser.add_argument("--workflow-version", default="v3")
    parser.add_argument("--ledger", default="")
    parser.add_argument("--fault", default="after_execute_before_checkpoint")
    args = parser.parse_args()
    if args.mode == "durable":
        paths = ProjectPaths.discover()
        frame = pd.read_parquet(paths.root / "data/silver/synthetic_cases_development.parquet")
        row = frame.loc[frame.case_id.eq(args.case_id)].iloc[0]
        controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
        workflow = GovernedWorkflow(
            frame,
            controls,
            ActionLedger(Path(args.ledger)),
            ApprovalAuthority(secrets.token_bytes(32)),
        )
        config = WorkflowConfig(
            retrieval=False,
            temporal_retrieval=False,
            reranker=False,
            ml_risk=False,
            calibration=False,
            anomaly=False,
            verifier=False,
            hitl=False,
        )
        try:
            DurableWorkflowRunner(workflow, Path(args.path)).run(row, config, ("LOW", "AUTO", True), fault=args.fault)
        except InjectedWorkflowCrash:
            os._exit(91)
        raise RuntimeError("durable fault was not injected")
    elif args.mode == "checkpoint":
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
