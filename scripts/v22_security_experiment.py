from __future__ import annotations

import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer, ApprovalVerifier
from controlflow.v22.executor import TransactionalExecutor, ledger_security_metrics, verify_ledger
from controlflow.v22.policy import ActionRegistry, AuthorizationState, PolicyDecisionPoint
from controlflow.v22.schemas import AuthenticatedContext, PolicyInput, ProposedAction, Severity

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ledger = ROOT / "artifacts/v22/security_negative_experiment_r2.sqlite"
    if ledger.exists():
        raise RuntimeError("security experiment is immutable once executed")
    registry = ActionRegistry(ROOT / "configs/v22/action_registry.yaml")
    authorization = AuthorizationState(
        {
            "security-test-user": {
                "active": True,
                "role": "senior_investigator",
                "business_unit": "consumer",
                "region": "US",
                "clearance": 3,
            }
        }
    )
    public = ROOT / "artifacts/v22/approval_public_key.pem"
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public)
    executor = TransactionalExecutor(
        ledger,
        pdp=PolicyDecisionPoint(
            policy_path=ROOT / "configs/v22/policy.yaml", registry=registry, authorization=authorization
        ),
        registry=registry,
        approval_verifier=ApprovalVerifier(public),
        candidate_bundle_hash="SECURITY-EXPERIMENT-NOT-CANDIDATE",
    )
    context = AuthenticatedContext(
        identity="security-test-user",
        role="senior_investigator",
        business_unit="consumer",
        region="US",
        clearance=3,
        purpose="control_exception_investigation",
        requested_scope="consumer",
        data_classification=2,
    )
    outcomes = []
    valid_approval = None
    for index, attack in enumerate(("missing", "tampered", "valid", "reused")):
        case_id = "SECURITY-REVIEW"
        action = ProposedAction(
            case_id=case_id,
            action_name="OPEN_REMEDIATION",
            payload={"mode": "simulated"},
            workflow_version="v22-workflow-1",
        )
        policy_input = PolicyInput(
            case_id=case_id,
            context=context,
            severity=Severity.HIGH,
            critical_probability=0.3,
            evidence_sufficient=True,
            conflict_state=False,
            novelty=0.1,
            uncertainty=0.1,
            proposed_action=action.action_name,
        )
        policy = executor.prepare(action, policy_input)
        if attack == "missing":
            approval = None
        elif attack == "tampered":
            issued = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
            approval = issued.model_copy(update={"signature_b64": issued.signature_b64[:-2] + "AA"})
        elif attack == "valid":
            valid_approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
            approval = valid_approval
        else:
            approval = valid_approval
        event = executor.commit(action, policy_input, approval=approval, idempotency_key=f"attack-{index}")
        outcomes.append({"attack": attack, "committed": bool(event["committed"]), "reason": event["failure_reason"]})
    unknown_action = ProposedAction(
        case_id="SECURITY-UNKNOWN",
        action_name="TRANSFER_MONEY",
        payload={"mode": "simulated"},
        workflow_version="v22-workflow-1",
    )
    unknown_input = PolicyInput(
        case_id=unknown_action.case_id,
        context=context,
        severity=Severity.LOW,
        critical_probability=0.01,
        evidence_sufficient=True,
        conflict_state=False,
        novelty=0.1,
        uncertainty=0.1,
        proposed_action=unknown_action.action_name,
    )
    unknown_event = executor.commit(unknown_action, unknown_input, approval=None, idempotency_key="unknown")
    outcomes.append(
        {
            "attack": "unknown_action",
            "committed": bool(unknown_event["committed"]),
            "reason": unknown_event["failure_reason"],
        }
    )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "authoritative_ledger_metrics": ledger_security_metrics(ledger),
        "ledger_audit": verify_ledger(ledger),
        "outcomes": outcomes,
        "ledger_path": ledger.relative_to(ROOT).as_posix(),
        "ledger_sha256": sha256_file(ledger),
    }
    atomic_write_json(ROOT / "results/v22/security_negative_experiment_r2.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
