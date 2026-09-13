from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from controlflow.schemas import ReviewDecision, Severity
from controlflow.v21.data import load_runtime_cases
from controlflow.v21.executor import ActionLedger, SimulatedActionExecutor, action_hash
from controlflow.v21.injection import detect_instruction_attack
from controlflow.v21.policy import PythonPolicyBackend
from controlflow.v21.rationale import decompose_claims, deterministic_rationale
from controlflow.v21.schemas import (
    ApprovalToken,
    CandidateDecision,
    PolicyDecision,
    PolicyInput,
    PolicyResult,
    ProposedAction,
)
from controlflow.v21.severity import hard_route_severity
from controlflow.v21.temporal import select_policy

CRITICAL_THRESHOLD = 0.2
WORKFLOW_VERSION = "v21-workflow-1"


@dataclass(frozen=True)
class RuntimeConfig:
    hard_critical_routing: bool = True
    mandatory_review_policy: bool = True
    temporal_filter: bool = True
    constrained_decoding: bool = True
    semantic_verifier: bool = True
    uncertainty_routing: bool = True


def run_candidate(
    runtime_path: Path,
    output_path: Path,
    ledger_path: Path,
    config: RuntimeConfig | None = None,
) -> Path:
    """Execute using runtime data only; this API intentionally has no truth-path argument."""
    config = config or RuntimeConfig()
    cases = load_runtime_cases(runtime_path)
    ledger = ActionLedger(ledger_path)
    executor = SimulatedActionExecutor(ledger)
    backend = PythonPolicyBackend()
    outputs = []
    for case in cases:
        started = time.perf_counter()
        if config.hard_critical_routing:
            severity = hard_route_severity(
                case.critical_probability, CRITICAL_THRESHOLD, case.noncritical_probabilities
            )
        else:
            composed = {
                **{
                    key: value * (1 - case.critical_probability)
                    for key, value in case.noncritical_probabilities.items()
                },
                "CRITICAL": case.critical_probability,
            }
            severity = Severity(max(composed, key=composed.__getitem__))
        policy_input = PolicyInput(
            case_id=case.case_id,
            identity=case.identity,
            role=case.role,
            business_unit=case.business_unit,
            requested_scope=case.requested_scope,
            region=case.region,
            clearance=case.clearance,
            purpose=case.purpose,
            data_classification=case.data_classification,
            critical_probability=case.critical_probability,
            severity=severity,
            evidence_sufficient=case.evidence_sufficient,
            evidence_conflict=case.evidence_conflict,
            uncertainty=case.uncertainty if config.uncertainty_routing else 0.0,
            novelty=case.novelty if config.uncertainty_routing else 0.0,
            requested_action=case.requested_action,
            action_risk_tier=case.action_risk_tier,
        )
        policy = backend.decide(policy_input)
        if not config.mandatory_review_policy and policy.decision is PolicyDecision.REQUIRE_REVIEW:
            policy = PolicyResult(
                decision=PolicyDecision.ALLOW,
                policy_version=policy.policy_version,
                reasons=("ablation_model_directed_auto",),
                permitted_actions=(case.requested_action,),
            )
        action = ProposedAction(
            case_id=case.case_id,
            action_type=case.requested_action,
            payload=case.action_payload,
            workflow_version=WORKFLOW_VERSION,
            policy_version=policy.policy_version,
        )
        now = datetime.now(UTC)
        approval = None
        if policy.decision.value == "REQUIRE_REVIEW":
            approval = ApprovalToken(
                token_id=str(uuid.uuid4()),
                case_id=case.case_id,
                action_hash=action_hash(action),
                reviewer_id="sim-reviewer",
                review_decision=ReviewDecision.APPROVE,
                policy_version=policy.policy_version,
                workflow_version=WORKFLOW_VERSION,
                issued_at=now,
                expires_at=now + timedelta(minutes=10),
                session_id="v21-run",
            )
        event = executor.execute(
            action,
            policy,
            approval=approval,
            session_id="v21-run",
            idempotency_key=f"{case.case_id}:{action_hash(action)}",
            now=now,
        )
        policy_id = select_policy(case.event_time, case.system_time) if config.temporal_filter else "policy-v3"
        evidence_ids = case.evidence_ids
        if not config.semantic_verifier and evidence_ids:
            evidence_ids = (*evidence_ids, "UNVERIFIED-EVIDENCE")
        rationale = deterministic_rationale(severity.value, policy.decision.value, case.evidence_ids)
        decision = CandidateDecision(
            case_id=case.case_id,
            severity=severity,
            disposition=policy.decision,
            action=case.requested_action,
            evidence_ids=evidence_ids,
            policy_id=policy_id,
            rationale=rationale,
            rationale_claims=decompose_claims(rationale),
            action_event_id=event.event_id,
            latency_seconds=time.perf_counter() - started,
        )
        record = decision.model_dump(mode="json")
        record["injection_detected"] = detect_instruction_attack(case.narrative)
        record["structured_output_valid"] = config.constrained_decoding or not detect_instruction_attack(case.narrative)
        outputs.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    run_candidate(args.runtime, args.output, args.ledger)


if __name__ == "__main__":
    main()
