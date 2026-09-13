from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from controlflow.v22.bundle import verify_bundle
from controlflow.v22.checkpoint import validate_checkpoint, write_checkpoint
from controlflow.v22.executor import TransactionalExecutor
from controlflow.v22.models import calibrated_probability
from controlflow.v22.retrieval import EvidenceRetriever
from controlflow.v22.schemas import (
    AuthenticatedContext,
    CandidateResult,
    ModelDecision,
    PolicyDecision,
    PolicyInput,
    PolicyResult,
    ProposedAction,
    RuntimeCase,
    Severity,
    SignedApproval,
)
from controlflow.v22.temporal import CandidateTemporalRetriever


class CandidateModelBundle:
    def __init__(self, bundle_path: Path, root: Path) -> None:
        self.root = root
        self.payload = verify_bundle(bundle_path, root)
        self.bundle_hash = str(self.payload["bundle_sha256"])
        self.threshold = float(self.payload["critical_threshold"])
        self.critical = joblib.load(root / self.payload["critical_model"]["path"])
        self.calibrator = joblib.load(root / self.payload["calibrator"]["path"])
        self.noncritical = joblib.load(root / self.payload["noncritical_model"]["path"])
        self.root_model = joblib.load(root / self.payload["root_model"]["path"])
        self.novelty = joblib.load(root / self.payload["novelty_model"]["path"])

    def artifact_path(self, name: str) -> Path:
        binding = self.payload[name]
        if not isinstance(binding, dict) or "path" not in binding:
            raise RuntimeError(f"CANDIDATE_BUNDLE_MISMATCH: {name}")
        return self.root / str(binding["path"])

    def infer(
        self, frame: pd.DataFrame, *, hard_critical_routing: bool = True, uncertainty_routing: bool = True
    ) -> ModelDecision:
        probability = float(calibrated_probability(self.critical, self.calibrator, frame)[0])
        noncritical_probabilities = self.noncritical.predict_proba(frame)[0]
        noncritical_labels = list(self.noncritical.classes_)
        noncritical_label = str(noncritical_labels[int(np.argmax(noncritical_probabilities))])
        if hard_critical_routing and probability >= self.threshold:
            severity = Severity.CRITICAL
        else:
            severity = Severity(noncritical_label)
        root = str(self.root_model.predict(frame)[0])
        transformed = self.novelty["preprocessor"].transform(frame)
        raw_novelty = float(-self.novelty["model"].score_samples(transformed)[0])
        novelty = 1.0 if raw_novelty >= float(self.novelty["threshold"]) else raw_novelty
        confidence = max(probability, 1 - probability, float(max(noncritical_probabilities)))
        uncertainty = 1.0 - confidence if uncertainty_routing else 0.0
        return ModelDecision(
            severity=severity,
            critical_probability=probability,
            root_cause=root,
            novelty=novelty,
            uncertainty=uncertainty,
        )


def _action(decision: ModelDecision, evidence_sufficient: bool) -> str:
    if not evidence_sufficient:
        return "REQUEST_EVIDENCE"
    if decision.severity is Severity.CRITICAL:
        return "ESCALATE_CRITICAL"
    if decision.severity is Severity.HIGH:
        return "OPEN_REMEDIATION"
    return "CLOSE_NO_ACTION"


class CandidateRunner:
    def __init__(
        self,
        *,
        bundle: CandidateModelBundle,
        evidence: EvidenceRetriever,
        temporal: CandidateTemporalRetriever,
        executor: TransactionalExecutor,
        workflow_version: str,
        explanation_client: Callable[[dict[str, Any]], tuple[str, bool, float, tuple[str, ...]]] | None = None,
    ) -> None:
        self.bundle = bundle
        self.evidence = evidence
        self.temporal = temporal
        self.executor = executor
        self.workflow_version = workflow_version
        self.explanation_client = explanation_client

    def prepare_case(
        self,
        row: dict[str, Any],
        *,
        temporal_filter: bool = True,
        hard_critical_routing: bool = True,
        uncertainty_routing: bool = True,
    ) -> PreparedCandidate:
        started = time.perf_counter()
        case = RuntimeCase.model_validate(row)
        frame = pd.DataFrame([case.model_dump(mode="json")])
        decision = self.bundle.infer(
            frame, hard_critical_routing=hard_critical_routing, uncertainty_routing=uncertainty_routing
        )
        context = AuthenticatedContext(
            identity=case.authenticated_identity,
            role=case.role,
            business_unit=case.business_unit,
            region=case.region,
            clearance=case.clearance,
            purpose=case.purpose,
            requested_scope=case.requested_scope,
            data_classification=case.data_classification,
        )
        evidence_ids = self.evidence.retrieve(case, context, temporal_filter=temporal_filter)
        evidence_sufficient = self.evidence.sufficient(case, evidence_ids)
        action_name = _action(decision, evidence_sufficient)
        action = ProposedAction(
            case_id=case.case_id,
            action_name=action_name,
            payload={"mode": "simulated", "control_family": case.control_family},
            workflow_version=self.workflow_version,
        )
        policy_input = PolicyInput(
            case_id=case.case_id,
            context=context,
            severity=decision.severity,
            critical_probability=decision.critical_probability,
            evidence_sufficient=evidence_sufficient,
            conflict_state="conflicting" in case.narrative.casefold(),
            novelty=decision.novelty,
            uncertainty=decision.uncertainty,
            proposed_action=action_name,
        )
        policy = self.executor.prepare(action, policy_input)
        return PreparedCandidate(
            started=started,
            case=case,
            decision=decision,
            action=action,
            policy_input=policy_input,
            proposal=policy,
            evidence_ids=evidence_ids,
            policy_id=self.temporal.select(case.event_time, case.system_time, temporal_filter=temporal_filter),
        )

    def complete_case(self, prepared: PreparedCandidate, *, approval: SignedApproval | None) -> CandidateResult:
        case = prepared.case
        decision = prepared.decision
        action = prepared.action
        policy = prepared.proposal
        evidence_ids = prepared.evidence_ids
        event = self.executor.commit(
            action,
            prepared.policy_input,
            approval=approval,
            idempotency_key=f"{case.case_id}:{action.action_name}:{action.workflow_version}",
        )
        deterministic = (
            f"Observed {decision.severity.value} control exception; root cause {decision.root_cause}; "
            f"retrieved {len(evidence_ids)} authorized evidence documents; policy {policy.decision.value}."
        )
        llm_latency: float | None = None
        structured_valid: bool | None = None
        rationale = deterministic
        rationale_claims: tuple[str, ...] = (
            f"severity={decision.severity.value}",
            f"root_cause={decision.root_cause}",
            f"evidence_count={len(evidence_ids)}",
            f"policy_decision={policy.decision.value}",
        )
        if self.explanation_client is not None:
            explanation, structured_valid, llm_latency, rationale_claims = self.explanation_client(
                {
                    "case_id": case.case_id,
                    "deterministic_summary": deterministic,
                    "evidence_ids": list(evidence_ids),
                    "policy_id": prepared.policy_id,
                }
            )
            rationale = explanation
        non_llm = time.perf_counter() - prepared.started - (llm_latency or 0.0)
        total = time.perf_counter() - prepared.started
        return CandidateResult(
            case_id=case.case_id,
            severity=decision.severity,
            critical_probability=decision.critical_probability,
            root_cause=decision.root_cause,
            novelty=decision.novelty,
            uncertainty=decision.uncertainty,
            disposition=PolicyDecision(str(event["pdp_decision"])),
            proposed_action=action.action_name,
            evidence_ids=evidence_ids,
            policy_id=prepared.policy_id,
            policy_decision_id=str(event["policy_decision_id"]),
            execution_event_id=str(event["event_id"]),
            committed=bool(event["committed"]),
            rationale=rationale,
            rationale_claims=rationale_claims,
            structured_output_valid=structured_valid,
            llm_latency_seconds=llm_latency,
            non_llm_latency_seconds=max(0.0, non_llm),
            total_latency_seconds=total,
        )


@dataclass(frozen=True)
class PreparedCandidate:
    started: float
    case: RuntimeCase
    decision: ModelDecision
    action: ProposedAction
    policy_input: PolicyInput
    proposal: PolicyResult
    evidence_ids: tuple[str, ...]
    policy_id: str | None


class CandidateExecutionWorkflow:
    """Orchestrates candidate and a logically separate reviewer callback.

    The candidate never receives an issuer or signing key. The callback is owned by
    the calling review process and may return a signed approval after inspection.
    """

    def __init__(
        self,
        candidate: CandidateRunner,
        *,
        reviewer: Callable[[PreparedCandidate], SignedApproval | None] | None,
    ) -> None:
        self.candidate = candidate
        self.reviewer = reviewer

    def run_case(self, row: dict[str, Any], **case_options: Any) -> CandidateResult:
        prepared = self.candidate.prepare_case(row, **case_options)
        approval = self.reviewer(prepared) if self.reviewer is not None else None
        return self.candidate.complete_case(prepared, approval=approval)

    def run_dataset(
        self,
        frame: pd.DataFrame,
        *,
        output_path: Path,
        partial_path: Path,
        checkpoint_path: Path,
        checkpoint_fields: dict[str, Any],
        checkpoint_interval: int = 25,
        **case_options: Any,
    ) -> list[CandidateResult]:
        completed: list[str] = []
        records: list[dict[str, Any]] = []
        if checkpoint_path.exists():
            completed = validate_checkpoint(checkpoint_path, checkpoint_fields)
            if not partial_path.exists():
                raise RuntimeError("CHECKPOINT_INCOMPATIBLE: partial output missing")
            records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
            partial_ids = [str(item["case_id"]) for item in records]
            if partial_ids[: len(completed)] != completed or len(partial_ids) != len(set(partial_ids)):
                raise RuntimeError("CHECKPOINT_INCOMPATIBLE: partial output does not match checkpoint")
            # A crash can occur after the durable partial append but before its
            # checkpoint update. Ledger idempotency makes adopting this suffix safe.
            completed = partial_ids
            write_checkpoint(checkpoint_path, checkpoint_fields, completed_case_ids=completed)
        else:
            write_checkpoint(checkpoint_path, checkpoint_fields, completed_case_ids=[])
        completed_set = set(completed)
        for row in frame.to_dict(orient="records"):
            if str(row["case_id"]) in completed_set:
                continue
            result = self.run_case(row, **case_options)
            serialized = result.model_dump(mode="json")
            with partial_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(serialized, sort_keys=True) + "\n")
            records.append(serialized)
            completed.append(result.case_id)
            if len(completed) % checkpoint_interval == 0:
                write_checkpoint(checkpoint_path, checkpoint_fields, completed_case_ids=completed)
        write_checkpoint(checkpoint_path, checkpoint_fields, completed_case_ids=completed)
        pd.DataFrame(records).to_parquet(output_path, index=False)
        return [CandidateResult.model_validate(item) for item in records]
