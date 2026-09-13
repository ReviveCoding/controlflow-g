from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from controlflow.schemas import ReviewDecision, Severity


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    DENY = "DENY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PepDecision(StrEnum):
    EXECUTE = "EXECUTE"
    BLOCK = "BLOCK"


class RuntimeCase(FrozenModel):
    case_id: str
    event_time: datetime
    system_time: datetime
    identity: str
    role: str
    business_unit: str
    requested_scope: str
    region: str
    clearance: int = Field(ge=0, le=5)
    purpose: str
    data_classification: int = Field(ge=0, le=5)
    narrative: str
    evidence_ids: tuple[str, ...]
    evidence_sufficient: bool
    evidence_conflict: bool
    critical_probability: float = Field(ge=0, le=1)
    noncritical_probabilities: dict[str, float]
    uncertainty: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    requested_action: str
    action_payload: dict[str, Any]
    action_risk_tier: int = Field(ge=0, le=3)


class PolicyInput(FrozenModel):
    case_id: str
    identity: str
    role: str
    business_unit: str
    requested_scope: str
    region: str
    clearance: int
    purpose: str
    data_classification: int
    critical_probability: float
    severity: Severity
    evidence_sufficient: bool
    evidence_conflict: bool
    uncertainty: float
    novelty: float
    requested_action: str
    action_risk_tier: int


class PolicyResult(FrozenModel):
    decision: PolicyDecision
    policy_version: str
    reasons: tuple[str, ...]
    permitted_actions: tuple[str, ...] = ()


class ProposedAction(FrozenModel):
    case_id: str
    action_type: str
    payload: dict[str, Any]
    workflow_version: str
    policy_version: str


class ApprovalToken(FrozenModel):
    token_id: str
    case_id: str
    action_hash: str
    reviewer_id: str
    review_decision: ReviewDecision
    policy_version: str
    workflow_version: str
    issued_at: datetime
    expires_at: datetime
    session_id: str


class ExecutionEvent(FrozenModel):
    event_id: str
    action_id: str
    case_id: str
    action_hash: str
    pdp_decision: PolicyDecision
    pep_decision: PepDecision
    approval_status: str
    approval_token_id: str | None = None
    idempotency_key: str
    attempted: bool
    committed: bool
    denied: bool
    failure_reason: str | None
    unauthorized_attempt: bool
    review_required: bool
    timestamp: datetime


class CandidateDecision(FrozenModel):
    case_id: str
    severity: Severity
    disposition: PolicyDecision
    action: str
    evidence_ids: tuple[str, ...]
    policy_id: str
    rationale: str
    rationale_claims: tuple[str, ...]
    action_event_id: str
    latency_seconds: float = Field(ge=0)
