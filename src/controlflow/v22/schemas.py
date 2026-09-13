from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    DENY = "DENY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RuntimeCase(FrozenModel):
    case_id: str
    entity_id: str
    event_time: datetime
    system_time: datetime
    authenticated_identity: str
    role: str
    business_unit: str
    region: str
    clearance: int = Field(ge=0, le=5)
    purpose: str
    requested_scope: str
    data_classification: int = Field(ge=0, le=5)
    control_family: str
    control_test_count: int = Field(ge=0)
    control_test_failures: int = Field(ge=0)
    historical_incidents: int = Field(ge=0)
    transaction_count: int = Field(ge=0)
    anomaly_count: int = Field(ge=0)
    privileged_event_count: int = Field(ge=0)
    repeat_exception_ratio: float = Field(ge=0, le=1)
    customer_impact_signal: float = Field(ge=0, le=1)
    policy_risk_signal: float = Field(ge=0, le=1)
    affected_customers: int = Field(ge=0)
    amount_variance: float
    scope_difference: float
    narrative: str
    evidence_query: str


class EvidenceDocument(FrozenModel):
    document_id: str
    case_id: str | None
    text: str
    control_family: str
    business_unit: str
    classification: int
    valid_from: datetime
    valid_to: datetime | None = None


class PolicyDocument(FrozenModel):
    policy_id: str
    text: str
    business_valid_from: datetime
    business_valid_to: datetime | None
    system_known_from: datetime
    system_known_to: datetime | None = None
    correction_rank: int = 0


class ModelDecision(FrozenModel):
    severity: Severity
    critical_probability: float = Field(ge=0, le=1)
    root_cause: str
    novelty: float = Field(ge=0)
    uncertainty: float = Field(ge=0, le=1)


class ProposedAction(FrozenModel):
    case_id: str
    action_name: str
    payload: dict[str, Any]
    workflow_version: str


class AuthenticatedContext(FrozenModel):
    identity: str
    role: str
    business_unit: str
    region: str
    clearance: int
    purpose: str
    requested_scope: str
    data_classification: int


class PolicyInput(FrozenModel):
    case_id: str
    context: AuthenticatedContext
    severity: Severity
    critical_probability: float
    evidence_sufficient: bool
    conflict_state: bool
    novelty: float
    uncertainty: float
    proposed_action: str


class PolicyResult(FrozenModel):
    policy_decision_id: str
    decision: PolicyDecision
    policy_version: str
    action_registry_version: str
    action_risk_tier: int | None
    reasons: tuple[str, ...]
    authorization_context_hash: str


class ApprovalPayload(FrozenModel):
    token_id: str
    case_id: str
    action_hash: str
    reviewer_id: str
    review_decision: str
    policy_decision_id: str
    authorization_snapshot_hash: str
    policy_version: str
    action_registry_version: str
    workflow_version: str
    issued_at: datetime
    expires_at: datetime
    nonce: str


class SignedApproval(FrozenModel):
    payload: ApprovalPayload
    signature_b64: str
    key_id: str


class CandidateResult(FrozenModel):
    case_id: str
    severity: Severity
    critical_probability: float
    root_cause: str
    novelty: float
    uncertainty: float
    disposition: PolicyDecision
    proposed_action: str
    evidence_ids: tuple[str, ...]
    policy_id: str | None
    policy_decision_id: str
    execution_event_id: str
    committed: bool
    rationale: str
    rationale_claims: tuple[str, ...]
    structured_output_valid: bool | None = None
    llm_latency_seconds: float | None = None
    non_llm_latency_seconds: float
    total_latency_seconds: float


class ExplanationOutput(FrozenModel):
    case_id: str
    summary: str
    claims: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    policy_id: str | None
