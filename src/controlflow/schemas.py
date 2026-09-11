from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Disposition(StrEnum):
    AUTO = "AUTO"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DENY = "DENY"


class AuthorizationOutcome(StrEnum):
    ALLOW = "ALLOW"
    ALLOW_READONLY = "ALLOW_READONLY"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    DENY = "DENY"


class ReviewDecision(StrEnum):
    APPROVE = "APPROVE"
    EDIT = "EDIT"
    REJECT = "REJECT"


class IdentityContext(FrozenModel):
    user_id: str
    role: str
    business_unit: str
    region: str
    clearance: int = Field(ge=0, le=5)
    purpose: str
    session_id: str


class TemporalEvidence(FrozenModel):
    evidence_id: str
    source: str
    text: str
    classification: int = Field(ge=0, le=5)
    business_valid_from: datetime
    business_valid_to: datetime | None = None
    system_known_from: datetime
    system_known_to: datetime | None = None
    authorized_roles: frozenset[str] = frozenset()
    content_sha256: str

    def valid_at(self, event_time: datetime, known_time: datetime) -> bool:
        business = self.business_valid_from <= event_time and (
            self.business_valid_to is None or event_time < self.business_valid_to
        )
        system = self.system_known_from <= known_time and (
            self.system_known_to is None or known_time < self.system_known_to
        )
        return business and system


class RiskPrediction(FrozenModel):
    model_id: str
    probabilities: dict[Severity, float]
    predicted_severity: Severity
    calibrated: bool
    prediction_time: datetime

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> RiskPrediction:
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        return self


class AnomalySignal(FrozenModel):
    model_id: str
    score: float
    threshold: float
    is_anomaly: bool


class ProposedAction(FrozenModel):
    action_type: str
    payload: dict[str, Any]
    risk_tier: int = Field(ge=0, le=3)
    rollback_available: bool


class ClaimVerification(FrozenModel):
    claim_id: str
    claim_text: str
    supporting_evidence_ids: tuple[str, ...]
    source_provenance: tuple[str, ...]
    temporal_validity: bool
    authorization_validity: bool
    conflict_status: bool
    verification_status: str
    confidence: float = Field(ge=0, le=1)


class VerificationResult(FrozenModel):
    claims: tuple[ClaimVerification, ...]
    all_verified: bool
    sufficient_evidence: bool


class AuthorizationResult(FrozenModel):
    outcome: AuthorizationOutcome
    policy_version: str
    reasons: tuple[str, ...]


class HumanDecision(FrozenModel):
    reviewer_id: str
    decision: ReviewDecision
    decided_at: datetime
    bound_action_hash: str
    edited_action: ProposedAction | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    case_id: str
    workflow_version: str
    event_time: datetime
    system_time: datetime
    identity_context: IdentityContext
    purpose: str
    case_context: dict[str, Any]
    applicable_controls: list[str] = Field(default_factory=list)
    applicable_regulations: list[str] = Field(default_factory=list)
    retrieved_evidence: list[TemporalEvidence] = Field(default_factory=list)
    structured_evidence: dict[str, Any] = Field(default_factory=dict)
    risk_prediction: RiskPrediction | None = None
    risk_uncertainty: float | None = Field(default=None, ge=0, le=1)
    anomaly_signal: AnomalySignal | None = None
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    verification_result: VerificationResult | None = None
    authorization_result: AuthorizationResult | None = None
    human_decision: HumanDecision | None = None
    final_disposition: Disposition | None = None


class FinalAgentOutput(FrozenModel):
    case_id: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    controls: tuple[str, ...]
    regulations: tuple[str, ...]
    evidence: tuple[str, ...]
    root_cause_hypotheses: tuple[str, ...]
    recommended_actions: tuple[ProposedAction, ...]
    automation_decision: Disposition
    human_review_required: bool

    @model_validator(mode="after")
    def decision_consistency(self) -> FinalAgentOutput:
        if self.automation_decision is Disposition.REVIEW_REQUIRED and not self.human_review_required:
            raise ValueError("REVIEW_REQUIRED must set human_review_required")
        if self.automation_decision is Disposition.AUTO and self.human_review_required:
            raise ValueError("AUTO cannot require human review")
        return self
