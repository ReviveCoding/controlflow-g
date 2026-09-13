from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from controlflow.schemas import AuthorizationOutcome, Disposition, Severity


class RootCauseCode(StrEnum):
    ROUTINE_VARIANCE = "ROUTINE_VARIANCE"
    OWNERSHIP_AMBIGUITY = "OWNERSHIP_AMBIGUITY"
    MATERIAL_CONTROL_BREAKDOWN = "MATERIAL_CONTROL_BREAKDOWN"
    NOVEL_THIRD_PARTY_FAILURE = "NOVEL_THIRD_PARTY_FAILURE"
    SOURCE_EVIDENCE_MISSING = "SOURCE_EVIDENCE_MISSING"
    AUTHORITATIVE_SOURCE_CONFLICT = "AUTHORITATIVE_SOURCE_CONFLICT"
    TEMPORAL_POLICY_MISMATCH = "TEMPORAL_POLICY_MISMATCH"
    AUTHORIZATION_SCOPE_VIOLATION = "AUTHORIZATION_SCOPE_VIOLATION"
    PROMPT_INJECTION_ATTEMPT = "PROMPT_INJECTION_ATTEMPT"


class RecommendedAction(StrEnum):
    CLOSE_NO_ACTION = "CLOSE_NO_ACTION"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    ESCALATE_CONTROL_OWNER = "ESCALATE_CONTROL_OWNER"
    INITIATE_REMEDIATION_REVIEW = "INITIATE_REMEDIATION_REVIEW"
    DENY_UNAUTHORIZED_ACTION = "DENY_UNAUTHORIZED_ACTION"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    source: str
    summary: str
    classification: int = Field(ge=0, le=5)
    authorized: bool
    business_valid_from: datetime
    business_valid_to: datetime | None = None
    system_known_from: datetime
    system_known_to: datetime | None = None

    def valid_at(self, event_time: datetime, known_time: datetime) -> bool:
        return (
            self.business_valid_from <= event_time
            and (self.business_valid_to is None or event_time < self.business_valid_to)
            and self.system_known_from <= known_time
            and (self.system_known_to is None or known_time < self.system_known_to)
        )


class RiskProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    critical: float = Field(ge=0, le=1)
    low: float = Field(ge=0, le=1)
    medium: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sum_to_one(self) -> RiskProbabilities:
        if abs(self.critical + self.low + self.medium + self.high - 1.0) > 1e-6:
            raise ValueError("risk probabilities must sum to one")
        return self


class CanonicalEvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    case_id: str
    event_time: datetime
    system_time: datetime
    narrative: str
    case_facts: dict[str, str | int | float | bool]
    risk_probabilities: RiskProbabilities
    critical_review_threshold: float = Field(ge=0, le=1)
    severity_candidate: Severity
    root_cause_candidate: RootCauseCode
    applicable_controls: tuple[str, ...]
    applicable_regulations: tuple[str, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    authorization_outcome: AuthorizationOutcome
    allowed_actions: tuple[RecommendedAction, ...]
    evidence_sufficient: bool


class InvestigationDecision(BaseModel):
    """The sole JSON-schema constrained analyst-facing generation contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Severity
    disposition: Disposition
    root_cause_code: RootCauseCode
    recommended_action: RecommendedAction
    supporting_evidence_ids: tuple[str, ...]
    rationale: str = Field(min_length=1, max_length=800)


class VerifiedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    syntactically_valid: bool
    semantic_valid: bool
    evidence_ids_exist: bool
    evidence_authorized: bool
    temporal_valid: bool
    action_allowed: bool
    coherent_insufficient_evidence: bool
    typed_candidates_match: bool
    evidence_set_complete: bool
    rationale_grounded: bool
    errors: tuple[str, ...] = ()


class AssembledDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    severity: Severity
    disposition: Disposition
    root_cause_code: RootCauseCode
    recommended_action: RecommendedAction
    supporting_evidence_ids: tuple[str, ...]
    rationale: str
    llm_semantic_valid: bool
    authorization_outcome: AuthorizationOutcome
