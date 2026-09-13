from __future__ import annotations

from controlflow.schemas import AuthorizationOutcome, Disposition, Severity
from controlflow.v2.schemas import (
    AssembledDecision,
    CanonicalEvidencePacket,
    InvestigationDecision,
    RecommendedAction,
    VerifiedDecision,
)


def deterministic_policy(
    *,
    severity: Severity,
    critical_probability: float,
    evidence_sufficient: bool,
    authorization: AuthorizationOutcome,
    review_threshold: float,
) -> tuple[Disposition, RecommendedAction]:
    if authorization is AuthorizationOutcome.DENY:
        return Disposition.DENY, RecommendedAction.DENY_UNAUTHORIZED_ACTION
    if not evidence_sufficient:
        return Disposition.INSUFFICIENT_EVIDENCE, RecommendedAction.REQUEST_EVIDENCE
    if authorization is AuthorizationOutcome.REQUIRE_REVIEW or severity in {Severity.HIGH, Severity.CRITICAL}:
        return Disposition.REVIEW_REQUIRED, RecommendedAction.INITIATE_REMEDIATION_REVIEW
    if critical_probability >= review_threshold:
        return Disposition.REVIEW_REQUIRED, RecommendedAction.ESCALATE_CONTROL_OWNER
    return Disposition.AUTO, RecommendedAction.CLOSE_NO_ACTION


def verify_generated_decision(
    packet: CanonicalEvidencePacket,
    decision: InvestigationDecision,
) -> VerifiedDecision:
    evidence = {record.evidence_id: record for record in packet.evidence_records}
    errors: list[str] = []
    ids_unique = len(decision.supporting_evidence_ids) == len(set(decision.supporting_evidence_ids))
    if not ids_unique:
        errors.append("duplicate_evidence_id")
    ids_exist = ids_unique and all(identifier in evidence for identifier in decision.supporting_evidence_ids)
    if not ids_exist:
        errors.append("unknown_evidence_id")
    authorized = ids_exist and all(evidence[identifier].authorized for identifier in decision.supporting_evidence_ids)
    if not authorized:
        errors.append("unauthorized_evidence")
    temporal = ids_exist and all(
        evidence[identifier].valid_at(packet.event_time, packet.system_time)
        for identifier in decision.supporting_evidence_ids
    )
    if not temporal:
        errors.append("temporally_invalid_evidence")
    action_allowed = decision.recommended_action in packet.allowed_actions
    if not action_allowed:
        errors.append("action_not_allowed")
    insufficient_coherent = not (
        decision.disposition is Disposition.INSUFFICIENT_EVIDENCE
        and (decision.supporting_evidence_ids or decision.recommended_action is not RecommendedAction.REQUEST_EVIDENCE)
    )
    if not insufficient_coherent:
        errors.append("incoherent_insufficient_evidence")
    expected_ids = {record.evidence_id for record in packet.evidence_records}
    evidence_complete = set(decision.supporting_evidence_ids) == expected_ids
    if not evidence_complete:
        errors.append("incomplete_evidence_set")
    disposition, expected_action = deterministic_policy(
        severity=packet.severity_candidate,
        critical_probability=packet.risk_probabilities.critical,
        evidence_sufficient=packet.evidence_sufficient,
        authorization=packet.authorization_outcome,
        review_threshold=packet.critical_review_threshold,
    )
    typed_match = (
        decision.severity is packet.severity_candidate
        and decision.root_cause_code is packet.root_cause_candidate
        and decision.disposition is disposition
        and decision.recommended_action is expected_action
    )
    if not typed_match:
        errors.append("typed_candidate_mismatch")
    rationale = decision.rationale.casefold()
    rationale_grounded = (
        decision.root_cause_code.value.casefold().replace("_", " ") in rationale
        or any(identifier.casefold() in rationale for identifier in decision.supporting_evidence_ids)
    ) and not any(token in rationale for token in ("unrestricted tool", "ignore policy", "bypass approval"))
    if not rationale_grounded:
        errors.append("ungrounded_rationale")
    semantic = all(
        (
            ids_exist,
            authorized,
            temporal,
            action_allowed,
            insufficient_coherent,
            evidence_complete,
            typed_match,
            rationale_grounded,
        )
    )
    return VerifiedDecision(
        syntactically_valid=True,
        semantic_valid=semantic,
        evidence_ids_exist=ids_exist,
        evidence_authorized=authorized,
        temporal_valid=temporal,
        action_allowed=action_allowed,
        coherent_insufficient_evidence=insufficient_coherent,
        typed_candidates_match=typed_match,
        evidence_set_complete=evidence_complete,
        rationale_grounded=rationale_grounded,
        errors=tuple(errors),
    )


def assemble_decision(
    packet: CanonicalEvidencePacket,
    generated: InvestigationDecision,
    verification: VerifiedDecision,
    *,
    severity: Severity,
    review_threshold: float,
) -> AssembledDecision:
    disposition, action = deterministic_policy(
        severity=severity,
        critical_probability=packet.risk_probabilities.critical,
        evidence_sufficient=packet.evidence_sufficient,
        authorization=packet.authorization_outcome,
        review_threshold=review_threshold,
    )
    evidence_ids = tuple(record.evidence_id for record in packet.evidence_records)
    if verification.semantic_valid:
        rationale = generated.rationale
    elif evidence_ids:
        rationale = f"{packet.root_cause_candidate.value.replace('_', ' ')} supported by {', '.join(evidence_ids)}."
    elif packet.authorization_outcome is AuthorizationOutcome.DENY:
        rationale = f"{packet.root_cause_candidate.value.replace('_', ' ')}; authorization policy denied the request."
    else:
        rationale = (
            f"{packet.root_cause_candidate.value.replace('_', ' ')}; source evidence is missing and must be requested."
        )
    return AssembledDecision(
        case_id=packet.case_id,
        severity=severity,
        disposition=disposition,
        root_cause_code=packet.root_cause_candidate,
        recommended_action=action,
        supporting_evidence_ids=evidence_ids,
        rationale=rationale,
        llm_semantic_valid=verification.semantic_valid,
        authorization_outcome=packet.authorization_outcome,
    )
