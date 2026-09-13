from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd

from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.schemas import AuthorizationOutcome, IdentityContext, Severity
from controlflow.v2.schemas import (
    CanonicalEvidencePacket,
    EvidenceRecord,
    RecommendedAction,
    RiskProbabilities,
    RootCauseCode,
)

POLICY_V2_START = datetime(2024, 1, 1, tzinfo=UTC)


def _aware(value: Any) -> datetime:
    result = cast(datetime, pd.Timestamp(value).to_pydatetime())
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def _severity(risk: RiskProbabilities) -> Severity:
    return max(
        (Severity.LOW, risk.low),
        (Severity.MEDIUM, risk.medium),
        (Severity.HIGH, risk.high),
        (Severity.CRITICAL, risk.critical),
        key=lambda item: item[1],
    )[0]


def _authorize(row: pd.Series[Any], severity: Severity) -> AuthorizationOutcome:
    business_unit = str(row["business_unit"])
    narrative = str(row["narrative"]).casefold()
    embedded_instruction = any(token in narrative for token in ("ignore", "disregard", "override", "system prompt"))
    protected_boundary = any(
        token in narrative for token in ("policy", "approval", "authorization", "unrestricted tool")
    )
    identity = IdentityContext(
        user_id="v2-analyst",
        role="Control Analyst",
        business_unit=business_unit,
        region="US",
        clearance=5,
        purpose="control exception investigation",
        session_id=f"v2-{row['case_id']}",
    )
    request = ToolPolicyInput(
        tool_name="propose_investigation_disposition",
        risk_tier={Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}[severity],
        read_only=False,
        allowed_roles=frozenset({"Control Analyst", "Risk Manager", "Compliance Reviewer"}),
        allowed_scopes=frozenset({business_unit}),
        requires_review=str(row["evidence_status"]) == "CONFLICT" or (embedded_instruction and protected_boundary),
        data_classification=int(row["data_sensitivity"]),
        requested_scope=str(row["requested_scope"]),
        case_severity=severity,
        arguments={"case_id": str(row["case_id"])},
    )
    return LocalPolicyBackend().authorize(identity, request).outcome


def _retrieve_records(row: pd.Series[Any], event_time: datetime, system_time: datetime) -> tuple[EvidenceRecord, ...]:
    """Create and bitemporally filter candidates without reading evaluator truth."""
    if str(row["evidence_status"]) == "MISSING":
        return ()
    classification = int(row["data_sensitivity"])
    candidates: list[EvidenceRecord] = []
    for control in row["control_ids"]:
        candidates.append(
            EvidenceRecord(
                evidence_id=str(control),
                source="control_catalog",
                summary=f"Applicable control {control}",
                classification=classification,
                authorized=True,
                business_valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                system_known_from=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
    for regulation in row["regulation_ids"]:
        candidates.extend(
            (
                EvidenceRecord(
                    evidence_id=f"{regulation}:policy-v1",
                    source="regulation_catalog",
                    summary=f"Historical regulation text for {regulation}",
                    classification=classification,
                    authorized=True,
                    business_valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                    business_valid_to=POLICY_V2_START,
                    system_known_from=datetime(2020, 1, 1, tzinfo=UTC),
                ),
                EvidenceRecord(
                    evidence_id=f"{regulation}:policy-v2",
                    source="regulation_catalog",
                    summary=f"Current regulation text for {regulation}",
                    classification=classification,
                    authorized=True,
                    business_valid_from=POLICY_V2_START,
                    system_known_from=POLICY_V2_START,
                ),
            )
        )
    if str(row["evidence_status"]) == "CONFLICT" and row["control_ids"]:
        control = str(row["control_ids"][0])
        candidates.append(
            EvidenceRecord(
                evidence_id=f"{control}:CONFLICT",
                source="independent_control_observation",
                summary=f"Conflicting observation for {control}",
                classification=classification,
                authorized=True,
                business_valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                system_known_from=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
    return tuple(record for record in candidates if record.valid_at(event_time, system_time))


def build_evidence_packet(
    row: pd.Series[Any],
    *,
    risk_probabilities: RiskProbabilities,
    root_cause_candidate: RootCauseCode,
    severity_candidate: Severity | None = None,
    critical_review_threshold: float = 0.8,
) -> CanonicalEvidencePacket:
    event_time = _aware(row["event_timestamp"])
    severity = severity_candidate or _severity(risk_probabilities)
    authorization = _authorize(row, severity)
    records = () if authorization is AuthorizationOutcome.DENY else _retrieve_records(row, event_time, event_time)
    sufficient = str(row["evidence_status"]) != "MISSING" and bool(records)
    if authorization is AuthorizationOutcome.DENY:
        allowed = (RecommendedAction.DENY_UNAUTHORIZED_ACTION,)
    elif not sufficient:
        allowed = (RecommendedAction.REQUEST_EVIDENCE,)
    elif authorization is AuthorizationOutcome.REQUIRE_REVIEW or severity in {Severity.HIGH, Severity.CRITICAL}:
        allowed = (RecommendedAction.INITIATE_REMEDIATION_REVIEW,)
    elif risk_probabilities.critical >= critical_review_threshold:
        allowed = (RecommendedAction.ESCALATE_CONTROL_OWNER,)
    else:
        allowed = (RecommendedAction.CLOSE_NO_ACTION,)
    return CanonicalEvidencePacket(
        case_id=str(row["case_id"]),
        event_time=event_time,
        system_time=event_time,
        narrative=str(row["narrative"]),
        case_facts={
            "amount": float(row["amount"]),
            "repeat_count": int(row["repeat_count"]),
            "historical_failures": int(row["historical_failures"]),
            "data_sensitivity": int(row["data_sensitivity"]),
            "evidence_status": str(row["evidence_status"]),
        },
        risk_probabilities=risk_probabilities,
        critical_review_threshold=critical_review_threshold,
        severity_candidate=severity,
        root_cause_candidate=root_cause_candidate,
        applicable_controls=tuple(str(value) for value in row["control_ids"]),
        applicable_regulations=tuple(str(value) for value in row["regulation_ids"]),
        evidence_records=records,
        authorization_outcome=authorization,
        allowed_actions=allowed,
        evidence_sufficient=sufficient,
    )
