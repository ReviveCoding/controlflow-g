from __future__ import annotations

from controlflow.schemas import Severity
from controlflow.v21.schemas import PolicyDecision, PolicyInput, PolicyResult

POLICY_VERSION = "v21-policy-1"
READ_ONLY_ACTIONS = frozenset({"CLOSE_NO_ACTION", "REQUEST_EVIDENCE"})


class PythonPolicyBackend:
    """Fail-closed PDP using runtime-observable fields only."""

    version = POLICY_VERSION

    def decide(self, value: PolicyInput) -> PolicyResult:
        authorized = value.requested_scope == value.business_unit and value.clearance >= value.data_classification
        if not authorized:
            return PolicyResult(
                decision=PolicyDecision.DENY, policy_version=self.version, reasons=("scope_or_clearance",)
            )
        if not value.evidence_sufficient:
            return PolicyResult(
                decision=PolicyDecision.INSUFFICIENT_EVIDENCE,
                policy_version=self.version,
                reasons=("required_evidence_missing",),
                permitted_actions=("REQUEST_EVIDENCE",),
            )
        reasons: list[str] = []
        if value.severity in {Severity.HIGH, Severity.CRITICAL}:
            reasons.append("high_or_critical_severity")
        if value.evidence_conflict:
            reasons.append("conflicting_evidence")
        if value.action_risk_tier > 0 or value.requested_action not in READ_ONLY_ACTIONS:
            reasons.append("state_changing_or_high_risk_action")
        if value.uncertainty >= 0.5 or value.novelty >= 0.5:
            reasons.append("uncertainty_or_novelty")
        if reasons:
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_REVIEW,
                policy_version=self.version,
                reasons=tuple(reasons),
                permitted_actions=(value.requested_action,),
            )
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            policy_version=self.version,
            reasons=("within_low_risk_scope",),
            permitted_actions=(value.requested_action,),
        )
