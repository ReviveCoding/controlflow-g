from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from controlflow.schemas import (
    AuthorizationOutcome,
    AuthorizationResult,
    IdentityContext,
    Severity,
)


@dataclass(frozen=True)
class ToolPolicyInput:
    tool_name: str
    risk_tier: int
    read_only: bool
    allowed_roles: frozenset[str]
    allowed_scopes: frozenset[str]
    requires_review: bool
    data_classification: int
    requested_scope: str
    case_severity: Severity
    arguments: dict[str, Any]


class LocalPolicyBackend:
    version = "local-policy-v1"

    def authorize(self, identity: IdentityContext, request: ToolPolicyInput) -> AuthorizationResult:
        reasons: list[str] = []
        if not identity.purpose.strip():
            reasons.append("missing_purpose")
        if identity.role not in request.allowed_roles:
            reasons.append("role_not_allowed")
        if request.requested_scope not in request.allowed_scopes:
            reasons.append("scope_not_allowed")
        if request.requested_scope != identity.business_unit:
            reasons.append("business_unit_mismatch")
        if identity.clearance < request.data_classification:
            reasons.append("insufficient_clearance")

        def keys(value: Any) -> set[str]:
            if isinstance(value, dict):
                return {str(key).casefold() for key in value} | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, (list, tuple)):
                return set().union(*(keys(item) for item in value)) if value else set()
            return set()

        if keys(request.arguments) & {"role", "clearance", "user_id", "business_unit", "region", "session_id"}:
            reasons.append("identity_argument_tampering")
        if reasons:
            return AuthorizationResult(
                outcome=AuthorizationOutcome.DENY,
                policy_version=self.version,
                reasons=tuple(sorted(reasons)),
            )
        if request.requires_review or request.risk_tier >= 2 or request.case_severity is Severity.CRITICAL:
            return AuthorizationResult(
                outcome=AuthorizationOutcome.REQUIRE_REVIEW,
                policy_version=self.version,
                reasons=("risk_requires_human_review",),
            )
        outcome = AuthorizationOutcome.ALLOW_READONLY if request.read_only else AuthorizationOutcome.ALLOW
        return AuthorizationResult(outcome=outcome, policy_version=self.version, reasons=("policy_allow",))
