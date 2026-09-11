from __future__ import annotations

from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.schemas import AuthorizationOutcome, IdentityContext, Severity


def identity(role: str = "Control Analyst", clearance: int = 2) -> IdentityContext:
    return IdentityContext(
        user_id="u1",
        role=role,
        business_unit="consumer",
        region="US",
        clearance=clearance,
        purpose="control investigation",
        session_id="s1",
    )


def request(**changes: object) -> ToolPolicyInput:
    values = {
        "tool_name": "query_case_data",
        "risk_tier": 0,
        "read_only": True,
        "allowed_roles": frozenset({"Control Analyst"}),
        "allowed_scopes": frozenset({"consumer"}),
        "requires_review": False,
        "data_classification": 2,
        "requested_scope": "consumer",
        "case_severity": Severity.MEDIUM,
        "arguments": {"case_id": "c1"},
    }
    values.update(changes)
    return ToolPolicyInput(**values)  # type: ignore[arg-type]


def test_readonly_allowed() -> None:
    result = LocalPolicyBackend().authorize(identity(), request())
    assert result.outcome is AuthorizationOutcome.ALLOW_READONLY


def test_privilege_escalation_and_role_spoofing_denied() -> None:
    assert LocalPolicyBackend().authorize(identity(clearance=1), request()).outcome is AuthorizationOutcome.DENY
    spoofed = request(arguments={"case_id": "c1", "role": "Risk Manager"})
    assert LocalPolicyBackend().authorize(identity(), spoofed).outcome is AuthorizationOutcome.DENY


def test_high_risk_requires_review() -> None:
    result = LocalPolicyBackend().authorize(identity(), request(risk_tier=2, read_only=False))
    assert result.outcome is AuthorizationOutcome.REQUIRE_REVIEW
