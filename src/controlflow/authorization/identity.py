from __future__ import annotations

import secrets
from dataclasses import dataclass

from controlflow.schemas import IdentityContext


@dataclass(frozen=True)
class SessionIdentityProvider:
    """Trusted local authentication adapter used by the production-like simulation."""

    _sessions: dict[str, IdentityContext]

    @classmethod
    def issue_for_business_units(cls, business_units: set[str]) -> tuple[SessionIdentityProvider, dict[str, str]]:
        sessions: dict[str, IdentityContext] = {}
        credentials: dict[str, str] = {}
        for unit in sorted(business_units):
            token = secrets.token_urlsafe(32)
            session_id = secrets.token_hex(16)
            credentials[unit] = token
            sessions[token] = IdentityContext(
                user_id=f"analyst-{unit}",
                role="Control Analyst",
                business_unit=unit,
                region="US",
                clearance=2,
                purpose="control exception investigation",
                session_id=session_id,
            )
        return cls(sessions), credentials

    def resolve(self, session_token: str) -> IdentityContext:
        try:
            return self._sessions[session_token]
        except KeyError as exc:
            raise PermissionError("invalid authenticated session") from exc
