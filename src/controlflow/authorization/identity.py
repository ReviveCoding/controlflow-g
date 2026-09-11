from __future__ import annotations

import hashlib
from dataclasses import dataclass

from controlflow.schemas import IdentityContext


@dataclass(frozen=True)
class SessionIdentityProvider:
    """Trusted local authentication adapter used by the production-like simulation."""

    sessions: dict[str, IdentityContext]

    @classmethod
    def for_business_units(cls, business_units: set[str]) -> SessionIdentityProvider:
        sessions: dict[str, IdentityContext] = {}
        for unit in sorted(business_units):
            token = hashlib.sha256(f"controlflow-authenticated-session:{unit}".encode()).hexdigest()
            sessions[token] = IdentityContext(
                user_id=f"analyst-{unit}",
                role="Control Analyst",
                business_unit=unit,
                region="US",
                clearance=2,
                purpose="control exception investigation",
                session_id=token,
            )
        return cls(sessions)

    def token_for_scope(self, business_unit: str) -> str:
        for token, identity in self.sessions.items():
            if identity.business_unit == business_unit:
                return token
        raise PermissionError(f"no authenticated session for scope {business_unit}")

    def resolve(self, session_token: str) -> IdentityContext:
        try:
            return self.sessions[session_token]
        except KeyError as exc:
            raise PermissionError("invalid authenticated session") from exc
