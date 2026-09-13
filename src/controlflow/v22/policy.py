from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import canonical_json
from controlflow.v22.schemas import PolicyDecision, PolicyInput, PolicyResult, Severity


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    risk_tier: int
    mode: str
    allowed_roles: frozenset[str]
    allowed_purposes: frozenset[str]
    allowed_scopes: frozenset[str]
    requires_review: bool
    reversible: bool
    state_transition: str


class ActionRegistry:
    def __init__(self, path: Path) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.version = str(payload["version"])
        self.actions = {
            name: ActionDefinition(
                name=name,
                risk_tier=int(value["risk_tier"]),
                mode=str(value["mode"]),
                allowed_roles=frozenset(value["allowed_roles"]),
                allowed_purposes=frozenset(value["allowed_purposes"]),
                allowed_scopes=frozenset(value["allowed_scopes"]),
                requires_review=bool(value["requires_review"]),
                reversible=bool(value["reversible"]),
                state_transition=str(value["state_transition"]),
            )
            for name, value in payload["actions"].items()
        }

    def canonicalize(self, action: str) -> str:
        return action.strip().upper().replace("-", "_").replace(" ", "_")

    def get(self, action: str) -> ActionDefinition | None:
        return self.actions.get(self.canonicalize(action))


class AuthorizationState:
    """Authoritative mutable identity lookup, external to candidate-provided fields."""

    def __init__(self, contexts: dict[str, dict[str, Any]]) -> None:
        self._contexts = {key: dict(value) for key, value in contexts.items()}
        self._lock = threading.RLock()

    def current(self, identity: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._contexts.get(identity)
            return None if value is None else dict(value)

    def replace(self, identity: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._contexts[identity] = dict(value)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the authoritative authorization snapshot through state commit."""
        with self._lock:
            yield


class PolicyDecisionPoint:
    def __init__(self, *, policy_path: Path, registry: ActionRegistry, authorization: AuthorizationState) -> None:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.version = str(payload["version"])
        self.registry = registry
        self.authorization = authorization
        self.novelty_threshold = float(payload["novelty_review_threshold"])
        self.uncertainty_threshold = float(payload["uncertainty_review_threshold"])
        self.critical_requires_review = bool(payload["critical_requires_review"])
        self.high_requires_review = bool(payload["high_requires_review"])
        self.conflict_requires_review = bool(payload["conflict_requires_review"])

    def decide(self, value: PolicyInput) -> PolicyResult:
        current = self.authorization.current(value.context.identity)
        context_payload = value.model_dump(mode="json")
        context_payload["authoritative"] = current
        context_hash = hashlib.sha256(canonical_json(context_payload)).hexdigest()
        action = self.registry.get(value.proposed_action)
        reasons: list[str] = []
        decision = PolicyDecision.ALLOW
        if action is None:
            decision, reasons = PolicyDecision.DENY, ["unknown_action"]
        elif current is None or not bool(current.get("active", False)):
            decision, reasons = PolicyDecision.DENY, ["identity_inactive_or_unknown"]
        elif (
            current.get("role") != value.context.role
            or current.get("business_unit") != value.context.business_unit
            or current.get("region") != value.context.region
            or int(current.get("clearance", -1)) != value.context.clearance
        ):
            decision, reasons = PolicyDecision.DENY, ["authenticated_context_mismatch"]
        elif (
            value.context.requested_scope != value.context.business_unit
            or value.context.clearance < value.context.data_classification
        ):
            decision, reasons = PolicyDecision.DENY, ["scope_or_clearance_denied"]
        elif (
            value.context.role not in action.allowed_roles
            or value.context.purpose not in action.allowed_purposes
            or value.context.requested_scope not in action.allowed_scopes
        ):
            decision, reasons = PolicyDecision.DENY, ["action_registry_authorization_denied"]
        elif not value.evidence_sufficient:
            decision, reasons = PolicyDecision.INSUFFICIENT_EVIDENCE, ["evidence_insufficient"]
        else:
            review_reasons = []
            if action.requires_review:
                review_reasons.append("action_registry_requires_review")
            if value.severity is Severity.CRITICAL and self.critical_requires_review:
                review_reasons.append("critical_severity")
            if value.severity is Severity.HIGH and self.high_requires_review:
                review_reasons.append("high_severity")
            if value.conflict_state and self.conflict_requires_review:
                review_reasons.append("evidence_conflict")
            if value.novelty >= self.novelty_threshold:
                review_reasons.append("novelty")
            if value.uncertainty >= self.uncertainty_threshold:
                review_reasons.append("uncertainty")
            if review_reasons:
                decision, reasons = PolicyDecision.REQUIRE_REVIEW, review_reasons
            else:
                reasons = ["authorized_low_risk_action"]
        return PolicyResult(
            policy_decision_id=str(uuid.uuid4()),
            decision=decision,
            policy_version=self.version,
            action_registry_version=self.registry.version,
            action_risk_tier=None if action is None else action.risk_tier,
            reasons=tuple(reasons),
            authorization_context_hash=context_hash,
        )
