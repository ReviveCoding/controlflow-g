from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel

from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.schemas import AuthorizationOutcome, IdentityContext, Severity

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class ToolSpec(Generic[InputT, OutputT]):
    name: str
    input_model: type[InputT]
    output_model: type[OutputT]
    risk_tier: int
    read_only: bool
    allowed_roles: frozenset[str]
    allowed_data_scopes: frozenset[str]
    human_review_required: bool
    timeout_seconds: float
    max_retries: int
    implementation: Callable[[InputT], OutputT]
    description: str = ""


class ToolDenied(PermissionError):
    pass


class ToolRegistry:
    def __init__(
        self, policy: LocalPolicyBackend, audit_sink: Callable[[str, str, dict[str, Any]], None] | None = None
    ) -> None:
        self.policy = policy
        self.audit_sink = audit_sink
        self._tools: dict[str, ToolSpec[Any, Any]] = {}
        self.audit_events: list[dict[str, Any]] = []

    def _audit(self, identity: IdentityContext, event: dict[str, Any]) -> None:
        self.audit_events.append(event)
        if self.audit_sink is not None:
            self.audit_sink("TOOL_CALL", identity.user_id, event)

    def register(self, spec: ToolSpec[Any, Any]) -> None:
        if re.search(r"ignore\s+(?:all\s+)?previous|override\s+system|unrestricted\s+tool", spec.description, re.I):
            raise ValueError("untrusted tool description contains instruction-like content")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self._tools[spec.name] = spec

    def invoke(
        self,
        name: str,
        raw_input: dict[str, Any],
        *,
        identity: IdentityContext,
        scope: str,
        data_classification: int,
        severity: Severity,
    ) -> BaseModel:
        if name not in self._tools:
            event = {"tool": name, "authorization": "DENY", "status": "unknown_tool"}
            self._audit(identity, event)
            raise ToolDenied(f"unknown tool: {name}")
        spec = self._tools[name]
        try:
            parsed = spec.input_model.model_validate(raw_input)
        except Exception:
            self._audit(identity, {"tool": name, "authorization": "DENY", "status": "invalid_arguments"})
            raise
        authorization = self.policy.authorize(
            identity,
            ToolPolicyInput(
                tool_name=name,
                risk_tier=spec.risk_tier,
                read_only=spec.read_only,
                allowed_roles=spec.allowed_roles,
                allowed_scopes=spec.allowed_data_scopes,
                requires_review=spec.human_review_required,
                data_classification=data_classification,
                requested_scope=scope,
                case_severity=severity,
                arguments=raw_input,
            ),
        )
        if authorization.outcome is AuthorizationOutcome.DENY:
            self._audit(identity, {"tool": name, "authorization": "DENY", "status": "denied"})
            raise ToolDenied(",".join(authorization.reasons))
        if authorization.outcome is AuthorizationOutcome.REQUIRE_REVIEW:
            self._audit(identity, {"tool": name, "authorization": "REQUIRE_REVIEW", "status": "denied"})
            raise ToolDenied("human_review_required")
        last_error: Exception | None = None
        attempts = 1 if not spec.read_only else spec.max_retries + 1
        for attempt in range(attempts):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
            future = executor.submit(spec.implementation, parsed)
            try:
                output = spec.output_model.model_validate(future.result(timeout=spec.timeout_seconds))
                self._audit(
                    identity,
                    {
                        "tool": name,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "status": "success",
                    },
                )
                executor.shutdown(wait=True, cancel_futures=True)
                return cast(BaseModel, output)
            except TimeoutError as exc:
                future.cancel()
                last_error = exc
                self._audit(
                    identity,
                    {
                        "tool": name,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "status": "timeout",
                    },
                )
            except Exception as exc:
                last_error = exc
                self._audit(
                    identity,
                    {
                        "tool": name,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    },
                )
            finally:
                # Threads cannot be forcibly cancelled. Waiting prevents
                # overlap before a read-only retry; writes are never retried.
                executor.shutdown(wait=True, cancel_futures=True)
        raise RuntimeError(f"tool {name} failed after {attempts} attempts") from last_error
