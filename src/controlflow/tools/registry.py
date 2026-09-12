from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel

from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.core.state import canonical_json
from controlflow.schemas import AuthorizationOutcome, IdentityContext, Severity
from controlflow.tools.deadline import ToolDeadlineLease, bind_tool_deadline, reset_tool_deadline

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
        started = time.perf_counter()
        base_event = {
            "tool": name,
            "argument_sha256": hashlib.sha256(canonical_json(raw_input)).hexdigest(),
            "requested_scope": scope,
            "data_classification": data_classification,
            "severity": severity.value,
            "policy_version": self.policy.version,
            "tool_schema_version": 1,
        }
        if name not in self._tools:
            event = {**base_event, "authorization": "DENY", "status": "unknown_tool"}
            self._audit(identity, event)
            raise ToolDenied(f"unknown tool: {name}")
        spec = self._tools[name]
        try:
            parsed = spec.input_model.model_validate(raw_input)
        except Exception:
            self._audit(identity, {**base_event, "authorization": "DENY", "status": "invalid_arguments"})
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
            self._audit(
                identity,
                {
                    **base_event,
                    "authorization": "DENY",
                    "authorization_reasons": list(authorization.reasons),
                    "status": "denied",
                },
            )
            raise ToolDenied(",".join(authorization.reasons))
        if authorization.outcome is AuthorizationOutcome.REQUIRE_REVIEW:
            self._audit(
                identity,
                {
                    **base_event,
                    "authorization": "REQUIRE_REVIEW",
                    "authorization_reasons": list(authorization.reasons),
                    "status": "denied",
                },
            )
            raise ToolDenied("human_review_required")
        last_error: Exception | None = None
        attempts = 1 if not spec.read_only else spec.max_retries + 1
        for attempt in range(attempts):
            outcome: Queue[tuple[bool, object]] = Queue(maxsize=1)
            lease = ToolDeadlineLease(time.monotonic() + spec.timeout_seconds)

            def run_tool(
                result_queue: Queue[tuple[bool, object]] = outcome,
                implementation: Callable[[BaseModel], BaseModel] = spec.implementation,
                input_value: BaseModel = parsed,
                deadline_lease: ToolDeadlineLease = lease,
            ) -> None:
                token = bind_tool_deadline(deadline_lease)
                try:
                    result_queue.put((True, implementation(input_value)))
                except BaseException as exc:  # propagated on the invoking thread
                    result_queue.put((False, exc))
                finally:
                    reset_tool_deadline(token)

            # A daemon worker makes the deadline enforceable for the caller and
            # process lifecycle. Implementations are bounded, typed local tools;
            # write implementations additionally rely on the idempotent ledger.
            Thread(target=run_tool, name=f"tool-{name}", daemon=True).start()
            try:
                succeeded, value = outcome.get(timeout=spec.timeout_seconds)
                if not succeeded:
                    if isinstance(value, BaseException):
                        raise value
                    raise RuntimeError("tool worker returned an invalid failure")
                output = spec.output_model.model_validate(value)
                self._audit(
                    identity,
                    {
                        **base_event,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "authorization_reasons": list(authorization.reasons),
                        "status": "success",
                        "output_sha256": hashlib.sha256(canonical_json(output.model_dump(mode="json"))).hexdigest(),
                        "duration_seconds": time.perf_counter() - started,
                    },
                )
                return cast(BaseModel, output)
            except Empty as exc:
                lease.cancel()
                last_error = exc
                self._audit(
                    identity,
                    {
                        **base_event,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "status": "timeout_indeterminate",
                        "duration_seconds": time.perf_counter() - started,
                    },
                )
                # A timed-out attempt is never overlapped by a retry. Read
                # calls abort this invocation; write commits recheck the lease
                # inside the only permitted action ledger boundary.
                break
            except PermissionError:
                self._audit(
                    identity,
                    {
                        **base_event,
                        "attempt": attempt + 1,
                        "authorization": "DENY",
                        "status": "argument_scope_denied",
                        "duration_seconds": time.perf_counter() - started,
                    },
                )
                raise
            except Exception as exc:
                last_error = exc
                self._audit(
                    identity,
                    {
                        **base_event,
                        "attempt": attempt + 1,
                        "authorization": authorization.outcome.value,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "duration_seconds": time.perf_counter() - started,
                    },
                )
        raise RuntimeError(f"tool {name} failed after {attempts} attempts") from last_error
