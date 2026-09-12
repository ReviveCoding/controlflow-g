from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Event


class ToolDeadlineExpired(RuntimeError):
    pass


@dataclass
class ToolDeadlineLease:
    deadline: float
    cancelled: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self.cancelled.set()

    def assert_active(self) -> None:
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise ToolDeadlineExpired("tool deadline lease expired before commit")


_CURRENT_LEASE: ContextVar[ToolDeadlineLease | None] = ContextVar("controlflow_tool_deadline", default=None)


def bind_tool_deadline(lease: ToolDeadlineLease) -> Token[ToolDeadlineLease | None]:
    return _CURRENT_LEASE.set(lease)


def reset_tool_deadline(token: Token[ToolDeadlineLease | None]) -> None:
    _CURRENT_LEASE.reset(token)


def assert_tool_deadline_active() -> None:
    lease = _CURRENT_LEASE.get()
    if lease is not None:
        lease.assert_active()
