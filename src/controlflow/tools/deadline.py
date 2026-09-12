from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Event, Lock


class ToolDeadlineExpired(RuntimeError):
    pass


@dataclass
class ToolDeadlineLease:
    deadline: float
    cancelled: Event = field(default_factory=Event)
    _commit_permit: Lock = field(default_factory=Lock)
    _commit_completed: bool = False

    def cancel(self) -> bool:
        """Cancel before commit, or report that a serialized commit completed."""
        with self._commit_permit:
            if self._commit_completed:
                return False
            self.cancelled.set()
            return True

    def assert_active(self) -> None:
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise ToolDeadlineExpired("tool deadline lease expired before commit")

    @property
    def commit_completed(self) -> bool:
        with self._commit_permit:
            return self._commit_completed

    @contextmanager
    def commit_section(self) -> Iterator[None]:
        """Serialize the irreversible commit with caller-side cancellation."""
        with self._commit_permit:
            self.assert_active()
            yield
            self._commit_completed = True


_CURRENT_LEASE: ContextVar[ToolDeadlineLease | None] = ContextVar("controlflow_tool_deadline", default=None)


def bind_tool_deadline(lease: ToolDeadlineLease) -> Token[ToolDeadlineLease | None]:
    return _CURRENT_LEASE.set(lease)


def reset_tool_deadline(token: Token[ToolDeadlineLease | None]) -> None:
    _CURRENT_LEASE.reset(token)


def assert_tool_deadline_active() -> None:
    lease = _CURRENT_LEASE.get()
    if lease is not None:
        lease.assert_active()


@contextmanager
def tool_commit_section() -> Iterator[None]:
    lease = _CURRENT_LEASE.get()
    if lease is None:
        yield
    else:
        with lease.commit_section():
            yield
