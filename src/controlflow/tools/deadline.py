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
    """Guard irreversible commits against timeout and caller cancellation.

    A cancellation waits for any active commit section to exit and then rejects
    future commit entries. Entering or leaving the section does not itself prove
    that an external side effect was durably committed; callers must reconcile
    the actual result with their authoritative store.
    """

    deadline: float
    cancelled: Event = field(default_factory=Event)
    _commit_permit: Lock = field(default_factory=Lock)
    _commit_started: bool = False
    _commit_completed: bool = False

    def cancel(self) -> bool:
        """Prevent further commits, waiting for any active commit section to exit.

        Return True if a commit section was entered, not if its side effects
        succeeded. Callers must reconcile a possible commit before retrying.
        """
        with self._commit_permit:
            commit_entered = self._commit_started
            self.cancelled.set()
            return commit_entered

    def assert_active(self) -> None:
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise ToolDeadlineExpired("tool deadline lease expired before commit")

    @property
    def commit_completed(self) -> bool:
        """Whether a commit section exited, even if its body raised an error."""
        with self._commit_permit:
            return self._commit_completed

    @contextmanager
    def commit_section(self) -> Iterator[None]:
        """Serialize an irreversible operation with caller-side cancellation.

        Once entered, the section runs to its normal exit or exception. The
        lease cannot interrupt or roll back an already-started side effect.
        """
        with self._commit_permit:
            self.assert_active()
            self._commit_started = True
            try:
                yield
            finally:
                # Once an irreversible commit zone starts, the caller must
                # reconcile it even when anchor export or result delivery
                # fails. Cancellation waits on this permit.
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
