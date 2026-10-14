from __future__ import annotations

import time
from threading import Event, Thread

import pytest

from controlflow.tools.deadline import (
    ToolDeadlineExpired,
    ToolDeadlineLease,
    bind_tool_deadline,
    reset_tool_deadline,
    tool_commit_section,
)


def active_lease() -> ToolDeadlineLease:
    return ToolDeadlineLease(time.monotonic() + 60)


def test_cancellation_before_commit_prevents_entry() -> None:
    lease = active_lease()

    assert lease.cancel() is False
    assert lease.cancel() is False
    assert lease.cancelled.is_set()
    assert lease.commit_completed is False
    with pytest.raises(ToolDeadlineExpired, match="before commit"):
        with lease.commit_section():
            pytest.fail("a cancelled commit must not start")


def test_expired_lease_prevents_commit_entry() -> None:
    lease = ToolDeadlineLease(time.monotonic() - 1)

    with pytest.raises(ToolDeadlineExpired, match="before commit"):
        with lease.commit_section():
            pytest.fail("an expired commit must not start")
    assert lease.commit_completed is False
    assert lease.cancel() is False


def test_commit_section_exit_does_not_imply_successful_side_effect() -> None:
    lease = active_lease()

    with pytest.raises(ValueError, match="write failed"):
        with lease.commit_section():
            raise ValueError("write failed")

    assert lease.commit_completed is True
    assert lease.cancel() is True  # Commit was entered, but durability is unknown.
    with pytest.raises(ToolDeadlineExpired, match="before commit"):
        with lease.commit_section():
            pytest.fail("a cancelled lease must not permit another commit")


def test_cancel_waits_for_active_commit_before_closing_lease() -> None:
    lease = active_lease()
    entered = Event()
    release = Event()
    cancel_invoked = Event()
    cancel_finished = Event()
    failures: list[BaseException] = []
    cancel_results: list[bool] = []

    def commit_worker() -> None:
        try:
            with lease.commit_section():
                entered.set()
                if not release.wait(timeout=5):
                    raise AssertionError("commit worker timed out")
        except BaseException as exc:
            failures.append(exc)

    def cancel_worker() -> None:
        cancel_invoked.set()
        cancel_results.append(lease.cancel())
        cancel_finished.set()

    commit_thread = Thread(target=commit_worker)
    cancel_thread = Thread(target=cancel_worker)
    commit_thread.start()
    try:
        assert entered.wait(timeout=2)
        cancel_thread.start()
        assert cancel_invoked.wait(timeout=2)
        assert not cancel_finished.wait(timeout=0.05)
    finally:
        release.set()
        commit_thread.join(timeout=2)
        if cancel_thread.ident is not None:
            cancel_thread.join(timeout=2)

    assert not commit_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert failures == []
    assert cancel_results == [True]
    assert lease.cancelled.is_set()
    assert lease.commit_completed is True
    with pytest.raises(ToolDeadlineExpired, match="before commit"):
        with lease.commit_section():
            pytest.fail("no new commit may start after cancellation")


def test_context_binding_is_reset_after_cancellation() -> None:
    lease = active_lease()
    token = bind_tool_deadline(lease)
    try:
        lease.cancel()
        with pytest.raises(ToolDeadlineExpired, match="before commit"):
            with tool_commit_section():
                pytest.fail("the bound, cancelled lease must reject commits")
    finally:
        reset_tool_deadline(token)

    with tool_commit_section():
        pass  # No active lease in this context.
