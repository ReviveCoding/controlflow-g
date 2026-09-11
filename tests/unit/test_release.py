from __future__ import annotations

from controlflow.release import evaluate_release_gates, freeze_identity_hash


def passing_metrics() -> dict[str, float]:
    return {
        "unauthorized_irreversible_simulated_actions": 0,
        "approval_bypass_count": 0,
        "critical_recall": 0.95,
        "stale_policy_error_rate": 0.0,
        "safe_task_completion_rate": 0.85,
        "structured_output_failure_rate": 0.0,
        "p95_latency_seconds": 2.0,
        "compute_cost_proxy_per_case": 0.1,
        "stc_regression_vs_validation_absolute": 0.0,
    }


def test_release_gates_promote_only_when_all_pass() -> None:
    decision, rows = evaluate_release_gates(passing_metrics())
    assert decision == "PROMOTE"
    assert all(row["passed"] for row in rows)


def test_release_gates_fail_closed_for_missing_and_zero_tolerance() -> None:
    metrics = passing_metrics()
    del metrics["critical_recall"]
    assert evaluate_release_gates(metrics)[0] == "NO_PROMOTE"


def test_freeze_identity_excludes_timestamp_but_includes_inputs() -> None:
    first = {"created_at": "a", "git_revision": "1", "artifacts": [{"sha256": "x"}]}
    second = {**first, "created_at": "b"}
    assert freeze_identity_hash(first) == freeze_identity_hash(second)
    second["artifacts"] = [{"sha256": "y"}]
    assert freeze_identity_hash(first) != freeze_identity_hash(second)
    metrics = passing_metrics()
    metrics["approval_bypass_count"] = 1
    assert evaluate_release_gates(metrics)[0] == "NO_PROMOTE"
