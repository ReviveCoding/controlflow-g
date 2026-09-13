from __future__ import annotations

from typing import Any


def _estimate(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return None
    estimate = value.get("estimate")
    return None if estimate is None else float(estimate)


def apply_gates(
    *,
    metrics: dict[str, Any],
    serving: dict[str, Any],
    integrity: dict[str, Any],
    reviews: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    security = metrics["security"]
    values: dict[str, float | int | None] = {
        "binary_critical_recall": _estimate(metrics, "binary_critical_recall"),
        "typed_critical_recall": _estimate(metrics, "typed_critical_recall"),
        "core_stc": _estimate(metrics, "core_stc"),
        "structured_output_failure_rate": float(serving["structured_failure_rate"]),
        "approval_bypass_commits": int(security["approval_bypass_commits"]),
        "unauthorized_committed_actions": int(security["unauthorized_committed_actions"]),
        "duplicate_commits": int(security["duplicate_commits"]),
        "temporal_policy_accuracy": _estimate(metrics, "temporal_policy_accuracy"),
        "stale_policy_error_rate": _estimate(metrics, "stale_policy_error_rate"),
        "evidence_completeness": _estimate(metrics, "evidence_completeness"),
        "concurrency_2_total_p95_seconds": float(serving["total_p95_seconds"]),
        "leakage_findings": int(integrity["leakage_findings"]),
        "bundle_violations": int(integrity["bundle_violations"]),
        "checkpoint_violations": int(integrity["checkpoint_violations"]),
        "ledger_tamper_verification_failures": int(integrity["ledger_tamper_verification_failures"]),
        "unresolved_blocker": int(reviews["counts"]["unresolved_BLOCKER"]),
        "unresolved_high": int(reviews["counts"]["unresolved_HIGH"]),
    }
    gates = {}
    for name, value in values.items():
        specification = gate_config["gates"][name]
        configured_operator = str(specification["operator"])
        operator = {"gte": ">=", "lte": "<=", "eq": "=="}[configured_operator]
        threshold = float(specification["value"])
        passed = False
        if value is not None:
            if operator == ">=":
                passed = value >= threshold
            elif operator == "<=":
                passed = value <= threshold
            else:
                passed = value == threshold
        gates[name] = {"value": value, "operator": operator, "threshold": threshold, "passed": passed}
    return {"gates": gates, "all_passed": all(item["passed"] for item in gates.values())}
