"""Apply only predeclared V2.4 gates after post-close verification."""

from __future__ import annotations

from typing import Any


def evaluate_gates(
    specification: dict[str, Any],
    metrics: dict[str, Any],
    denominators: dict[str, Any],
    *,
    leakage_findings: int,
    checkpoint_violations: int,
    artifact_binding_violations: int,
    ledger_failures: int,
    unresolved_blocker: int,
    unresolved_high: int,
) -> dict[str, Any]:
    ratios = denominators["ratios"]
    values: dict[str, float | int | None] = {
        "binary_critical_recall": ratios["binary_critical_recall"]["estimate"],
        "typed_critical_recall": ratios["typed_critical_recall"]["estimate"],
        "core_stc": ratios["core_stc"]["estimate"],
        "structured_output_failure_rate": ratios["structured_output_failure_rate"]["estimate"],
        "approval_bypass_commits": ratios["approval_bypass_rate"]["numerator"],
        "unauthorized_committed_actions": ratios["unauthorized_commit_rate"]["numerator"],
        "duplicate_commits": ratios["duplicate_commit_rate"]["numerator"],
        "temporal_policy_accuracy": ratios["temporal_policy_accuracy"]["estimate"],
        "stale_policy_error_rate": ratios["stale_policy_error_rate"]["estimate"],
        "evidence_completeness": ratios["evidence_completeness"]["estimate"],
        "concurrency_2_total_p95_seconds": metrics["latency"].get("total_p95_seconds")
        if metrics["latency"].get("concurrency") == 2
        else None,
        "leakage_findings": leakage_findings,
        "checkpoint_violations": checkpoint_violations,
        "artifact_binding_violations": artifact_binding_violations,
        "ledger_tamper_verification_failures": ledger_failures,
        "unresolved_blocker": unresolved_blocker,
        "unresolved_high": unresolved_high,
    }
    results: dict[str, Any] = {}
    for name, gate in specification["gates"].items():
        value = values[name]
        minimum = gate.get("minimum_denominator")
        ratio_name = {
            "approval_bypass_commits": "approval_bypass_rate",
            "unauthorized_committed_actions": "unauthorized_commit_rate",
            "duplicate_commits": "duplicate_commit_rate",
        }.get(name, name)
        denominator = ratios[ratio_name]["denominator"] if minimum is not None else None
        denominator_valid = minimum is None or (
            denominator is not None
            and denominator >= int(minimum)
            and ratios[ratio_name]["denominator_contract_status"] == "VALID"
        )
        operator = gate["operator"]
        threshold = gate["value"]
        passed = bool(
            value is not None
            and denominator_valid
            and (
                (operator == "gte" and value >= threshold)
                or (operator == "lte" and value <= threshold)
                or (operator == "eq" and value == threshold)
            )
        )
        results[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "denominator": denominator,
            "minimum_denominator": minimum,
            "null_policy": gate["null_policy"],
            "evidence_source": gate["evidence_source"],
            "passed": passed,
        }
    return {"gates": results, "all_passed": all(row["passed"] for row in results.values())}
