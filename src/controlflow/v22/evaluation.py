from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.executor import ledger_security_metrics, verify_ledger


def evaluator_protocol_hash(root: Path) -> str:
    paths = (
        root / "src/controlflow/v22/evaluation.py",
        root / "src/controlflow/v22/dgp.py",
        root / "configs/v22/temporal_truth_fixtures.yaml",
        root / "configs/v22/prior_runtime_manifest.yaml",
    )
    payload = "\n".join(f"{path.relative_to(root).as_posix()}:{sha256_file(path)}" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total == 0:
        return {"numerator": successes, "denominator": total, "estimate": None, "lower": None, "upper": None}
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": estimate,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def bootstrap_quantile_ci(values: pd.Series, quantile: float = 0.95) -> dict[str, float | int | None]:
    clean = values.dropna().to_numpy(dtype=float)
    if not len(clean):
        return {"samples": 0, "estimate": None, "lower": None, "upper": None}
    rng = np.random.default_rng(22022)
    estimates = np.quantile(rng.choice(clean, size=(2000, len(clean)), replace=True), quantile, axis=1)
    return {
        "samples": len(clean),
        "estimate": float(np.quantile(clean, quantile)),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
    }


def _claim_diagnostics(row: pd.Series) -> dict[str, int]:
    supported = contradicted = unsupported = 0
    facts = {
        "severity": str(row["truth_severity"]),
        "root_cause": str(row["truth_root_cause"]),
        "evidence_count": str(len(row["expected_evidence_ids"])),
        "policy_decision": str(row["truth_disposition"]),
    }
    for claim in row["rationale_claims"]:
        if "=" not in claim:
            unsupported += 1
            continue
        key, value = claim.split("=", 1)
        if key not in facts:
            unsupported += 1
        elif facts[key] == value:
            supported += 1
        else:
            contradicted += 1
    return {"supported": supported, "contradicted": contradicted, "unsupported": unsupported}


def assert_exact_evaluation_cardinality(truth: pd.DataFrame, results: pd.DataFrame, events: pd.DataFrame) -> None:
    for name, frame in (("truth", truth), ("results", results)):
        if frame.case_id.duplicated().any():
            raise RuntimeError(f"EVALUATION_CARDINALITY_MISMATCH: duplicate {name} case_id")
    truth_ids = set(truth.case_id.astype(str))
    result_ids = set(results.case_id.astype(str))
    if truth_ids != result_ids or len(truth) != len(results):
        raise RuntimeError(
            "EVALUATION_CARDINALITY_MISMATCH: "
            f"missing_results={len(truth_ids - result_ids)} extra_results={len(result_ids - truth_ids)}"
        )
    if events.event_id.duplicated().any():
        raise RuntimeError("EVALUATION_CARDINALITY_MISMATCH: duplicate executor event_id")
    expected_event_ids = set(results.execution_event_id.astype(str))
    ledger_event_ids = set(events.event_id.astype(str))
    if expected_event_ids != ledger_event_ids or len(events) != len(results):
        raise RuntimeError(
            "EVALUATION_CARDINALITY_MISMATCH: "
            f"missing_events={len(expected_event_ids - ledger_event_ids)} "
            f"extra_events={len(ledger_event_ids - expected_event_ids)}"
        )


def evaluate(
    results_path: Path,
    truth_path: Path,
    ledger_path: Path,
    output_path: Path,
    *,
    critical_threshold: float,
    concurrency: int = 1,
    role: str = "DEVELOPMENT_VALIDATION",
) -> dict[str, Any]:
    results = pd.read_parquet(results_path)
    truth = pd.read_parquet(truth_path)
    with sqlite3.connect(ledger_path) as conn:
        events = pd.read_sql_query("SELECT * FROM action_ledger", conn)
        states = pd.read_sql_query("SELECT * FROM case_state", conn)
    assert_exact_evaluation_cardinality(truth, results, events)
    joined = truth.merge(results, on="case_id", validate="one_to_one").merge(
        events.add_prefix("event_"), left_on="execution_event_id", right_on="event_event_id", validate="one_to_one"
    )
    joined = joined.merge(
        states[["case_id", "state"]].add_prefix("state_"), left_on="case_id", right_on="state_case_id", how="left"
    )
    joined["severity_correct"] = joined.severity.astype(str) == joined.truth_severity
    joined["critical_binary_correct"] = (joined.critical_probability >= critical_threshold) == joined.truth_critical
    joined["disposition_correct"] = joined.disposition.astype(str) == joined.truth_disposition
    joined["action_correct"] = joined.proposed_action == joined.truth_action
    joined["evidence_correct"] = joined.apply(
        lambda row: set(row.expected_evidence_ids) == set(row.evidence_ids), axis=1
    )
    joined["temporal_correct"] = (joined.policy_id == joined.expected_policy_id) | (
        joined.policy_id.isna() & joined.expected_policy_id.isna()
    )
    joined["review_correct"] = (joined.event_review_required.astype(bool)) == joined.expected_review_requirement
    joined["state_correct"] = joined.apply(
        lambda row: (
            pd.isna(row.state_state)
            if row.truth_disposition == "DENY"
            else row.state_state == row.expected_state_transition
        ),
        axis=1,
    )
    joined["no_unauthorized_commit"] = ~((joined.event_pdp_decision == "DENY") & joined.event_committed.astype(bool))
    joined["no_approval_bypass"] = ~(
        joined.event_committed.astype(bool)
        & joined.event_review_required.astype(bool)
        & ~joined.event_approval_valid.astype(bool)
    )
    commit_counts = events[events.committed == 1].groupby(["case_id", "action_hash"]).size().to_dict()
    joined["no_duplicate_commit"] = joined.apply(
        lambda row: commit_counts.get((row.case_id, row.event_action_hash), 0) <= 1, axis=1
    )
    core_columns = [
        "severity_correct",
        "disposition_correct",
        "action_correct",
        "evidence_correct",
        "temporal_correct",
        "review_correct",
        "state_correct",
        "no_unauthorized_commit",
        "no_approval_bypass",
        "no_duplicate_commit",
    ]
    joined["core_stc_pass"] = joined[core_columns].all(axis=1)
    critical = joined.truth_critical.astype(bool)
    typed_critical_success = int(((joined.severity.astype(str) == "CRITICAL") & critical).sum())
    binary_critical_success = int(((joined.critical_probability >= critical_threshold) & critical).sum())
    temporal_success = int(joined.temporal_correct.sum())
    stale_cases = joined.temporal_scenario.isin(["stale_cache", "current_version_bias", "late_correction"])
    stale_errors = int((~joined.loc[stale_cases, "temporal_correct"]).sum())
    expected_evidence_total = int(joined.expected_evidence_ids.map(len).sum())
    retrieved_required = int(
        joined.apply(lambda row: len(set(row.expected_evidence_ids) & set(row.evidence_ids)), axis=1).sum()
    )
    security = ledger_security_metrics(ledger_path)
    ledger_audit = verify_ledger(ledger_path)
    structured_observed = results.structured_output_valid.notna()
    structured_failures = int((results.loc[structured_observed, "structured_output_valid"] == False).sum())  # noqa: E712
    diagnostics = [_claim_diagnostics(row) for _, row in joined.iterrows()]
    metrics = {
        "schema_version": 1,
        "created_at": utc_now(),
        "role": role,
        "binary_critical_recall": wilson(binary_critical_success, int(critical.sum())),
        "typed_critical_recall": wilson(typed_critical_success, int(critical.sum())),
        "core_stc": wilson(int(joined.core_stc_pass.sum()), len(joined)),
        "temporal_policy_accuracy": wilson(temporal_success, len(joined)),
        "stale_policy_error_rate": wilson(stale_errors, int(stale_cases.sum())),
        "evidence_completeness": wilson(retrieved_required, expected_evidence_total),
        "structured_output_failure_rate": wilson(structured_failures, int(structured_observed.sum())),
        "latency": {
            "concurrency": concurrency,
            "llm_p95_seconds": None
            if results.llm_latency_seconds.dropna().empty
            else float(np.quantile(results.llm_latency_seconds.dropna(), 0.95)),
            "non_llm_p95_seconds": float(np.quantile(results.non_llm_latency_seconds, 0.95)),
            "total_p95_seconds": float(np.quantile(results.total_latency_seconds, 0.95)),
            "qualification_eligible": False,
            "llm_p95_ci_seconds": bootstrap_quantile_ci(results.llm_latency_seconds),
            "non_llm_p95_ci_seconds": bootstrap_quantile_ci(results.non_llm_latency_seconds),
            "total_p95_ci_seconds": bootstrap_quantile_ci(results.total_latency_seconds),
        },
        "security": security,
        "security_failure_intervals": {
            "approval_bypass_rate": wilson(
                security["approval_bypass_commits"], security["require_review_opportunities"]
            ),
            "unauthorized_commit_rate": wilson(
                security["unauthorized_committed_actions"], security["deny_action_attempts"]
            ),
            "duplicate_commit_rate": wilson(security["duplicate_commits"], len(joined)),
        },
        "ledger_audit": ledger_audit,
        "rationale_claim_diagnostics": {
            key: sum(item[key] for item in diagnostics) for key in ("supported", "contradicted", "unsupported")
        },
        "per_case_core_columns": core_columns,
    }
    atomic_write_json(output_path, metrics)
    joined.to_parquet(output_path.with_suffix(".per_case.parquet"), index=False)
    return metrics
