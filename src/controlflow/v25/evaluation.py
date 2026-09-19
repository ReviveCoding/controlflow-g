"""V2.5 scoring adapter for the explicit stale-challenge truth contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.core.state import atomic_write_json
from controlflow.v22.evaluation import evaluate as evaluate_core
from controlflow.v22.evaluation import wilson


def evaluate(
    results_path: Path,
    truth_path: Path,
    ledger_path: Path,
    output_path: Path,
    *,
    critical_threshold: float,
    concurrency: int,
    role: str,
) -> dict[str, Any]:
    """Use the preserved evaluator, then score V2.5's explicit stale marker."""
    metrics = evaluate_core(
        results_path,
        truth_path,
        ledger_path,
        output_path,
        critical_threshold=critical_threshold,
        concurrency=concurrency,
        role=role,
    )
    truth = pd.read_parquet(truth_path)[["case_id", "is_stale_challenge", "expected_policy_id"]]
    candidate = pd.read_parquet(results_path)[["case_id", "policy_id"]]
    joined = truth.merge(candidate, on="case_id", validate="one_to_one")
    if len(joined) != len(truth) or len(joined) != len(candidate):
        raise RuntimeError("V25_STALE_EVALUATION_CARDINALITY_INVALID")
    stale = joined.is_stale_challenge.astype(bool)
    temporal_correct = (joined.policy_id == joined.expected_policy_id) | (
        joined.policy_id.isna() & joined.expected_policy_id.isna()
    )
    metrics["stale_policy_error_rate"] = wilson(int((~temporal_correct & stale).sum()), int(stale.sum()))
    if metrics["stale_policy_error_rate"]["estimate"] is None:
        raise RuntimeError("V25_STALE_EVALUATION_NULL_DENOMINATOR")
    atomic_write_json(output_path, metrics)
    return metrics
