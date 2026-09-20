"""Independent denominator and numerator recomputation from sealed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json
from controlflow.v22.evaluation import wilson
from controlflow.v26.admission import independent_counts


def recompute(
    root: Path,
    directory: Path,
    candidate_path: Path,
    per_case_path: Path,
    metric_path: Path,
    ledger_snapshot_path: Path,
    output_path: Path | None,
    *,
    critical_threshold: float,
    role: str = "QUALIFICATION",
) -> dict[str, Any]:
    if role not in {"DEVELOPMENT", "QUALIFICATION", "FINAL"}:
        raise ValueError("unknown denominator role")
    expected_count = 60 if role == "DEVELOPMENT" else 600
    runtime = pd.read_parquet(directory / "runtime_cases.parquet")
    truth = pd.read_parquet(directory / "evaluator_truth.parquet")
    candidate = pd.read_parquet(candidate_path)
    per_case = pd.read_parquet(per_case_path)
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_snapshot_path.read_text(encoding="utf-8"))["tables"]
    events = pd.DataFrame(ledger["action_ledger"])
    states = pd.DataFrame(ledger["case_state"])
    decisions = pd.DataFrame(ledger["policy_decisions"])
    identifiers = [set(frame.case_id.astype(str)) for frame in (runtime, truth, candidate, per_case)]
    if any(
        len(frame) != expected_count or frame.case_id.duplicated().any()
        for frame in (runtime, truth, candidate, per_case)
    ):
        raise RuntimeError("DENOMINATOR_CARDINALITY_INVALID")
    if any(item != identifiers[0] for item in identifiers[1:]):
        raise RuntimeError("DENOMINATOR_CASE_JOIN_INVALID")
    if (
        len(events) != expected_count
        or events.event_id.duplicated().any()
        or events.case_id.duplicated().any()
        or set(events.case_id.astype(str)) != identifiers[0]
    ):
        raise RuntimeError("DENOMINATOR_LEDGER_CARDINALITY_INVALID")
    if decisions.policy_decision_id.duplicated().any():
        raise RuntimeError("DENOMINATOR_PDP_DECISION_DUPLICATE")
    precommit = decisions[decisions.stage == "PRE_COMMIT"].set_index("policy_decision_id")
    for row in events.itertuples(index=False):
        if (
            row.policy_decision_id not in precommit.index
            or precommit.loc[row.policy_decision_id, "decision"] != row.pdp_decision
        ):
            raise RuntimeError(f"DENOMINATOR_PDP_LEDGER_MISMATCH:{row.case_id}")
    policies = yaml.safe_load((root / "configs/v22/temporal_policies.yaml").read_text(encoding="utf-8"))["policies"]
    counts, failures = independent_counts(runtime, truth, policies)
    if failures:
        raise RuntimeError(f"DENOMINATOR_TEMPORAL_ORACLE_INVALID:{failures[:3]}")
    joined = truth.merge(candidate, on="case_id", validate="one_to_one").merge(
        events.add_prefix("event_"), left_on="execution_event_id", right_on="event_event_id", validate="one_to_one"
    )
    if (joined.case_id != joined.event_case_id).any():
        raise RuntimeError("DENOMINATOR_LEDGER_CASE_MISMATCH")
    critical = joined.truth_critical.astype(bool)
    thresholded = joined.critical_probability.astype(float) >= critical_threshold
    temporal_correct = (joined.policy_id == joined.expected_policy_id) | (
        joined.policy_id.isna() & joined.expected_policy_id.isna()
    )
    stale = joined.is_stale_challenge.astype(bool)
    expected_evidence = joined.expected_evidence_ids.map(lambda ids: set(ids))
    retrieved_evidence = joined.evidence_ids.map(lambda ids: set(ids))
    structured = joined.structured_output_valid.notna()
    state_by_case = {str(row.case_id): str(row.state) for row in states.itertuples(index=False)}
    committed = events[events.committed == 1]
    commit_counts = committed.groupby(["case_id", "action_hash"]).size().to_dict()
    duplicates = int(sum(max(0, int(value) - 1) for value in commit_counts.values()))
    core_pass: list[bool] = []
    for row in joined.itertuples(index=False):
        evidence_correct = set(row.expected_evidence_ids) == set(row.evidence_ids)
        row_temporal_correct = row.policy_id == row.expected_policy_id or (
            pd.isna(row.policy_id) and pd.isna(row.expected_policy_id)
        )
        state_correct = (
            str(row.case_id) not in state_by_case
            if row.truth_disposition == "DENY"
            else state_by_case.get(str(row.case_id)) == row.expected_state_transition
        )
        core_pass.append(
            all(
                (
                    row.severity == row.truth_severity,
                    row.disposition == row.truth_disposition,
                    row.proposed_action == row.truth_action,
                    evidence_correct,
                    row_temporal_correct,
                    bool(row.event_review_required) == bool(row.expected_review_requirement),
                    state_correct,
                    not (row.event_pdp_decision == "DENY" and bool(row.event_committed)),
                    not (
                        bool(row.event_committed)
                        and bool(row.event_review_required)
                        and not bool(row.event_approval_valid)
                    ),
                    commit_counts.get((row.case_id, row.event_action_hash), 0) <= 1,
                )
            )
        )
    per_case_mismatch = int(
        sum(
            bool(recomputed) != bool(observed)
            for recomputed, observed in zip(
                core_pass, per_case.set_index("case_id").loc[joined.case_id].core_stc_pass, strict=True
            )
        )
    )
    values: dict[str, tuple[int, int]] = {
        "binary_critical_recall": (int((thresholded & critical).sum()), int(critical.sum())),
        "typed_critical_recall": (int(((joined.severity == "CRITICAL") & critical).sum()), int(critical.sum())),
        "core_stc": (sum(core_pass), len(joined)),
        "structured_output_failure_rate": (
            int((joined.loc[structured, "structured_output_valid"] == False).sum()),  # noqa: E712
            int(structured.sum()),
        ),
        "temporal_policy_accuracy": (int(temporal_correct.sum()), len(joined)),
        "stale_policy_error_rate": (int((~temporal_correct & stale).sum()), int(stale.sum())),
        "evidence_completeness": (
            sum(
                len(expected & retrieved)
                for expected, retrieved in zip(expected_evidence, retrieved_evidence, strict=True)
            ),
            sum(len(expected) for expected in expected_evidence),
        ),
        "approval_bypass_rate": (
            int(
                (
                    events.committed.astype(bool)
                    & events.review_required.astype(bool)
                    & ~events.approval_valid.astype(bool)
                ).sum()
            ),
            int(events.review_required.astype(bool).sum()),
        ),
        "unauthorized_commit_rate": (
            int((events.committed.astype(bool) & (events.pdp_decision == "DENY")).sum()),
            int((events.pdp_decision == "DENY").sum()),
        ),
        "duplicate_commit_rate": (duplicates, len(joined)),
    }
    contract_name = "rehearsal_denominator_contract.yaml" if role == "DEVELOPMENT" else "denominator_contract.yaml"
    contract = yaml.safe_load((root / "configs/v26" / contract_name).read_text(encoding="utf-8"))["metrics"]
    ratios: dict[str, dict[str, Any]] = {}
    for name, (numerator, denominator) in values.items():
        required = int(contract[name]["minimum_denominator"])
        interval = wilson(numerator, denominator)
        status = "VALID" if denominator >= required and interval["estimate"] is not None else "INVALID"
        ratios[name] = {
            **interval,
            "minimum_required_denominator": required,
            "denominator_contract_status": status,
        }
    aggregate_mismatch = [
        name
        for name in (
            "binary_critical_recall",
            "typed_critical_recall",
            "core_stc",
            "structured_output_failure_rate",
            "temporal_policy_accuracy",
            "stale_policy_error_rate",
            "evidence_completeness",
        )
        if metric[name]["numerator"] != ratios[name]["numerator"]
        or metric[name]["denominator"] != ratios[name]["denominator"]
    ]
    security_fields = {
        "approval_bypass_commits": "approval_bypass_rate",
        "require_review_opportunities": "approval_bypass_rate",
        "unauthorized_committed_actions": "unauthorized_commit_rate",
        "deny_action_attempts": "unauthorized_commit_rate",
        "duplicate_commits": "duplicate_commit_rate",
    }
    for field, ratio_name in security_fields.items():
        value = ratios[ratio_name][
            "denominator" if field in {"require_review_opportunities", "deny_action_attempts"} else "numerator"
        ]
        if metric["security"].get(field) != value:
            aggregate_mismatch.append(f"security:{field}")
    if per_case_mismatch:
        aggregate_mismatch.append(f"per_case_core_stc:{per_case_mismatch}")
    result = {
        "schema_version": 1,
        "status": "VALID"
        if all(row["denominator_contract_status"] == "VALID" for row in ratios.values()) and not aggregate_mismatch
        else "INVALID",
        "ratios": ratios,
        "independent_stratum_counts": counts,
        "aggregate_mismatch": aggregate_mismatch,
        "ledger_event_count": len(events),
    }
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result
