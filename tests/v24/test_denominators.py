from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from controlflow.core.state import atomic_write_json
from controlflow.v24.denominators import recompute


def test_independent_recompute_rejects_zero_stale_denominator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path / "dataset"
    directory.mkdir()
    event = "2023-11-07T15:40:00+00:00"
    system = "2025-05-16T12:30:00+00:00"
    current_event = "2026-10-06T20:05:00+00:00"
    current_system = "2026-10-12T09:45:00+00:00"
    events = [event if i < 150 else current_event for i in range(600)]
    systems = [system if i < 150 else current_system for i in range(600)]
    policies = ["policy-2023-correction" if i < 150 else "policy-2026" for i in range(600)]
    runtime = pd.DataFrame(
        {
            "case_id": [f"C-{i:03d}" for i in range(600)],
            "event_time": events,
            "system_time": systems,
            "requested_scope": ["consumer"] * 600,
        }
    )
    truth = pd.DataFrame(
        {
            "case_id": runtime.case_id,
            "event_time": events,
            "system_time": systems,
            "expected_policy_id": policies,
            "temporal_scenario": ["v23_late_correction" if i < 150 else "v23_current_2026" for i in range(600)],
            "is_stale_challenge": [i < 150 for i in range(600)],
            "stale_candidate_ids": [["policy-2023", "policy-2026"] if i < 150 else [] for i in range(600)],
            "truth_critical": [i < 120 for i in range(600)],
            "truth_severity": ["CRITICAL" if i < 120 else "LOW" for i in range(600)],
            "truth_disposition": ["REQUIRE_REVIEW" if i < 120 else "DENY" if i < 180 else "ALLOW" for i in range(600)],
            "truth_action": ["OPEN_REMEDIATION" if i < 120 else "CLOSE_NO_ACTION" for i in range(600)],
            "expected_review_requirement": [i < 120 for i in range(600)],
            "expected_state_transition": ["REMEDIATION_OPEN" if i < 120 else "CLOSED_NO_ACTION" for i in range(600)],
            "expected_evidence_ids": [[f"E-{i:03d}"] for i in range(600)],
            "latent": [json.dumps({"evidence_quality": 0.9, "conflict_state": False})] * 600,
            "is_adversarial": [False] * 600,
        }
    )
    candidate = pd.DataFrame(
        {
            "case_id": runtime.case_id,
            "critical_probability": [0.9 if i < 120 else 0.0 for i in range(600)],
            "severity": ["CRITICAL" if i < 120 else "LOW" for i in range(600)],
            "disposition": ["REQUIRE_REVIEW" if i < 120 else "DENY" if i < 180 else "ALLOW" for i in range(600)],
            "proposed_action": ["OPEN_REMEDIATION" if i < 120 else "CLOSE_NO_ACTION" for i in range(600)],
            "execution_event_id": [f"EV-{i:03d}" for i in range(600)],
            "policy_id": policies,
            "evidence_ids": [[f"E-{i:03d}"] for i in range(600)],
            "structured_output_valid": [True] * 600,
        }
    )
    per_case = pd.DataFrame(
        {
            "case_id": runtime.case_id,
            "core_stc_pass": [True] * 600,
            "event_committed": [not 120 <= i < 180 for i in range(600)],
            "event_review_required": [i < 120 for i in range(600)],
            "event_approval_valid": [True] * 600,
            "event_pdp_decision": ["REQUIRE_REVIEW" if i < 120 else "DENY" if i < 180 else "ALLOW" for i in range(600)],
        }
    )
    runtime.to_parquet(directory / "runtime_cases.parquet", index=False)
    truth.to_parquet(directory / "evaluator_truth.parquet", index=False)
    candidate_path = tmp_path / "candidate.parquet"
    per_case_path = tmp_path / "per_case.parquet"
    candidate.to_parquet(candidate_path, index=False)
    per_case.to_parquet(per_case_path, index=False)
    metric_path = tmp_path / "metrics.json"
    atomic_write_json(
        metric_path,
        {
            **{
                name: {"numerator": numerator, "denominator": denominator}
                for name, numerator, denominator in (
                    ("binary_critical_recall", 120, 120),
                    ("typed_critical_recall", 120, 120),
                    ("core_stc", 600, 600),
                    ("structured_output_failure_rate", 0, 600),
                    ("temporal_policy_accuracy", 600, 600),
                    ("evidence_completeness", 600, 600),
                )
            },
            "security": {
                "approval_bypass_commits": 0,
                "require_review_opportunities": 120,
                "unauthorized_committed_actions": 0,
                "deny_action_attempts": 60,
                "duplicate_commits": 0,
            },
        },
    )
    ledger_path = tmp_path / "ledger_snapshot.json"
    atomic_write_json(
        ledger_path,
        {
            "schema_version": 1,
            "tables": {
                "action_ledger": [
                    {
                        "case_id": f"C-{i:03d}",
                        "event_id": f"EV-{i:03d}",
                        "policy_decision_id": f"PDP-{i:03d}",
                        "pdp_decision": "REQUIRE_REVIEW" if i < 120 else "DENY" if i < 180 else "ALLOW",
                        "committed": int(not 120 <= i < 180),
                        "review_required": int(i < 120),
                        "approval_valid": 1,
                        "action_hash": f"HASH-{i:03d}",
                    }
                    for i in range(600)
                ],
                "case_state": [
                    {"case_id": f"C-{i:03d}", "state": "REMEDIATION_OPEN" if i < 120 else "CLOSED_NO_ACTION"}
                    for i in range(600)
                    if not 120 <= i < 180
                ],
                "policy_decisions": [
                    {
                        "policy_decision_id": f"PDP-{i:03d}",
                        "stage": "PRE_COMMIT",
                        "decision": "REQUIRE_REVIEW" if i < 120 else "DENY" if i < 180 else "ALLOW",
                    }
                    for i in range(600)
                ],
                "ledger_head": [],
            },
        },
    )
    valid = recompute(
        root,
        directory,
        candidate_path,
        per_case_path,
        metric_path,
        ledger_path,
        tmp_path / "denom.json",
        critical_threshold=0.5,
    )
    assert valid["status"] == "VALID"
    assert valid["ratios"]["stale_policy_error_rate"]["denominator"] == 150
    truth["is_stale_challenge"] = False
    truth["stale_candidate_ids"] = [[] for _ in range(600)]
    truth["temporal_scenario"] = "v23_current_2026"
    truth["expected_policy_id"] = "policy-2026"
    truth["event_time"] = current_event
    truth["system_time"] = current_system
    runtime["event_time"] = truth["event_time"]
    runtime["system_time"] = truth["system_time"]
    runtime.to_parquet(directory / "runtime_cases.parquet", index=False)
    candidate["policy_id"] = "policy-2026"
    candidate.to_parquet(candidate_path, index=False)
    truth.to_parquet(directory / "evaluator_truth.parquet", index=False)
    invalid = recompute(
        root,
        directory,
        candidate_path,
        per_case_path,
        metric_path,
        ledger_path,
        tmp_path / "denom2.json",
        critical_threshold=0.5,
    )
    assert invalid["status"] == "INVALID"
    assert invalid["ratios"]["stale_policy_error_rate"]["estimate"] is None
