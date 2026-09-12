from __future__ import annotations

import pandas as pd
import pytest

from controlflow.agents.experiments import evaluate_trace
from controlflow.agents.workflow import WorkflowTrace
from controlflow.eval.rescore_governed_claims import _rescore_traces
from controlflow.final_evaluation import FinalCheckpointTrace


def test_invalid_governed_claims_can_be_deterministically_rescored() -> None:
    frame = pd.DataFrame(
        [
            {
                "architecture_mode": "governed",
                "llm_analysis_valid": False,
                "llm_analysis_supported": True,
                "llm_analysis_support_checked": True,
                "safe_task_completion": True,
            },
            {
                "architecture_mode": "rules",
                "llm_analysis_valid": True,
                "llm_analysis_supported": False,
                "llm_analysis_support_checked": False,
                "safe_task_completion": True,
            },
        ]
    )
    result = _rescore_traces(frame)
    assert result.root_cause_correct.tolist() == [False, True]
    assert result.safe_task_completion.tolist() == [False, True]
    assert "llm_analysis_evidence_valid" in result


def test_valid_governed_claim_requires_full_text_re_evaluation() -> None:
    frame = pd.DataFrame(
        [
            {
                "architecture_mode": "governed",
                "llm_analysis_valid": True,
                "llm_analysis_supported": True,
                "llm_analysis_support_checked": True,
                "safe_task_completion": True,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="full text-preserving"):
        _rescore_traces(frame)


def test_final_checkpoint_schema_accepts_actual_evaluator_record() -> None:
    row = pd.Series(
        {
            "case_id": "case",
            "entity_id": "entity",
            "required_evidence": [],
            "authorization_outcome": "ALLOW",
            "severity": "LOW",
            "expected_disposition": "INSUFFICIENT_EVIDENCE",
            "expected_actions": [],
            "case_type": "missing_evidence",
            "control_ids": ["AC-2"],
            "permitted_tools": [],
        }
    )
    trace = WorkflowTrace(
        case_id="case",
        predicted_severity="LOW",
        predicted_disposition="INSUFFICIENT_EVIDENCE",
        risk_confidence=0.5,
        anomaly_score=0.0,
        retrieved_ids=[],
        tool_calls=[],
        evidence_verified=False,
        temporal_correct=True,
        feature_event_timestamp="2025-01-01T00:00:00+00:00",
        feature_system_known_at="2025-01-01T00:00:00+00:00",
        authorization_outcome="ALLOW",
        human_review_requested=False,
        structured_output_valid=True,
        action_executed=False,
        action_performed_this_invocation=False,
        action_id=None,
        injection_detected=False,
        latency_seconds=0.1,
        llm_analysis_evidence_valid=True,
        llm_analysis_evidence_checked=True,
    )
    usage = {
        "llm_latency_seconds": 0.1,
        "input_tokens": 1.0,
        "output_tokens": 1.0,
        "architecture_mode": "governed",
        "analysis_valid": True,
        "root_cause_hypothesis": "Required evidence is unavailable",
        "recommended_action": "REQUEST_EVIDENCE",
    }
    record = evaluate_trace("AG6_controlflow_g", row, trace, usage)
    validated = FinalCheckpointTrace.model_validate(record)
    assert validated.root_cause_correct
