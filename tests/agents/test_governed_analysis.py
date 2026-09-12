from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from controlflow.agents.experiments import GovernedAnalysis, _root_cause_compatible


def test_insufficient_evidence_may_use_empty_citations() -> None:
    analysis = GovernedAnalysis.model_validate(
        {
            "severity": "LOW",
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause_hypothesis": "Evidence is unavailable",
            "recommended_action": "REQUEST_EVIDENCE",
            "supporting_evidence_ids": [],
        }
    )
    assert analysis.supporting_evidence_ids == []


def test_non_insufficient_analysis_requires_unique_citations() -> None:
    base = {
        "severity": "HIGH",
        "disposition": "REVIEW_REQUIRED",
        "root_cause_hypothesis": "AC-2 control execution variance",
        "recommended_action": "ESCALATE",
    }
    with pytest.raises(ValidationError, match="only INSUFFICIENT_EVIDENCE"):
        GovernedAnalysis.model_validate({**base, "supporting_evidence_ids": []})
    with pytest.raises(ValidationError, match="must be unique"):
        GovernedAnalysis.model_validate({**base, "supporting_evidence_ids": ["AC-2", "AC-2"]})
    with pytest.raises(ValidationError, match="must request evidence"):
        GovernedAnalysis.model_validate(
            {
                **base,
                "disposition": "INSUFFICIENT_EVIDENCE",
                "recommended_action": "ESCALATE",
                "supporting_evidence_ids": [],
            }
        )


def test_root_cause_compatibility_uses_benchmark_predicates() -> None:
    row = pd.Series({"case_type": "normal", "control_ids": ["AC-2"]})
    base = {"architecture_mode": "governed", "analysis_valid": True}
    assert _root_cause_compatible(
        row,
        {**base, "root_cause_hypothesis": "AC-2 routine variance caused the exception"},
    )
    assert not _root_cause_compatible(
        row,
        {**base, "root_cause_hypothesis": "AC-2 caused an unrelated event"},
    )
    assert not _root_cause_compatible(
        row,
        {**base, "root_cause_hypothesis": "AC-2 routine variance does not apply"},
    )
    missing = pd.Series({"case_type": "missing_evidence", "control_ids": ["AC-2"]})
    assert _root_cause_compatible(
        missing,
        {**base, "root_cause_hypothesis": "Required evidence is unavailable"},
    )
