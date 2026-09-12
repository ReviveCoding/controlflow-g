from __future__ import annotations

import pytest
from pydantic import ValidationError

from controlflow.agents.experiments import GovernedAnalysis


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
