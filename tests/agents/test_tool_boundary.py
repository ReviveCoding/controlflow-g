from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig
from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.policy import LocalPolicyBackend
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import IdentityContext, Severity
from controlflow.tools.registry import ToolRegistry, ToolSpec


class Probe(BaseModel):
    value: int


def test_exact_identifier_is_not_oracle_injected_after_retrieval_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = pd.DataFrame(
        [
            {"control_id": "AC-2", "title": "unrelated", "description": "unrelated", "family": "AC"},
            *[
                {
                    "control_id": f"AU-{index}",
                    "title": "payment exception investigation",
                    "description": "payment exception investigation evidence",
                    "family": "AU",
                }
                for index in range(1, 12)
            ],
        ]
    )
    training = pd.DataFrame([{"case_id": "train", "business_unit": "consumer"}])
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "miss.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=object(),  # type: ignore[arg-type]
    )
    row = pd.Series(
        {
            "case_id": "case",
            "entity_id": "entity",
            "business_unit": "consumer",
            "event_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
            "narrative": "AC-2 payment exception investigation evidence",
            "evidence_status": "AVAILABLE",
        }
    )
    monkeypatch.setattr("controlflow.retrieval.core.BM25Retriever.search", lambda *_: [])
    evidence, _ = workflow._evidence(row, WorkflowConfig(reranker=False), workflow.identity(row))
    assert "AC-2" not in {item.evidence_id for item in evidence}


def test_denied_or_malformed_tool_never_executes_implementation() -> None:
    called = False

    def implementation(value: Probe) -> Probe:
        nonlocal called
        called = True
        return value

    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "probe",
            Probe,
            Probe,
            0,
            True,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            1.0,
            0,
            implementation,
        )
    )
    identity = IdentityContext(
        user_id="analyst",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=1,
        purpose="investigation",
        session_id="session",
    )
    with pytest.raises(ValueError):
        registry.invoke(
            "probe", {"bad": 1}, identity=identity, scope="consumer", data_classification=0, severity=Severity.LOW
        )
    assert not called
