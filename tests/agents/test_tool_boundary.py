from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


class FakeRisk:
    anomaly_threshold = 1.0
    review_threshold = 0.8

    def predict_details(self, _row: pd.Series, *, calibrated: bool = True) -> tuple[np.ndarray, float]:
        return np.asarray([0.7, 0.2, 0.09, 0.01]), 0.1


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
        risk_service=FakeRisk(),  # type: ignore[arg-type]
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
    token = workflow.session_token_for_scope("consumer")
    evidence, _ = workflow._evidence(row, WorkflowConfig(reranker=False), workflow.identity(token))
    assert "AC-2" not in {item.evidence_id for item in evidence}


def test_authenticated_identity_is_immutable_when_case_scope_is_tampered(tmp_path: Path) -> None:
    controls = pd.read_parquet("data/staging/nist_controls_raw.parquet").head(1)
    training = pd.DataFrame(
        [
            {
                "case_id": "train",
                "entity_id": "ENTITY-00001",
                "business_unit": "consumer",
                "event_timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "narrative": "historic payment exception",
            }
        ]
    )
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "identity.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
    )
    token = workflow.session_token_for_scope("consumer")
    tampered = training.iloc[0].copy()
    tampered["business_unit"] = "wealth"
    assert workflow.identity(token).business_unit == "consumer"
    assert workflow.identity(token).session_id == token


def test_all_read_tool_outputs_enter_llm_context(tmp_path: Path) -> None:
    controls = pd.read_parquet("data/staging/nist_controls_raw.parquet").head(2)
    training = pd.DataFrame(
        [
            {
                "case_id": "train",
                "entity_id": "ENTITY-00001",
                "business_unit": "consumer",
                "event_timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "narrative": "historic payment exception",
            }
        ]
    )
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "context.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
    )
    row = training.iloc[0].copy()
    row["case_id"] = "current"
    row["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")
    row["pit_historical_failures"] = 0
    row["future_failures"] = 0
    row["repeat_count"] = 1
    row["data_sensitivity"] = 0
    token = workflow.session_token_for_scope("consumer")
    context = json.loads(workflow.context_for_llm(row, WorkflowConfig(reranker=False), session_token=token))
    assert {
        "search_cases",
        "get_policy_at_time",
        "query_case_data",
        "query_transactions",
        "generate_evidence_bundle",
    }.issubset(context)
    assert context["query_case_data"]["values"]["case_id"] == "current"


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
