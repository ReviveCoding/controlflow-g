from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig
from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.identity import SessionIdentityProvider
from controlflow.authorization.policy import LocalPolicyBackend
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import IdentityContext, Severity
from controlflow.tools.deadline import tool_commit_section
from controlflow.tools.registry import ToolRegistry, ToolSpec


class Probe(BaseModel):
    value: int


class FakeRisk:
    anomaly_threshold = 1.0
    review_threshold = 0.8

    def predict_details(self, _row: pd.Series, *, calibrated: bool = True) -> tuple[np.ndarray, float]:
        return np.asarray([0.7, 0.2, 0.09, 0.01]), 0.1


def _sessions(*units: str) -> tuple[SessionIdentityProvider, dict[str, str]]:
    return SessionIdentityProvider.issue_for_business_units(set(units))


def test_exact_identifier_is_not_oracle_injected_after_retrieval_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_corpus: Path
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
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "miss.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
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
    token = credentials["consumer"]
    evidence, _ = workflow._evidence(row, WorkflowConfig(reranker=False), workflow.identity(token))
    assert "AC-2" not in {item.evidence_id for item in evidence}


def test_authenticated_identity_is_immutable_when_case_scope_is_tampered(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(1)
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
    provider, credentials = _sessions("consumer", "wealth")
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "identity.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    token = credentials["consumer"]
    tampered = training.iloc[0].copy()
    tampered["business_unit"] = "wealth"
    assert workflow.identity(token).business_unit == "consumer"
    assert workflow.identity(token).session_id != token
    with pytest.raises(PermissionError):
        workflow.context_for_llm(tampered, WorkflowConfig(reranker=False), session_token=token)
    with pytest.raises(PermissionError):
        workflow.execute(tampered, WorkflowConfig(retrieval=False), ("LOW", "AUTO", True), session_token=token)
    with pytest.raises(PermissionError):
        workflow.execute(
            training.iloc[0], WorkflowConfig(retrieval=False), ("LOW", "AUTO", True), session_token="forged"
        )


def test_all_read_tool_outputs_enter_llm_context(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(2)
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
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "context.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    row = training.iloc[0].copy()
    row["case_id"] = "current"
    row["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")
    row["pit_historical_failures"] = 0
    row["future_failures"] = 0
    row["repeat_count"] = 1
    row["data_sensitivity"] = 0
    token = credentials["consumer"]
    context = json.loads(workflow.context_for_llm(row, WorkflowConfig(reranker=False), session_token=token))
    assert {
        "search_cases",
        "get_policy_at_time",
        "query_case_data",
        "query_transactions",
        "generate_evidence_bundle",
    }.issubset(context)
    assert context["query_case_data"]["values"]["case_id"] == "current"


def test_agentic_context_executes_only_selected_tools(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(2)
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
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "lazy-context.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    row = training.iloc[0].copy()
    row["case_id"] = "current"
    row["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")
    row["pit_historical_failures"] = 0
    row["future_failures"] = 0
    row["repeat_count"] = 1
    row["data_sensitivity"] = 0
    context = json.loads(
        workflow.context_for_llm(
            row,
            WorkflowConfig(reranker=False),
            session_token=credentials["consumer"],
            selected_tools=frozenset({"query_case_data"}),
        )
    )
    assert {
        "search_controls",
        "search_regulations",
        "compute_risk",
        "compute_anomaly",
        "query_case_data",
        "tool_steps",
        "_context_id",
    }.issubset(context)
    assert context["search_controls"] == []
    assert context["search_regulations"] == []
    assert context["compute_risk"] == {}
    assert context["compute_anomaly"] == {}
    assert context["query_case_data"]["values"]["case_id"] == "current"
    assert "query_transactions" not in context


def test_duplicate_plan_is_rejected_before_any_tool_executes(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(1)
    row = pd.Series(
        {
            "case_id": "case",
            "entity_id": "entity",
            "business_unit": "consumer",
            "narrative": "exception",
            "event_timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "pit_historical_failures": 0,
            "future_failures": 0,
            "repeat_count": 0,
            "data_sensitivity": 0,
        }
    )
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        pd.DataFrame([row]),
        controls,
        ActionLedger(tmp_path / "duplicates.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    context = json.loads(
        workflow.context_for_llm(
            row,
            WorkflowConfig(reranker=False),
            session_token=credentials["consumer"],
            selected_tool_calls=[
                ("query_case_data", {"case_id": "case"}),
                ("query_case_data", {"case_id": "case"}),
            ],
        )
    )
    assert context["plan"]["error"] == "duplicate_tool_request_rejected_before_execution"
    assert context["tool_steps"] == []
    malformed = json.loads(
        workflow.context_for_llm(
            row,
            WorkflowConfig(reranker=False),
            session_token=credentials["consumer"],
            selected_tool_calls=[("query_case_data", {"__invalid__": True})],
        )
    )
    assert malformed["plan"]["error"] == "malformed_tool_plan_rejected_before_execution"
    assert malformed["tool_steps"] == []


def test_case_scoped_context_rejects_wrong_case_argument(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(1)
    training = pd.DataFrame(
        [{"case_id": "case", "entity_id": "entity", "business_unit": "consumer", "narrative": "exception"}]
    )
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        training,
        controls,
        ActionLedger(tmp_path / "case-binding.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    row = training.iloc[0].copy()
    row["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")
    row["pit_historical_failures"] = row["future_failures"] = 0
    row["repeat_count"] = row["data_sensitivity"] = 0
    context = json.loads(
        workflow.context_for_llm(
            row,
            WorkflowConfig(reranker=False),
            session_token=credentials["consumer"],
            selected_tools=frozenset({"query_case_data"}),
            selected_tool_arguments={"query_case_data": {"case_id": "other-case"}},
        )
    )
    assert context["query_case_data"] == {"error": "argument_or_authorization_rejected"}


def test_replayed_action_trace_reports_achieved_state_without_duplicate_side_effect(
    tmp_path: Path, tool_corpus: Path
) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(1)
    row = pd.Series(
        {
            "case_id": "case",
            "entity_id": "entity",
            "business_unit": "consumer",
            "narrative": "exception",
            "event_timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "pit_historical_failures": 0,
            "future_failures": 0,
            "repeat_count": 0,
            "data_sensitivity": 0,
        }
    )
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        pd.DataFrame([row]),
        controls,
        ActionLedger(tmp_path / "replay-trace.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    config = WorkflowConfig(retrieval=False, ml_risk=False, anomaly=False, verifier=False, hitl=False)
    token = credentials["consumer"]
    first = workflow.execute(row, config, ("LOW", "AUTO", True), session_token=token)
    second = workflow.execute(row, config, ("LOW", "AUTO", True), session_token=token)
    assert first.action_executed and first.action_performed_this_invocation
    assert second.action_executed and not second.action_performed_this_invocation


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


def test_tool_timeout_returns_within_declared_wall_clock() -> None:
    def implementation(value: Probe) -> Probe:
        Event().wait(0.03)
        return value

    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "slow_probe",
            Probe,
            Probe,
            0,
            True,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            0.02,
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
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="failed after"):
        registry.invoke(
            "slow_probe",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    elapsed = time.perf_counter() - started
    assert 0.03 <= elapsed < 0.2
    assert registry.audit_events[-1]["status"] == "timeout_completed_without_detached_worker"


def test_timed_out_write_cannot_commit_later(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "timeout-write.sqlite")
    authority = ApprovalAuthority(b"timeout-write-secret")
    payload = {"status": "investigated"}
    evidence_hash = "evidence"
    token = authority.issue_system(
        case_id="case",
        action_type="propose_case_update",
        payload=payload,
        workflow_version="workflow",
        policy_version="local-policy-v1",
        evidence_hash=evidence_hash,
        risk_tier=1,
        authorization_outcome="ALLOW",
    )

    def implementation(value: Probe) -> Probe:
        time.sleep(0.05)
        ledger.execute_simulated(
            case_id="case",
            action_type="propose_case_update",
            payload=payload,
            workflow_version="workflow",
            authorization_token=token,
            approval_authority=authority,
            evidence_hash=evidence_hash,
        )
        return value

    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "slow_write",
            Probe,
            Probe,
            1,
            False,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            0.01,
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
    with pytest.raises(RuntimeError, match="failed after"):
        registry.invoke(
            "slow_write",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    time.sleep(0.1)
    with ledger._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM action_ledger").fetchone()[0] == 0


def test_timeout_racing_entered_commit_waits_and_reconciles() -> None:
    entered = Event()
    release = Event()
    caller_done = Event()
    committed: list[int] = []
    returned: list[Probe] = []

    def implementation(value: Probe) -> Probe:
        with tool_commit_section():
            entered.set()
            release.wait(timeout=1)
            committed.append(value.value)
        return value

    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "commit_race",
            Probe,
            Probe,
            1,
            False,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            0.2,
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

    def invoke() -> None:
        returned.append(
            registry.invoke(
                "commit_race",
                {"value": 7},
                identity=identity,
                scope="consumer",
                data_classification=0,
                severity=Severity.LOW,
            )  # type: ignore[arg-type]
        )
        caller_done.set()

    caller = Thread(target=invoke)
    caller.start()
    assert entered.wait(timeout=1)
    # Leave ample time for a cold Windows test worker to enter the commit
    # section, then hold it beyond the declared deadline deterministically.
    time.sleep(0.25)
    assert not caller_done.is_set()
    release.set()
    caller.join(timeout=1)
    assert not caller.is_alive()
    assert committed == [7]
    assert returned == [Probe(value=7)]
    assert registry.audit_events[-1]["status"] == "success"


def test_context_id_is_bound_to_exact_ordered_tool_arguments(tmp_path: Path, tool_corpus: Path) -> None:
    controls = pd.read_parquet(tool_corpus / "data/staging/nist_controls_raw.parquet").head(1)
    row = pd.Series(
        {
            "case_id": "case",
            "entity_id": "entity",
            "business_unit": "consumer",
            "narrative": "exception",
            "event_timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "pit_historical_failures": 0,
            "future_failures": 0,
            "repeat_count": 0,
            "data_sensitivity": 0,
        }
    )
    provider, credentials = _sessions("consumer")
    workflow = GovernedWorkflow(
        pd.DataFrame([row]),
        controls,
        ActionLedger(tmp_path / "plan-binding.sqlite"),
        ApprovalAuthority(b"secret"),
        risk_service=FakeRisk(),  # type: ignore[arg-type]
        identity_provider=provider,
        corpus_root=tool_corpus,
    )
    context = json.loads(
        workflow.context_for_llm(
            row,
            WorkflowConfig(retrieval=False),
            session_token=credentials["consumer"],
            selected_tool_calls=[("query_case_data", {"case_id": "case"})],
        )
    )
    with pytest.raises(PermissionError, match="context-bound ordered plan"):
        workflow.execute(
            row,
            WorkflowConfig(retrieval=False),
            ("LOW", "AUTO", True),
            ["query_case_data"],
            [{"case_id": "different"}],
            session_token=credentials["consumer"],
            context_id=context["_context_id"],
        )
    with pytest.raises(PermissionError, match="not issued"):
        workflow.execute(
            row,
            WorkflowConfig(retrieval=False),
            ("LOW", "AUTO", True),
            ["query_case_data"],
            [{"case_id": "case"}],
            session_token=credentials["consumer"],
            context_id="never-issued",
        )
    workflow.execute(
        row,
        WorkflowConfig(retrieval=False),
        ("LOW", "AUTO", True),
        ["query_case_data"],
        [{"case_id": "case"}],
        session_token=credentials["consumer"],
        context_id=context["_context_id"],
    )
    with pytest.raises(PermissionError, match="not issued"):
        workflow.execute(
            row,
            WorkflowConfig(retrieval=False),
            ("LOW", "AUTO", True),
            ["query_case_data"],
            [{"case_id": "case"}],
            session_token=credentials["consumer"],
            context_id=context["_context_id"],
        )
