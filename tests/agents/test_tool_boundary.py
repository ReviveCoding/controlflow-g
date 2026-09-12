from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event, Thread
from threading import enumerate as enumerate_threads

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

import controlflow.tools.registry as registry_module
from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig
from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.identity import SessionIdentityProvider
from controlflow.authorization.policy import LocalPolicyBackend
from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import ProjectPaths
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import IdentityContext, Severity
from controlflow.tools.deadline import tool_commit_section
from controlflow.tools.registry import MAX_TOOL_WORKERS, ToolRegistry, ToolSpec


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
    assert 0.02 <= elapsed < 0.2
    assert registry.audit_events[-1]["status"] == "timeout_precommit_cancelled"
    deadline = time.monotonic() + 1.0
    while any(thread.name == "controlflow-tool-slow_probe" for thread in enumerate_threads()):
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_success_is_not_published_before_worker_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_entered = Event()
    allow_cleanup = Event()
    caller_done = Event()
    result_values: list[int] = []
    original_unregister = registry_module.unregister_tool_worker

    def blocked_unregister() -> None:
        cleanup_entered.set()
        assert allow_cleanup.wait(timeout=2)
        original_unregister()

    monkeypatch.setattr(registry_module, "unregister_tool_worker", blocked_unregister)
    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "cleanup_handshake",
            Probe,
            Probe,
            0,
            True,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            1.0,
            0,
            lambda value: value,
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

    def invoke_first() -> None:
        result = registry.invoke(
            "cleanup_handshake",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
        result_values.append(result.value)  # type: ignore[attr-defined]
        caller_done.set()

    caller = Thread(target=invoke_first)
    caller.start()
    assert cleanup_entered.wait(timeout=1)
    assert not caller_done.is_set()
    allow_cleanup.set()
    caller.join(timeout=1)
    assert not caller.is_alive()
    assert result_values == [1]
    assert (
        registry.invoke(
            "cleanup_handshake",
            {"value": 2},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        ).value
        == 2
    )


def test_never_returning_precommit_work_is_bounded_and_cancelled() -> None:
    release = Event()

    def implementation(value: Probe) -> Probe:
        release.wait()
        return value

    registry = ToolRegistry(LocalPolicyBackend())
    registry.register(
        ToolSpec(
            "hung_probe",
            Probe,
            Probe,
            0,
            True,
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
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="failed after"):
        registry.invoke(
            "hung_probe",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    elapsed = time.perf_counter() - started
    release.set()
    deadline = time.monotonic() + 1.0
    while any(thread.name == "controlflow-tool-hung_probe" for thread in enumerate_threads()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert elapsed < 0.2
    assert registry.audit_events[-1]["status"] == "timeout_precommit_cancelled"


def test_hung_worker_capacity_opens_fail_closed_circuit_breaker() -> None:
    release = Event()
    identity = IdentityContext(
        user_id="analyst",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=1,
        purpose="investigation",
        session_id="session",
    )

    def implementation(value: Probe) -> Probe:
        release.wait()
        return value

    registries: list[ToolRegistry] = []
    try:
        for index in range(MAX_TOOL_WORKERS):
            registry = ToolRegistry(LocalPolicyBackend())
            registry.register(
                ToolSpec(
                    f"hung_{index}",
                    Probe,
                    Probe,
                    0,
                    True,
                    frozenset({"Control Analyst"}),
                    frozenset({"consumer"}),
                    False,
                    0.005,
                    0,
                    implementation,
                )
            )
            with pytest.raises(RuntimeError, match="failed after"):
                registry.invoke(
                    f"hung_{index}",
                    {"value": index},
                    identity=identity,
                    scope="consumer",
                    data_classification=0,
                    severity=Severity.LOW,
                )
            registries.append(registry)
        overflow = ToolRegistry(LocalPolicyBackend())
        overflow.register(
            ToolSpec(
                "overflow",
                Probe,
                Probe,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({"consumer"}),
                False,
                0.005,
                0,
                implementation,
            )
        )
        with pytest.raises(RuntimeError, match="circuit breaker"):
            overflow.invoke(
                "overflow",
                {"value": 99},
                identity=identity,
                scope="consumer",
                data_classification=0,
                severity=Severity.LOW,
            )
        assert overflow.audit_events[-1]["status"] == "worker_capacity_exhausted"
    finally:
        release.set()
        deadline = time.monotonic() + 1.0
        while any(thread.name.startswith("controlflow-tool-hung_") for thread in enumerate_threads()):
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_gpu_lease_remains_locked_until_timed_out_worker_exits(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.state.mkdir(parents=True)
    release = Event()
    timeout_observed = Event()
    gpu_scope_exited = Event()
    identity = IdentityContext(
        user_id="analyst",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=1,
        purpose="investigation",
        session_id="session",
    )

    def implementation(value: Probe) -> Probe:
        release.wait()
        return value

    def owner() -> None:
        registry = ToolRegistry(LocalPolicyBackend())
        registry.register(
            ToolSpec(
                "gpu_hung",
                Probe,
                Probe,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({"consumer"}),
                False,
                0.01,
                0,
                implementation,
            )
        )
        with GpuSemaphore(paths):
            with pytest.raises(RuntimeError, match="failed after"):
                registry.invoke(
                    "gpu_hung",
                    {"value": 1},
                    identity=identity,
                    scope="consumer",
                    data_classification=0,
                    severity=Severity.LOW,
                )
            timeout_observed.set()
        gpu_scope_exited.set()

    owner_thread = Thread(target=owner)
    owner_thread.start()
    assert timeout_observed.wait(timeout=1)
    time.sleep(0.02)
    assert not gpu_scope_exited.is_set()
    with pytest.raises(RuntimeError, match="GPU is locked"):
        GpuSemaphore(paths).__enter__()
    release.set()
    owner_thread.join(timeout=1)
    assert not owner_thread.is_alive()
    assert gpu_scope_exited.is_set()
    with GpuSemaphore(paths):
        pass


def test_worker_admission_cannot_race_gpu_scope_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ProjectPaths(tmp_path)
    paths.state.mkdir(parents=True)
    release_entered = Event()
    allow_release = Event()
    implementation_started = Event()
    gpu = GpuSemaphore(paths)
    original_release = gpu._lock.release

    def blocked_release(*args: object, **kwargs: object) -> None:
        release_entered.set()
        assert allow_release.wait(timeout=2)
        original_release(*args, **kwargs)

    monkeypatch.setattr(gpu._lock, "release", blocked_release)

    def owner() -> None:
        with gpu:
            pass

    owner_thread = Thread(target=owner)
    owner_thread.start()
    assert release_entered.wait(timeout=1)

    registry = ToolRegistry(LocalPolicyBackend())

    def implementation(value: Probe) -> Probe:
        implementation_started.set()
        return value

    registry.register(
        ToolSpec(
            "release_race",
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
    with pytest.raises(RuntimeError, match="admission gate"):
        registry.invoke(
            "release_race",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    assert registry.audit_events[-1]["status"] == "worker_admission_quarantined"
    assert not implementation_started.is_set()
    allow_release.set()
    owner_thread.join(timeout=1)
    assert not owner_thread.is_alive()
    assert (
        registry.invoke(
            "release_race",
            {"value": 2},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        ).value
        == 2
    )


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
    errors: list[str] = []

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
        try:
            registry.invoke(
                "commit_race",
                {"value": 7},
                identity=identity,
                scope="consumer",
                data_classification=0,
                severity=Severity.LOW,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
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
    assert errors == ["tool commit_race failed after 1 attempts"]
    assert registry.audit_events[-1]["status"] == "timeout_after_commit_reconciled"


def test_timeout_waits_for_external_anchor_and_anchor_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = ActionLedger(tmp_path / "anchor-timeout.sqlite")
    authority = ApprovalAuthority(b"anchor-timeout-secret")
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
    entered_anchor = Event()
    release_anchor = Event()
    caller_done = Event()
    original_sync = ledger._sync_action_anchor

    def blocked_sync() -> None:
        entered_anchor.set()
        release_anchor.wait(timeout=2)
        original_sync()

    monkeypatch.setattr(ledger, "_sync_action_anchor", blocked_sync)

    def implementation(value: Probe) -> Probe:
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
            "anchor_write",
            Probe,
            Probe,
            1,
            False,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            0.1,
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
        with pytest.raises(RuntimeError, match="failed after"):
            registry.invoke(
                "anchor_write",
                {"value": 1},
                identity=identity,
                scope="consumer",
                data_classification=0,
                severity=Severity.LOW,
            )
        caller_done.set()

    caller = Thread(target=invoke)
    caller.start()
    assert entered_anchor.wait(timeout=1)
    time.sleep(0.15)
    assert not caller_done.is_set()
    release_anchor.set()
    caller.join(timeout=1)
    assert caller_done.is_set()
    assert registry.audit_events[-1]["status"] == "timeout_after_commit_reconciled"
    assert ledger.verify_event_chain()

    broken = ActionLedger(tmp_path / "anchor-failure.sqlite")
    broken_token = authority.issue_system(
        case_id="broken",
        action_type="propose_case_update",
        payload=payload,
        workflow_version="workflow",
        policy_version="local-policy-v1",
        evidence_hash=evidence_hash,
        risk_tier=1,
        authorization_outcome="ALLOW",
    )
    monkeypatch.setattr(broken, "_sync_action_anchor", lambda: (_ for _ in ()).throw(OSError("anchor unavailable")))

    def broken_implementation(value: Probe) -> Probe:
        broken.execute_simulated(
            case_id="broken",
            action_type="propose_case_update",
            payload=payload,
            workflow_version="workflow",
            authorization_token=broken_token,
            approval_authority=authority,
            evidence_hash=evidence_hash,
        )
        return value

    broken_registry = ToolRegistry(LocalPolicyBackend())
    broken_registry.register(
        ToolSpec(
            "broken_anchor_write",
            Probe,
            Probe,
            1,
            False,
            frozenset({"Control Analyst"}),
            frozenset({"consumer"}),
            False,
            1.0,
            0,
            broken_implementation,
        )
    )
    with pytest.raises(RuntimeError, match="failed after"):
        broken_registry.invoke(
            "broken_anchor_write",
            {"value": 1},
            identity=identity,
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    assert not broken.verify_event_chain()


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
