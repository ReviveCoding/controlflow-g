from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from controlflow.v22.approval import ApprovalIssuer, ApprovalVerifier
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateRunner
from controlflow.v22.checkpoint import REQUIRED_FIELDS
from controlflow.v22.executor import InjectedCrash, TransactionalExecutor, ledger_security_metrics, verify_ledger
from controlflow.v22.policy import ActionRegistry, AuthorizationState, PolicyDecisionPoint
from controlflow.v22.retrieval import EvidenceRetriever
from controlflow.v22.schemas import (
    AuthenticatedContext,
    ModelDecision,
    PolicyDecision,
    PolicyInput,
    ProposedAction,
    Severity,
)
from controlflow.v22.temporal import CandidateTemporalRetriever, load_policy_corpus

ROOT = Path(__file__).resolve().parents[2]


def setup_executor(tmp_path: Path, *, fault: str | None = None):
    registry = ActionRegistry(ROOT / "configs/v22/action_registry.yaml")
    authorization = AuthorizationState(
        {
            "alice": {
                "active": True,
                "role": "senior_investigator",
                "business_unit": "consumer",
                "region": "US",
                "clearance": 3,
            }
        }
    )
    pdp = PolicyDecisionPoint(
        policy_path=ROOT / "configs/v22/policy.yaml", registry=registry, authorization=authorization
    )
    public = tmp_path / "public.pem"
    issuer = ApprovalIssuer.create_ephemeral(tmp_path / "private", public)
    executor = TransactionalExecutor(
        tmp_path / "state.sqlite",
        pdp=pdp,
        registry=registry,
        approval_verifier=ApprovalVerifier(public),
        candidate_bundle_hash="a" * 64,
        fault=fault,
    )
    context = AuthenticatedContext(
        identity="alice",
        role="senior_investigator",
        business_unit="consumer",
        region="US",
        clearance=3,
        purpose="control_exception_investigation",
        requested_scope="consumer",
        data_classification=2,
    )
    action = ProposedAction(
        case_id="CASE-1",
        action_name="OPEN_REMEDIATION",
        payload={"mode": "simulated"},
        workflow_version="v22-workflow-1",
    )
    policy_input = PolicyInput(
        case_id=action.case_id,
        context=context,
        severity=Severity.HIGH,
        critical_probability=0.2,
        evidence_sufficient=True,
        conflict_state=False,
        novelty=0.1,
        uncertainty=0.1,
        proposed_action=action.action_name,
    )
    return executor, issuer, authorization, action, policy_input


def test_signed_review_commit_is_atomic_exactly_once_and_auditable(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    assert policy.decision is PolicyDecision.REQUIRE_REVIEW
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    first = executor.commit(action, policy_input, approval=approval, idempotency_key="key-1")
    second = executor.commit(action, policy_input, approval=approval, idempotency_key="key-1")
    assert first["committed"] == 1
    assert second["event_id"] == first["event_id"]
    assert len(executor.rows("case_state")) == 1
    assert len(executor.rows("approval_consumption")) == 1
    assert verify_ledger(executor.path)["valid"]
    assert ledger_security_metrics(executor.path)["duplicate_commits"] == 0


@pytest.mark.parametrize("mode", ["missing", "tampered", "reused"])
def test_review_approval_cannot_be_bypassed_tampered_or_reused(tmp_path: Path, mode: str) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    if mode == "missing":
        selected = None
    elif mode == "tampered":
        selected = approval.model_copy(update={"signature_b64": approval.signature_b64[:-2] + "AA"})
    else:
        first = executor.commit(action, policy_input, approval=approval, idempotency_key="key-first")
        assert first["committed"] == 1
        selected = approval
    result = executor.commit(action, policy_input, approval=selected, idempotency_key=f"key-{mode}")
    assert result["committed"] == 0
    assert result["failure_reason"] in {
        "missing_approval",
        "invalid_approval_signature",
        "approval_already_consumed",
        "logical_action_already_committed",
    }


def test_noncommitted_retry_reuses_the_authoritative_event(tmp_path: Path) -> None:
    executor, _, _, action, policy_input = setup_executor(tmp_path)
    executor.prepare(action, policy_input)
    first = executor.commit(action, policy_input, approval=None, idempotency_key="deny-once")
    second = executor.commit(action, policy_input, approval=None, idempotency_key="deny-once")
    assert first["committed"] == 0
    assert second["event_id"] == first["event_id"]
    assert len(executor.rows("action_ledger")) == 1


def test_authorization_is_revalidated_immediately_before_commit(tmp_path: Path) -> None:
    executor, issuer, authorization, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    authorization.replace(
        "alice",
        {"active": True, "role": "investigator", "business_unit": "consumer", "region": "US", "clearance": 1},
    )
    result = executor.commit(action, policy_input, approval=approval, idempotency_key="changed-auth")
    assert result["committed"] == 0
    assert result["failure_reason"] == "authorization_changed_since_proposal"


@pytest.mark.parametrize("change", ["risk", "policy_version"])
def test_material_risk_or_policy_version_change_invalidates_old_approval(tmp_path: Path, change: str) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    if change == "risk":
        policy_input = policy_input.model_copy(update={"critical_probability": 0.99, "severity": Severity.CRITICAL})
    else:
        executor.pdp.version = "v22-policy-2"
    result = executor.commit(action, policy_input, approval=approval, idempotency_key=f"changed-{change}")
    assert result["committed"] == 0
    assert result["failure_reason"] == "authorization_changed_since_proposal"


def test_unknown_action_denied_by_authoritative_registry(tmp_path: Path) -> None:
    executor, _, _, action, policy_input = setup_executor(tmp_path)
    action = action.model_copy(update={"action_name": "TRANSFER_MONEY"})
    policy_input = policy_input.model_copy(update={"proposed_action": "TRANSFER_MONEY"})
    result = executor.commit(action, policy_input, approval=None, idempotency_key="unknown")
    assert result["pdp_decision"] == "DENY"
    assert result["committed"] == 0


@pytest.mark.parametrize(
    "fault", ["after_approval_verification", "after_state_mutation_before_commit", "during_ledger_write"]
)
def test_precommit_crashes_roll_back_state_ledger_and_token(tmp_path: Path, fault: str) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path, fault=fault)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    with pytest.raises(InjectedCrash, match=fault):
        executor.commit(action, policy_input, approval=approval, idempotency_key="crash")
    assert executor.rows("case_state") == []
    assert executor.rows("action_ledger") == []
    assert executor.rows("approval_consumption") == []


def test_immediate_after_commit_crash_recovers_idempotently(tmp_path: Path) -> None:
    executor, issuer, authorization, action, policy_input = setup_executor(tmp_path, fault="immediately_after_commit")
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    with pytest.raises(InjectedCrash, match="immediately_after_commit"):
        executor.commit(action, policy_input, approval=approval, idempotency_key="after-commit")
    registry = executor.registry
    recovered = TransactionalExecutor(
        executor.path,
        pdp=PolicyDecisionPoint(
            policy_path=ROOT / "configs/v22/policy.yaml", registry=registry, authorization=authorization
        ),
        registry=registry,
        approval_verifier=executor.approval_verifier,
        candidate_bundle_hash="a" * 64,
    )
    row = recovered.commit(action, policy_input, approval=approval, idempotency_key="after-commit")
    assert row["committed"] == 1
    assert len(recovered.rows("action_ledger")) == 1


def test_ledger_verifier_detects_modified_authorization_provenance(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    executor.commit(action, policy_input, approval=approval, idempotency_key="tamper")
    with sqlite3.connect(executor.path) as conn:
        conn.execute("UPDATE action_ledger SET authorization_context_hash='tampered'")
    audit = verify_ledger(executor.path)
    assert not audit["valid"]
    assert any(item.startswith("modified_event") for item in audit["failures"])


@pytest.mark.parametrize("fault", ["before_pdp", "after_pdp"])
def test_proposal_fault_points_write_no_decision(tmp_path: Path, fault: str) -> None:
    executor, _, _, action, policy_input = setup_executor(tmp_path, fault=fault)
    with pytest.raises(InjectedCrash, match=fault):
        executor.prepare(action, policy_input)
    assert executor.rows("policy_decisions") == []


def test_before_transaction_fault_writes_no_business_or_ledger_state(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path, fault="before_transaction")
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    with pytest.raises(InjectedCrash, match="before_transaction"):
        executor.commit(action, policy_input, approval=approval, idempotency_key="before-transaction")
    assert executor.rows("case_state") == []
    assert executor.rows("action_ledger") == []


def test_restart_fault_is_deterministic(tmp_path: Path) -> None:
    with pytest.raises(InjectedCrash, match="during_restart"):
        setup_executor(tmp_path, fault="during_restart")
    executor, _, _, _, _ = setup_executor(tmp_path)
    assert verify_ledger(executor.path)["valid"]


def test_idempotency_key_is_bound_to_case_and_action(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    assert executor.commit(action, policy_input, approval=approval, idempotency_key="bound")["committed"] == 1
    other = action.model_copy(update={"case_id": "CASE-2"})
    other_input = policy_input.model_copy(update={"case_id": "CASE-2"})
    with pytest.raises(RuntimeError, match="IDEMPOTENCY_KEY_BINDING_MISMATCH"):
        executor.commit(other, other_input, approval=None, idempotency_key="bound")


def test_cross_case_policy_input_fails_closed(tmp_path: Path) -> None:
    executor, _, _, action, policy_input = setup_executor(tmp_path)
    with pytest.raises(RuntimeError, match="POLICY_ACTION_CASE_MISMATCH"):
        executor.commit(
            action,
            policy_input.model_copy(update={"case_id": "OTHER"}),
            approval=None,
            idempotency_key="cross-case",
        )


def test_same_logical_action_with_different_keys_commits_once(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approvals = [
        issuer.issue(action=action, policy=policy, reviewer_id=f"reviewer-{index}", approve=True) for index in range(2)
    ]

    def commit(index: int):
        return executor.commit(action, policy_input, approval=approvals[index], idempotency_key=f"race-{index}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(commit, range(2)))
    assert sum(int(row["committed"]) for row in rows) == 1
    assert len(executor.rows("case_state")) == 1
    assert ledger_security_metrics(executor.path)["duplicate_commits"] == 0


@pytest.mark.parametrize("mode", ["expired", "rejected", "wrong_action", "wrong_workflow", "wrong_key"])
def test_invalid_approval_variants_fail_closed(tmp_path: Path, mode: str) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    now = datetime.now(UTC)
    if mode == "expired":
        approval = issuer.issue(
            action=action,
            policy=policy,
            reviewer_id="reviewer",
            approve=True,
            now=now - timedelta(hours=1),
            lifetime=timedelta(minutes=1),
        )
    elif mode == "rejected":
        approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=False)
    elif mode == "wrong_action":
        approval = issuer.issue(
            action=action.model_copy(update={"payload": {"mode": "different"}}),
            policy=policy,
            reviewer_id="reviewer",
            approve=True,
        )
    elif mode == "wrong_workflow":
        approval = issuer.issue(
            action=action.model_copy(update={"workflow_version": "wrong"}),
            policy=policy,
            reviewer_id="reviewer",
            approve=True,
        )
    else:
        approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True).model_copy(
            update={"key_id": "unknown-key"}
        )
    row = executor.commit(action, policy_input, approval=approval, idempotency_key=f"invalid-{mode}", now=now)
    assert row["committed"] == 0


def test_registry_version_change_invalidates_approval(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    executor.registry.version = "changed-registry"
    row = executor.commit(action, policy_input, approval=approval, idempotency_key="registry-change")
    assert row["committed"] == 0
    assert row["failure_reason"] == "authorization_changed_since_proposal"


def test_unverified_approval_is_not_consumed_on_allow_path(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    action = action.model_copy(update={"action_name": "CLOSE_NO_ACTION"})
    policy_input = policy_input.model_copy(
        update={"severity": Severity.LOW, "proposed_action": "CLOSE_NO_ACTION", "uncertainty": 0.0}
    )
    policy = executor.prepare(action, policy_input)
    assert policy.decision is PolicyDecision.ALLOW
    unrelated = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True).model_copy(
        update={"signature_b64": "not-a-signature"}
    )
    row = executor.commit(action, policy_input, approval=unrelated, idempotency_key="allow-unverified")
    assert row["committed"] == 1
    assert executor.rows("approval_consumption") == []


def test_foreign_keys_and_deleted_chain_detection(tmp_path: Path) -> None:
    executor, issuer, _, action, policy_input = setup_executor(tmp_path)
    policy = executor.prepare(action, policy_input)
    approval = issuer.issue(action=action, policy=policy, reviewer_id="reviewer", approve=True)
    executor.commit(action, policy_input, approval=approval, idempotency_key="fk-chain")
    with sqlite3.connect(executor.path) as conn:
        assert conn.execute("PRAGMA foreign_key_list(action_ledger)").fetchall()
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM approval_consumption")
        conn.execute("DELETE FROM action_ledger")
    audit = verify_ledger(executor.path)
    assert not audit["valid"]
    assert "head_without_events" in audit["failures"]


def test_actual_candidate_runner_recovers_after_durable_commit_before_partial_output(tmp_path: Path) -> None:
    executor, _, _, _, _ = setup_executor(tmp_path, fault="immediately_after_commit")

    class ObservableBundle:
        bundle_hash = "a" * 64

        @staticmethod
        def infer(_frame, **_options):
            return ModelDecision(
                severity=Severity.LOW,
                critical_probability=0.01,
                root_cause="PROCESS",
                novelty=0.1,
                uncertainty=0.1,
            )

    candidate = CandidateRunner(
        bundle=ObservableBundle(),  # type: ignore[arg-type]
        evidence=EvidenceRetriever([]),
        temporal=CandidateTemporalRetriever(load_policy_corpus(ROOT / "configs/v22/temporal_policies.yaml")),
        executor=executor,
        workflow_version="v22-workflow-1",
    )
    workflow = CandidateExecutionWorkflow(candidate, reviewer=None)
    frame = pd.DataFrame(
        [
            {
                "case_id": "CASE-RUNNER",
                "entity_id": "ENTITY-RUNNER",
                "event_time": "2026-05-01T00:00:00Z",
                "system_time": "2026-09-01T00:00:00Z",
                "authenticated_identity": "alice",
                "role": "senior_investigator",
                "business_unit": "consumer",
                "region": "US",
                "clearance": 3,
                "purpose": "control_exception_investigation",
                "requested_scope": "consumer",
                "data_classification": 2,
                "control_family": "AC",
                "control_test_count": 20,
                "control_test_failures": 1,
                "historical_incidents": 0,
                "transaction_count": 100,
                "anomaly_count": 1,
                "privileged_event_count": 0,
                "repeat_exception_ratio": 0.1,
                "customer_impact_signal": 0.1,
                "policy_risk_signal": 0.1,
                "affected_customers": 0,
                "amount_variance": 0.0,
                "scope_difference": 0.0,
                "narrative": "observable low risk case",
                "evidence_query": "incident-runner AC consumer process 2026",
            }
        ]
    )
    checkpoint_fields = {name: "bound" for name in REQUIRED_FIELDS}
    partial = tmp_path / "actual.partial.jsonl"
    checkpoint = tmp_path / "actual.checkpoint.json"
    output = tmp_path / "actual.parquet"
    with pytest.raises(InjectedCrash, match="immediately_after_commit"):
        workflow.run_dataset(
            frame,
            output_path=output,
            partial_path=partial,
            checkpoint_path=checkpoint,
            checkpoint_fields=checkpoint_fields,
        )
    assert len([row for row in executor.rows("action_ledger") if row["committed"]]) == 1
    executor.fault = None
    results = workflow.run_dataset(
        frame,
        output_path=output,
        partial_path=partial,
        checkpoint_path=checkpoint,
        checkpoint_fields=checkpoint_fields,
    )
    assert len(results) == 1
    assert len(executor.rows("action_ledger")) == 1
    record = json.loads(partial.read_text(encoding="utf-8"))
    record["severity"] = "HIGH"
    partial.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed partial records were modified"):
        workflow.run_dataset(
            frame,
            output_path=output,
            partial_path=partial,
            checkpoint_path=checkpoint,
            checkpoint_fields=checkpoint_fields,
        )
