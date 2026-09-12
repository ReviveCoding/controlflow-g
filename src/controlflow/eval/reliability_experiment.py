from __future__ import annotations

import hashlib
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel

from controlflow.agents.durable import DurableWorkflowRunner
from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig
from controlflow.audit.ledger import ActionLedger
from controlflow.audit.recovery import configured_recovery_authority
from controlflow.authorization.identity import SessionIdentityProvider
from controlflow.authorization.policy import LocalPolicyBackend
from controlflow.core.resources import wait_for_tool_workers
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import HumanDecision, IdentityContext, ReviewDecision, Severity
from controlflow.tools.registry import ToolRegistry, ToolSpec
from controlflow.tools.sql_safety import validate_readonly_sql


class Probe(BaseModel):
    value: int


class FaultingModelEndpoint:
    def invoke(self, value: Probe, *, unavailable: bool) -> Probe:
        if unavailable:
            raise RuntimeError("injected model endpoint unavailable")
        time.sleep(0.03)
        return value


class FaultingRetriever:
    def search(self, value: Probe) -> Probe:
        time.sleep(0.03)
        return value


class FaultingWarehouse:
    def query(self, value: Probe) -> Probe:
        validate_readonly_sql("SELECT case_id FROM gold.fact_case LIMIT 1", frozenset({"gold.fact_case"}))
        time.sleep(0.03)
        return value


def _identity() -> IdentityContext:
    return IdentityContext(
        user_id="u",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=2,
        purpose="reliability test",
        session_id="fault-session",
    )


def _tool_fault(kind: str, tool_name: str = "probe") -> tuple[bool, int]:
    attempts = {"count": 0}
    registry = ToolRegistry(LocalPolicyBackend())

    def implementation(value: Probe) -> Probe:
        attempts["count"] += 1
        if tool_name == "llm_inference":
            return FaultingModelEndpoint().invoke(value, unavailable=kind == "unavailable")
        if tool_name == "retrieval_search":
            return FaultingRetriever().search(value)
        if tool_name == "readonly_sql_query":
            return FaultingWarehouse().query(value)
        return FaultingModelEndpoint().invoke(value, unavailable=True)

    registry.register(
        ToolSpec(
            name=tool_name,
            input_model=Probe,
            output_model=Probe,
            risk_tier=0,
            read_only=True,
            allowed_roles=frozenset({"Control Analyst"}),
            allowed_data_scopes=frozenset({"consumer"}),
            human_review_required=False,
            timeout_seconds=0.005,
            max_retries=1,
            implementation=implementation,
        )
    )
    try:
        registry.invoke(
            tool_name,
            {"value": 1},
            identity=_identity(),
            scope="consumer",
            data_classification=0,
            severity=Severity.LOW,
        )
    except RuntimeError:
        # A wall-clock timeout returns promptly while its worker is quarantined.
        # Recovery is complete only after that worker exits and releases the
        # process/GPU capacity circuit; measure that drain as part of latency.
        wait_for_tool_workers()
        if kind == "timeout":
            detected = (
                len(registry.audit_events) == 1 and registry.audit_events[0]["status"] == "timeout_precommit_cancelled"
            )
        else:
            detected = len(registry.audit_events) == 2 and all(
                event["status"] == "error" for event in registry.audit_events
            )
        return detected, attempts["count"] - 1
    return False, attempts["count"] - 1


def run() -> str:
    paths = ProjectPaths.discover()
    recovery_authority = configured_recovery_authority()
    ledger = ActionLedger(
        paths.root / "artifacts/reliability_action_ledger_v12.sqlite",
        recovery_authority=recovery_authority,
    )
    authority = ApprovalAuthority(
        secrets.token_bytes(32), reviewer_entitlements={"reviewer": ("Risk Manager", "enterprise")}
    )
    rows = []
    development = pd.read_parquet(paths.root / "data/silver/synthetic_cases_development.parquet")
    recovery_case = development[development.case_type.eq("normal") & development.severity.isin(["LOW", "MEDIUM"])].iloc[
        0
    ]
    controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
    recovery_config = WorkflowConfig(
        retrieval=False,
        temporal_retrieval=False,
        reranker=False,
        ml_risk=False,
        calibration=False,
        anomaly=False,
        verifier=False,
        hitl=False,
    )
    with PhaseRun("P20", paths) as phase:
        for index, failure in enumerate(
            (
                "LLM timeout",
                "retrieval timeout",
                "SQL timeout",
                "model unavailable",
                "agent crash",
                "crash before approval",
                "crash after approval",
                "crash after action",
                "duplicate resume",
                "partial pipeline failure",
                "checkpoint recovery",
                "audit anchor crash",
            )
        ):
            started = time.perf_counter()
            retries = 0
            duplicate = False
            detected = False
            injected = False
            status = "not_run"
            if "timeout" in failure.casefold():
                boundary = {
                    "LLM timeout": "llm_inference",
                    "retrieval timeout": "retrieval_search",
                    "SQL timeout": "readonly_sql_query",
                }[failure]
                recovery, retries = _tool_fault("timeout", boundary)
                detected, injected, status = recovery, True, "ok"
            elif failure == "model unavailable":
                recovery, retries = _tool_fault("unavailable")
                detected, injected, status = recovery, True, "ok"
            elif failure in {"agent crash", "partial pipeline failure", "checkpoint recovery"}:
                checkpoint = paths.root / f"build/fault-checkpoint-v11-{index}.json"
                workflow_ledger = paths.root / f"artifacts/reliability_workflow_v11_{index}.sqlite"
                if checkpoint.exists():
                    crash_detected = True
                else:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "controlflow.eval.recovery_worker",
                            "durable",
                            str(checkpoint),
                            str(recovery_case.case_id),
                            "--ledger",
                            str(workflow_ledger),
                            "--fault",
                            "before_execute" if failure == "agent crash" else "after_execute_before_checkpoint",
                        ],
                        check=False,
                        timeout=20,
                    )
                    crash_detected = process.returncode == 91
                identity_provider, credentials = SessionIdentityProvider.issue_for_business_units(
                    set(development["business_unit"].astype(str))
                )
                resumed_workflow = GovernedWorkflow(
                    development,
                    controls,
                    ActionLedger(workflow_ledger, recovery_authority=configured_recovery_authority()),
                    ApprovalAuthority(secrets.token_bytes(32)),
                    identity_provider=identity_provider,
                )
                resumed = DurableWorkflowRunner(resumed_workflow, checkpoint).resume(
                    recovery_case,
                    recovery_config,
                    ("LOW", "AUTO", True),
                    session_token=credentials[str(recovery_case.business_unit)],
                )
                with ActionLedger(
                    workflow_ledger, recovery_authority=configured_recovery_authority()
                )._connect() as connection:
                    executed_count = int(
                        connection.execute("SELECT COUNT(*) FROM action_ledger WHERE status='EXECUTED'").fetchone()[0]
                    )
                recovery = crash_detected and resumed.predicted_disposition == "AUTO" and executed_count == 1
                detected, injected, status = crash_detected, True, "ok"
                retries = 1
            elif failure == "audit anchor crash":
                ledger.record_system_event("PRE_CRASH_EVENT", "reliability-runner", {"injected": True})
                ledger.system_head_path.unlink()
                reason = "injected database-commit-before-anchor-write crash"
                binding, evidence_hash = ledger.recovery_binding(reason)
                token = recovery_authority.issue(
                    case_id="AUDIT-RECOVERY",
                    action_type="reconcile_audit_anchors",
                    payload=binding,
                    workflow_version="audit-protocol-v1",
                    reviewer_id="audit-recovery-reviewer",
                    reviewer_role="Risk Manager",
                    reviewer_scope="enterprise",
                    decision=ReviewDecision.APPROVE,
                    policy_version="audit-recovery-v1",
                    evidence_hash=evidence_hash,
                )
                ledger.reconcile_external_anchors(
                    authorization_token=token,
                    actor_id="audit-recovery-reviewer",
                    reason=reason,
                )
                recovery = ledger.verify_system_event_chain()
                detected, injected, status = True, True, "ok"
                retries = 1
            else:
                args: dict[str, Any] = dict(
                    case_id=f"fault-{index}",
                    action_type="case_update",
                    payload={"state": "investigated"},
                    workflow_version="v3",
                )
                evidence_hash = "fault-evidence"
                if failure == "crash before approval":
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "controlflow.eval.recovery_worker",
                            "review",
                            str(ledger.path),
                            args["case_id"],
                        ],
                        check=False,
                        timeout=20,
                    )
                    pending = ledger.request_review(**args)
                    if pending.status == "EXECUTED":
                        recovery = True
                    else:
                        token = authority.issue(
                            **args,
                            reviewer_id="reviewer",
                            reviewer_role="Risk Manager",
                            reviewer_scope="enterprise",
                            decision=ReviewDecision.APPROVE,
                            policy_version="local-policy-v1",
                            evidence_hash=evidence_hash,
                        )
                        ledger.record_review(
                            pending.action_id,
                            HumanDecision(
                                reviewer_id="reviewer",
                                reviewer_role="Risk Manager",
                                reviewer_scope="enterprise",
                                decision=ReviewDecision.APPROVE,
                                decided_at=datetime.now(UTC),
                                bound_action_hash=pending.idempotency_key,
                            ),
                            authorization_token=token,
                            approval_authority=authority,
                            policy_version="local-policy-v1",
                            evidence_hash=evidence_hash,
                        )
                        receipt = ledger.execute_simulated(
                            **args,
                            authorization_token=token,
                            approval_authority=authority,
                            evidence_hash=evidence_hash,
                        )
                        recovery = pending.status == "PENDING_REVIEW" and receipt.executed
                    detected, injected, status = process.returncode == 91, True, "ok"
                elif failure == "crash after approval":
                    secret = secrets.token_bytes(32)
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "controlflow.eval.recovery_worker",
                            "approval",
                            str(ledger.path),
                            args["case_id"],
                            "--secret",
                            secret.hex(),
                        ],
                        check=False,
                        timeout=20,
                    )
                    action_authority = ApprovalAuthority(
                        secret, reviewer_entitlements={"reviewer": ("Risk Manager", "enterprise")}
                    )
                    token = action_authority.issue(
                        **args,
                        reviewer_id="reviewer",
                        reviewer_role="Risk Manager",
                        reviewer_scope="enterprise",
                        decision=ReviewDecision.APPROVE,
                        policy_version="local-policy-v1",
                        evidence_hash=evidence_hash,
                    )
                    receipt = ledger.execute_simulated(
                        **args,
                        authorization_token=token,
                        approval_authority=action_authority,
                        evidence_hash=evidence_hash,
                    )
                    recovery = receipt.executed or receipt.status == "EXECUTED"
                    detected, injected, status = process.returncode == 91, True, "ok"
                else:
                    secret = secrets.token_bytes(32)
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "controlflow.eval.recovery_worker",
                            "action",
                            str(ledger.path),
                            args["case_id"],
                            "--secret",
                            secret.hex(),
                        ],
                        check=False,
                        timeout=20,
                    )
                    action_authority = ApprovalAuthority(secret)
                    token = action_authority.issue_system(
                        **args,
                        policy_version="local-policy-v1",
                        evidence_hash=evidence_hash,
                        risk_tier=1,
                        authorization_outcome="ALLOW",
                    )
                    first = ledger.execute_simulated(
                        **args,
                        authorization_token=token,
                        approval_authority=action_authority,
                        evidence_hash=evidence_hash,
                    )
                    second = ledger.execute_simulated(
                        **args,
                        authorization_token=token,
                        approval_authority=action_authority,
                        evidence_hash=evidence_hash,
                    )
                    duplicate = second.executed
                    recovery = (first.executed or first.status == "EXECUTED") and not duplicate
                    detected, injected, status = process.returncode == 91, True, "ok"
                retries = 1
            rows.append(
                {
                    "experiment_id": f"reliability-{index:02d}",
                    "config_hash": hashlib.sha256(canonical_json({"failure": failure, "suite": 2})).hexdigest(),
                    "dataset_hash": "fault-suite-v2",
                    "split_identifier": "fault_validation",
                    "seed": 17,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": status,
                    "failure": failure,
                    "injected": injected,
                    "detected": detected,
                    "recovery_success": bool(recovery),
                    "duplicate_execution": duplicate,
                    "retry_count": retries,
                    "timeout_count": int("timeout" in failure.casefold()),
                    "latency_seconds": time.perf_counter() - started,
                }
            )
        target = paths.root / "results/operations.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
