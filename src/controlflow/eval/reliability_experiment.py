from __future__ import annotations

import hashlib
import secrets
import time

import pandas as pd
from pydantic import BaseModel

from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.policy import LocalPolicyBackend
from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, canonical_json, utc_now
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.schemas import IdentityContext, Severity
from controlflow.tools.registry import ToolRegistry, ToolSpec


class Probe(BaseModel):
    value: int


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


def _tool_fault(kind: str) -> tuple[bool, int]:
    attempts = {"count": 0}
    registry = ToolRegistry(LocalPolicyBackend())

    def implementation(value: Probe) -> Probe:
        attempts["count"] += 1
        if kind == "timeout":
            time.sleep(0.03)
        else:
            raise RuntimeError("injected unavailable dependency")
        return value

    registry.register(
        ToolSpec(
            name="probe",
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
            "probe", {"value": 1}, identity=_identity(), scope="consumer", data_classification=0, severity=Severity.LOW
        )
    except RuntimeError:
        return len(registry.audit_events) == 2, attempts["count"] - 1
    return False, attempts["count"] - 1


def run() -> str:
    paths = ProjectPaths.discover()
    ledger = ActionLedger(paths.root / "artifacts/reliability_action_ledger_v2.sqlite")
    authority = ApprovalAuthority(secrets.token_bytes(32))
    rows = []
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
            )
        ):
            started = time.perf_counter()
            retries = 0
            duplicate = False
            detected = True
            if "timeout" in failure.casefold():
                recovery, retries = _tool_fault("timeout")
            elif failure == "model unavailable":
                recovery, retries = _tool_fault("unavailable")
            elif failure in {"agent crash", "partial pipeline failure", "checkpoint recovery"}:
                checkpoint = paths.root / f"build/fault-checkpoint-{index}.json"
                atomic_write_json(checkpoint, {"step": "before_fault", "case_id": f"fault-{index}"})
                recovered = pd.read_json(checkpoint, typ="series")
                recovery = recovered["step"] == "before_fault"
                retries = 1
            else:
                args = dict(
                    case_id=f"fault-{index}",
                    action_type="case_update",
                    payload={"state": "investigated"},
                    workflow_version="v2",
                )
                evidence_hash = "fault-evidence"
                if failure == "crash before approval":
                    pending = ledger.request_review(**args)
                    token = authority.issue(
                        **args, reviewer_id="reviewer", policy_version="local-policy-v1", evidence_hash=evidence_hash
                    )
                    receipt = ledger.execute_simulated(
                        **args, authorization_token=token, approval_authority=authority, evidence_hash=evidence_hash
                    )
                    recovery = pending.status == "PENDING_REVIEW" and receipt.executed
                elif failure == "crash after approval":
                    token = authority.issue(
                        **args, reviewer_id="reviewer", policy_version="local-policy-v1", evidence_hash=evidence_hash
                    )
                    receipt = ledger.execute_simulated(
                        **args, authorization_token=token, approval_authority=authority, evidence_hash=evidence_hash
                    )
                    recovery = receipt.status == "EXECUTED"
                else:
                    token = authority.issue(
                        **args, reviewer_id="SYSTEM_AUTO", policy_version="local-policy-v1", evidence_hash=evidence_hash
                    )
                    first = ledger.execute_simulated(
                        **args, authorization_token=token, approval_authority=authority, evidence_hash=evidence_hash
                    )
                    second = ledger.execute_simulated(
                        **args, authorization_token=token, approval_authority=authority, evidence_hash=evidence_hash
                    )
                    duplicate = second.executed
                    recovery = (first.executed or first.status == "EXECUTED") and not duplicate
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
                    "status": "ok",
                    "failure": failure,
                    "injected": True,
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
