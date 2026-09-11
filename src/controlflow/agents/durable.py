from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.agents.workflow import GovernedWorkflow, WorkflowConfig, WorkflowTrace
from controlflow.core.state import atomic_write_json


class InjectedWorkflowCrash(RuntimeError):
    pass


class DurableWorkflowRunner:
    """Checkpointed wrapper that resumes safely across process failure.

    If execution completed but the process died before checkpoint commit, the
    underlying action ledger makes the replay side effect idempotent.
    """

    def __init__(self, workflow: GovernedWorkflow, checkpoint: Path) -> None:
        self.workflow = workflow
        self.checkpoint = checkpoint

    def run(
        self,
        row: pd.Series,
        config: WorkflowConfig,
        prediction: tuple[str, str, bool],
        *,
        session_token: str,
        fault: str | None = None,
    ) -> WorkflowTrace:
        atomic_write_json(
            self.checkpoint,
            {"status": "RUNNING", "case_id": str(row.case_id), "completed_nodes": []},
        )
        if fault == "before_execute":
            raise InjectedWorkflowCrash("injected crash before workflow execution")
        trace = self.workflow.execute(row, config, prediction, session_token=session_token)
        if fault == "after_execute_before_checkpoint":
            raise InjectedWorkflowCrash("injected crash after execution before checkpoint commit")
        self._complete(trace)
        return trace

    def resume(
        self,
        row: pd.Series,
        config: WorkflowConfig,
        prediction: tuple[str, str, bool],
        *,
        session_token: str,
        recovery_approval: tuple[str, str, str] | None = None,
    ) -> WorkflowTrace:
        # Explicit recovery reconciles a committed audit event whose external
        # signed checkpoint could not be exported before process death.
        if not self.workflow.ledger.verify_event_chain() or not self.workflow.ledger.verify_system_event_chain():
            if recovery_approval is None:
                raise RuntimeError("audit recovery requires a separately supplied operator approval")
            token, actor_id, reason = recovery_approval
            self.workflow.ledger.reconcile_external_anchors(
                authorization_token=token,
                actor_id=actor_id,
                reason=reason,
            )
        state: dict[str, Any] = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        if state.get("case_id") != str(row.case_id):
            raise ValueError("checkpoint belongs to another case")
        if state.get("status") == "COMPLETE":
            return WorkflowTrace(**state["trace"])
        trace = self.workflow.execute(row, config, prediction, session_token=session_token)
        self._complete(trace)
        return trace

    def _complete(self, trace: WorkflowTrace) -> None:
        atomic_write_json(
            self.checkpoint,
            {
                "status": "COMPLETE",
                "case_id": trace.case_id,
                "completed_nodes": [
                    "features",
                    "retrieval",
                    "risk",
                    "verification",
                    "authorization",
                    "action",
                ],
                "trace": asdict(trace),
            },
        )
