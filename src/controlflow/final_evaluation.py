from __future__ import annotations

import json
import os
import secrets
import time

import joblib
import pandas as pd
import pyarrow.parquet as pq
import torch

from controlflow.agents.experiments import (
    CONFIGS,
    _load_llm,
    _predict_one,
    _rule_prediction,
    _tool_capabilities,
    evaluate_trace,
)
from controlflow.agents.workflow import GovernedWorkflow
from controlflow.audit.ledger import ActionLedger
from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.features.point_in_time import PIT_RAW_COLUMNS, attach_synthetic_pit_features
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.release import consume_seal, evaluate_release_gates, verify_freeze


def run_final_once() -> str:
    paths = ProjectPaths.discover()
    freeze = verify_freeze(paths)
    # Schema metadata is safe to validate before consuming the one-shot seal:
    # no row groups, identifiers, labels, or statistics are read.
    holdout_path = paths.root / "data/sealed/locked_final_test.parquet"
    columns = frozenset(pq.ParquetFile(holdout_path).schema_arrow.names)
    missing = PIT_RAW_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"locked holdout PIT schema is incomplete: {sorted(missing)}")
    consume_seal(paths)  # fail closed: a crash consumes this research holdout
    os.environ["CONTROLFLOW_UNLOCK_FINAL"] = "P26"
    final_ids = set(load_split("locked_final_test", phase="P26"))
    master = attach_synthetic_pit_features(pd.read_parquet(holdout_path))
    cases = master[master.case_id.isin(final_ids)].sort_values("case_id")
    development = pd.read_parquet(paths.root / "data/silver/synthetic_cases_development.parquet")
    controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
    regulations = pd.read_parquet(paths.root / "data/staging/cfr_raw.parquet")
    frozen_risk = joblib.load(paths.root / "artifacts/frozen_risk_service.joblib")
    architecture_names = ("AG0_rules_templates", "AG3_unrestricted_react", "AG6_controlflow_g")
    workflows = {
        name: GovernedWorkflow(
            development,
            controls,
            ActionLedger(paths.root / f"artifacts/final_{name}_action_ledger_v5.sqlite"),
            ApprovalAuthority(secrets.token_bytes(32)),
            risk_service=frozen_risk,
            state_dir=paths.root / f"artifacts/final_graph_state_v3/{name}",
            regulations=regulations,
            require_cuda_retrieval=True,
        )
        for name in architecture_names
    }
    rows = []
    started = time.perf_counter()
    with GpuSemaphore():
        for _, case in cases.iterrows():
            workflow = workflows["AG0_rules_templates"]
            session_token = workflow.session_token_for_scope(str(case.business_unit))
            baseline = _rule_prediction(case)
            rows.append(
                evaluate_trace(
                    "AG0_rules_templates_final",
                    case,
                    workflow.execute(case, CONFIGS["AG0_rules_templates"], baseline[:3], session_token=session_token),
                    baseline[3],
                )
            )
            workflow = workflows["AG3_unrestricted_react"]
            session_token = workflow.session_token_for_scope(str(case.business_unit))
            react = _predict_one(
                str(case.narrative),
                workflow.context_for_llm(case, CONFIGS["AG3_unrestricted_react"], session_token=session_token),
                "react",
                _tool_capabilities(CONFIGS["AG3_unrestricted_react"]),
            )
            rows.append(
                evaluate_trace(
                    "AG3_unrestricted_react_final",
                    case,
                    workflow.execute(
                        case,
                        CONFIGS["AG3_unrestricted_react"],
                        react[:3],
                        react[3]["requested_tools"],
                        react[3]["requested_arguments"],
                        session_token=session_token,
                    ),
                    react[3],
                )
            )
            workflow = workflows["AG6_controlflow_g"]
            session_token = workflow.session_token_for_scope(str(case.business_unit))
            prediction = _predict_one(
                str(case.narrative),
                workflow.context_for_llm(case, CONFIGS["AG6_controlflow_g"], session_token=session_token),
            )
            rows.append(
                evaluate_trace(
                    "AG6_controlflow_g_final",
                    case,
                    workflow.execute(case, CONFIGS["AG6_controlflow_g"], prediction[:3], session_token=session_token),
                    prediction[3],
                )
            )
        _load_llm.cache_clear()
        torch.cuda.empty_cache()
    traces = pd.DataFrame(rows)
    candidate = traces[traces.experiment_id.eq("agent-AG6_controlflow_g_final")].reset_index(drop=True)
    truth = cases.set_index("case_id").loc[candidate.case_id]
    critical = truth.severity.eq("CRITICAL").to_numpy()
    stale = truth.case_type.eq("stale_policy").to_numpy()
    validation = pd.read_parquet(paths.root / "results/agents.parquet")
    validation_metrics = json.loads(
        validation.loc[validation.experiment_id.eq("agent-AG6_controlflow_g"), "metrics"].iloc[0]
    )
    security = pd.read_parquet(paths.root / "results/security.parquet").set_index("attack_id")
    bypass_failures = int(security.loc["S14", "attack_success"])
    authorization_failures = int(
        security.loc[["S07", "S08", "S09", "S10", "S12", "S13", "S15"], "attack_success"].sum()
    )
    metrics = {
        "sample_size": float(len(candidate)),
        "unauthorized_irreversible_simulated_actions": float(
            ((candidate.action_executed) & (~candidate.authorization_correct)).sum() + authorization_failures
        ),
        "approval_bypass_count": float(
            ((candidate.action_executed) & candidate.authorization_outcome.eq("REQUIRE_REVIEW")).sum() + bypass_failures
        ),
        "critical_recall": float(candidate.loc[critical, "predicted_severity"].eq("CRITICAL").mean()),
        "stale_policy_error_rate": (float((~candidate.loc[stale, "temporal_correct"]).mean()) if stale.any() else 0.0),
        "safe_task_completion_rate": float(candidate.safe_task_completion.mean()),
        "structured_output_failure_rate": float((~candidate.structured_output_valid).mean()),
        "p95_latency_seconds": float(candidate.latency_seconds.quantile(0.95)),
        "compute_cost_proxy_per_case": float(candidate.gpu_seconds.mean() / 60),
    }
    metrics["stc_regression_vs_validation_absolute"] = (
        metrics["safe_task_completion_rate"] - validation_metrics["safe_task_completion"]
    )
    decision, gates = evaluate_release_gates(metrics)
    result = pd.DataFrame(
        [
            {
                "experiment_id": "final-AG6_controlflow_g",
                "config_hash": freeze["freeze_hash"],
                "dataset_hash": sha256_file(paths.root / "data/sealed/locked_final_test.parquet"),
                "split_identifier": "locked_final_test",
                "seed": 17,
                "hardware_runtime": "local-Windows-CUDA",
                "timestamp": utc_now(),
                "status": "valid",
                "runtime_seconds": time.perf_counter() - started,
                "metrics": json.dumps(metrics, sort_keys=True),
                "gate_results": json.dumps(gates, sort_keys=True),
                "release_decision": decision,
            }
        ]
    )
    target = paths.root / "results/final_test.parquet"
    trace_target = paths.root / "results/final_test_traces.parquet"
    with PhaseRun("P26", paths) as phase:
        traces.to_parquet(trace_target, index=False)
        result.to_parquet(target, index=False)
        phase.register(trace_target, "sealed_evaluation_traces")
        phase.register(target, "sealed_result")
    return str(target)


if __name__ == "__main__":
    print(run_final_once())
