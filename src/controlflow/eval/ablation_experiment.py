from __future__ import annotations

import hashlib
import secrets
from dataclasses import replace

import joblib
import pandas as pd
import torch

from controlflow.agents.experiments import CONFIGS, _load_llm, _predict_one, evaluate_trace
from controlflow.agents.workflow import GovernedWorkflow
from controlflow.audit.ledger import ActionLedger
from controlflow.audit.recovery import configured_recovery_authority
from controlflow.authorization.identity import SessionIdentityProvider
from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.hitl.approval import ApprovalAuthority

REMOVALS = {
    "AB01_no_temporal_retrieval": {"temporal_retrieval": False},
    "AB02_no_reranker": {"reranker": False},
    "AB03_no_ml_risk": {"ml_risk": False},
    "AB04_no_calibration": {"calibration": False},
    "AB05_no_anomaly": {"anomaly": False},
    "AB06_no_verifier": {"verifier": False},
    "AB07_no_authorization": {"authorization": False},
    "AB08_no_hitl": {"hitl": False},
    "AB09_no_structured_output_validation": {"structured_output": False},
    "AB10_no_point_in_time_features": {"point_in_time_features": False},
    "AB11_unrestricted_tools": {"bounded_tools": False, "authorization": False},
    "AB12_full_controlflow_g": {},
    "INT_no_verifier_no_authorization": {"verifier": False, "authorization": False},
    "INT_no_calibration_no_hitl": {"calibration": False, "hitl": False},
    "INT_no_temporal_no_verifier": {"temporal_retrieval": False, "verifier": False},
}


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data/silver/synthetic_cases_development.parquet"
    frame = pd.read_parquet(source)
    train = frame[frame.case_id.isin(set(load_split("train", phase="P22")))]
    validation = frame[frame.case_id.isin(set(load_split("validation", phase="P22")))]
    cases = validation.groupby("case_type", group_keys=False).head(10).sort_values("case_id")
    controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
    regulations = pd.read_parquet(paths.root / "data/staging/cfr_raw.parquet")
    risk_service = joblib.load(paths.root / "artifacts/calibrated_risk_service.joblib")
    identity_provider, session_credentials = SessionIdentityProvider.issue_for_business_units(
        set(frame["business_unit"].astype(str))
    )
    trace_target = paths.root / "results/ablation_traces.repeat7.inprogress.parquet"
    summary_target = paths.root / "results/ablation.repeat7.inprogress.parquet"
    traces: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    completed: set[str] = set()
    if trace_target.exists() and summary_target.exists():
        traces = pd.read_parquet(trace_target).to_dict(orient="records")
        summaries = pd.read_parquet(summary_target).to_dict(orient="records")
        completed = {str(row["ablation"]) for row in summaries}
    prediction_cache: dict[tuple[str, str], tuple[str, str, bool, dict[str, object]]] = {}
    shared_retriever_cache: dict[object, object] = {}
    with GpuSemaphore(), PhaseRun("P22", paths) as phase:
        for name, changes in REMOVALS.items():
            if name in completed:
                continue
            config = replace(CONFIGS["AG6_controlflow_g"], **changes)
            workflow = GovernedWorkflow(
                train,
                controls,
                ActionLedger(
                    paths.root / f"artifacts/ablation_{name}_action_ledger_v9.sqlite",
                    recovery_authority=configured_recovery_authority(),
                ),
                ApprovalAuthority(secrets.token_bytes(32)),
                risk_service=risk_service,
                state_dir=paths.root / f"artifacts/ablation_graph_state_v4/{name}",
                regulations=regulations,
                require_cuda_retrieval=True,
                identity_provider=identity_provider,
            )
            workflow._retriever_cache = shared_retriever_cache  # type: ignore[assignment]
            current = []
            for _, row in cases.iterrows():
                session_token = session_credentials[str(row.business_unit)]
                context = workflow.context_for_llm(row, config, session_token=session_token)
                cache_key = (str(row.case_id), context)
                if cache_key not in prediction_cache:
                    prediction_cache[cache_key] = _predict_one(str(row.narrative), context)
                prediction = prediction_cache[cache_key]
                observed = evaluate_trace(
                    name,
                    row,
                    workflow.execute(row, config, prediction[:3], session_token=session_token),
                    prediction[3],
                )
                observed["experiment_id"] = f"ablation-{name}"
                current.append(observed)
                traces.append(observed)
            stc = pd.DataFrame(current).safe_task_completion
            summaries.append(
                {
                    "experiment_id": f"ablation-{name}",
                    "config_hash": hashlib.sha256(canonical_json(config.__dict__)).hexdigest(),
                    "dataset_hash": sha256_file(source),
                    "split_identifier": "validation_agent_scenario_sample",
                    "seed": 17,
                    "hardware_runtime": "local-Windows-CUDA",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "ablation": name,
                    "sample_size": len(stc),
                    "safe_task_completion": float(stc.mean()),
                }
            )
            pd.DataFrame(traces).to_parquet(trace_target, index=False)
            pd.DataFrame(summaries).to_parquet(summary_target, index=False)
        target = paths.root / "results/ablation.parquet"
        final_trace_target = paths.root / "results/ablation_traces.parquet"
        pd.DataFrame(summaries).to_parquet(target, index=False)
        pd.DataFrame(traces).to_parquet(final_trace_target, index=False)
        phase.register(target, "result_table")
        phase.register(final_trace_target, "evaluation_traces")
        _load_llm.cache_clear()
        torch.cuda.empty_cache()
    return str(target)


if __name__ == "__main__":
    print(run())
