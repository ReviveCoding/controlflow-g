from __future__ import annotations

import hashlib
import secrets
from dataclasses import replace

import joblib
import pandas as pd

from controlflow.agents.experiments import CONFIGS, evaluate_trace
from controlflow.agents.workflow import GovernedWorkflow
from controlflow.audit.ledger import ActionLedger
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
    llm_traces = pd.read_parquet(paths.root / "results/agent_traces.parquet")
    llm = llm_traces[llm_traces.experiment_id == "agent-AG1_single_llm"].set_index("case_id")
    cases = frame[frame.case_id.isin(llm.index)].set_index("case_id").loc[llm.index]
    controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
    workflow = GovernedWorkflow(
        train,
        controls,
        ActionLedger(paths.root / "artifacts/ablation_action_ledger.sqlite"),
        ApprovalAuthority(secrets.token_bytes(32)),
        risk_service=joblib.load(paths.root / "artifacts/calibrated_risk_service.joblib"),
    )
    traces, summaries = [], []
    with PhaseRun("P22", paths) as phase:
        for name, changes in REMOVALS.items():
            config = replace(CONFIGS["AG6_controlflow_g"], **changes)
            current = []
            for case_id, row in cases.iterrows():
                row = row.copy()
                row["case_id"] = case_id
                prediction = (
                    str(llm.loc[case_id].predicted_severity),
                    str(llm.loc[case_id].predicted_disposition),
                    bool(llm.loc[case_id].structured_output_valid),
                )
                observed = evaluate_trace(
                    name,
                    row,
                    workflow.execute(row, config, prediction),
                    {"llm_latency_seconds": 0.0, "input_tokens": 0.0, "output_tokens": 0.0},
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
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "ablation": name,
                    "sample_size": len(stc),
                    "safe_task_completion": float(stc.mean()),
                }
            )
        target = paths.root / "results/ablation.parquet"
        trace_target = paths.root / "results/ablation_traces.parquet"
        pd.DataFrame(summaries).to_parquet(target, index=False)
        pd.DataFrame(traces).to_parquet(trace_target, index=False)
        phase.register(target, "result_table")
        phase.register(trace_target, "evaluation_traces")
    return str(target)


if __name__ == "__main__":
    print(run())
