from __future__ import annotations

import json

import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths


def _rescore_traces(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "llm_analysis_evidence_valid" not in result:
        result = result.rename(
            columns={
                "llm_analysis_supported": "llm_analysis_evidence_valid",
                "llm_analysis_support_checked": "llm_analysis_evidence_checked",
            }
        )
    governed = result["architecture_mode"].eq("governed")
    if "llm_root_cause_code" not in result:
        result["llm_root_cause_code"] = ""
    if "llm_root_cause_hypothesis" not in result:
        result["llm_root_cause_hypothesis"] = ""
    valid_without_code = governed & result["llm_analysis_valid"] & result["llm_root_cause_code"].fillna("").eq("")
    if bool(valid_without_code.any()):
        raise RuntimeError("valid governed claims require full text-preserving re-evaluation")
    if "root_cause_correct" not in result:
        result["root_cause_correct"] = False
    result.loc[~governed, "root_cause_correct"] = False
    result["safe_task_completion"] = (
        result["nominal_success"]
        & result["evidence_correct"]
        & result["temporal_correct"]
        & result["authorization_correct"]
        & result["structured_output_valid"]
        & (~result["llm_analysis_evidence_checked"] | result["llm_analysis_evidence_valid"])
    )
    return result


def run() -> tuple[str, str]:
    paths = ProjectPaths.discover()
    agent_trace_path = paths.root / "results/agent_traces.parquet"
    ablation_trace_path = paths.root / "results/ablation_traces.parquet"
    agents_path = paths.root / "results/agents.parquet"
    ablation_path = paths.root / "results/ablation.parquet"

    agent_traces = _rescore_traces(pd.read_parquet(agent_trace_path))
    ablation_traces = _rescore_traces(pd.read_parquet(ablation_trace_path))
    agent_traces.to_parquet(agent_trace_path, index=False)
    ablation_traces.to_parquet(ablation_trace_path, index=False)

    agents = pd.read_parquet(agents_path)
    for index, row in agents.iterrows():
        metrics = json.loads(str(row["metrics"]))
        if "llm_analysis_supported_rate" in metrics:
            metrics["llm_analysis_evidence_valid_rate"] = metrics.pop("llm_analysis_supported_rate")
        group = agent_traces.loc[agent_traces.experiment_id == row["experiment_id"]]
        metrics["root_cause_correct_rate"] = float(group.root_cause_correct.mean())
        metrics["safe_task_completion"] = float(group.safe_task_completion.mean())
        agents.at[index, "metrics"] = json.dumps(metrics, sort_keys=True)
    agents.to_parquet(agents_path, index=False)

    ablation = pd.read_parquet(ablation_path)
    for index, row in ablation.iterrows():
        group = ablation_traces.loc[ablation_traces.experiment_id == row["experiment_id"]]
        ablation.at[index, "safe_task_completion"] = float(group.safe_task_completion.mean())
    ablation.to_parquet(ablation_path, index=False)

    with PhaseRun("P17", paths) as phase:
        phase.register(agent_trace_path, "evaluation_traces")
        phase.register(agents_path, "result_table")
    with PhaseRun("P22", paths) as phase:
        phase.register(ablation_trace_path, "evaluation_traces")
        phase.register(ablation_path, "result_table")
    return str(agents_path), str(ablation_path)


if __name__ == "__main__":
    print(run())
