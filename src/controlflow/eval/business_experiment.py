from __future__ import annotations

import hashlib
import json

import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now

SCENARIOS = {
    "low_review_cost": {"review_minutes": 3.0, "review_cost_units_per_minute": 1.0, "critical_error_units": 100.0},
    "base_review_cost": {
        "review_minutes": 8.0,
        "review_cost_units_per_minute": 2.0,
        "critical_error_units": 1_000.0,
    },
    "high_safety_cost": {
        "review_minutes": 15.0,
        "review_cost_units_per_minute": 3.0,
        "critical_error_units": 10_000.0,
    },
}


def run() -> str:
    paths = ProjectPaths.discover()
    trace_path = paths.root / "results/agent_traces.parquet"
    traces = pd.read_parquet(trace_path)
    cases = pd.read_parquet(paths.root / "data/silver/synthetic_cases_development.parquet").set_index("case_id")
    rows: list[dict[str, object]] = []
    for experiment_id, group in traces.groupby("experiment_id"):
        truth = cases.loc[group.case_id]
        critical = truth.severity.eq("CRITICAL").to_numpy()
        critical_errors = int((critical & ~group.safe_task_completion.to_numpy()).sum())
        review_rate = float(group.human_review_requested.mean())
        automation_rate = float(group.predicted_disposition.eq("AUTO").mean())
        mean_latency = float(group.latency_seconds.mean())
        for scenario, assumptions in SCENARIOS.items():
            review_minutes_per_case = review_rate * assumptions["review_minutes"]
            mean_resolution_minutes = mean_latency / 60 + review_minutes_per_case
            compute_units = float(group.gpu_seconds.mean() / 60)
            review_units = review_minutes_per_case * assumptions["review_cost_units_per_minute"]
            error_units = critical_errors / max(1, len(group)) * assumptions["critical_error_units"]
            resolved = max(1, int(group.safe_task_completion.sum()))
            metrics = {
                "sample_size": len(group),
                "cases_per_hour_sequential": 3600 / max(mean_latency, 1e-9),
                "automation_percentage": 100 * automation_rate,
                "review_percentage": 100 * review_rate,
                "review_minutes_per_case": review_minutes_per_case,
                "critical_errors_per_1000_cases": 1000 * critical_errors / max(1, len(group)),
                "mean_time_to_resolution_minutes": mean_resolution_minutes,
                "cost_proxy_per_resolved_case": (len(group) * (compute_units + review_units + error_units) / resolved),
                "operational_utility_units_per_case": -(compute_units + review_units + error_units),
            }
            rows.append(
                {
                    "experiment_id": f"business-{experiment_id}-{scenario}",
                    "config_hash": hashlib.sha256(canonical_json({"scenario": scenario, **assumptions})).hexdigest(),
                    "dataset_hash": sha256_file(trace_path),
                    "split_identifier": "validation_agent_scenario_sample",
                    "seed": 17,
                    "hardware_runtime": "derived_from_observed_validation_traces",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "architecture": experiment_id,
                    "scenario": scenario,
                    "assumptions": json.dumps(assumptions, sort_keys=True),
                    "metrics": json.dumps(metrics, sort_keys=True),
                }
            )
    target = paths.root / "results/business_metrics.parquet"
    with PhaseRun("P21", paths) as phase:
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "business_sensitivity_result")
    return str(target)


if __name__ == "__main__":
    print(run())
