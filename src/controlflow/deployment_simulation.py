from __future__ import annotations

import json

import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now


def run() -> str:
    paths = ProjectPaths.discover()
    final_path = paths.root / "results/final_test.parquet"
    final = pd.read_parquet(final_path).iloc[0]
    metrics = json.loads(final.metrics)
    gate_rows = json.loads(final.gate_results)
    rollback = any(
        not gate["passed"]
        and gate["gate"]
        in {
            "unauthorized_irreversible_simulated_actions",
            "approval_bypass_count",
            "safe_task_completion_rate",
            "p95_latency_seconds",
            "compute_cost_proxy_per_case",
        }
        for gate in gate_rows
    )
    stages = [
        ("DEV", 1.00, "completed"),
        ("STAGING", 1.00, "completed"),
        ("SHADOW", 0.00, "completed"),
        ("CANARY", 0.05, "rolled_back" if rollback else "completed"),
        ("PRODUCTION_LIKE_SIMULATION", 0.00 if rollback else 1.00, "blocked" if rollback else "completed"),
    ]
    rows = []
    for stage, traffic_fraction, status in stages:
        config = {"stage": stage, "traffic_fraction": traffic_fraction, "rollback_policy": "release-gates-v1"}
        rows.append(
            {
                "experiment_id": f"deployment-{stage.casefold()}",
                "config_hash": sha256_file(paths.root / "configs/release_gates.yaml"),
                "dataset_hash": str(final.dataset_hash),
                "split_identifier": "locked_final_test-derived-no-reexecution",
                "seed": 17,
                "hardware_runtime": "simulation",
                "timestamp": utc_now(),
                "status": status,
                "traffic_fraction": traffic_fraction,
                "rollback_triggered": rollback and stage == "CANARY",
                "release_decision": final.release_decision,
                "metrics": json.dumps(metrics, sort_keys=True),
                "simulation_config": canonical_json(config).decode(),
            }
        )
    target = paths.root / "results/deployment.parquet"
    with PhaseRun("P27", paths) as phase:
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "deployment_simulation")
    return str(target)


if __name__ == "__main__":
    print(run())
