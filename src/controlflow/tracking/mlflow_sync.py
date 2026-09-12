from __future__ import annotations

import json

import mlflow
import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, utc_now

GROUPS = {
    "data_quality": "data",
    "ml_models": "risk_model",
    "ml_models_advanced": "risk_model",
    "calibration": "risk_model",
    "anomaly": "anomaly",
    "semi_supervised": "semi_supervised",
    "retrieval": "retrieval",
    "chunking": "retrieval",
    "retrieval_scaling": "retrieval",
    "agents": "agents",
    "hitl": "agents",
    "security": "security",
    "ablation": "ablation",
    "operations": "operations",
    "scaling": "operations",
    "data_failures": "data",
    "drift": "operations",
    "risk_coverage": "risk_model",
    "validation_statistics": "ablation",
}


def run() -> str:
    paths = ProjectPaths.discover()
    database = paths.root / "artifacts/mlflow.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    recorded: list[dict[str, str]] = []
    for stem, group in GROUPS.items():
        artifact = paths.root / "results" / f"{stem}.parquet"
        if not artifact.exists():
            continue
        mlflow.set_experiment(f"/controlflow/{group}")
        frame = pd.read_parquet(artifact)
        for _, row in frame.iterrows():
            experiment_id = str(row.get("experiment_id", f"{stem}-{len(recorded)}"))
            with mlflow.start_run(run_name=experiment_id) as active:
                for key in ("config_hash", "dataset_hash", "split_identifier", "seed", "hardware_runtime", "status"):
                    if key in row and pd.notna(row[key]):
                        mlflow.set_tag(key, str(row[key]))
                mlflow.set_tags(
                    {
                        "llm_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                        "agent_graph_version": "agent-graph-v4",
                        "tool_schema_version": "1",
                        "authorization_policy_version": "local-policy-v1",
                    }
                )
                metrics = {}
                if "metrics" in row and isinstance(row.metrics, str):
                    try:
                        metrics = json.loads(row.metrics)
                    except json.JSONDecodeError:
                        metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and pd.notna(value):
                        mlflow.log_metric(key, float(value))
                for key, value in row.items():
                    if key not in metrics and isinstance(value, (int, float)) and pd.notna(value):
                        mlflow.log_metric(str(key), float(value))
                mlflow.log_artifact(str(artifact), artifact_path="machine_readable_results")
                recorded.append(
                    {
                        "experiment_id": experiment_id,
                        "mlflow_run_id": active.info.run_id,
                        "group": f"/controlflow/{group}",
                    }
                )
    target = paths.state / "mlflow_manifest.json"
    atomic_write_json(target, {"created_at": utc_now(), "tracking_uri": tracking_uri, "runs": recorded})
    return str(target)


if __name__ == "__main__":
    print(run())
