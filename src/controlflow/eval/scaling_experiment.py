from __future__ import annotations

import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now


def run() -> str:
    paths = ProjectPaths.discover()
    spark_path = paths.root / "results" / "spark_scaling.json"
    if not spark_path.exists():
        raise FileNotFoundError("run pipelines/scaling_spark.py in the selected WSL Spark runtime first")
    records = json.loads(spark_path.read_text(encoding="utf-8"))
    with PhaseRun("P21", paths) as phase:
        for scale in (100_000, 1_000_000, 5_000_000):
            started = time.perf_counter()
            values = np.arange(scale, dtype=np.int64)
            frame = pd.DataFrame({"group_id": values % 1000, "amount": np.log1p(values % 10000)})
            result_rows = len(frame.groupby("group_id", sort=False).amount.sum())
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "implementation": "pandas",
                    "scale": scale,
                    "wall_clock_seconds": elapsed,
                    "rows_per_second": scale / elapsed,
                    "result_rows": result_rows,
                    "runtime": f"pandas {pd.__version__} / {platform.platform()}",
                }
            )
        timestamp = utc_now()
        for record in records:
            record.update(
                {
                    "experiment_id": f"scaling-{record['implementation']}-{record['scale']}",
                    "config_hash": hashlib.sha256(
                        canonical_json({"implementation": record["implementation"], "scale": record["scale"]})
                    ).hexdigest(),
                    "dataset_hash": "deterministic-range-v1",
                    "split_identifier": "scaling",
                    "seed": 0,
                    "hardware_runtime": record.pop("runtime"),
                    "timestamp": timestamp,
                    "status": "ok",
                }
            )
        target = paths.root / "results" / "scaling.parquet"
        pd.DataFrame(records).to_parquet(target, index=False)
        phase.register(target, "result_table")
        phase.register(spark_path, "raw_benchmark")
    return str(target)


if __name__ == "__main__":
    print(run())
