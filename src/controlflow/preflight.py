from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from controlflow.core.resources import GpuSemaphore, resource_snapshot
from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, utc_now


def _gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    index, name, used, total, utilization, temperature = [part.strip() for part in output.split(",")]
    return {
        "timestamp": utc_now(),
        "index": int(index),
        "name": name,
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
    }


class GpuMonitor:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
                self.samples.append(_gpu_sample())
            self.stop.wait(self.interval)

    def __enter__(self) -> GpuMonitor:
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.stop.set()
        self.thread.join(timeout=5)
        return False


def torch_smoke() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("project-local PyTorch CUDA is unavailable")
    with GpuSemaphore(), GpuMonitor() as monitor:
        left = torch.randn((6144, 6144), device="cuda", dtype=torch.float16)
        right = torch.randn((6144, 6144), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = left @ right
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        finite = bool(torch.isfinite(result).all().item())
    if not result.is_cuda or not finite:
        raise RuntimeError("CUDA tensor proof failed")
    return {
        "status": "passed",
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "operation": "6144x6144 fp16 matrix multiply",
        "result_cuda": result.is_cuda,
        "finite": finite,
        "elapsed_seconds": elapsed,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": monitor.samples,
    }


def xgboost_smoke() -> dict[str, Any]:
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(1729)
    features = rng.normal(size=(300_000, 32)).astype("float32")
    labels = ((features[:, 0] + features[:, 1] * 0.5 + features[:, 2] ** 2) > 1).astype("int32")
    matrix = xgb.QuantileDMatrix(features, labels)
    parameters = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "binary:logistic",
        "max_depth": 8,
        "eta": 0.08,
        "seed": 1729,
    }
    with GpuSemaphore(), GpuMonitor() as monitor, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        booster = xgb.train(parameters, matrix, num_boost_round=120)
        elapsed = time.perf_counter() - started
    warning_text = [str(item.message) for item in caught]
    if any("fall" in message.casefold() and "cpu" in message.casefold() for message in warning_text):
        raise RuntimeError(f"XGBoost CUDA fallback detected: {warning_text}")
    config = json.loads(booster.save_config())
    device = config["learner"]["generic_param"]["device"]
    peak_used = max((sample["memory_used_mib"] for sample in monitor.samples), default=0)
    peak_utilization = max((sample["utilization_percent"] for sample in monitor.samples), default=0)
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"XGBoost booster device is {device}")
    return {
        "status": "passed",
        "xgboost_version": xgb.__version__,
        "rows": len(features),
        "columns": features.shape[1],
        "boost_rounds": 120,
        "booster_device": device,
        "elapsed_seconds": elapsed,
        "peak_memory_used_mib": peak_used,
        "peak_utilization_percent": peak_utilization,
        "warnings": warning_text,
        "telemetry": monitor.samples,
    }


def spark_smoke(root: Path) -> dict[str, Any]:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as functions

    output = root / "artifacts" / "preflight" / "spark_sql_parquet"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("controlflow-g-preflight")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    try:
        frame = spark.range(0, 100_000).withColumn("bucket", functions.col("id") % 10)
        frame.createOrReplaceTempView("preflight_events")
        result = spark.sql(
            "SELECT bucket, COUNT(*) AS rows, SUM(id) AS id_sum FROM preflight_events GROUP BY bucket ORDER BY bucket"
        )
        rows = [row.asDict() for row in result.collect()]
        result.write.mode("overwrite").parquet(str(output))
        read_count = spark.read.parquet(str(output)).count()
        spark_version = spark.version
    finally:
        spark.stop()
    if read_count != 10 or sum(row["rows"] for row in rows) != 100_000:
        raise RuntimeError("Spark SQL or Parquet round-trip proof failed")
    return {
        "status": "passed",
        "spark_version": spark_version,
        "java_home": os.environ.get("JAVA_HOME", "unknown"),
        "input_rows": 100_000,
        "grouped_rows": read_count,
        "sql_sum_rows": sum(row["rows"] for row in rows),
        "parquet_path": output.relative_to(root).as_posix(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_core() -> Path:
    paths = ProjectPaths.discover()
    target = paths.state / "preflight_gpu_smoke.json"
    phase = PhaseRun("P00", paths)
    phase.journal("checkpoint_started", "Core CUDA and Spark execution proofs started")
    try:
        result = {
            "schema_version": 1,
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "before": resource_snapshot(),
            "torch_cuda": torch_smoke(),
            "xgboost_cuda": xgboost_smoke(),
            "after": resource_snapshot(),
        }
        atomic_write_json(target, result)
        phase.register(target, "environment_proof")
        phase.journal(
            "checkpoint_completed",
            "Project-local PyTorch and XGBoost CUDA execution proofs passed",
            [target.relative_to(paths.root).as_posix()],
        )
    except Exception as exc:
        phase.journal("checkpoint_failed", f"Core execution proof failed: {type(exc).__name__}: {exc}")
        raise
    return target


def run_spark_only() -> Path:
    paths = ProjectPaths.discover()
    target = paths.state / "preflight_spark_smoke.json"
    phase = PhaseRun("P00", paths)
    phase.journal("checkpoint_started", "Spark SQL execution proof started")
    try:
        result = {
            "schema_version": 1,
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "spark_sql": spark_smoke(paths.root),
        }
        atomic_write_json(target, result)
        phase.register(target, "environment_proof")
        phase.journal(
            "checkpoint_completed",
            "Spark SQL execution proof passed",
            [target.relative_to(paths.root).as_posix()],
        )
    except Exception as exc:
        phase.journal("checkpoint_failed", f"Spark proof failed: {type(exc).__name__}: {exc}")
        raise
    return target


if __name__ == "__main__":
    print(run_core())
