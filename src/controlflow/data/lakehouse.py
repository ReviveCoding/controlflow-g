from __future__ import annotations

import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, utc_now


def delta_session(app_name: str, warehouse: Path) -> Any:
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    builder = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.databricks.delta.snapshotPartitions", "4")
        .config("spark.driver.memory", "2g")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def run_delta_smoke() -> Path:
    from delta.tables import DeltaTable
    from pyspark.sql import functions as functions

    paths = ProjectPaths.discover()
    output = paths.root / "artifacts" / "preflight" / "delta_table"
    warehouse = paths.root / "artifacts" / "preflight" / "spark_warehouse"
    target = paths.state / "preflight_delta_smoke.json"
    phase = PhaseRun("P00", paths)
    started = time.perf_counter()
    spark = delta_session("controlflow-g-delta-preflight", warehouse)
    try:
        source = spark.range(0, 10_000).withColumn("value", functions.col("id") * 2)
        source.write.format("delta").mode("overwrite").save(str(output))
        table = DeltaTable.forPath(spark, str(output))
        table.update(condition="id % 2 = 0", set={"value": "value + 1"})
        current = spark.read.format("delta").load(str(output))
        version_zero = spark.read.format("delta").option("versionAsOf", 0).load(str(output))
        current_sum = int(current.agg(functions.sum("value")).first()[0])
        version_zero_sum = int(version_zero.agg(functions.sum("value")).first()[0])
        history = [row.asDict(recursive=True) for row in table.history().select("version", "operation").collect()]
        result = {
            "schema_version": 1,
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "spark_version": spark.version,
            "delta_python_version": version("delta-spark"),
            "rows": current.count(),
            "version_zero_sum": version_zero_sum,
            "current_sum": current_sum,
            "delta_history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "path": output.relative_to(paths.root).as_posix(),
            "status": "passed",
        }
    finally:
        spark.stop()
    if result["rows"] != 10_000 or result["current_sum"] - result["version_zero_sum"] != 5_000:
        raise RuntimeError("Delta update or versioned read proof failed")
    atomic_write_json(target, result)
    phase.register(target, "environment_proof")
    phase.journal(
        "checkpoint_completed",
        "Delta write/update/history/version-read proof passed",
        [target.relative_to(paths.root).as_posix()],
    )
    return target
