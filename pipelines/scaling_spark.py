from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]


def session() -> SparkSession:
    builder = (
        SparkSession.builder.master("local[4]")
        .appName("ControlFlow-G-P21")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def run() -> None:
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")
    rows = []
    for scale in (100_000, 1_000_000, 5_000_000):
        frame = (
            spark.range(scale)
            .withColumn("group_id", F.col("id") % 1000)
            .withColumn("amount", F.log1p(F.col("id") % 10000))
        )
        for implementation in ("naive_pyspark", "optimized_spark_sql"):
            started = time.perf_counter()
            if implementation == "naive_pyspark":
                checksum = frame.groupBy("group_id").agg(F.sum("amount")).count()
            else:
                frame.repartition(8, "group_id").createOrReplaceTempView("events")
                checksum = spark.sql("SELECT group_id, SUM(amount) total FROM events GROUP BY group_id").count()
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "implementation": implementation,
                    "scale": scale,
                    "wall_clock_seconds": elapsed,
                    "rows_per_second": scale / elapsed,
                    "result_rows": checksum,
                    "runtime": f"Spark {spark.version} / {platform.platform()}",
                }
            )
    delta_target = str(ROOT / "data" / "lakehouse" / "scaling_incremental")
    base = spark.range(100_000).withColumn("batch", F.lit(0))
    started = time.perf_counter()
    base.write.format("delta").mode("overwrite").save(delta_target)
    initial = time.perf_counter() - started
    increment = spark.range(100_000, 110_000).withColumn("batch", F.lit(1))
    started = time.perf_counter()
    increment.write.format("delta").mode("append").save(delta_target)
    refresh = time.perf_counter() - started
    rows.extend(
        [
            {
                "implementation": "incremental_delta_initial",
                "scale": 100_000,
                "wall_clock_seconds": initial,
                "rows_per_second": 100_000 / initial,
                "result_rows": 100_000,
                "runtime": f"Spark {spark.version}",
            },
            {
                "implementation": "incremental_delta_refresh",
                "scale": 10_000,
                "wall_clock_seconds": refresh,
                "rows_per_second": 10_000 / refresh,
                "result_rows": 10_000,
                "runtime": f"Spark {spark.version}",
            },
        ]
    )
    target = ROOT / "results" / "spark_scaling.json"
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    spark.stop()


if __name__ == "__main__":
    run()
