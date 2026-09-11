from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from controlflow.core.state import (
    PhaseRun,
    ProjectPaths,
    atomic_write_json,
    canonical_json,
    utc_now,
)
from controlflow.data.lakehouse import delta_session
from controlflow.data.temporal import apply_bitemporal_event, point_in_time_lookup


def _write_event_batches(directory: Path) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.json")):
        return {"base": 120, "duplicates": 10, "late": 5, "out_of_order": 24}
    start = datetime(2025, 1, 1, tzinfo=UTC)
    events: list[dict[str, Any]] = []
    for index in range(120):
        timestamp = start + timedelta(minutes=index)
        if index % 24 == 0:
            timestamp -= timedelta(days=3)
        events.append(
            {
                "event_id": f"STREAM-{index:05d}",
                "entity_id": f"CASE-{index % 20:04d}",
                "event_type": [
                    "case_create",
                    "case_update",
                    "control_test",
                    "policy_update",
                    "ownership_change",
                    "transaction_event",
                    "human_review",
                ][index % 7],
                "event_timestamp": timestamp.isoformat(),
                "sequence": index // 20,
                "payload": json.dumps({"ordinal": index}, sort_keys=True),
            }
        )
    duplicated = [*events, *events[:10]]
    for batch, offset in enumerate(range(0, len(duplicated), 45)):
        rows = duplicated[offset : offset + 45]
        (directory / f"batch-{batch:03d}.json").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    return {"base": 120, "duplicates": 10, "late": 5, "out_of_order": 24}


def _temporal_proof() -> dict[str, Any]:
    def dt(value: str) -> datetime:
        return datetime.fromisoformat(value)

    versions = apply_bitemporal_event(
        [],
        entity_id="policy-1",
        value={"version": "v2"},
        business_valid_from=dt("2024-01-01"),
        system_known_from=dt("2024-01-02"),
    )
    versions = apply_bitemporal_event(
        versions,
        entity_id="policy-1",
        value={"version": "v1"},
        business_valid_from=dt("2023-01-01"),
        system_known_from=dt("2024-03-01"),
    )
    before = point_in_time_lookup(versions, "policy-1", dt("2023-06-01"), dt("2024-02-01"))
    historical = point_in_time_lookup(versions, "policy-1", dt("2023-06-01"), dt("2024-04-01"))
    current = point_in_time_lookup(versions, "policy-1", dt("2024-06-01"), dt("2024-04-01"))
    if (
        before is not None
        or historical is None
        or historical.value["version"] != "v1"
        or current is None
        or current.value["version"] != "v2"
    ):
        raise RuntimeError("bitemporal late-arrival reconstruction failed")
    return {
        "version_rows": len(versions),
        "unknown_before_late_arrival": before is None,
        "historical_version_after_arrival": historical.value["version"],
        "current_version_preserved": current.value["version"],
    }


def run() -> Path:
    from pyspark.sql import functions as functions
    from pyspark.sql.types import LongType, StringType, StructField, StructType, TimestampType
    from pyspark.sql.window import Window

    paths = ProjectPaths.discover()
    root = paths.root
    phase = PhaseRun("P06", paths)
    phase.__enter__()
    input_dir = root / "data" / "stream" / "input-v1"
    output_dir = root / "data" / "lakehouse" / "silver" / "stream_case_events"
    checkpoint = root / "data" / "checkpoints" / "case-events-v1"
    input_metrics = _write_event_batches(input_dir)
    schema = StructType(
        [
            StructField("event_id", StringType(), False),
            StructField("entity_id", StringType(), False),
            StructField("event_type", StringType(), False),
            StructField("event_timestamp", TimestampType(), False),
            StructField("sequence", LongType(), False),
            StructField("payload", StringType(), False),
        ]
    )
    spark = delta_session("controlflow-g-streaming", root / "data/lakehouse/warehouse")
    try:
        source = spark.readStream.schema(schema).json(str(input_dir))
        deduplicated = source.withWatermark("event_timestamp", "1 day").dropDuplicates(["event_id"])
        query = (
            deduplicated.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", str(checkpoint))
            .trigger(availableNow=True)
            .start(str(output_dir))
        )
        query.awaitTermination()
        after_first = spark.read.format("delta").load(str(output_dir)).count()
        query = (
            deduplicated.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", str(checkpoint))
            .trigger(availableNow=True)
            .start(str(output_dir))
        )
        query.awaitTermination()
        stored = spark.read.format("delta").load(str(output_dir))
        after_restart = stored.count()
        latest = (
            stored.withColumn(
                "rank",
                functions.row_number().over(
                    Window.partitionBy("entity_id").orderBy(
                        functions.col("sequence").desc(), functions.col("event_timestamp").desc()
                    )
                ),
            )
            .filter("rank = 1")
            .drop("rank")
        )
        latest_dir = root / "data/lakehouse/gold/latest_case_state"
        latest.write.format("delta").mode("overwrite").save(str(latest_dir))
        latest_count = latest.count()
    finally:
        spark.stop()
    if after_first != input_metrics["base"] or after_restart != after_first:
        raise RuntimeError(f"stream dedup/restart failed: first={after_first}, restart={after_restart}")
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "experiment_id": "P06-structured-streaming-replay-v1",
        "config_hash": __import__("hashlib")
        .sha256(canonical_json({"watermark": "1 day", "dedup": "event_id", "trigger": "availableNow"}))
        .hexdigest(),
        "status": "ok",
        "input_events": input_metrics["base"] + input_metrics["duplicates"],
        "unique_event_ids": input_metrics["base"],
        "stored_after_first_run": after_first,
        "stored_after_checkpoint_restart": after_restart,
        "duplicate_events_removed": input_metrics["duplicates"],
        "synthetic_late_events": input_metrics["late"],
        "synthetic_out_of_order_events": input_metrics["out_of_order"],
        "latest_entity_states": latest_count,
        "checkpoint_recovery": after_restart == after_first,
        "business_action_exactly_once_delegated_to": (
            "ActionLedger; stream exactly-once is insufficient for side effects"
        ),
        "bitemporal": _temporal_proof(),
    }
    target = root / "results" / "streaming_temporal.json"
    atomic_write_json(target, result)
    phase.register(target, "streaming_temporal_result")
    phase.__exit__(None, None, None)
    return target


if __name__ == "__main__":
    print(run())
