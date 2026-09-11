from __future__ import annotations

import hashlib
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

BRONZE = {
    "nist_controls_raw": "nist_controls_raw.parquet",
    "nist_assessments_raw": "nist_assessments_raw.parquet",
    "cfpb_complaints_raw": "cfpb_complaints_raw.parquet",
    "cfr_raw": "cfr_raw.parquet",
    "sec_filings_raw": "sec_filings_raw.parquet",
    "transactions_raw": "transactions_raw.parquet",
    "control_events_raw": "control_events_raw.parquet",
}


def run() -> Path:
    from pyspark.sql import Window
    from pyspark.sql import functions as functions
    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    paths = ProjectPaths.discover()
    root = paths.root
    lakehouse = root / "data" / "lakehouse"
    warehouse = lakehouse / "warehouse"
    target = root / "results" / "lakehouse_manifest.json"
    phase = PhaseRun("P05", paths)
    phase.journal("checkpoint_started", "Delta medallion build started")
    spark = delta_session("controlflow-g-medallion", warehouse)
    tables: list[dict[str, Any]] = []

    def save(layer: str, name: str, frame: Any) -> Any:
        location = lakehouse / layer / name
        frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(str(location))
        loaded = spark.read.format("delta").load(str(location))
        count = loaded.count()
        schema = loaded.schema.jsonValue()
        tables.append(
            {
                "table": f"{layer}.{name}",
                "path": location.relative_to(root).as_posix(),
                "rows": count,
                "schema_hash": hashlib.sha256(canonical_json(schema)).hexdigest(),
            }
        )
        loaded.createOrReplaceTempView(f"{layer}_{name}")
        return loaded

    try:
        bronze: dict[str, Any] = {}
        for name, filename in BRONZE.items():
            frame = (
                spark.read.parquet(str(root / "data" / "staging" / filename))
                .withColumn("_bronze_ingested_at", functions.to_timestamp("ingested_at"))
                .withColumn("_bronze_source_table", functions.lit(name))
            )
            expected = {
                "nist_controls_raw": "nist_controls_raw",
                "nist_assessments_raw": "nist_control_assessments_raw",
                "cfpb_complaints_raw": "cfpb_complaints_raw",
                "cfr_raw": "cfr_raw",
                "sec_filings_raw": "sec_filings_raw",
                "transactions_raw": "transactions_raw",
                "control_events_raw": "control_events_raw",
            }[name]
            bronze[name] = save("bronze", expected, frame)

        controls = (
            bronze["nist_controls_raw"]
            .select(
                functions.upper("control_id").alias("control_id"),
                functions.upper("family").alias("family"),
                functions.trim("title").alias("title"),
                functions.trim("description").alias("description"),
                "relationship_count",
                functions.to_timestamp("ingested_at").alias("system_known_from"),
            )
            .dropDuplicates(["control_id"])
        )
        controls = controls.withColumn("system_known_to", functions.to_timestamp(functions.lit("9999-12-31 23:59:59")))
        controls = save("silver", "controls", controls)

        assessment = (
            bronze["nist_assessments_raw"]
            .select(
                functions.upper("identifier").alias("control_id"),
                "family",
                "control_name",
                "assessment_objective",
                "examine",
                "interview",
                "test",
                functions.to_timestamp("ingested_at").alias("system_known_from"),
            )
            .dropDuplicates(["control_id", "assessment_objective"])
        )
        assessment = save("silver", "control_assessments", assessment)

        complaints = (
            bronze["cfpb_complaints_raw"]
            .select(
                functions.col("complaint_id").cast("string"),
                functions.to_date("date_received").alias("received_date"),
                "product",
                "sub_product",
                "issue",
                "sub_issue",
                functions.col("consumer_complaint_narrative").alias("narrative"),
                "company",
                "state",
                "submitted_via",
                "timely_response",
                "sampling_design",
            )
            .filter(functions.col("complaint_id").isNotNull())
            .dropDuplicates(["complaint_id"])
        )
        complaints = save("silver", "complaints", complaints)

        raw_regulations = (
            bronze["cfr_raw"]
            .withColumn(
                "section_number",
                functions.regexp_extract("text", r"(?:§|Sec\.)\s*([0-9]+\.[0-9A-Za-z.-]+)", 1),
            )
            .withColumn(
                "stable_regulation_id",
                functions.when(
                    functions.length("section_number") > 0,
                    functions.concat(functions.lit("12CFR-"), functions.col("section_number")),
                ).otherwise(functions.col("regulation_id")),
            )
            .withColumn("business_valid_from", functions.to_timestamp("business_valid_from"))
            .withColumn("system_known_from", functions.to_timestamp("ingested_at"))
        )
        window = Window.partitionBy("stable_regulation_id").orderBy("business_valid_from", "system_known_from")
        regulations = (
            raw_regulations.withColumn(
                "business_valid_to",
                functions.coalesce(
                    functions.lead("business_valid_from").over(window),
                    functions.to_timestamp(functions.lit("9999-12-31 23:59:59")),
                ),
            )
            .withColumn("system_known_to", functions.to_timestamp(functions.lit("9999-12-31 23:59:59")))
            .select(
                functions.col("stable_regulation_id").alias("regulation_id"),
                "source_type",
                "source_file",
                "section_number",
                "text",
                "business_valid_from",
                "business_valid_to",
                "system_known_from",
                "system_known_to",
                "source_hash",
            )
        )
        regulations = save("silver", "regulation_versions", regulations)

        documents = (
            bronze["sec_filings_raw"]
            .select(
                "document_id",
                "issuer",
                "form",
                functions.to_date("filing_date").alias("filing_date"),
                "text",
                "source_file",
                "source_hash",
                functions.length("text").alias("text_characters"),
            )
            .filter(functions.length("text") > 100)
            .dropDuplicates(["document_id"])
        )
        documents = save("silver", "financial_documents", documents)

        transactions = (
            bronze["transactions_raw"]
            .select(
                "transaction_id",
                "entity_id",
                functions.to_timestamp("event_timestamp").alias("event_timestamp"),
                functions.col("amount").cast("double"),
                "channel",
                "injected_anomaly",
                "source_hash",
            )
            .dropDuplicates(["transaction_id"])
        )
        transactions = save("silver", "transactions", transactions)

        events = (
            bronze["control_events_raw"]
            .select(
                "event_id",
                "case_id",
                functions.col("sequence").cast("long"),
                "event_type",
                functions.to_timestamp("event_timestamp").alias("event_timestamp"),
                "payload",
            )
            .dropDuplicates(["event_id"])
        )
        events = save("silver", "case_events", events)

        development = root / "data/silver/synthetic_cases_development.parquet"
        if development.exists():
            synthetic_cases = spark.read.parquet(str(development))
            for column in ("event_timestamp", "feature_event_timestamp", "feature_system_known_at"):
                synthetic_cases = synthetic_cases.withColumn(
                    column,
                    (functions.col(column) / functions.lit(1_000_000_000)).cast("timestamp"),
                )
        else:
            synthetic_schema = StructType(
                [
                    StructField("case_id", StringType(), False),
                    StructField("event_timestamp", TimestampType(), False),
                    StructField("severity", StringType(), False),
                    StructField("expected_disposition", StringType(), False),
                ]
            )
            synthetic_cases = spark.createDataFrame([], synthetic_schema)
        synthetic_cases = save("silver", "synthetic_cases", synthetic_cases)

        roles = spark.sql("""SELECT * FROM VALUES
            ('Control Analyst', 2, false), ('Business Owner', 2, true),
            ('Risk Manager', 4, true), ('Compliance Reviewer', 4, true),
            ('Data Scientist', 3, false), ('ML Engineer', 3, false),
            ('Platform Engineer', 4, true), ('Auditor', 5, false)
            AS roles(role, max_clearance, may_approve)""")
        roles = save("silver", "user_roles", roles)

        save("gold", "dim_control", controls.select("control_id", "family", "title"))
        save(
            "gold",
            "dim_regulation",
            regulations.select("regulation_id", "source_type").dropDuplicates(),
        )
        business_units = spark.sql(
            "SELECT * FROM VALUES ('consumer'),('payments'),('wealth'),('treasury') AS units(business_unit)"
        )
        save("gold", "dim_business_unit", business_units)
        save("gold", "dim_user_role", roles)
        case_types = spark.sql(
            "SELECT * FROM VALUES ('normal'),('difficult'),('critical'),('rare'),"
            "('missing_evidence'),('conflicting_evidence'),('stale_policy'),('privilege'),('adversarial') "
            "AS types(case_type)"
        )
        save("gold", "dim_case_type", case_types)

        fact_case = synthetic_cases.select(
            "case_id",
            functions.col("event_timestamp").alias("created_at"),
            functions.col("event_timestamp").alias("updated_at"),
            functions.lit(1).alias("event_count"),
            "severity",
        )
        fact_case = save("gold", "fact_case", fact_case)
        save("gold", "fact_case_event", events)
        save(
            "gold",
            "fact_control_test",
            events.filter(functions.col("event_type") == "control_test"),
        )

        empty_specs = {
            "fact_model_prediction": "prediction_id STRING, case_id STRING, model_id STRING, predicted_at TIMESTAMP",
            "fact_agent_run": "run_id STRING, case_id STRING, started_at TIMESTAMP, disposition STRING",
            "fact_tool_call": (
                "call_id STRING, run_id STRING, tool_name STRING, authorized BOOLEAN, called_at TIMESTAMP"
            ),
            "fact_human_review": (
                "review_id STRING, case_id STRING, reviewer_id STRING, decision STRING, decided_at TIMESTAMP"
            ),
        }
        for name, schema_ddl in empty_specs.items():
            trace_path = root / "results/agent_traces.parquet"
            if trace_path.exists() and name in {
                "fact_model_prediction",
                "fact_agent_run",
                "fact_tool_call",
                "fact_human_review",
            }:
                traces = spark.read.parquet(str(trace_path))
                traces = traces.join(
                    synthetic_cases.select("case_id", "event_timestamp"),
                    "case_id",
                    "left",
                )
                if name == "fact_model_prediction":
                    table = traces.selectExpr(
                        "concat(experiment_id, '-', case_id) prediction_id",
                        "case_id",
                        "experiment_id model_id",
                        "event_timestamp predicted_at",
                    )
                elif name == "fact_agent_run":
                    table = traces.selectExpr(
                        "concat(experiment_id, '-', case_id) run_id",
                        "case_id",
                        "event_timestamp started_at",
                        "predicted_disposition disposition",
                    )
                elif name == "fact_tool_call":
                    table = traces.withColumn(
                        "tool_name", functions.explode(functions.from_json("tool_calls", "array<string>"))
                    ).selectExpr(
                        "concat(experiment_id, '-', case_id, '-', tool_name) call_id",
                        "concat(experiment_id, '-', case_id) run_id",
                        "tool_name",
                        "authorization_correct authorized",
                        "event_timestamp called_at",
                    )
                else:
                    table = traces.filter("human_review_requested").selectExpr(
                        "concat(experiment_id, '-', case_id) review_id",
                        "case_id",
                        "'simulated-reviewer' reviewer_id",
                        "'REVIEW_REQUIRED' decision",
                        "event_timestamp decided_at",
                    )
                save("gold", name, table)
            else:
                save("gold", name, spark.createDataFrame([], schema_ddl))
        case_360 = fact_case.join(events.groupBy("case_id").pivot("event_type").count(), "case_id", "left")
        save("gold", "case_360", case_360)

        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "runtime": {"spark": spark.version, "delta": "3.3.2", "profile": "SMOKE"},
            "tables": sorted(tables, key=lambda item: item["table"]),
        }
        atomic_write_json(target, manifest)
        phase.register(target, "lakehouse_manifest")
        phase.journal(
            "checkpoint_completed",
            f"Built {len(tables)} Delta Bronze/Silver/Gold tables",
            [target.relative_to(root).as_posix()],
        )
        phase.__exit__(None, None, None)
    except Exception as exc:
        phase.__exit__(type(exc), exc, exc.__traceback__)
        raise
    finally:
        spark.stop()
    return target


if __name__ == "__main__":
    print(run())
