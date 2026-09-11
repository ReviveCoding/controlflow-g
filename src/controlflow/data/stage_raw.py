from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from controlflow.core.state import PhaseRun, ProjectPaths, sha256_file, utc_now
from controlflow.data.contracts import ColumnRule, QualityAction, SourceClass, validate_frame

CONTRACTS = {
    "nist_controls_raw": (
        (
            ColumnRule("control_id", "string"),
            ColumnRule("family", "string"),
            ColumnRule("title", "string"),
            ColumnRule("description", "string"),
            ColumnRule("relationship_count", "int64"),
            ColumnRule("source_hash", "string"),
            ColumnRule("ingested_at", "string"),
        ),
        "control_id",
        SourceClass.STRICT,
    ),
    "nist_assessments_raw": ((ColumnRule("source_hash", "string"),), None, SourceClass.EVOLVING),
    "cfpb_complaints_raw": ((ColumnRule("complaint_id", "string"),), "complaint_id", SourceClass.EVOLVING),
    "cfr_raw": ((ColumnRule("regulation_id", "string"), ColumnRule("text", "string")), None, SourceClass.EVOLVING),
    "sec_filings_raw": (
        (ColumnRule("document_id", "string"), ColumnRule("text", "string")),
        "document_id",
        SourceClass.EVOLVING,
    ),
    "transactions_raw": (
        (ColumnRule("transaction_id", "string"), ColumnRule("amount", "float64")),
        "transaction_id",
        SourceClass.EVOLVING,
    ),
    "control_events_raw": ((ColumnRule("event_id", "string"),), "event_id", SourceClass.EVOLVING),
}


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def nist_controls(path: Path) -> pd.DataFrame:
    catalog = json.loads(path.read_text(encoding="utf-8"))["catalog"]
    source_hash = sha256_file(path)
    ingested_at = utc_now()
    rows: list[dict[str, Any]] = []
    for group in catalog.get("groups", []):
        for control in group.get("controls", []):
            stack = [control, *control.get("controls", [])]
            for item in stack:
                prose = " ".join(part.get("prose", "") for part in item.get("parts", []))
                rows.append(
                    {
                        "control_id": item["id"].upper(),
                        "family": group["id"].upper(),
                        "title": item.get("title", ""),
                        "description": prose,
                        "relationship_count": len(item.get("links", [])),
                        "source_hash": source_hash,
                        "ingested_at": ingested_at,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("control_id")


def nist_assessments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="cp1252", low_memory=False).astype("string")
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_") for column in frame.columns]
    frame["source_hash"] = sha256_file(path)
    frame["ingested_at"] = utc_now()
    return frame


def cfpb_sample(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=rows, low_memory=False)
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_") for column in frame.columns]
    frame["source_hash"] = sha256_file(path)
    frame["ingested_at"] = utc_now()
    frame["sampling_design"] = f"first_{rows}_rows_smoke_profile"
    return frame


def cfr_sections(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        date_match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", path.name)
        valid_from = date_match.group(0) if date_match else f"{path.parent.name}-01-01"
        source_type = "ecfr" if "ecfr" in path.as_posix() else "govinfo"
        source_hash = sha256_file(path)
        ingested_at = utc_now()
        tree = ET.parse(path)
        for element in tree.iter():
            tag = element.tag.rsplit("}", 1)[-1].upper()
            if tag in {"DIV8", "SECTION"}:
                text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
                if len(text) >= 20:
                    section_match = re.search(r"(?:§|Sec\.)\s*([0-9]+\.[0-9A-Za-z.-]+)", text)
                    identity_source = section_match.group(1) if section_match else text[:200].casefold()
                    identifier = hashlib.sha256(identity_source.encode()).hexdigest()[:20]
                    rows.append(
                        {
                            "regulation_id": f"12CFR-{identifier}",
                            "source_type": source_type,
                            "source_file": path.name,
                            "business_valid_from": valid_from,
                            "text": text,
                            "source_hash": source_hash,
                            "ingested_at": ingested_at,
                        }
                    )
        del tree
    return pd.DataFrame(rows).drop_duplicates(["regulation_id", "business_valid_from"])


def sec_documents(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        match = re.match(r"(\d{4}-\d{2}-\d{2})_(10-[KQ])_", path.name)
        rows.append(
            {
                "document_id": hashlib.sha256(path.as_posix().encode()).hexdigest()[:20],
                "issuer": path.parent.name,
                "form": match.group(2) if match else "UNKNOWN",
                "filing_date": match.group(1) if match else None,
                "text": text,
                "source_file": path.name,
                "source_hash": sha256_file(path),
                "ingested_at": utc_now(),
            }
        )
    return pd.DataFrame(rows)


def transactions(count: int, seed: int = 1729) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    entity = rng.integers(0, 5000, count)
    amount = rng.lognormal(5.5, 1.4, count)
    injected = np.arange(count) % 997 == 0
    amount[injected] *= 40
    return pd.DataFrame(
        {
            "transaction_id": [f"TX-{index:09d}" for index in range(count)],
            "entity_id": [f"ENTITY-{value:05d}" for value in entity],
            "event_timestamp": [
                start + timedelta(seconds=int(value)) for value in np.sort(rng.integers(0, 365 * 86400, count))
            ],
            "amount": np.round(amount, 2),
            "channel": np.asarray(["ACH", "CARD", "WIRE", "CHECK"])[rng.integers(0, 4, count)],
            "injected_anomaly": injected,
            "source_hash": hashlib.sha256(f"transactions:{seed}:{count}".encode()).hexdigest(),
            "ingested_at": utc_now(),
        }
    )


def control_events(count: int = 5000) -> pd.DataFrame:
    types = [
        "case_create",
        "case_update",
        "control_test",
        "policy_update",
        "ownership_change",
        "human_review",
    ]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "event_id": [f"CE-{index:08d}" for index in range(count)],
            "case_id": [f"CASE-{index % 2000:07d}" for index in range(count)],
            "sequence": [index // 2000 for index in range(count)],
            "event_type": [types[index % len(types)] for index in range(count)],
            "event_timestamp": [start + timedelta(minutes=index * 13) for index in range(count)],
            "payload": [json.dumps({"ordinal": index}, sort_keys=True) for index in range(count)],
            "ingested_at": utc_now(),
        }
    )


def run(cfpb_rows: int = 20_000, transaction_rows: int = 100_000) -> Path:
    paths = ProjectPaths.discover()
    root = paths.root
    phase = PhaseRun("P05", paths)
    phase.__enter__()
    outputs = root / "data" / "staging"
    try:
        tables = {
            "nist_controls_raw": nist_controls(root / "data/raw/nist_oscal_5_1_1/source.json"),
            "nist_assessments_raw": nist_assessments(root / "data/raw/nist_800_53a_r5/source.csv"),
            "cfpb_complaints_raw": cfpb_sample(root / "data/raw/cfpb_complaints/source.zip", cfpb_rows),
            "cfr_raw": cfr_sections(
                sorted((root / "data/raw/ecfr_title_12").glob("*.xml"))
                + sorted((root / "data/raw/govinfo_title_12/2024").glob("*.xml"))
            ),
            "sec_filings_raw": sec_documents(
                sorted(
                    path
                    for path in (root / "data/raw/sec_filings").glob("*/*")
                    if path.suffix.lower() in {".htm", ".html"}
                )
            ),
            "transactions_raw": transactions(transaction_rows),
            "control_events_raw": control_events(),
        }
        manifest_rows: list[dict[str, Any]] = []
        previous_manifest = outputs / "manifest.json"
        prior_rows = {}
        if previous_manifest.exists():
            prior_rows = {
                item["table"]: item for item in json.loads(previous_manifest.read_text(encoding="utf-8"))["tables"]
            }
        for name, frame in tables.items():
            rules, primary_key, source_class = CONTRACTS[name]
            contract = validate_frame(
                frame,
                rules,
                source_class=source_class,
                invalid_action=QualityAction.QUARANTINE,
                primary_key=primary_key,
            )
            frame = contract.accepted
            quarantine = outputs / "quarantine" / f"{name}.parquet"
            _write(contract.quarantined, quarantine)
            target = outputs / f"{name}.parquet"
            parquet = pq.ParquetFile(target) if target.exists() else None
            existing_rows = parquet.metadata.num_rows if parquet is not None else None
            parquet_schema = parquet.schema_arrow if parquet is not None else None
            incompatible_timestamp = parquet_schema is not None and any(
                getattr(field.type, "unit", None) == "ns" for field in parquet_schema
            )
            source_hashes = sorted(
                str(value) for value in frame.get("source_hash", pd.Series(dtype=str)).dropna().unique()
            )
            input_fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
            prior_fingerprint = prior_rows.get(name, {}).get("input_fingerprint")
            existing_source_hashes: list[str] = []
            if target.exists() and "source_hash" in (parquet.schema_arrow.names if parquet else []):
                existing_source_hashes = sorted(
                    str(value.as_py())
                    for value in pq.read_table(target, columns=["source_hash"])["source_hash"].unique()
                )
            source_changed = prior_fingerprint not in {None, input_fingerprint} or (
                prior_fingerprint is None and existing_source_hashes != source_hashes
            )
            transform_marker = outputs / f"{name}.transform-v2"
            transform_changed = (
                name == "cfr_raw"
                and prior_rows.get(name, {}).get("transform_version") != 2
                and not transform_marker.exists()
            )
            if existing_rows != len(frame) or incompatible_timestamp or source_changed or transform_changed:
                _write(frame, target)
                if name == "cfr_raw":
                    transform_marker.write_text("canonical-cfr-identity-v2\n", encoding="utf-8")
            manifest_rows.append(
                {
                    "table": name,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "sha256": sha256_file(target),
                    "path": target.relative_to(root).as_posix(),
                    "input_fingerprint": input_fingerprint,
                    "transform_version": 2 if name == "cfr_raw" else 1,
                    "contract_metrics": contract.metrics,
                    "quarantine_path": quarantine.relative_to(root).as_posix(),
                }
            )
            phase.register(target, "staging_table")
            phase.register(quarantine, "quarantine_table")
        manifest = outputs / "manifest.json"
        manifest.write_text(
            json.dumps(
                {"created_at": utc_now(), "profile": "SMOKE", "tables": manifest_rows},
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        phase.register(manifest, "staging_manifest")
    except Exception as exc:
        phase.__exit__(type(exc), exc, exc.__traceback__)
        raise
    phase.journal("checkpoint_completed", "Bounded raw staging tables created", ["data/staging/manifest.json"])
    phase.__exit__(None, None, None)
    return manifest


if __name__ == "__main__":
    print(run())
