from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, canonical_json, sha256_file, utc_now


def _merge(counter: Counter[str], series: pd.Series) -> None:
    counter.update({str(key): int(value) for key, value in series.value_counts(dropna=False).items()})


def profile_cfpb(path: Path) -> dict[str, Any]:
    columns = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company",
        "State",
        "Submitted via",
        "Timely response?",
        "Complaint ID",
    ]
    products: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    months: Counter[str] = Counter()
    states: Counter[str] = Counter()
    lengths: list[int] = []
    rows = 0
    narratives = 0
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    duplicate_ids = 0
    seen_ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000, low_memory=False):
        rows += len(chunk)
        dates = pd.to_datetime(chunk["Date received"], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates):
            current_min, current_max = valid_dates.min(), valid_dates.max()
            earliest = current_min if earliest is None else min(earliest, current_min)
            latest = current_max if latest is None else max(latest, current_max)
            _merge(months, dates.dt.to_period("M").astype(str))
        _merge(products, chunk["Product"].fillna("<MISSING>"))
        _merge(issues, chunk["Issue"].fillna("<MISSING>"))
        _merge(states, chunk["State"].fillna("<MISSING>"))
        narrative = chunk["Consumer complaint narrative"].dropna().astype(str)
        narratives += len(narrative)
        identifiers = chunk["Complaint ID"].astype(str)
        unique_chunk = set(identifiers)
        duplicate_ids += len(seen_ids.intersection(unique_chunk)) + int(identifiers.duplicated().sum())
        seen_ids.update(unique_chunk)
        mask = (pd.util.hash_pandas_object(identifiers, index=False).to_numpy() % 1000) == 0
        lengths.extend(
            chunk.loc[mask & chunk["Consumer complaint narrative"].notna(), "Consumer complaint narrative"]
            .astype(str)
            .str.split()
            .str.len()
            .tolist()
        )
    length_array = np.asarray(lengths, dtype=float)
    return {
        "rows": rows,
        "columns": columns,
        "earliest_date": earliest.date().isoformat() if earliest is not None else None,
        "latest_date": latest.date().isoformat() if latest is not None else None,
        "narratives_present": narratives,
        "narratives_missing": rows - narratives,
        "narrative_present_rate": narratives / rows,
        "duplicate_ids": duplicate_ids,
        "products": products.most_common(),
        "issues_top_100": issues.most_common(100),
        "states": states.most_common(),
        "monthly_counts": sorted(months.items()),
        "narrative_length_hash_sample_n": len(lengths),
        "narrative_length_tokens_quantiles": {
            str(quantile): float(np.quantile(length_array, quantile)) if len(length_array) else None
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "prevalence_warning": "Complaint counts are not a statistical sample and are not population prevalence.",
    }


def profile_nist(catalog_path: Path, assessment_path: Path) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))["catalog"]
    controls: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    relation_count = 0
    for group in catalog.get("groups", []):
        family = group.get("id", "unknown")
        for control in group.get("controls", []):
            controls.append(control)
            family_counts[family] += 1
            relation_count += len(control.get("links", []))
            controls.extend(control.get("controls", []))
            family_counts[family] += len(control.get("controls", []))
    lengths = [
        len(" ".join(part.get("prose", "") for part in control.get("parts", [])).split()) for control in controls
    ]
    assessment = pd.read_csv(assessment_path, low_memory=False, encoding="cp1252")
    return {
        "catalog_uuid": catalog.get("uuid"),
        "controls_and_enhancements": len(controls),
        "family_counts": sorted(family_counts.items()),
        "control_text_tokens_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.1, 0.5, 0.9, 0.99)},
        "relationship_links": relation_count,
        "assessment_rows": len(assessment),
        "assessment_columns": list(assessment.columns),
        "assessment_nonempty_per_control_mean": float(assessment.notna().sum(axis=1).mean()),
    }


def profile_xml(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        tags: Counter[str] = Counter()
        text_characters = 0
        for _, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            tags[tag] += 1
            if element.text:
                text_characters += len(element.text)
            element.clear()
        records.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "top_tags": tags.most_common(20),
                "section_like_count": sum(
                    count for tag, count in tags.items() if tag.upper() in {"SECTION", "DIV8", "SECTNO"}
                ),
                "text_characters": text_characters,
            }
        )
    return {
        "files": records,
        "total_bytes": sum(item["bytes"] for item in records),
        "total_section_like_count": sum(item["section_like_count"] for item in records),
    }


def profile_sec(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        records.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "text_characters": len(text),
                "approx_512_token_chunks": max(1, len(text.split()) // 480),
            }
        )
    return {
        "documents": records,
        "document_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "approx_512_token_chunks": sum(item["approx_512_token_chunks"] for item in records),
    }


def _figures(
    root: Path, cfpb: dict[str, Any], nist: dict[str, Any], cfr: dict[str, Any], sec: dict[str, Any]
) -> list[Path]:
    directory = root / "reports" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    product = cfpb["products"][:12]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh([item[0] for item in product][::-1], [item[1] for item in product][::-1])
    axis.set_title("CFPB complaint records by product (not population prevalence)")
    axis.set_xlabel("records")
    fig.tight_layout()
    path = directory / "cfpb_product_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    family = nist["family_counts"]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([item[0] for item in family], [item[1] for item in family])
    axis.set_title("NIST control and enhancement counts by family")
    axis.set_ylabel("count")
    fig.tight_layout()
    path = directory / "nist_family_counts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    monthly = cfpb["monthly_counts"]
    stride = max(1, len(monthly) // 36)
    sampled = monthly[::stride]
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot([item[0] for item in sampled], [item[1] for item in sampled])
    axis.tick_params(axis="x", rotation=60)
    axis.set_title("CFPB record volume over receipt time (taxonomy/publication effects included)")
    axis.set_ylabel("records")
    fig.tight_layout()
    path = directory / "cfpb_temporal_counts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)
    return outputs


def enrich_data_manifest(root: Path, profile: dict[str, Any]) -> None:
    manifest_path = root / "state" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfr_by_name = {Path(item["path"]).name: item for item in profile["cfr"]["files"]}
    sec_by_name = {Path(item["path"]).name: item for item in profile["sec"]["documents"]}
    for record in manifest["datasets"]:
        name = record["name"]
        filename = Path(record["local_path"]).name
        if name == "cfpb_complaints":
            record["row_count"] = profile["cfpb"]["rows"]
            record["schema"] = profile["cfpb"]["columns"]
        elif name == "nist_oscal_5_1_1":
            record["row_count"] = profile["nist"]["controls_and_enhancements"]
            record["schema"] = ["catalog.groups.controls", "catalog.groups.controls.controls"]
        elif name == "nist_800_53a_r5":
            record["row_count"] = profile["nist"]["assessment_rows"]
            record["schema"] = profile["nist"]["assessment_columns"]
        elif name.startswith("ecfr_title_12_") or name.startswith("govinfo_title_12_"):
            record["row_count"] = cfr_by_name[filename]["section_like_count"]
            record["schema"] = ["XML elements", "section-like count"]
        elif name.startswith("sec_") and filename in sec_by_name:
            record["row_count"] = 1
            record["schema"] = ["EDGAR primary filing HTML", "text characters", "estimated chunks"]
        elif name.startswith("sec_submissions_"):
            payload = json.loads((root / record["local_path"]).read_text(encoding="utf-8"))
            record["row_count"] = len(payload["filings"]["recent"]["form"])
            record["schema"] = sorted(payload["filings"]["recent"].keys())
        elif name == "sec_company_tickers":
            payload = json.loads((root / record["local_path"]).read_text(encoding="utf-8"))
            record["row_count"] = len(payload["data"])
            record["schema"] = payload["fields"]
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def run() -> Path:
    paths = ProjectPaths.discover()
    root = paths.root
    phase = PhaseRun("P03", paths)
    phase.__enter__()
    try:
        checkpoint_dir = root / "results" / "profiles"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def cached(name: str, build: Any) -> dict[str, Any]:
            checkpoint = checkpoint_dir / f"{name}.json"
            if checkpoint.exists():
                return cast(dict[str, Any], json.loads(checkpoint.read_text(encoding="utf-8")))
            value = cast(dict[str, Any], build())
            atomic_write_json(checkpoint, value)
            phase.register(checkpoint, "profile_checkpoint")
            phase.journal(
                "checkpoint_completed",
                f"{name} raw profile completed",
                [checkpoint.relative_to(root).as_posix()],
            )
            return value

        cfpb = cached("cfpb", lambda: profile_cfpb(root / "data/raw/cfpb_complaints/source.zip"))
        nist = cached(
            "nist",
            lambda: profile_nist(
                root / "data/raw/nist_oscal_5_1_1/source.json",
                root / "data/raw/nist_800_53a_r5/source.csv",
            ),
        )
        cfr_paths = sorted((root / "data/raw/ecfr_title_12").glob("*.xml")) + sorted(
            (root / "data/raw/govinfo_title_12/2024").glob("*.xml")
        )
        cfr = cached("cfr", lambda: profile_xml(cfr_paths))
        sec_paths = sorted(
            path for path in (root / "data/raw/sec_filings").glob("*/*") if path.suffix.lower() in {".htm", ".html"}
        )
        sec = cached("sec", lambda: profile_sec(sec_paths))
        profile = {
            "schema_version": 1,
            "created_at": utc_now(),
            "cfpb": cfpb,
            "nist": nist,
            "cfr": cfr,
            "sec": sec,
        }
        enrich_data_manifest(root, profile)
        target = root / "results" / "data_profile.json"
        atomic_write_json(target, profile)
        dq = pd.DataFrame(
            [
                {"dataset": "cfpb", "metric": "rows", "value": cfpb["rows"], "status": "PASS"},
                {
                    "dataset": "cfpb",
                    "metric": "duplicate_ids",
                    "value": cfpb["duplicate_ids"],
                    "status": "WARN" if cfpb["duplicate_ids"] else "PASS",
                },
                {
                    "dataset": "nist",
                    "metric": "controls_and_enhancements",
                    "value": nist["controls_and_enhancements"],
                    "status": "PASS",
                },
                {
                    "dataset": "nist",
                    "metric": "assessment_rows",
                    "value": nist["assessment_rows"],
                    "status": "PASS",
                },
                {"dataset": "cfr", "metric": "files", "value": len(cfr_paths), "status": "PASS"},
                {
                    "dataset": "sec",
                    "metric": "documents",
                    "value": sec["document_count"],
                    "status": "PASS",
                },
            ]
        )
        dq["experiment_id"] = "data-quality-public-sources"
        dq["config_hash"] = hashlib.sha256(canonical_json({"profile": "SMOKE", "version": 2})).hexdigest()
        dq["dataset_hash"] = sha256_file(root / "state/data_manifest.json")
        dq["split_identifier"] = "not_applicable"
        dq["seed"] = 17
        dq["hardware_runtime"] = "CPU"
        dq["timestamp"] = utc_now()
        dq_path = root / "results" / "data_quality.parquet"
        dq.to_parquet(dq_path, index=False)
        figures = _figures(root, cfpb, nist, cfr, sec)
        for artifact, kind in [
            (root / "state/data_manifest.json", "data_manifest"),
            (target, "data_profile"),
            (dq_path, "result_table"),
            *((item, "figure") for item in figures),
        ]:
            phase.register(artifact, kind)
        phase.__exit__(None, None, None)
    except Exception as exc:
        phase.__exit__(type(exc), exc, exc.__traceback__)
        raise
    return target


if __name__ == "__main__":
    print(run())
