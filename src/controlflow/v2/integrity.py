from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.critical import CATEGORICAL, NUMERIC


def audit_development_integrity(dataset: Path) -> Path:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    split_dir = paths.root / "data/v2/splits"
    splits = {item.stem: set(item.read_text(encoding="utf-8").splitlines()) for item in split_dir.glob("*.ids")}
    overlaps: list[dict[str, Any]] = []
    names = sorted(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = splits[left] & splits[right]
            if shared:
                overlaps.append({"left": left, "right": right, "count": len(shared)})
    model_features = set(NUMERIC + CATEGORICAL + ["narrative"])
    feature_audit = json.loads((paths.state / "v2_feature_audit.json").read_text(encoding="utf-8"))
    prohibited = set(str(value) for value in feature_audit["prohibited_generator_or_truth_features"])
    leakage_features = sorted(model_features & prohibited)
    packet_source = (paths.root / "src/controlflow/v2/evidence.py").read_text(encoding="utf-8")
    packet_inputs = sorted(set(re.findall(r'row\["([^"]+)"\]', packet_source)))
    packet_leakage = sorted(set(packet_inputs) & prohibited)
    forbidden_data_paths = sorted(
        str(item.relative_to(paths.root))
        for item in (paths.root / "data/v2").rglob("*")
        if item.is_file() and any(token in item.name.casefold() for token in ("final", "test"))
    )
    all_split_ids = set().union(*splits.values()) if splits else set()
    findings: list[str] = []
    if overlaps:
        findings.append("development_split_overlap")
    if leakage_features:
        findings.append("prohibited_model_feature")
    if packet_leakage:
        findings.append("prohibited_runtime_packet_input")
    if forbidden_data_paths:
        findings.append("premature_final_or_test_artifact")
    if not frame["case_id"].astype(str).str.startswith("V2DEV-").all():
        findings.append("non_v2dev_case_namespace")
    if all_split_ids != set(frame["case_id"].astype(str)):
        findings.append("split_membership_not_exhaustive")
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(frame),
        "case_namespace": "V2DEV-*",
        "split_counts": {key: len(value) for key, value in sorted(splits.items())},
        "split_overlaps": overlaps,
        "model_features": sorted(model_features),
        "prohibited_feature_intersection": leakage_features,
        "runtime_packet_inputs": packet_inputs,
        "prohibited_runtime_packet_intersection": packet_leakage,
        "forbidden_data_paths": forbidden_data_paths,
        "v1_final_ids_or_labels_read_for_comparison": False,
        "findings": findings,
        "leakage_finding_count": int(bool(leakage_features or packet_leakage)),
        "passed": not findings,
    }
    target = paths.state / "v2_development_integrity.json"
    atomic_write_json(target, record)
    return target


def audit_internal_eligibility(runtime_path: Path, truth_path: Path) -> Path:
    """Verify the untouched eligibility boundary without exposing truth to runtime code."""
    paths = ProjectPaths.discover()
    runtime = pd.read_parquet(runtime_path)
    truth = pd.read_parquet(truth_path)
    development_ids = set(
        pd.read_parquet(paths.root / "data/v2/development/cases.parquet", columns=["case_id"])["case_id"]
    )
    runtime_ids = set(runtime["case_id"].astype(str))
    truth_ids = set(truth["case_id"].astype(str))
    prohibited = set(
        json.loads((paths.state / "v2_feature_audit.json").read_text(encoding="utf-8"))[
            "prohibited_generator_or_truth_features"
        ]
    )
    shared_columns = (set(runtime.columns) & set(truth.columns)) - {"case_id"}
    findings: list[str] = []
    if runtime_ids != truth_ids:
        findings.append("runtime_truth_id_mismatch")
    if runtime_ids & development_ids:
        findings.append("development_id_overlap")
    if shared_columns:
        findings.append("runtime_truth_column_overlap")
    if set(runtime.columns) & prohibited:
        findings.append("prohibited_runtime_column")
    if not runtime["case_id"].astype(str).str.match(r"V2ELIG\d?-\d{7}").all():
        findings.append("wrong_internal_eligibility_namespace")
    base = json.loads(
        audit_development_integrity(paths.root / "data/v2/development/cases.parquet").read_text(encoding="utf-8")
    )
    base.update(
        {
            "created_at": utc_now(),
            "internal_eligibility_runtime_sha256": sha256_file(runtime_path),
            "internal_eligibility_truth_sha256": sha256_file(truth_path),
            "internal_eligibility_rows": len(runtime),
            "runtime_truth_shared_columns_except_case_id": sorted(shared_columns),
            "internal_eligibility_findings": findings,
            "leakage_finding_count": int(base["leakage_finding_count"]) + len(findings),
            "passed": bool(base["passed"]) and not findings,
        }
    )
    target = paths.state / "v2_development_integrity.json"
    atomic_write_json(target, base)
    return target
