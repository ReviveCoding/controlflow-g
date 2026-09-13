from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from controlflow.core.state import atomic_write_json, utc_now
from controlflow.v22.dgp import contamination_against_prior, generate_split

ROOT = Path(__file__).resolve().parents[1]


def _prior_runtime_paths(excluded: Path) -> list[Path]:
    paths = []
    for path in (ROOT / "data").rglob("*.parquet"):
        if path.resolve() == excluded.resolve():
            continue
        columns = set(pd.read_parquet(path).columns)
        if {"case_id", "narrative"} <= columns:
            paths.append(path)
    return paths


def main() -> None:
    qualification = json.loads((ROOT / "state/v22_qualification_manifest.json").read_text(encoding="utf-8"))
    post_review = json.loads((ROOT / "state/v22_postqualification_review.json").read_text(encoding="utf-8"))
    if qualification.get("status") != "PASS_POST_REVIEW_REQUIRED":
        raise RuntimeError("FINAL_GENERATION_PROHIBITED: qualification did not pass")
    if post_review.get("status") != "CLEAR" or post_review.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("FINAL_GENERATION_PROHIBITED: post-qualification review not clear")
    directory = ROOT / "data/v22/final/V22FINAL"
    if directory.exists():
        raise RuntimeError("V22 final namespace already exists and is immutable")
    manifest = generate_split(directory, role="FINAL", count=500, seed=23901, prefix="V22FINAL")
    runtime_path = directory / "runtime_cases.parquet"
    contamination = contamination_against_prior(runtime_path, _prior_runtime_paths(runtime_path))
    atomic_write_json(ROOT / "results/v22/final_contamination.json", contamination)
    status = "GENERATED_SEALED_NOT_RUN" if contamination["leakage_findings"] == 0 else "INVALID_CONTAMINATED"
    atomic_write_json(
        ROOT / "state/v22_final_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": status,
            "one_shot_executed": False,
            "dataset": manifest,
            "contamination": contamination,
        },
    )
    if contamination["leakage_findings"]:
        raise RuntimeError("FINAL_GENERATION_INVALID: contamination detected")


if __name__ == "__main__":
    main()
