"""Open exactly one fresh V25QUAL identity after the committed freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v25.admission import structural_admission
from controlflow.v25.contamination import check_contamination
from controlflow.v25.freeze_validation import verify_freeze
from controlflow.v25.generator import generate_v25_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    freeze = verify_freeze(ROOT)
    directory = ROOT / "data/v25/qualification/V25QUAL"
    manifest_path = ROOT / "state/v25_qualification_manifest.json"
    if (
        directory.exists()
        or manifest_path.exists()
        or (ROOT / "results/v25/qualification").exists()
        or (ROOT / "artifacts/v25/qualification").exists()
        or any((ROOT / "state").glob("v25_qualification*.json"))
        or any((ROOT / "state").glob("v25_postclose*.json"))
        or any((ROOT / "state").glob("v25_unbound*.json"))
    ):
        raise RuntimeError("V25QUAL_ALREADY_GENERATED_OR_CONSUMED")
    reviews: dict[str, Any] = json.loads(
        (ROOT / "state/v25_prequalification_review_findings.json").read_text(encoding="utf-8")
    )
    if reviews.get("status") != "PRE_QUALIFICATION_CLEAR" or reviews.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("V25QUAL_REVIEW_NOT_CLEAR")
    config = yaml.safe_load((ROOT / "configs/v25/strata.yaml").read_text(encoding="utf-8"))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "GENERATION_OPENED",
            "created_at": utc_now(),
            "one_shot_opened": True,
            "qualification_executed": False,
            "freeze_hash": freeze["freeze_hash"],
            "dataset_identity": "V25QUAL",
        },
    )
    try:
        dataset = generate_v25_split(
            directory, role="QUALIFICATION", seed=int(config["qualification_seed"]), root=ROOT, prefix="V25QUAL"
        )
        contamination = check_contamination(ROOT, directory / "runtime_cases.parquet")
        results = ROOT / "results/v25/qualification"
        results.mkdir(parents=True, exist_ok=True)
        contamination_path = results / "contamination.json"
        atomic_write_json(contamination_path, contamination)
        admission = structural_admission(
            directory, ROOT, contamination, results / "structural_admission.json", role="QUALIFICATION"
        )
        status = "ADMITTED_NOT_EXECUTED" if admission["status"] == "ADMITTED" else "V25_GENERATION_INVALID"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": status,
                "one_shot_opened": True,
                "qualification_executed": False,
                "freeze_hash": freeze["freeze_hash"],
                "dataset_identity": "V25QUAL",
                "dataset_manifest_sha256": sha256_file(directory / "manifest.json"),
                "dataset": dataset,
                "structural_admission": admission,
                "contamination_sha256": sha256_file(contamination_path),
            },
        )
        if status != "ADMITTED_NOT_EXECUTED":
            raise RuntimeError(f"V25_GENERATION_INVALID:{admission['failures']}")
    except BaseException:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current["status"] == "GENERATION_OPENED":
            current["status"] = "V25_GENERATION_INVALID"
            atomic_write_json(manifest_path, current)
        raise


if __name__ == "__main__":
    main()
