"""One-shot V24QUAL generation and structural admission after freeze."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v24.admission import structural_admission
from controlflow.v24.contamination import check_contamination
from controlflow.v24.freeze_validation import verify_freeze
from controlflow.v24.generator import generate_v24_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    freeze = verify_freeze(ROOT)
    directory = ROOT / "data/v24/qualification/V24QUAL"
    manifest_path = ROOT / "state/v24_qualification_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("status") != "NOT_GENERATED" or directory.exists():
        raise RuntimeError("V24QUAL_ALREADY_GENERATED_OR_CONSUMED")
    config = yaml.safe_load((ROOT / "configs/v24/strata.yaml").read_text(encoding="utf-8"))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "GENERATION_OPENED",
            "created_at": utc_now(),
            "one_shot_opened": True,
            "qualification_executed": False,
            "freeze_hash": freeze["freeze_hash"],
            "dataset_identity": "V24QUAL",
        },
    )
    try:
        dataset = generate_v24_split(
            directory, role="QUALIFICATION", seed=int(config["qualification_seed"]), root=ROOT, prefix="V24QUAL"
        )
        contamination = check_contamination(ROOT, directory / "runtime_cases.parquet")
        results = ROOT / "results/v24/qualification"
        results.mkdir(parents=True, exist_ok=True)
        contamination_path = results / "contamination.json"
        atomic_write_json(contamination_path, contamination)
        admission = structural_admission(directory, ROOT, contamination, results / "structural_admission.json")
        status = "ADMITTED_NOT_EXECUTED" if admission["status"] == "ADMITTED" else "V24_GENERATION_INVALID"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": status,
                "one_shot_opened": True,
                "qualification_executed": False,
                "freeze_hash": freeze["freeze_hash"],
                "dataset_identity": "V24QUAL",
                "dataset_manifest_sha256": sha256_file(directory / "manifest.json"),
                "dataset": dataset,
                "structural_admission": admission,
                "contamination_sha256": sha256_file(contamination_path),
            },
        )
        if status != "ADMITTED_NOT_EXECUTED":
            raise RuntimeError(f"V24_GENERATION_INVALID:{admission['failures']}")
    except BaseException:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current["status"] == "GENERATION_OPENED":
            current["status"] = "V24_GENERATION_INVALID"
            atomic_write_json(manifest_path, current)
        raise


if __name__ == "__main__":
    main()
