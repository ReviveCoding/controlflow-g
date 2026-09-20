"""Open exactly one qualification or final identity and generate its sealed evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v26.admission import structural_admission
from controlflow.v26.closure_receipt import verify_receipt
from controlflow.v26.freeze_validation import verify_freeze
from controlflow.v26.generator import generate_v26_split
from controlflow.v26.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v26/prior_runtime_manifest.yaml"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _qualification_clear() -> dict[str, Any]:
    freeze = verify_freeze(ROOT, role="qualification")
    review = _json(ROOT / "state/v26_prequalification_review_findings.json")
    if review.get("status") != "PRE_QUALIFICATION_CLEAR" or review.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("V26QUAL_REVIEW_NOT_CLEAR")
    return freeze


def _final_clear() -> dict[str, Any]:
    freeze = _qualification_clear()
    qualification_path = ROOT / "state/v26_qualification_manifest.json"
    qualification = _json(qualification_path)
    review = _json(ROOT / "state/v26_postqualification_review_findings.json")
    graph = ROOT / "state/v26_qualification_bindings.json"
    receipt = ROOT / "state/v26_qualification_closure_receipt.json"
    proposed = ROOT / "state/v26_qualification_manifest.proposed.json"
    anchor = ROOT / "state/v26_qualification_terminal_anchor.json"
    postclose = _json(ROOT / "state/v26_postclose_verification.json")
    decision = _json(ROOT / "results/v26/qualification/gate_decision.json")
    if (
        qualification.get("status") != "COMPLETE"
        or qualification.get("all_passed") is not True
        or qualification.get("freeze_hash") != freeze["freeze_hash"]
        or decision.get("all_passed") is not True
        or postclose.get("status") != "PASS"
        or review.get("status") != "POST_QUALIFICATION_CLEAR"
        or review.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or review.get("qualification_manifest_sha256") != sha256_file(qualification_path)
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V26FINAL_REQUIRES_ADMISSIBLE_SEALED_QUALIFICATION_AND_REVIEW")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("qualification", "final"), required=True)
    role = parser.parse_args().role
    freeze = _final_clear() if role == "final" else _qualification_clear()
    identity = "V26FINAL" if role == "final" else "V26QUAL"
    directory = ROOT / f"data/v26/{role}/{identity}"
    results = ROOT / f"results/v26/{role}"
    artifacts = ROOT / f"artifacts/v26/{role}"
    manifest_path = ROOT / f"state/v26_{role}_manifest.json"
    if (
        directory.exists()
        or results.exists()
        or artifacts.exists()
        or manifest_path.exists()
        or any((ROOT / "state").glob(f"v26_{role}*.json"))
    ):
        raise RuntimeError(f"{identity}_ALREADY_OPENED_OR_CONSUMED")
    strata = yaml.safe_load((ROOT / "configs/v26/strata.yaml").read_text(encoding="utf-8"))
    seed = int(strata[f"{role}_seed"])
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "GENERATION_OPENED",
            "created_at": utc_now(),
            "one_shot_opened": True,
            "execution_started": False,
            "dataset_identity": identity,
            "seed": seed,
            "qualification_freeze_hash": freeze["freeze_hash"],
            **({"freeze_hash": freeze["freeze_hash"]} if role == "qualification" else {}),
        },
    )
    try:
        dataset = generate_v26_split(directory, role=role.upper(), seed=seed, root=ROOT, prefix=identity)
        results.mkdir(parents=True)
        runtime = directory / "runtime_cases.parquet"
        inventory = results / "prior_inventory.json"
        stored = results / "contamination.json"
        fresh = results / "contamination_recheck.json"
        receipt = results / "transition_receipt.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.v26_contamination",
                "generate",
                str(runtime),
                str(CONFIG),
                str(inventory),
                str(stored),
                str(fresh),
                str(manifest_path),
                str(receipt),
            ],
            cwd=ROOT,
            check=True,
        )
        contamination = _json(stored)
        admission = structural_admission(
            directory, ROOT, contamination, results / "structural_admission.json", role=role.upper()
        )
        current = _json(manifest_path)
        current.update(
            {
                "status": "ADMITTED_PENDING_TRANSITION" if admission["status"] == "ADMITTED" else "NO_PROMOTE",
                "dataset": dataset,
                "dataset_manifest_sha256": sha256_file(directory / "manifest.json"),
                "contamination_sha256": sha256_file(stored),
                "contamination_semantic_digest": contamination["semantic_digest"],
                "prior_inventory_sha256": sha256_file(inventory),
                "structural_admission": admission,
            }
        )
        atomic_write_json(manifest_path, current)
        if admission["status"] != "ADMITTED":
            raise RuntimeError(f"V26_SEALED_GENERATION_ADMISSION_INVALID:{admission['failures']}")
    except BaseException:
        current = _json(manifest_path)
        current["status"] = "NO_PROMOTE"
        atomic_write_json(manifest_path, current)
        raise


if __name__ == "__main__":
    main()
