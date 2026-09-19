"""Open one final holdout only after sealed V25QUAL and independent review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v25.admission import structural_admission
from controlflow.v25.closure_receipt import verify_receipt
from controlflow.v25.contamination import check_contamination
from controlflow.v25.freeze_validation import verify_freeze
from controlflow.v25.generator import generate_v25_split
from controlflow.v25.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    qualification_freeze = verify_freeze(ROOT)
    directory = ROOT / "data/v25/final/V25FINAL"
    manifest_path = ROOT / "state/v25_final_manifest.json"
    if (
        directory.exists()
        or manifest_path.exists()
        or (ROOT / "results/v25/final").exists()
        or (ROOT / "artifacts/v25/final").exists()
        or any((ROOT / "state").glob("v25_final*.json"))
    ):
        raise RuntimeError("V25FINAL_ALREADY_GENERATED_OR_CONSUMED")
    qualification_path = ROOT / "state/v25_qualification_manifest.json"
    qualification = _json(qualification_path)
    review = _json(ROOT / "state/v25_postqualification_review_findings.json")
    graph = ROOT / "state/v25_qualification_bindings.json"
    receipt = ROOT / "state/v25_qualification_closure_receipt.json"
    proposed = ROOT / "state/v25_qualification_manifest.proposed.json"
    anchor = ROOT / "state/v25_qualification_terminal_anchor.json"
    postclose = _json(ROOT / "state/v25_postclose_verification.json")
    decision = _json(ROOT / "results/v25/qualification/gate_decision.json")
    if (
        qualification.get("status") != "COMPLETE"
        or qualification.get("all_passed") is not True
        or qualification.get("freeze_hash") != qualification_freeze["freeze_hash"]
        or decision.get("all_passed") is not True
        or postclose.get("status") != "PASS"
        or review.get("status") != "POST_QUALIFICATION_CLEAR"
        or review.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or review.get("qualification_manifest_sha256") != sha256_file(qualification_path)
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V25FINAL_REQUIRES_SEALED_QUALIFICATION_AND_REVIEW")
    config = yaml.safe_load((ROOT / "configs/v25/strata.yaml").read_text(encoding="utf-8"))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "GENERATION_OPENED",
            "created_at": utc_now(),
            "one_shot_opened": True,
            "final_executed": False,
            "qualification_freeze_hash": qualification_freeze["freeze_hash"],
            "dataset_identity": "V25FINAL",
        },
    )
    try:
        dataset = generate_v25_split(
            directory, role="FINAL", seed=int(config["final_seed"]), root=ROOT, prefix="V25FINAL"
        )
        contamination = check_contamination(ROOT, directory / "runtime_cases.parquet")
        results = ROOT / "results/v25/final"
        results.mkdir(parents=True, exist_ok=True)
        contamination_path = results / "contamination.json"
        atomic_write_json(contamination_path, contamination)
        admission = structural_admission(
            directory, ROOT, contamination, results / "structural_admission.json", role="FINAL"
        )
        status = "ADMITTED_PENDING_FREEZE" if admission["status"] == "ADMITTED" else "V25_GENERATION_INVALID"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": status,
                "one_shot_opened": True,
                "final_executed": False,
                "qualification_freeze_hash": qualification_freeze["freeze_hash"],
                "dataset_identity": "V25FINAL",
                "dataset_manifest_sha256": sha256_file(directory / "manifest.json"),
                "dataset": dataset,
                "structural_admission": admission,
                "contamination_sha256": sha256_file(contamination_path),
            },
        )
        if status != "ADMITTED_PENDING_FREEZE":
            raise RuntimeError(f"V25_GENERATION_INVALID:{admission['failures']}")
    except BaseException:
        current = _json(manifest_path)
        if current["status"] == "GENERATION_OPENED":
            current["status"] = "V25_GENERATION_INVALID"
            atomic_write_json(manifest_path, current)
        raise


if __name__ == "__main__":
    main()
