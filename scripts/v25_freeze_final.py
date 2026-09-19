"""Freeze an admitted, fresh final dataset after qualification review."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v25.artifact_closure import binding
from controlflow.v25.closure_receipt import verify_receipt
from controlflow.v25.freeze_validation import verify_freeze
from controlflow.v25.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    output = ROOT / "state/v25_final_freeze_manifest.json"
    if output.exists():
        raise RuntimeError("V25_FINAL_FREEZE_ALREADY_CREATED")
    qualification_freeze = verify_freeze(ROOT)
    final_manifest = _json(ROOT / "state/v25_final_manifest.json")
    review_path = ROOT / "state/v25_postqualification_review_findings.json"
    review = _json(review_path)
    qualification_path = ROOT / "state/v25_qualification_manifest.json"
    qualification = _json(qualification_path)
    graph = ROOT / "state/v25_qualification_bindings.json"
    receipt = ROOT / "state/v25_qualification_closure_receipt.json"
    proposed = ROOT / "state/v25_qualification_manifest.proposed.json"
    anchor = ROOT / "state/v25_qualification_terminal_anchor.json"
    if (
        final_manifest.get("status") != "ADMITTED_PENDING_FREEZE"
        or final_manifest.get("structural_admission", {}).get("status") != "ADMITTED"
        or qualification.get("status") != "COMPLETE"
        or qualification.get("all_passed") is not True
        or qualification.get("freeze_hash") != qualification_freeze["freeze_hash"]
        or review.get("status") != "POST_QUALIFICATION_CLEAR"
        or review.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or review.get("qualification_manifest_sha256") != sha256_file(qualification_path)
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V25_FINAL_FREEZE_PRECONDITION_INVALID")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("V25_FINAL_EXECUTABLE_DIRTY")
    data = ROOT / "data/v25/final/V25FINAL"
    results = ROOT / "results/v25/final"
    if sha256_file(data / "manifest.json") != final_manifest.get("dataset_manifest_sha256") or sha256_file(
        results / "contamination.json"
    ) != final_manifest.get("contamination_sha256"):
        raise RuntimeError("V25_FINAL_DATASET_BINDING_INVALID")
    paths = [
        ROOT / "state/v25_freeze_manifest.json",
        qualification_path,
        graph,
        receipt,
        proposed,
        anchor,
        ROOT / "state/v25_postclose_verification.json",
        ROOT / "results/v25/qualification/gate_decision.json",
        review_path,
        *sorted(data.glob("*")),
        results / "structural_admission.json",
        results / "contamination.json",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "role": "FINAL",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "qualification_freeze_hash": qualification_freeze["freeze_hash"],
        "qualification_manifest_sha256": sha256_file(qualification_path),
        "checkpoint_contract_sha256": qualification_freeze["checkpoint_contract_sha256"],
        "checkpoint_schema_version": qualification_freeze["checkpoint_schema_version"],
        "final_dataset_manifest_sha256": sha256_file(data / "manifest.json"),
        "gate_config_sha256": qualification_freeze["gate_config_sha256"],
        "bindings": [*qualification_freeze["bindings"], *[binding(ROOT, path) for path in sorted(set(paths))]],
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(output, manifest)
    final_manifest["freeze_hash"] = manifest["freeze_hash"]
    final_manifest["status"] = "ADMITTED_NOT_EXECUTED"
    atomic_write_json(ROOT / "state/v25_final_manifest.json", final_manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
