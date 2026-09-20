"""Freeze an admitted final identity after independent postqualification review."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v26.artifact_closure import binding
from controlflow.v26.closure_receipt import verify_receipt
from controlflow.v26.freeze_validation import verify_freeze
from controlflow.v26.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    output = ROOT / "state/v26_final_freeze_manifest.json"
    if output.exists():
        raise RuntimeError("V26_FINAL_FREEZE_ALREADY_CREATED")
    qualification_freeze = verify_freeze(ROOT, role="qualification")
    final_path = ROOT / "state/v26_final_manifest.json"
    final = _json(final_path)
    qualification_path = ROOT / "state/v26_qualification_manifest.json"
    qualification = _json(qualification_path)
    review_path = ROOT / "state/v26_postqualification_review_findings.json"
    review = _json(review_path)
    graph = ROOT / "state/v26_qualification_bindings.json"
    receipt = ROOT / "state/v26_qualification_closure_receipt.json"
    proposed = ROOT / "state/v26_qualification_manifest.proposed.json"
    anchor = ROOT / "state/v26_qualification_terminal_anchor.json"
    if (
        final.get("status") != "ADMITTED_PENDING_FREEZE"
        or final.get("structural_admission", {}).get("status") != "ADMITTED"
        or qualification.get("status") != "COMPLETE"
        or qualification.get("all_passed") is not True
        or qualification.get("freeze_hash") != qualification_freeze["freeze_hash"]
        or review.get("status") != "POST_QUALIFICATION_CLEAR"
        or review.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or review.get("qualification_manifest_sha256") != sha256_file(qualification_path)
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V26_FINAL_FREEZE_PRECONDITION_INVALID")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("V26_FINAL_EXECUTABLE_BOUNDARY_DIRTY")
    data = ROOT / "data/v26/final/V26FINAL"
    results = ROOT / "results/v26/final"
    if (
        sha256_file(data / "manifest.json") != final.get("dataset_manifest_sha256")
        or sha256_file(results / "contamination.json") != final.get("contamination_sha256")
        or sha256_file(results / "contamination_recheck.json") != final.get("fresh_contamination_sha256")
        or sha256_file(results / "transition_receipt.json") != final.get("transition_receipt_sha256")
    ):
        raise RuntimeError("V26_FINAL_GENERATION_BINDING_INVALID")
    paths = [
        ROOT / "state/v26_freeze_manifest.json",
        qualification_path,
        graph,
        receipt,
        proposed,
        anchor,
        ROOT / "state/v26_postclose_verification.json",
        ROOT / "results/v26/qualification/gate_decision.json",
        review_path,
        *sorted(data.glob("*")),
        results / "structural_admission.json",
        results / "contamination.json",
        results / "contamination_recheck.json",
        results / "prior_inventory.json",
        results / "transition_receipt.json",
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
        "qwen_model": qualification_freeze["qwen_model"],
        "qwen_revision": qualification_freeze["qwen_revision"],
        "vllm_version": qualification_freeze["vllm_version"],
        "serving": qualification_freeze["serving"],
        "bindings": [*qualification_freeze["bindings"], *[binding(ROOT, path) for path in sorted(set(paths))]],
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(output, manifest)
    final["freeze_hash"] = manifest["freeze_hash"]
    final["status"] = "ADMITTED_NOT_EXECUTED"
    atomic_write_json(final_path, final)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
