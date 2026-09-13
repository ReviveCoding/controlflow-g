from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.bundle import verify_bundle
from controlflow.v22.checkpoint import git_state

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("FINAL_FREEZE_REQUIRES_CLEAN_WORKTREE")
    final_manifest = json.loads((ROOT / "state/v22_final_manifest.json").read_text(encoding="utf-8"))
    if final_manifest.get("status") != "GENERATED_SEALED_NOT_RUN":
        raise RuntimeError("FINAL_FREEZE_PROHIBITED: sealed final is absent")
    bundle = verify_bundle(ROOT / "state/v22_model_bundle.json", ROOT)
    commit, dirty_hash = git_state(ROOT)
    bindings: dict[str, Any] = {
        "git_commit": commit,
        "dirty_state_hash": dirty_hash,
        "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "candidate_bundle_sha256": sha256_file(ROOT / "state/v22_model_bundle.json"),
        "candidate_bundle_hash": bundle["bundle_sha256"],
        "qualification_gates_sha256": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
        "final_runtime_sha256": final_manifest["dataset"]["runtime_sha256"],
        "final_truth_sha256": final_manifest["dataset"]["truth_sha256"],
        "final_evidence_sha256": final_manifest["dataset"]["evidence_sha256"],
        "final_authorization_sha256": final_manifest["dataset"]["authorization_sha256"],
        "integrity_sha256": sha256_file(ROOT / "state/v22_integrity.json"),
        "postqualification_review_sha256": sha256_file(ROOT / "state/v22_postqualification_review.json"),
    }
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "FROZEN_FINAL_NOT_RUN",
        "bindings": bindings,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(ROOT / "state/v22_freeze_manifest.json", manifest)


if __name__ == "__main__":
    main()
