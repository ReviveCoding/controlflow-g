from __future__ import annotations

import json
import subprocess
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("GATE_FREEZE_REQUIRES_CLEAN_COMMITTED_WORKTREE")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    reviews = json.loads((ROOT / "state/v22_review_findings.json").read_text(encoding="utf-8"))
    if reviews.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}:
        raise RuntimeError("GATE_FREEZE_REQUIRES_REVIEW_CLEARANCE")
    atomic_write_json(
        ROOT / "state/v22_qualification_gate_freeze.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "FROZEN_BEFORE_QUALIFICATION",
            "source_commit": commit,
            "gate_config_path": "configs/v22/qualification_gates.yaml",
            "gate_config_sha256": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
            "qualification_results_present_at_freeze": (ROOT / "results/v22/qualification_metrics.json").exists(),
        },
    )


if __name__ == "__main__":
    main()
