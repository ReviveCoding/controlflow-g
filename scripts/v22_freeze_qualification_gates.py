from __future__ import annotations

import subprocess
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("GATE_FREEZE_REQUIRES_CLEAN_COMMITTED_WORKTREE")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if (ROOT / "results/v22/qualification_metrics.json").exists():
        raise RuntimeError("GATE_FREEZE_MUST_PRECEDE_QUALIFICATION_RESULTS")
    atomic_write_json(
        ROOT / "configs/v22/qualification_gate_freeze.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "FROZEN_BEFORE_QUALIFICATION",
            "source_commit": commit,
            "gate_config_path": "configs/v22/qualification_gates.yaml",
            "gate_config_sha256": sha256_file(ROOT / "configs/v22/qualification_gates.yaml"),
            "qualification_results_present_at_freeze": False,
        },
    )


if __name__ == "__main__":
    main()
