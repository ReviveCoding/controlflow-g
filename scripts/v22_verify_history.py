from __future__ import annotations

import subprocess
from pathlib import Path

from controlflow.core.state import atomic_write_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("controlflow-g-v1-frozen", "controlflow-g-v2-no-go", "controlflow-g-v21-no-go")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    branch = _git("branch", "--show-current")
    if branch != "v2.2-development":
        raise RuntimeError(f"wrong branch: {branch}")
    tags = []
    for tag in TAGS:
        object_hash = _git("rev-parse", f"refs/tags/{tag}")
        peeled_hash = _git("rev-parse", f"{tag}^{{}}")
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", peeled_hash, "HEAD"], cwd=ROOT).returncode == 0
        if not ancestor:
            raise RuntimeError(f"historical tag is not an ancestor: {tag}")
        tags.append({"tag": tag, "object_hash": object_hash, "peeled_hash": peeled_hash, "ancestor_of_head": True})
    atomic_write_json(
        ROOT / "state/v22_historical_boundary.json",
        {
            "schema_version": 1,
            "verified_at": utc_now(),
            "branch": branch,
            "head": _git("rev-parse", "HEAD"),
            "tags": tags,
        },
    )


if __name__ == "__main__":
    main()
