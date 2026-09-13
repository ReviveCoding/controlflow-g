from __future__ import annotations

import hashlib
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.checkpoint import git_state

ROOT = Path(__file__).resolve().parents[1]


def test_source_hash() -> str:
    return hashlib.sha256(
        "\n".join(
            f"{path.relative_to(ROOT).as_posix()}:{sha256_file(path)}"
            for path in sorted((ROOT / "tests").rglob("test_*.py"))
        ).encode()
    ).hexdigest()


def main() -> None:
    junit = ROOT / "results/v22/v22_tests.xml"
    commit, dirty_hash = git_state(ROOT)
    atomic_write_json(
        ROOT / "state/v22_test_evidence.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty_state_hash": dirty_hash,
            "test_source_hash": test_source_hash(),
            "junit_path": junit.relative_to(ROOT).as_posix(),
            "junit_sha256": sha256_file(junit),
        },
    )


if __name__ == "__main__":
    main()
