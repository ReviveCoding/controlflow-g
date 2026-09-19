"""Run and record the complete V2.4 static and pytest gate."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".venv/Scripts"


def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    scripts = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "scripts").glob("v24_*.py"))]
    temp = ROOT / "results/v24" / f"pytest_admission_{run_id}"
    jobs = [
        ("ruff_format", [str(BIN / "ruff.exe"), "format", "--check", "src", "scripts", "tests"]),
        ("ruff_check", [str(BIN / "ruff.exe"), "check", "src", "scripts", "tests"]),
        ("strict_mypy", [str(BIN / "mypy.exe"), "--strict", "src/controlflow", *scripts]),
        (
            "full_pytest",
            [
                str(BIN / "pytest.exe"),
                "-q",
                "--junitxml=results/v24/v24_tests.xml",
                f"--basetemp={temp.relative_to(ROOT).as_posix()}",
            ],
        ),
    ]
    outcomes: dict[str, Any] = {}
    for name, argv in jobs:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        outcomes[name] = {
            "argv": argv,
            "exit_code": result.returncode,
            "output_tail": (result.stdout + result.stderr)[-3000:],
        }
        if result.returncode:
            break
    passed = len(outcomes) == len(jobs) and all(row["exit_code"] == 0 for row in outcomes.values())
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "run_id": run_id,
        "checks": outcomes,
        "junit_sha256": sha256_file(ROOT / "results/v24/v24_tests.xml")
        if passed and (ROOT / "results/v24/v24_tests.xml").is_file()
        else None,
    }
    atomic_write_json(ROOT / "state/v24_static_quality.json", report)
    print(report["status"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
