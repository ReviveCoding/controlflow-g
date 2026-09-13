from __future__ import annotations

import os
import subprocess

from controlflow.core.state import ProjectPaths, atomic_write_json, utc_now


def main() -> None:
    paths = ProjectPaths.discover()
    qualification_temp = paths.root / "build" / f"v2-qualification-{os.getpid()}"
    commands = [
        [str(paths.root / ".venv/Scripts/python.exe"), "-m", "ruff", "check", "src", "tests", "scripts"],
        [str(paths.root / ".venv/Scripts/python.exe"), "-m", "mypy", "src/controlflow/v2"],
        [
            str(paths.root / ".venv/Scripts/python.exe"),
            "-m",
            "pytest",
            "-q",
            "--basetemp",
            str(qualification_temp),
        ],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=paths.root, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4_000:],
                "stderr": completed.stderr[-4_000:],
            }
        )
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "passed": all(item["returncode"] == 0 for item in results),
        "checks": results,
    }
    target = paths.state / "v2_quality_checks.json"
    atomic_write_json(target, record)
    print(target)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
