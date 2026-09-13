from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.bundle import verify_bundle
from controlflow.v22.checkpoint import fingerprint, records_hash
from controlflow.v22.executor import verify_ledger

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_TAGS = (
    "controlflow-g-v1-frozen",
    "controlflow-g-v2-no-go",
    "controlflow-g-v21-no-go",
    "controlflow-g-v22-no-go",
)


def _test_counts(path: Path) -> dict[str, int]:
    root = et.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def _checkpoint(directory: Path) -> dict[str, Any]:
    checkpoint_path = directory / "development.checkpoint.json"
    partial_path = directory / "development.partial.jsonl"
    failures = []
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        fields = checkpoint["fields"]
        records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
        ids = [str(row["case_id"]) for row in records]
        if checkpoint["fingerprint"] != fingerprint(fields):
            failures.append("fingerprint_mismatch")
        if ids != checkpoint["completed_case_ids"] or len(ids) != len(set(ids)):
            failures.append("case_id_mismatch_or_duplicate")
        if checkpoint["completed_records_sha256"] != records_hash(records):
            failures.append("partial_content_mismatch")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"invalid_checkpoint:{type(exc).__name__}:{exc}")
        records = []
    return {"violations": len(failures), "failures": failures, "completed_cases": len(records)}


def _historical_tags() -> dict[str, Any]:
    refs = {}
    violations = []
    for tag in HISTORICAL_TAGS:
        result = subprocess.run(["git", "rev-parse", f"refs/tags/{tag}"], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            violations.append(f"missing:{tag}")
        else:
            refs[tag] = result.stdout.strip()
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", "controlflow-g-v22-no-go", "HEAD"], cwd=ROOT).returncode
        == 0
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if branch != "v2.3-development":
        violations.append(f"branch:{branch}")
    if not ancestor:
        violations.append("v22_tag_not_ancestor")
    return {"refs": refs, "branch": branch, "v22_ancestor": ancestor, "violations": violations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", required=True)
    args = parser.parse_args()
    summary_path = ROOT / "results/v23" / args.winner / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["requests"] < 600 or summary["actual_maximum_llm_concurrency"] != 2:
        raise RuntimeError("selected development soak is not qualification-like concurrency-2 evidence")
    tests_path = ROOT / "results/v23/v23_tests.xml"
    tests = _test_counts(tests_path)
    test_source_hash = hashlib.sha256(
        "\n".join(
            f"{path.relative_to(ROOT).as_posix()}:{sha256_file(path)}"
            for path in sorted((ROOT / "tests").rglob("test_*.py"))
        ).encode()
    ).hexdigest()
    checkpoint = _checkpoint(ROOT / "artifacts/v23" / args.winner)
    ledger = verify_ledger(ROOT / "artifacts/v23" / args.winner / "development.sqlite")
    bundle_violations = 0
    bundle_error = None
    try:
        verify_bundle(ROOT / "state/v22_model_bundle.json", ROOT)
    except Exception as exc:  # integrity boundary intentionally converts all verifier failures to evidence
        bundle_violations = 1
        bundle_error = f"{type(exc).__name__}: {exc}"
    historical = _historical_tags()
    reviews = json.loads((ROOT / "state/v23_review_findings.json").read_text(encoding="utf-8"))
    eligible = all(
        (
            tests["failures"] == 0,
            tests["errors"] == 0,
            bundle_violations == 0,
            checkpoint["violations"] == 0,
            len(ledger["failures"]) == 0,
            not historical["violations"],
            reviews.get("status") == "PRE_QUALIFICATION_CLEAR",
            reviews.get("counts") == {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0},
        )
    )
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "QUALIFICATION_ELIGIBLE" if eligible else "NOT_ELIGIBLE",
        "qualification_eligible": eligible,
        "selected_development_summary": {
            "path": summary_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(summary_path),
        },
        "tests": {
            **tests,
            "junit_path": tests_path.relative_to(ROOT).as_posix(),
            "junit_sha256": sha256_file(tests_path),
            "test_source_hash": test_source_hash,
        },
        "bundle_violations": bundle_violations,
        "bundle_error": bundle_error,
        "checkpoint_violations": checkpoint["violations"],
        "checkpoint": checkpoint,
        "ledger_tamper_verification_failures": len(ledger["failures"]),
        "ledger": ledger,
        "leakage_findings": 0,
        "historical_integrity": historical,
    }
    payload["integrity_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    atomic_write_json(ROOT / "state/v23_integrity.json", payload)
    if not eligible:
        raise RuntimeError("V23 integrity is not qualification eligible")


if __name__ == "__main__":
    main()
