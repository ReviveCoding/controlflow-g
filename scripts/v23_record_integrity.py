from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as et
from datetime import datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.bundle import verify_bundle
from controlflow.v22.checkpoint import fingerprint, records_hash
from controlflow.v22.dgp import contamination_report
from controlflow.v22.executor import verify_ledger
from controlflow.v23.integrity import (
    file_binding,
    verify_file_bindings,
    verify_hash_bindings,
    verify_historical_boundary,
    verify_request_diagnostics,
)
from controlflow.v23.telemetry import PERIODIC_METRICS

ROOT = Path(__file__).resolve().parents[1]


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
    exact = verify_historical_boundary(ROOT, ROOT / "state/v23_historical_boundary.json")
    refs = {item["tag"]: item["object_hash"] for item in exact["observed"]}
    violations = list(exact["failures"])
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
    return {
        "refs": refs,
        "exact_identity": exact,
        "branch": branch,
        "v22_ancestor": ancestor,
        "violations": violations,
    }


def _interrupted_namespace() -> dict[str, Any]:
    binding_path = ROOT / "state/v23_interrupted_namespace_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    failures = verify_file_bindings(ROOT, binding["bindings"])
    marker = json.loads((ROOT / binding["bindings"][0]["path"]).read_text(encoding="utf-8"))
    ledger_path = ROOT / "artifacts/v23" / binding["namespace"] / "development.sqlite"
    ledger = verify_ledger(ledger_path)
    if marker.get("status") != "INCOMPLETE_NOT_ADMISSIBLE":
        failures.append("interrupted_marker_status_changed")
    if (
        marker.get("selection_use_permitted") is not False
        or marker.get("qualification_evidence_use_permitted") is not False
    ):
        failures.append("interrupted_marker_admission_changed")
    if ledger != binding["ledger"]:
        failures.append("interrupted_ledger_state_changed")
    return {"valid": not failures, "failures": failures, "binding": file_binding(ROOT, binding_path)}


def _cohort(config_id: str) -> dict[str, Any]:
    result_dir = ROOT / "results/v23" / config_id
    artifact_dir = ROOT / "artifacts/v23" / config_id
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = int(summary["requests"])
    checkpoint = _checkpoint(artifact_dir)
    ledger_path = artifact_dir / "development.sqlite"
    ledger = verify_ledger(ledger_path)
    diagnostics = verify_request_diagnostics(
        ROOT,
        artifact_dir / "request_diagnostics.partial.jsonl",
        artifact_dir / "development.partial.jsonl",
        result_dir / "candidate_results.parquet",
        result_dir / "request_attribution.json",
        expected,
    )
    summary_artifact_failures = verify_hash_bindings(ROOT, summary["artifacts"].values())
    telemetry_path = ROOT / summary["artifacts"]["telemetry"]["path"]
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    if telemetry.get("errors"):
        summary_artifact_failures.append("telemetry_sampling_errors")
    diagnostic_rows = [
        json.loads(line)
        for line in (artifact_dir / "request_diagnostics.partial.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    request_start = min(float(row["request_start"]) for row in diagnostic_rows)
    request_end = max(float(row["request_end"]) for row in diagnostic_rows)
    telemetry_epochs = [
        datetime.fromisoformat(row["sampled_at_utc"]).timestamp() for row in telemetry.get("samples", [])
    ]
    if not telemetry_epochs or min(telemetry_epochs) > request_start + 2 or max(telemetry_epochs) < request_end - 2:
        summary_artifact_failures.append("telemetry_measurement_window_coverage")
    periodic_binding = summary["artifacts"].get("periodic_metrics")
    periodic = None
    durable_sampler_bindings: list[dict[str, Any]] = []
    if periodic_binding is None:
        summary_artifact_failures.append("periodic_vllm_metrics_missing")
    else:
        periodic_path = ROOT / periodic_binding["path"]
        periodic = json.loads(periodic_path.read_text(encoding="utf-8"))
        if periodic.get("errors") or len(periodic.get("samples", [])) < 2:
            summary_artifact_failures.append("periodic_vllm_metrics_coverage_or_errors")
        periodic_epochs = [
            datetime.fromisoformat(row["captured_at"]).timestamp() for row in periodic.get("samples", [])
        ]
        if not periodic_epochs or min(periodic_epochs) > request_start + 6 or max(periodic_epochs) < request_end - 6:
            summary_artifact_failures.append("periodic_vllm_metrics_measurement_window_coverage")
        observed_names = {
            str(sample["name"]) for snapshot in periodic.get("samples", []) for sample in snapshot.get("samples", [])
        }
        missing_metrics = PERIODIC_METRICS - observed_names
        if missing_metrics:
            summary_artifact_failures.append(f"periodic_vllm_metrics_missing_names:{sorted(missing_metrics)}")
        for path in (
            periodic_path.with_suffix(periodic_path.suffix + ".partial.jsonl"),
            telemetry_path.with_suffix(telemetry_path.suffix + ".partial.jsonl"),
        ):
            if path.is_file():
                durable_sampler_bindings.append(file_binding(ROOT, path))
            else:
                summary_artifact_failures.append(f"durable_sampler_file_missing:{path.name}")
    violations = [
        *(f"checkpoint:{item}" for item in checkpoint["failures"]),
        *(f"ledger:{item}" for item in ledger["failures"]),
        *(f"diagnostics:{item}" for item in diagnostics["violations"]),
        *(f"summary_artifact:{item}" for item in summary_artifact_failures),
    ]
    return {
        "config_id": config_id,
        "valid": not violations,
        "violations": violations,
        "summary": file_binding(ROOT, summary_path),
        "checkpoint": checkpoint,
        "checkpoint_binding": file_binding(ROOT, artifact_dir / "development.checkpoint.json"),
        "ledger": {
            "valid": ledger["valid"],
            "event_count": ledger["event_count"],
            "head": ledger["head"],
            "failures": ledger["failures"],
            "binding": file_binding(ROOT, ledger_path),
        },
        "request_diagnostics": diagnostics,
        "summary_artifact_failures": summary_artifact_failures,
        "summary_artifact_bindings": [
            file_binding(ROOT, ROOT / item["path"]) for item in summary["artifacts"].values()
        ],
        "telemetry_sample_count": len(telemetry.get("samples", [])),
        "periodic_vllm_metric_sample_count": len(periodic.get("samples", [])) if periodic else 0,
        "durable_sampler_bindings": durable_sampler_bindings,
        "runtime_sha256": summary["provenance"]["runtime"]["sha256"],
        "runtime_path": summary["provenance"]["runtime"]["path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", required=True)
    parser.add_argument("--independent", required=True)
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
    cohorts = [_cohort(args.winner), _cohort(args.independent)]
    independent_runtime = cohorts[0]["runtime_sha256"] != cohorts[1]["runtime_sha256"]
    development_contamination = contamination_report(
        {
            "primary": ROOT / cohorts[0]["runtime_path"],
            "independent": ROOT / cohorts[1]["runtime_path"],
        }
    )
    bundle_violations = 0
    bundle_error = None
    try:
        verify_bundle(ROOT / "state/v22_model_bundle.json", ROOT)
    except Exception as exc:  # integrity boundary intentionally converts all verifier failures to evidence
        bundle_violations = 1
        bundle_error = f"{type(exc).__name__}: {exc}"
    historical = _historical_tags()
    interrupted = _interrupted_namespace()
    reviews = json.loads((ROOT / "state/v23_review_findings.json").read_text(encoding="utf-8"))
    eligible = all(
        (
            tests["failures"] == 0,
            tests["errors"] == 0,
            bundle_violations == 0,
            all(cohort["valid"] for cohort in cohorts),
            independent_runtime,
            development_contamination["leakage_findings"] == 0,
            not historical["violations"],
            interrupted["valid"],
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
        "checkpoint_violations": sum(cohort["checkpoint"]["violations"] for cohort in cohorts),
        "ledger_tamper_verification_failures": sum(len(cohort["ledger"]["failures"]) for cohort in cohorts),
        "request_diagnostic_violations": sum(len(cohort["request_diagnostics"]["violations"]) for cohort in cohorts),
        "development_cohorts": cohorts,
        "development_runtime_hashes_distinct": independent_runtime,
        "leakage_findings": development_contamination["leakage_findings"],
        "development_contamination": development_contamination,
        "historical_integrity": historical,
        "interrupted_namespace": interrupted,
    }
    payload["integrity_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    atomic_write_json(ROOT / "state/v23_integrity.json", payload)
    if not eligible:
        raise RuntimeError("V23 integrity is not qualification eligible")


if __name__ == "__main__":
    main()
