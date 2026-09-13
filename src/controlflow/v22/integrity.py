from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.bundle import verify_bundle
from controlflow.v22.checkpoint import fingerprint, git_state
from controlflow.v22.evaluation import evaluator_protocol_hash
from controlflow.v22.executor import verify_ledger


def verify_integrity_report(path: Path, *, root: Path, ledger_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("integrity_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if claimed != actual:
        raise RuntimeError("INTEGRITY_REPORT_HASH_MISMATCH")
    recomputed = derive_integrity(
        root,
        ledger_path=ledger_path,
        created_at=str(payload["created_at"]),
        write=False,
    )
    if canonical_json(recomputed) != canonical_json(payload):
        raise RuntimeError("INTEGRITY_REPORT_EVIDENCE_MISMATCH")
    if not payload.get("qualification_eligible"):
        raise RuntimeError("QUALIFICATION_PROHIBITED: integrity report is not eligible")
    return cast(dict[str, Any], payload)


def _checkpoint_evidence(root: Path, checkpoint_path: Path, partial_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    if not checkpoint_path.is_file() or not partial_path.is_file():
        return {"violations": 1, "failures": ["checkpoint_or_partial_missing"]}
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    fields = checkpoint.get("fields", {})
    try:
        if checkpoint.get("fingerprint") != fingerprint(fields):
            failures.append("stored_fingerprint_invalid")
    except (TypeError, ValueError) as exc:
        failures.append(f"fingerprint_fields_invalid:{exc}")
    bundle_path = root / "state/v22_model_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "state/v22_dataset_manifest.json").read_text(encoding="utf-8"))
    validation = manifest["splits"]["VALIDATION"]
    git_commit, dirty_hash = git_state(root)
    expected = dict(fields)
    expected.update(
        {
            "git_commit": git_commit,
            "dirty_state_hash": dirty_hash,
            "dependency_lock_hash": sha256_file(root / "uv.lock"),
            "runtime_dataset_hash": sha256_file(root / validation["runtime_path"]),
            "evidence_corpus_hash": sha256_file(root / validation["evidence_path"]),
            "authorization_state_hash": sha256_file(root / validation["authorization_path"]),
            "candidate_bundle_hash": sha256_file(bundle_path),
            "model_hashes": {
                name: bundle[name]["sha256"]
                for name in ("critical_model", "calibrator", "noncritical_model", "root_model", "novelty_model")
            },
            "critical_threshold": bundle["critical_threshold"],
            "qwen_revision": bundle["qwen_revision"],
            "vllm_version": bundle["vllm_version"],
            "structured_backend": bundle["structured_output_backend"],
            "prompt_schema_hashes": {
                "prompt": bundle["prompt"]["sha256"],
                "schema": bundle["schema"]["sha256"],
            },
            "retrieval_config_hash": bundle["retrieval_config"]["sha256"],
            "temporal_config_hash": bundle["temporal_config"]["sha256"],
            "pdp_action_registry_hashes": {
                "pdp": bundle["policy_config"]["sha256"],
                "actions": bundle["action_registry"]["sha256"],
            },
            "evaluator_protocol_hash": evaluator_protocol_hash(root),
            "gate_config_hash": sha256_file(root / "configs/v22/qualification_gates.yaml"),
            "seed": validation["seed"],
            "concurrency": 1,
        }
    )
    try:
        if fingerprint(expected) != checkpoint.get("fingerprint"):
            failures.append("checkpoint_not_current")
    except (TypeError, ValueError) as exc:
        failures.append(f"current_fingerprint_invalid:{exc}")
    partial = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
    partial_ids = [str(row["case_id"]) for row in partial]
    if partial_ids != checkpoint.get("completed_case_ids") or len(partial_ids) != len(set(partial_ids)):
        failures.append("partial_checkpoint_cardinality_mismatch")
    return {"violations": len(failures), "failures": failures, "completed_cases": len(partial_ids)}


def _test_counts(path: Path) -> dict[str, int]:
    root = et.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def derive_integrity(
    root: Path,
    *,
    ledger_path: Path,
    created_at: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    dataset_manifest = json.loads((root / "state/v22_dataset_manifest.json").read_text(encoding="utf-8"))
    contamination = json.loads((root / dataset_manifest["contamination_report"]["path"]).read_text(encoding="utf-8"))
    reviews = json.loads((root / "state/v22_review_findings.json").read_text(encoding="utf-8"))
    tests_path = root / "results/v22/v22_tests.xml"
    tests = _test_counts(tests_path)
    test_source_hash = hashlib.sha256(
        "\n".join(
            f"{path.relative_to(root).as_posix()}:{sha256_file(path)}"
            for path in sorted((root / "tests").rglob("test_*.py"))
        ).encode()
    ).hexdigest()
    bundle_violations = 0
    bundle_error = None
    try:
        verify_bundle(root / "state/v22_model_bundle.json", root)
    except Exception as exc:
        bundle_violations = 1
        bundle_error = f"{type(exc).__name__}: {exc}"
    checkpoint_path = ledger_path.parent / "validation.checkpoint.json"
    partial_path = ledger_path.parent / "validation.partial.jsonl"
    checkpoint = _checkpoint_evidence(root, checkpoint_path, partial_path)
    checkpoint_violations = int(checkpoint["violations"])
    ledger = verify_ledger(ledger_path)
    serving_path = root / "state/v22_vllm_structured_c2.json"
    serving = json.loads(serving_path.read_text(encoding="utf-8")) if serving_path.exists() else None
    evidence = {
        "contamination": contamination,
        "tests": {
            **tests,
            "junit_sha256": sha256_file(tests_path),
            "test_source_hash": test_source_hash,
        },
        "bundle": {"violations": bundle_violations, "error": bundle_error},
        "checkpoint": checkpoint,
        "ledger": ledger,
        "reviews": reviews.get("counts", {}),
        "serving": None
        if serving is None
        else {
            "actual_live_vllm": serving["actual_live_vllm"],
            "total_structured_failures": serving["total_structured_failures"],
            "requests": serving["requests"],
            "concurrency": serving["concurrency"],
        },
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at or utc_now(),
        "evidence": evidence,
        "leakage_findings": int(contamination["leakage_findings"]) + int(contamination["runtime_schema_violations"]),
        "bundle_violations": bundle_violations,
        "checkpoint_violations": checkpoint_violations,
        "pdp_pep_security_test_failures": tests["failures"] + tests["errors"],
        "ledger_tamper_verification_failures": len(ledger["failures"]),
        "serving_evidence_present": serving is not None,
        "unresolved_blocker": int(reviews.get("counts", {}).get("unresolved_BLOCKER", 0)),
        "unresolved_high": int(reviews.get("counts", {}).get("unresolved_HIGH", 0)),
    }
    eligibility_fields = (
        "leakage_findings",
        "bundle_violations",
        "checkpoint_violations",
        "pdp_pep_security_test_failures",
        "ledger_tamper_verification_failures",
        "unresolved_blocker",
        "unresolved_high",
    )
    serving_valid = bool(
        serving
        and serving.get("actual_live_vllm") is True
        and serving.get("concurrency") == 2
        and int(serving.get("requests", 0)) > 0
        and int(serving.get("failures", {}).get("http_failure", -1)) == 0
    )
    report["serving_evidence_valid"] = serving_valid
    report["qualification_eligible"] = all(report[field] == 0 for field in eligibility_fields) and serving_valid
    report["integrity_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    if write:
        atomic_write_json(root / "state/v22_integrity.json", report)
    return report
