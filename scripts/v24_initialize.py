from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "0040359672c9cce826b16390feb0c0d9d91d58b2"
TAGS = (
    "controlflow-g-v1-frozen",
    "controlflow-g-v2-no-go",
    "controlflow-g-v21-no-go",
    "controlflow-g-v22-no-go",
    "controlflow-g-v23-no-go",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    if git("branch", "--show-current") != "v2.4-development":
        raise RuntimeError("HISTORICAL_BOUNDARY: incorrect branch")
    head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, head], cwd=ROOT).returncode:
        raise RuntimeError("HISTORICAL_BOUNDARY: HEAD does not descend from V2.3 terminal")
    tags: dict[str, dict[str, str]] = {}
    for tag in TAGS:
        tags[tag] = {"object_sha256_git": git("rev-parse", tag), "peeled_commit": git("rev-parse", f"{tag}^{{}}")}
    if tags[TAGS[-1]]["peeled_commit"] != EXPECTED_HEAD:
        raise RuntimeError("HISTORICAL_BOUNDARY: V2.3 tag points elsewhere")
    release = json.loads((ROOT / "state/v23_release_decision.json").read_text(encoding="utf-8"))
    execution = json.loads((ROOT / "state/v23_execution_state.json").read_text(encoding="utf-8"))
    if not (
        release["decision"] == "NO_PROMOTE"
        and release["final_generated"] is False
        and release["final_executed"] is False
        and release["qualification_one_shot_consumed"] is True
        and execution["v23qual_role"] == "DIAGNOSTIC_ONLY_PERMANENTLY_CONSUMED"
    ):
        raise RuntimeError("HISTORICAL_BOUNDARY: V23QUAL immutability evidence invalid")
    boundary: dict[str, Any] = {
        "schema_version": 1,
        "status": "VERIFIED",
        "branch": "v2.4-development",
        "starting_commit": EXPECTED_HEAD,
        "observed_head": head,
        "tags": tags,
        "v23qual_role": "DIAGNOSTIC_ONLY_PERMANENTLY_CONSUMED",
        "v23_release_decision_sha256": sha256_file(ROOT / "state/v23_release_decision.json"),
    }
    atomic_write_json(ROOT / "state/v24_historical_boundary.json", boundary)
    typed_path = ROOT / "state/v23_typed_core_manifest.json"
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    bindings = typed["bindings"]
    preserved = {
        name: row
        for name, row in bindings.items()
        if name not in {"executor_implementation", "evaluation_implementation"}
    }
    failures = [name for name, row in preserved.items() if sha256_file(ROOT / row["path"]) != row["sha256"]]
    if sha256_file(ROOT / typed["source_bundle"]["path"]) != typed["source_bundle"]["sha256"]:
        failures.append("source_bundle")
    if failures:
        raise RuntimeError(f"FROZEN_V23_CORE_CHANGED: {failures}")
    lifecycle_repairs = {
        name: {
            "path": row["path"],
            "v23_sha256": row["sha256"],
            "v24_sha256": sha256_file(ROOT / row["path"]),
            "reason": "explicit SQLite connection closure; independently audited correctness defect",
        }
        for name, row in bindings.items()
        if name in {"executor_implementation", "evaluation_implementation"}
    }
    candidate = {
        "schema_version": 1,
        "status": "V23_CORE_BOUND_WITH_LIFECYCLE_REPAIR",
        "v23_typed_manifest_sha256": sha256_file(typed_path),
        "preserved_bindings": preserved,
        "lifecycle_repairs": lifecycle_repairs,
        "source_bundle": typed["source_bundle"],
        "critical_threshold": typed["critical_threshold"],
        "embedding_revision": typed["embedding_revision"],
        "qwen_model": typed["qwen_model"],
        "qwen_revision": typed["qwen_revision"],
        "vllm_version": typed["vllm_version"],
    }
    candidate["manifest_sha256"] = hashlib.sha256(canonical_json(candidate)).hexdigest()
    atomic_write_json(ROOT / "state/v24_candidate_manifest.json", candidate)
    contract_path = ROOT / "configs/v24/denominator_contract.yaml"
    atomic_write_json(
        ROOT / "state/v24_denominator_contract.json",
        {"schema_version": 1, "status": "DRAFT_REVIEW_REQUIRED", "config_sha256": sha256_file(contract_path)},
    )
    for folder in ("data/v24", "results/v24", "artifacts/v24", "reports/v24", "tests/v24"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    for filename, status in (
        ("v24_integrity.json", "IN_PROGRESS"),
        ("v24_review_findings.json", "NOT_RUN"),
        ("v24_prequalification_review_findings.json", "NOT_RUN"),
        ("v24_freeze_manifest.json", "NOT_FROZEN"),
        ("v24_qualification_manifest.json", "NOT_GENERATED"),
        ("v24_release_decision.json", "PENDING"),
    ):
        target = ROOT / "state" / filename
        if not target.exists():
            atomic_write_json(target, {"schema_version": 1, "status": status})
    atomic_write_json(
        ROOT / "state/v24_execution_state.json",
        {
            "schema_version": 1,
            "status": "DEVELOPMENT_IN_PROGRESS",
            "current_phase": "V24-INTEGRITY-IMPLEMENTATION",
            "v23qual_role": "DIAGNOSTIC_ONLY_PERMANENTLY_CONSUMED",
            "qualification_generation_permitted": False,
            "qualification_executed": False,
            "final_generated": False,
            "final_executed": False,
            "updated_at": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
