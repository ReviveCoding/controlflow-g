from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TAGS = {
    "controlflow-g-v1-frozen": (
        "26c6069091cea22f9964d64f360de756b5840c73",
        "80678792c3dbadeffb78902e65a14bdaafe6200d",
    ),
    "controlflow-g-v2-no-go": (
        "16b1be5ffcfaedc45364160439c490bcab4012d2",
        "2d21a70c1585669a456929c554bdbfc997f708ac",
    ),
    "controlflow-g-v21-no-go": (
        "3938c03ef2a15da35ab76e1372b8fcbd82a2a137",
        "532c181b47ed56703130d54d73db4cc12cf4e046",
    ),
    "controlflow-g-v22-no-go": (
        "86a4556d92ef2144c49b2c5ed3301a47e0773a7f",
        "7b41478d60696200db79b4a967821f37d20b1f42",
    ),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_history() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    if branch != "v2.3-development":
        raise RuntimeError(f"wrong branch: {branch}")
    tags = []
    for name, (expected_object, expected_peeled) in EXPECTED_TAGS.items():
        object_hash = git("rev-parse", name)
        peeled_hash = git("rev-parse", f"{name}^{{}}")
        ancestor = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", peeled_hash, "HEAD"], cwd=ROOT, capture_output=True
            ).returncode
            == 0
        )
        if (object_hash, peeled_hash) != (expected_object, expected_peeled) or not ancestor:
            raise RuntimeError(f"historical tag integrity failure: {name}")
        tags.append(
            {
                "tag": name,
                "object_hash": object_hash,
                "peeled_hash": peeled_hash,
                "ancestor_of_head": ancestor,
            }
        )
    return {
        "schema_version": 1,
        "verified_at": utc_now(),
        "branch": branch,
        "head": git("rev-parse", "HEAD"),
        "tags": tags,
    }


def _binding(path: str) -> dict[str, str]:
    target = ROOT / path
    if not target.is_file():
        raise RuntimeError(f"required V2.2 core artifact absent: {path}")
    return {"path": path, "sha256": sha256_file(target)}


def freeze_core() -> dict[str, Any]:
    v22 = json.loads((ROOT / "state/v22_model_bundle.json").read_text(encoding="utf-8"))
    bindings = {
        name: v22[name]
        for name in (
            "critical_model",
            "calibrator",
            "noncritical_model",
            "root_model",
            "novelty_model",
            "retrieval_config",
            "temporal_config",
            "policy_config",
            "action_registry",
        )
    }
    bindings.update(
        {
            "approval_verification_logic": _binding("src/controlflow/v22/approval.py"),
            "executor_implementation": _binding("src/controlflow/v22/executor.py"),
            "candidate_implementation": _binding("src/controlflow/v22/candidate.py"),
            "evaluation_implementation": _binding("src/controlflow/v22/evaluation.py"),
        }
    )
    for name, binding in bindings.items():
        if sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"V2.2 core hash mismatch: {name}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "V22_TYPED_DECISION_CORE_REUSED_EXACTLY",
        "source_bundle": _binding("state/v22_model_bundle.json"),
        "critical_threshold": v22["critical_threshold"],
        "embedding_revision": v22["embedding_revision"],
        "qwen_model": v22["qwen_model"],
        "qwen_revision": v22["qwen_revision"],
        "vllm_version": v22["vllm_version"],
        "tabular_inference_device": "cpu",
        "bindings": bindings,
        "selection_uses_v22qual_labels": False,
    }
    return payload


def historical_failures() -> dict[str, Any]:
    results_path = ROOT / "results/v22/qualification_candidate_results.parquet"
    frame = pd.read_parquet(results_path)
    failed = frame[frame.structured_output_valid == False]  # noqa: E712
    rows = []
    for row in failed.to_dict(orient="records"):
        rationale = str(row["rationale"])
        if rationale.startswith("{"):
            mechanism = "JSON_PARSE"
            basis = "returned text is visibly malformed JSON in diagnostic-only candidate output"
        else:
            mechanism = "OTHER_UNRESOLVED"
            basis = "fallback-shaped output lacks retained client diagnostics; no stronger attribution is claimed"
        rows.append(
            {
                "case_id": row["case_id"],
                "mechanism": mechanism,
                "basis": basis,
                "llm_latency_seconds": float(row["llm_latency_seconds"]),
                "total_latency_seconds": float(row["total_latency_seconds"]),
            }
        )
    manifest = json.loads((ROOT / "state/v22_qualification_manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "role": "HISTORICAL_DIAGNOSTIC_ONLY",
        "prohibited_uses": ["model_selection", "qualification_evidence", "fresh_evidence", "warmup"],
        "source": _binding("results/v22/qualification_candidate_results.parquet"),
        "source_manifest": _binding("state/v22_qualification_manifest.json"),
        "source_status": manifest["status"],
        "observed_failure_count": len(rows),
        "failures": rows,
    }


def main() -> None:
    for relative in ("results/v23", "artifacts/v23", "reports/v23", "data/v23"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    history = verify_history()
    core = freeze_core()
    atomic_write_json(ROOT / "state/v23_historical_boundary.json", history)
    atomic_write_json(ROOT / "state/v23_typed_core_manifest.json", core)
    atomic_write_json(ROOT / "results/v23/v22qual_structured_failure_diagnostics.json", historical_failures())
    empty = {"schema_version": 1, "created_at": utc_now(), "status": "NOT_RUN"}
    for name in (
        "v23_latency_attribution.json",
        "v23_serving_tournament.json",
        "v23_review_findings.json",
        "v23_integrity.json",
        "v23_qualification_manifest.json",
        "v23_freeze_manifest.json",
    ):
        path = ROOT / "state" / name
        if not path.exists():
            atomic_write_json(path, empty)
    atomic_write_json(
        ROOT / "state/v23_execution_state.json",
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "status": "V23_DEVELOPMENT_IN_PROGRESS",
            "current_phase": "V23-00",
            "completed_phases": ["V23-HISTORICAL-INTEGRITY", "V23-TYPED-CORE-FREEZE"],
            "v22qual_role": "DIAGNOSTIC_ONLY_PERMANENTLY_CONSUMED",
            "qualification_generation_permitted": False,
            "qualification_executed": False,
            "final_generation_permitted": False,
            "final_executed": False,
            "release_decision": None,
        },
    )


if __name__ == "__main__":
    main()
