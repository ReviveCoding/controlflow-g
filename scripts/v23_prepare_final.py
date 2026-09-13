from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now
from controlflow.v22.dgp import contamination_against_prior, generate_split, resolve_prior_runtime_paths

ROOT = Path(__file__).resolve().parents[1]


def _binding(path: Path) -> dict[str, int | str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    qualification_path = ROOT / "state/v23_qualification_manifest.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("status") != "PASS_POST_REVIEW_REQUIRED" or not qualification.get("all_passed"):
        raise RuntimeError("FINAL_PROHIBITED: V23QUAL did not pass")
    reviews_path = ROOT / "state/v23_review_findings.json"
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    if reviews.get("status") != "POST_QUALIFICATION_CLEAR" or reviews.get("counts") != {
        "unresolved_BLOCKER": 0,
        "unresolved_HIGH": 0,
    }:
        raise RuntimeError("FINAL_PROHIBITED: post-qualification review has not cleared")
    final_manifest_path = ROOT / "state/v23_final_manifest.json"
    if final_manifest_path.is_file():
        prior = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if prior.get("one_shot_opened") or prior.get("final_executed"):
            raise RuntimeError("V23 final one-shot namespace has already been opened")
    qualification_freeze = json.loads((ROOT / "state/v23_freeze_manifest.json").read_text(encoding="utf-8"))
    if qualification_freeze.get("status") != "QUALIFICATION_PROTOCOL_FROZEN":
        raise RuntimeError("qualification protocol freeze is missing")
    directory = ROOT / "data/v23/final/V23FINAL"
    prior_paths = resolve_prior_runtime_paths(
        ROOT,
        ROOT / "configs/v23/prior_runtime_manifest.yaml",
        excluded_directory=directory,
    )
    manifest = generate_split(directory, role="FINAL", count=500, seed=23999, prefix="V23FINAL")
    runtime_path = directory / "runtime_cases.parquet"
    contamination = contamination_against_prior(runtime_path, prior_paths)
    contamination_path = ROOT / "results/v23/final_contamination.json"
    atomic_write_json(contamination_path, contamination)
    if contamination["leakage_findings"]:
        atomic_write_json(
            final_manifest_path,
            {
                "schema_version": 1,
                "status": "INVALID_CONTAMINATED_NOT_EXECUTED",
                "one_shot_opened": False,
                "final_executed": False,
                "dataset": manifest,
                "contamination": contamination,
            },
        )
        raise RuntimeError("FINAL_PROHIBITED: contamination detected")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    final_bindings = [
        _binding(directory / "runtime_cases.parquet"),
        _binding(directory / "evaluator_truth.parquet"),
        _binding(directory / "evidence_corpus.parquet"),
        _binding(directory / "authorization_state.parquet"),
        _binding(directory / "manifest.json"),
        _binding(contamination_path),
        _binding(qualification_path),
        _binding(reviews_path),
        _binding(ROOT / "configs/v23/qualification_gates.yaml"),
        _binding(ROOT / "configs/v23/serving.yaml"),
        _binding(ROOT / "configs/v23/explanation_schema_minimal.json"),
        _binding(ROOT / "configs/v23/explanation_prompt_minimal.txt"),
        _binding(ROOT / "state/v23_typed_core_manifest.json"),
        _binding(ROOT / "uv.lock"),
    ]
    freeze_body = {
        **{key: value for key, value in qualification_freeze.items() if key not in {"freeze_hash", "created_at"}},
        "status": "FINAL_PROTOCOL_FROZEN",
        "source_commit": git_commit,
        "qualification_protocol_freeze_hash": qualification_freeze["freeze_hash"],
        "qualification_metrics_sha256": qualification["metrics_sha256"],
        "post_qualification_review_sha256": sha256_file(reviews_path),
        "final_dataset": manifest,
        "final_contamination": contamination,
        "final_bindings": final_bindings,
        "final_seed": 23999,
        "final_count": 500,
        "final_not_generated": False,
        "final_executed_at_freeze": False,
    }
    freeze_body["freeze_hash"] = hashlib.sha256(canonical_json(freeze_body)).hexdigest()
    freeze_body["created_at"] = utc_now()
    atomic_write_json(ROOT / "state/v23_freeze_manifest.json", freeze_body)
    atomic_write_json(
        final_manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "FINAL_DATASET_GENERATED_PROTOCOL_FROZEN",
            "one_shot_opened": False,
            "final_executed": False,
            "dataset": manifest,
            "contamination": contamination,
            "freeze_hash": freeze_body["freeze_hash"],
        },
    )


if __name__ == "__main__":
    main()
