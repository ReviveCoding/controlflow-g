"""Freeze the reviewed executable qualification boundary before V24QUAL exists."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v24.artifact_closure import binding

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if (ROOT / "data/v24/qualification/V24QUAL").exists():
        raise RuntimeError("FREEZE_PROHIBITED: V24QUAL already exists")
    reviews = json.loads((ROOT / "state/v24_prequalification_review_findings.json").read_text(encoding="utf-8"))
    static = json.loads((ROOT / "state/v24_static_quality.json").read_text(encoding="utf-8"))
    development = json.loads((ROOT / "results/v24/development_generator_validation.json").read_text(encoding="utf-8"))
    gates = yaml.safe_load((ROOT / "configs/v24/qualification_gates.yaml").read_text(encoding="utf-8"))
    if (
        reviews.get("status") != "PRE_QUALIFICATION_CLEAR"
        or reviews.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or static.get("status") != "PASS"
        or len(development.get("outcomes", [])) < 3
        or any(row["status"] != "ADMITTED" for row in development["outcomes"])
        or gates.get("frozen_before_qualification") is not True
    ):
        raise RuntimeError("FREEZE_PROHIBITED: prequalification gate incomplete")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("FREEZE_PROHIBITED: executable boundary not committed")
    candidate = json.loads((ROOT / "state/v24_candidate_manifest.json").read_text(encoding="utf-8"))
    source_paths = [
        *sorted((ROOT / "src/controlflow/v24").glob("*.py")),
        *sorted((ROOT / "src/controlflow/v22").glob("*.py")),
        *sorted((ROOT / "src/controlflow/v23").glob("*.py")),
        *sorted((ROOT / "scripts").glob("v24_*.py")),
        ROOT / "scripts/v24_server.sh",
        *sorted((ROOT / "configs/v24").glob("*.yaml")),
        *sorted((ROOT / "tests/v24").glob("*.py")),
        ROOT / "src/controlflow/core/state.py",
        ROOT / "src/controlflow/v22/executor.py",
        ROOT / "src/controlflow/v22/evaluation.py",
        ROOT / "configs/v23/serving.yaml",
        ROOT / "configs/v23/explanation_prompt_minimal.txt",
        ROOT / "configs/v23/explanation_schema_minimal.json",
        ROOT / "scripts/v23_benchmark.py",
        ROOT / "state/v24_candidate_manifest.json",
        ROOT / "state/v24_historical_boundary.json",
        ROOT / "state/v24_environment_manifest.json",
        ROOT / "state/v24_wsl_pip_freeze.txt",
        ROOT / "state/v24_prequalification_review_findings.json",
        ROOT / "state/v24_static_quality.json",
        ROOT / "results/v24/development_generator_validation.json",
        ROOT / "results/v24/development_regression/summary.json",
        ROOT / "results/v24/v24_tests.xml",
        ROOT / "tests/v22/test_data_bundle_checkpoint.py",
        ROOT / "uv.lock",
    ]
    source_paths += [ROOT / row["path"] for row in candidate["preserved_bindings"].values()]
    source_paths.append(ROOT / candidate["source_bundle"]["path"])
    unique_paths = sorted(set(source_paths))
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "source_commit": source_commit,
        "bindings": [binding(ROOT, path) for path in unique_paths],
        "candidate_manifest_sha256": sha256_file(ROOT / "state/v24_candidate_manifest.json"),
        "denominator_contract_sha256": sha256_file(ROOT / "configs/v24/denominator_contract.yaml"),
        "gate_config_sha256": sha256_file(ROOT / "configs/v24/qualification_gates.yaml"),
        "qwen_model": serving["model"],
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "serving": {
            "dtype": serving["dtype"],
            "structured_output_backend": serving["structured_output_backend"],
            "performance_mode": "interactivity",
            "optimization_level": 2,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 2,
            "max_model_len": 4096,
            "max_tokens": 128,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "host": serving["host"],
            "port": serving["port"],
        },
        "warmup_request_count": 8,
        "qualification_count": 600,
        "qualification_seed": 24041,
        "final_seed": 24042,
        "qualification_results_present_at_freeze": False,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(ROOT / "state/v24_freeze_manifest.json", manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
