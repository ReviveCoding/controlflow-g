"""Freeze the reviewed V2.5 executable qualification boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v25.artifact_closure import binding
from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v25.closure_receipt import verify_receipt
from controlflow.v25.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "development_rehearsal_25013"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    if (
        (ROOT / "data/v25/qualification/V25QUAL").exists()
        or (ROOT / "results/v25/qualification").exists()
        or (ROOT / "artifacts/v25/qualification").exists()
        or any((ROOT / "state").glob("v25_qualification*.json"))
        or any((ROOT / "state").glob("v25_postclose*.json"))
        or any((ROOT / "state").glob("v25_unbound*.json"))
    ):
        raise RuntimeError("V25_FREEZE_PROHIBITED: V25QUAL already exists")
    reviews = _json(ROOT / "state/v25_prequalification_review_findings.json")
    static = _json(ROOT / "state/v25_static_quality.json")
    rehearsal = _json(ROOT / f"state/v25_{IDENTITY}_manifest.json")
    graph = ROOT / f"state/v25_{IDENTITY}_bindings.json"
    receipt = ROOT / f"state/v25_{IDENTITY}_closure_receipt.json"
    proposed = ROOT / f"state/v25_{IDENTITY}_manifest.proposed.json"
    anchor = ROOT / f"state/v25_{IDENTITY}_terminal_anchor.json"
    if (
        reviews.get("status") != "PRE_QUALIFICATION_CLEAR"
        or reviews.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or static.get("status") != "PASS"
        or rehearsal.get("status") != "DEVELOPMENT_REHEARSAL_PASS"
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V25_FREEZE_PROHIBITED: development/review/static gate incomplete")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise RuntimeError("V25_FREEZE_PROHIBITED: executable boundary not committed")
    gates24 = yaml.safe_load((ROOT / "configs/v24/qualification_gates.yaml").read_text(encoding="utf-8"))
    gates25 = yaml.safe_load((ROOT / "configs/v25/qualification_gates.yaml").read_text(encoding="utf-8"))
    contract24 = yaml.safe_load((ROOT / "configs/v24/denominator_contract.yaml").read_text(encoding="utf-8"))
    contract25 = yaml.safe_load((ROOT / "configs/v25/denominator_contract.yaml").read_text(encoding="utf-8"))
    if gates24 != gates25 or any(
        contract25["metrics"][name]["minimum_denominator"] != row["minimum_denominator"]
        for name, row in contract24["metrics"].items()
    ):
        raise RuntimeError("V25_FREEZE_PROHIBITED: prior qualification gates weakened")
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    strata = yaml.safe_load((ROOT / "configs/v25/strata.yaml").read_text(encoding="utf-8"))
    checkpoint_path = ROOT / "artifacts/v25/development/rehearsal_25013/rehearsal.checkpoint.json"
    checkpoint = CheckpointEnvelope.load(checkpoint_path)
    if len(checkpoint.completed_case_ids) != 60 or checkpoint.seed != 25013 or checkpoint.concurrency != 2:
        raise RuntimeError("V25_FREEZE_PROHIBITED: real producer fixture invalid")
    historical = _json(ROOT / "state/v24_candidate_manifest.json")
    for name, row in historical["preserved_bindings"].items():
        if name == "candidate_implementation":
            continue  # V2.5 changes the resume parser; model and policy evidence remain frozen.
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"V25_PRESERVED_CANDIDATE_BINDING_CHANGED:{name}")
    source_paths = [
        *sorted((ROOT / "src/controlflow").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("v25_*.py")),
        ROOT / "scripts/v25_server.sh",
        *sorted((ROOT / "configs/v25").glob("*.yaml")),
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "tests/v25").glob("*.py")),
        ROOT / "src/controlflow/v22/candidate.py",
        ROOT / "src/controlflow/v22/checkpoint.py",
        ROOT / "src/controlflow/v22/evaluation.py",
        ROOT / "src/controlflow/v22/executor.py",
        ROOT / "src/controlflow/v24/sqlite_lifecycle.py",
        ROOT / "src/controlflow/v24/sqlite_finalization.py",
        ROOT / "src/controlflow/v24/hash_stability.py",
        ROOT / "src/controlflow/v24/ledger_snapshot.py",
        ROOT / "configs/v23/serving.yaml",
        ROOT / "configs/v23/explanation_prompt_minimal.txt",
        ROOT / "configs/v23/explanation_schema_minimal.json",
        ROOT / "scripts/v23_benchmark.py",
        ROOT / "state/v24_candidate_manifest.json",
        ROOT / "state/v22_model_bundle.json",
        ROOT / "state/v25_prequalification_review_findings.json",
        ROOT / "state/v25_static_quality.json",
        ROOT / "results/v25/v25_tests.xml",
        ROOT / f"state/v25_{IDENTITY}_manifest.json",
        ROOT / f"state/v25_{IDENTITY}_bindings.json",
        ROOT / f"state/v25_{IDENTITY}_closure_receipt.json",
        ROOT / f"state/v25_{IDENTITY}_terminal_anchor.json",
        ROOT / f"state/v25_{IDENTITY}_manifest.proposed.json",
        ROOT / f"state/v25_{IDENTITY}_postclose_verification.json",
        ROOT / f"state/v25_{IDENTITY}_unbound_artifacts.json",
        ROOT / f"state/v25_{IDENTITY}_protocol.json",
        ROOT / f"state/v25_{IDENTITY}_environment_manifest.json",
        *sorted((ROOT / "data/v25/development/rehearsal_25013").glob("*")),
        *sorted((ROOT / "results/v25/development/rehearsal_25013").glob("*")),
        *sorted((ROOT / "artifacts/v25/development/rehearsal_25013").glob("*")),
        ROOT / "uv.lock",
    ]
    source_paths += [
        ROOT / row["path"]
        for name, row in historical["preserved_bindings"].items()
        if name != "candidate_implementation"
    ]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "source_commit": source_commit,
        "bindings": [binding(ROOT, path) for path in sorted(set(source_paths))],
        "checkpoint_contract_sha256": sha256_file(ROOT / "src/controlflow/v25/checkpoint_contract.py"),
        "checkpoint_schema_version": checkpoint.schema_version,
        "positive_producer_fixture": binding(ROOT, checkpoint_path),
        "positive_producer_fixture_count": len(checkpoint.completed_case_ids),
        "positive_producer_fixture_fingerprint": checkpoint.fingerprint,
        "schema_mutation_test_source_sha256": sha256_file(ROOT / "tests/v25/test_checkpoint_contract.py"),
        "schema_mutation_test_junit_sha256": static["junit_sha256"],
        "candidate_manifest_sha256": sha256_file(ROOT / "state/v24_candidate_manifest.json"),
        "denominator_contract_sha256": sha256_file(ROOT / "configs/v25/denominator_contract.yaml"),
        "gate_config_sha256": sha256_file(ROOT / "configs/v25/qualification_gates.yaml"),
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
        "qualification_seed": int(strata["qualification_seed"]),
        "final_seed": int(strata["final_seed"]),
        "qualification_results_present_at_freeze": False,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(ROOT / "state/v25_freeze_manifest.json", manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
