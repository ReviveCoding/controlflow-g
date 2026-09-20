"""Freeze the reviewed V2.6 executable boundary before opening V26QUAL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v26.artifact_closure import binding
from controlflow.v26.closure_receipt import verify_receipt
from controlflow.v26.terminal_decision import verify_terminal

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-seed", type=int, required=True)
    seed = parser.parse_args().development_seed
    if (
        (ROOT / "state/v26_freeze_manifest.json").exists()
        or (ROOT / "data/v26/qualification").exists()
        or (ROOT / "results/v26/qualification").exists()
        or (ROOT / "artifacts/v26/qualification").exists()
        or any((ROOT / "state").glob("v26_qualification*.json"))
    ):
        raise RuntimeError("V26_FREEZE_PROHIBITED_QUALIFICATION_ALREADY_OPENED")
    identity = f"development_rehearsal_{seed}"
    review = _json(ROOT / "state/v26_prequalification_review_findings.json")
    static = _json(ROOT / "state/v26_static_quality.json")
    rehearsal = _json(ROOT / f"state/v26_{identity}_manifest.json")
    graph = ROOT / f"state/v26_{identity}_bindings.json"
    receipt = ROOT / f"state/v26_{identity}_closure_receipt.json"
    proposed = ROOT / f"state/v26_{identity}_manifest.proposed.json"
    anchor = ROOT / f"state/v26_{identity}_terminal_anchor.json"
    if (
        review.get("status") != "PRE_QUALIFICATION_CLEAR"
        or review.get("counts") != {"unresolved_BLOCKER": 0, "unresolved_HIGH": 0}
        or review.get("development_seed") != seed
        or static.get("status") != "PASS"
        or static.get("development_seed") != seed
        or rehearsal.get("status") != "DEVELOPMENT_REHEARSAL_PASS"
        or rehearsal.get("all_passed") is not True
        or verify_receipt(ROOT, graph, receipt)["status"] != "PASS"
        or verify_terminal(ROOT, graph, receipt, proposed, anchor, after=True)["status"] != "PASS"
    ):
        raise RuntimeError("V26_FREEZE_DEVELOPMENT_REVIEW_STATIC_GATE_INVALID")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("V26_FREEZE_EXECUTABLE_BOUNDARY_DIRTY")
    checkpoint = CheckpointEnvelope.load(ROOT / f"artifacts/v26/development/rehearsal_{seed}/rehearsal.checkpoint.json")
    if len(checkpoint.completed_case_ids) != 60 or checkpoint.seed != seed or checkpoint.concurrency != 2:
        raise RuntimeError("V26_FREEZE_REAL_PRODUCER_CHECKPOINT_INVALID")
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    strata = yaml.safe_load((ROOT / "configs/v26/strata.yaml").read_text(encoding="utf-8"))
    sources = [
        *sorted((ROOT / "src/controlflow").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("v26_*.py")),
        ROOT / "scripts/v26_server.sh",
        ROOT / "scripts/v23_benchmark.py",
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "configs").rglob("*.json")),
        *sorted((ROOT / "configs").rglob("*.txt")),
        *sorted((ROOT / "tests/v26").glob("*.py")),
        ROOT / "state/v24_candidate_manifest.json",
        ROOT / "state/v22_model_bundle.json",
        ROOT / "state/v26_prequalification_review_findings.json",
        ROOT / "state/v26_static_quality.json",
        ROOT / f"state/v26_{identity}_manifest.json",
        graph,
        receipt,
        proposed,
        anchor,
        ROOT / f"state/v26_{identity}_postclose_verification.json",
        ROOT / f"state/v26_{identity}_unbound_artifacts.json",
        ROOT / f"state/v26_{identity}_protocol.json",
        ROOT / f"state/v26_{identity}_environment_manifest.json",
        *sorted((ROOT / f"data/v26/development/rehearsal_{seed}").glob("*")),
        *sorted((ROOT / f"results/v26/development/rehearsal_{seed}").glob("*")),
        *sorted((ROOT / f"artifacts/v26/development/rehearsal_{seed}").glob("*")),
        ROOT / "uv.lock",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "bindings": [binding(ROOT, path) for path in sorted(set(sources))],
        "development_seed": seed,
        "checkpoint_contract_sha256": sha256_file(ROOT / "src/controlflow/v25/checkpoint_contract.py"),
        "checkpoint_schema_version": checkpoint.schema_version,
        "positive_producer_fixture": binding(
            ROOT, ROOT / f"artifacts/v26/development/rehearsal_{seed}/rehearsal.checkpoint.json"
        ),
        "positive_producer_fixture_count": len(checkpoint.completed_case_ids),
        "positive_producer_fixture_fingerprint": checkpoint.fingerprint,
        "denominator_contract_sha256": sha256_file(ROOT / "configs/v26/denominator_contract.yaml"),
        "gate_config_sha256": sha256_file(ROOT / "configs/v26/qualification_gates.yaml"),
        "qwen_model": serving["model"],
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "serving": {
            "host": serving["host"],
            "port": serving["port"],
            "max_model_len": 4096,
            "max_num_seqs": 2,
        },
        "qualification_count": 600,
        "qualification_seed": int(strata["qualification_seed"]),
        "final_seed": int(strata["final_seed"]),
        "qualification_results_present_at_freeze": False,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(ROOT / "state/v26_freeze_manifest.json", manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
