"""Bind the admitted development cohort and executable rehearsal protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v26.artifact_closure import binding

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    seed = parser.parse_args().seed
    data = ROOT / f"data/v26/development/rehearsal_{seed}"
    results = ROOT / f"results/v26/development/rehearsal_{seed}"
    output = ROOT / f"state/v26_development_rehearsal_{seed}_protocol.json"
    if output.exists():
        raise RuntimeError("V26_DEVELOPMENT_PROTOCOL_ALREADY_BOUND")
    transition = json.loads(
        (ROOT / f"state/v26_development_rehearsal_{seed}_manifest.json").read_text(encoding="utf-8")
    )
    if transition.get("status") != "TRANSITION_PASS":
        raise RuntimeError("V26_DEVELOPMENT_TRANSITION_NOT_ADMITTED")
    admission = json.loads((results / "structural_admission.json").read_text(encoding="utf-8"))
    contamination = json.loads((results / "contamination.json").read_text(encoding="utf-8"))
    if admission["status"] != "ADMITTED" or contamination["semantic"]["leakage_findings"] != 0:
        raise RuntimeError("V26_DEVELOPMENT_COHORT_NOT_ADMITTED")
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    paths = [
        *sorted((ROOT / "src/controlflow").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("v26_*.py")),
        ROOT / "scripts/v26_server.sh",
        ROOT / "scripts/v23_benchmark.py",
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "configs").rglob("*.json")),
        *sorted((ROOT / "configs").rglob("*.txt")),
        ROOT / "state/v24_candidate_manifest.json",
        ROOT / "state/v22_model_bundle.json",
        *sorted(data.glob("*")),
        results / "structural_admission.json",
        results / "contamination.json",
        results / "prior_inventory.json",
        ROOT / "uv.lock",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_PROTOCOL_BOUND",
        "role": "DEVELOPMENT_ONLY",
        "bindings": [binding(ROOT, path) for path in sorted(set(paths))],
        "checkpoint_schema_version": 1,
        "checkpoint_contract_sha256": sha256_file(ROOT / "src/controlflow/v25/checkpoint_contract.py"),
        "gate_config_sha256": sha256_file(ROOT / "configs/v26/development_gates.yaml"),
        "qwen_model": serving["model"],
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "serving": {
            "max_model_len": 4096,
            "max_num_seqs": 2,
            "host": serving["host"],
            "port": serving["port"],
        },
        "development_seed": seed,
        "development_count": 60,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(output, manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
