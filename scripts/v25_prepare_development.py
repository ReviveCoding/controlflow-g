"""Bind the admitted development cohort and executable rehearsal protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v25.artifact_closure import binding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v25/development/rehearsal_25013"
RESULTS = ROOT / "results/v25/development/rehearsal_25013"
OUTPUT = ROOT / "state/v25_development_rehearsal_25013_protocol.json"


def main() -> None:
    if OUTPUT.exists() or (ROOT / "state/v25_development_rehearsal_25013_manifest.json").exists():
        raise RuntimeError("V25_DEVELOPMENT_PROTOCOL_ALREADY_BOUND")
    admission = json.loads((RESULTS / "structural_admission.json").read_text(encoding="utf-8"))
    contamination = json.loads((RESULTS / "contamination.json").read_text(encoding="utf-8"))
    if admission["status"] != "ADMITTED" or contamination["leakage_findings"] != 0:
        raise RuntimeError("V25_DEVELOPMENT_COHORT_NOT_ADMITTED")
    serving = yaml.safe_load((ROOT / "configs/v23/serving.yaml").read_text(encoding="utf-8"))
    paths = [
        *sorted((ROOT / "src/controlflow/v25").glob("*.py")),
        ROOT / "src/controlflow/v22/candidate.py",
        ROOT / "src/controlflow/v22/checkpoint.py",
        ROOT / "src/controlflow/v22/evaluation.py",
        ROOT / "src/controlflow/v22/executor.py",
        ROOT / "src/controlflow/v24/sqlite_finalization.py",
        ROOT / "src/controlflow/v24/hash_stability.py",
        ROOT / "src/controlflow/v24/sqlite_lifecycle.py",
        ROOT / "src/controlflow/v24/ledger_snapshot.py",
        ROOT / "scripts/v25_run.py",
        ROOT / "scripts/v25_server.sh",
        ROOT / "scripts/v25_prepare_development.py",
        *sorted((ROOT / "configs/v25").glob("*.yaml")),
        ROOT / "configs/v23/serving.yaml",
        ROOT / "state/v24_candidate_manifest.json",
        ROOT / "state/v22_model_bundle.json",
        *sorted(DATA.glob("*")),
        RESULTS / "structural_admission.json",
        RESULTS / "contamination.json",
        ROOT / "uv.lock",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_PROTOCOL_BOUND",
        "role": "DEVELOPMENT_ONLY",
        "bindings": [binding(ROOT, path) for path in sorted(set(paths))],
        "checkpoint_schema_version": 1,
        "checkpoint_contract_sha256": sha256_file(ROOT / "src/controlflow/v25/checkpoint_contract.py"),
        "gate_config_sha256": sha256_file(ROOT / "configs/v25/development_gates.yaml"),
        "qwen_model": serving["model"],
        "qwen_revision": serving["revision"],
        "vllm_version": serving["vllm_version"],
        "serving": {
            "max_model_len": 4096,
            "max_num_seqs": 2,
            "host": serving["host"],
            "port": serving["port"],
        },
        "development_seed": 25013,
        "development_count": 60,
    }
    manifest["freeze_hash"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_write_json(OUTPUT, manifest)
    print(manifest["freeze_hash"])


if __name__ == "__main__":
    main()
