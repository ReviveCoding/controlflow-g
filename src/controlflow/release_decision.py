from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pandas as pd

from controlflow.audit.final_attestation import load_verified_final_artifacts
from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, utc_now


def run() -> str:
    paths = ProjectPaths.discover()
    verified = load_verified_final_artifacts(paths)
    final_bytes = verified.artifacts["results/final_test.parquet"]
    final = pd.read_parquet(BytesIO(final_bytes)).iloc[0]
    decision = str(final.release_decision)
    if decision not in {"PROMOTE", "CONDITIONAL_PROMOTE", "NO_PROMOTE"}:
        raise ValueError("invalid release decision")
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "decision": decision,
        "final_result_sha256": hashlib.sha256(final_bytes).hexdigest(),
        "freeze_hash": str(final.config_hash),
        "gate_results": json.loads(final.gate_results),
        "scope": "local production-like simulation only",
    }
    target = paths.root / "results/release_decision.json"
    with PhaseRun("P29", paths) as phase:
        atomic_write_json(target, payload)
        phase.register(target, "release_decision")
    return str(target)


if __name__ == "__main__":
    print(run())
