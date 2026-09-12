from __future__ import annotations

import json

import pandas as pd

from controlflow.audit.final_attestation import verify_final_run_outputs
from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, sha256_file, utc_now


def run() -> str:
    paths = ProjectPaths.discover()
    verify_final_run_outputs(paths)
    final_path = paths.root / "results/final_test.parquet"
    final = pd.read_parquet(final_path).iloc[0]
    decision = str(final.release_decision)
    if decision not in {"PROMOTE", "CONDITIONAL_PROMOTE", "NO_PROMOTE"}:
        raise ValueError("invalid release decision")
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "decision": decision,
        "final_result_sha256": sha256_file(final_path),
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
