from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from controlflow.audit.final_attestation import FinalRunAttestor, verify_final_run_outputs
from controlflow.core.state import ProjectPaths, canonical_json, sha256_file


def test_final_attestation_rejects_payload_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROLFLOW_AUDIT_TRUST_DIR", str(tmp_path / "trust"))
    attestor = FinalRunAttestor()
    envelope = attestor.envelope({"run_id": "run", "count": 1})
    envelope["payload"]["count"] = 2
    with pytest.raises(RuntimeError, match="signature mismatch"):
        attestor.verify_envelope(envelope)


def test_every_final_consumer_verifier_rejects_result_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLFLOW_AUDIT_TRUST_DIR", str(tmp_path / "trust"))
    state = tmp_path / "state"
    results = tmp_path / "results"
    state.mkdir()
    results.mkdir()
    result = results / "final_test.parquet"
    traces = results / "final_test_traces.parquet"
    result.write_bytes(b"result")
    traces.write_bytes(b"traces")
    run = {"run_id": "run", "freeze_hash": "freeze", "status": "complete"}
    (state / "final_run.json").write_text(json.dumps(run), encoding="utf-8")
    attestor = FinalRunAttestor()
    checkpoint = {"run_id": "run", "freeze_hash": "freeze", "count": 0, "records": {}}
    attestor.write_anchor("final-checkpoints-run.json", checkpoint)
    attestor.write_anchor(
        "final-result-run.json",
        {
            "run_id": "run",
            "freeze_hash": "freeze",
            "checkpoint_head": hashlib.sha256(canonical_json(checkpoint)).hexdigest(),
            "artifacts": {
                "results/final_test.parquet": sha256_file(result),
                "results/final_test_traces.parquet": sha256_file(traces),
            },
        },
    )
    paths = ProjectPaths(tmp_path)
    verify_final_run_outputs(paths)
    result.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="integrity verification"):
        verify_final_run_outputs(paths)
