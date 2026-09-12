from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event, Thread

import pytest

from controlflow import final_evaluation
from controlflow.audit.final_attestation import (
    FinalRunAttestor,
    load_verified_final_artifacts,
    verify_final_run_outputs,
)
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
    progress = attestor.initialize_progress(
        "run",
        "freeze",
        {
            "records": {},
            "active_work_id": None,
            "attempt_started_at": "2026-01-01T00:00:00+00:00",
            "accumulated_runtime_seconds": 0.0,
            "retry_count": 0,
            "transition": "test",
        },
    )
    checkpoint = {
        "run_id": "run",
        "freeze_hash": "freeze",
        "count": 0,
        "records": {},
        "progress_head": hashlib.sha256(canonical_json(progress)).hexdigest(),
    }
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
    verified = load_verified_final_artifacts(paths)
    result.write_bytes(b"replacement")
    assert verified.artifacts["results/final_test.parquet"] == b"result"
    with pytest.raises(RuntimeError, match="integrity verification"):
        verify_final_run_outputs(paths)


def test_progress_chain_recovers_head_lag_but_rejects_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROLFLOW_AUDIT_TRUST_DIR", str(tmp_path / "trust"))
    attestor = FinalRunAttestor()
    initial = attestor.initialize_progress("run", "freeze", {"records": {}, "active_work_id": None})
    advanced = attestor.append_progress("run", "freeze", {"records": {"work": "hash"}, "active_work_id": None})
    head_path = attestor.trust / "final-progress-run.json"
    archived_head = attestor.envelope(
        {
            "run_id": "run",
            "freeze_hash": "freeze",
            "sequence": 0,
            "entry_hash": hashlib.sha256(canonical_json(initial)).hexdigest(),
        }
    )
    head_path.write_text(json.dumps(archived_head), encoding="utf-8")
    # A fully signed contiguous successor is the entry-durable/head-update
    # crash window and is recovered by advancing the authenticated head.
    assert attestor.read_progress("run", "freeze")[-1] == advanced
    assert attestor.read_anchor("final-progress-run.json")["sequence"] == 1
    attestor.write_anchor(
        "final-progress-run.json",
        {
            "run_id": "run",
            "freeze_hash": "freeze",
            "sequence": 1,
            "entry_hash": hashlib.sha256(canonical_json(advanced)).hexdigest(),
        },
    )
    (attestor.trust / "final-progress-run-00000001.json").unlink()
    with pytest.raises(RuntimeError, match=r"head rollback|entry deletion|missing protected|head exceeds"):
        attestor.read_progress("run", "freeze")


def test_progress_chain_rejects_divergent_signed_orphan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROLFLOW_AUDIT_TRUST_DIR", str(tmp_path / "trust"))
    attestor = FinalRunAttestor()
    attestor.initialize_progress("run", "freeze", {"records": {}, "active_work_id": None})
    divergent = {
        "run_id": "run",
        "freeze_hash": "freeze",
        "sequence": 1,
        "previous_entry_hash": "NOT_THE_HEAD",
        "records": {},
        "active_work_id": None,
    }
    (attestor.trust / "final-progress-run-00000001.json").write_text(
        json.dumps(attestor.envelope(divergent)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="chain validation"):
        attestor.read_progress("run", "freeze")


def test_final_evaluation_rejects_live_duplicate_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "state").mkdir()
    entered = Event()
    release = Event()
    completed: list[str] = []

    def held_run() -> str:
        entered.set()
        release.wait(2)
        return "complete"

    monkeypatch.setattr(final_evaluation.ProjectPaths, "discover", lambda: ProjectPaths(tmp_path))
    monkeypatch.setattr(final_evaluation, "_run_final_once_locked", held_run)
    owner = Thread(target=lambda: completed.append(final_evaluation.run_final_once()))
    owner.start()
    assert entered.wait(1)
    with pytest.raises(RuntimeError, match="another final evaluation owner"):
        final_evaluation.run_final_once()
    release.set()
    owner.join(2)
    assert completed == ["complete"]
