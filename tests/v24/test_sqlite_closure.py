from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from controlflow.v24 import hash_stability
from controlflow.v24.sqlite_finalization import _sidecar_sizes, finalize
from controlflow.v24.sqlite_lifecycle import assert_all_closed, owned_connection


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE action_ledger(sequence_id INTEGER, event_hash TEXT, previous_event_hash TEXT, "
            "review_required INTEGER, approval_valid INTEGER, committed INTEGER)"
        )
        connection.execute("CREATE TABLE ledger_head(singleton INTEGER, event_hash TEXT, sequence_id INTEGER)")
        connection.execute("CREATE TABLE probe(value INTEGER)")
        connection.commit()
    finally:
        connection.close()


def test_owned_connection_closes_on_exception(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    with pytest.raises(RuntimeError, match="SQLITE_CONNECTIONS_OPEN"), owned_connection(path):
        assert_all_closed(path)
    assert_all_closed(path)


def test_finalizer_success_and_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    report = finalize(path, tmp_path / "finalization.json", expected_events=0)
    assert report["checkpoint"][0] == 0
    assert report["status"] == "FINALIZED"
    with pytest.raises(RuntimeError, match="DB_MISSING"):
        finalize(tmp_path / "missing.sqlite", tmp_path / "missing.json", expected_events=0)


def test_finalizer_rejects_tampered_ledger(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    with owned_connection(path) as connection:
        connection.execute("INSERT INTO action_ledger VALUES (1,'tampered','GENESIS',0,0,0)")
    with pytest.raises(RuntimeError, match="LEDGER_INVALID"):
        finalize(path, tmp_path / "report.json", expected_events=1)
    assert not (tmp_path / "report.json").exists()


def test_finalizer_rejects_open_reader(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    writer = sqlite3.connect(path)
    reader = sqlite3.connect(path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM probe").fetchall()
        writer.execute("INSERT INTO probe VALUES (1)")
        writer.commit()
        with pytest.raises(RuntimeError):
            finalize(path, tmp_path / "report.json", expected_events=0)
    finally:
        reader.close()
        writer.close()


def test_finalizer_rejects_open_writer(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    writer = sqlite3.connect(path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO probe VALUES (1)")
        with pytest.raises(RuntimeError, match="CHECKPOINT_BUSY"):
            finalize(path, tmp_path / "report.json", expected_events=0)
        assert not (tmp_path / "report.json").exists()
    finally:
        writer.rollback()
        writer.close()


@pytest.mark.parametrize("fault", ["before_checkpoint", "after_checkpoint_before_close"])
def test_finalizer_crash_never_admits(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    with pytest.raises(RuntimeError, match="INJECTED_FINALIZER_CRASH"):
        finalize(path, tmp_path / "report.json", expected_events=0, fault=fault)
    assert not (tmp_path / "report.json").exists()


def test_finalizer_checkpoints_wal_frames(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    _database(path)
    keeper = sqlite3.connect(path)
    try:
        keeper.execute("INSERT INTO probe VALUES (10)")
        keeper.commit()
        assert path.with_name(path.name + "-wal").stat().st_size > 0
    finally:
        keeper.close()
    report = finalize(path, tmp_path / "report.json", expected_events=0)
    assert report["checkpoint"][0] == 0
    assert _sidecar_sizes(path) == (0, 0)


def test_unexpected_sidecars_are_not_empty(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    path.write_bytes(b"database")
    path.with_name(path.name + "-wal").write_bytes(b"uncheckpointed")
    path.with_name(path.name + "-shm").write_bytes(b"unexpected")
    assert _sidecar_sizes(path) == (14, 10)


def test_stability_detects_mutation_between_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "test.sqlite"
    path.write_bytes(b"first")

    def mutate(_seconds: float) -> None:
        path.write_bytes(b"second")

    monkeypatch.setattr(hash_stability.time, "sleep", mutate)
    with pytest.raises(RuntimeError, match="FINALIZATION_UNSTABLE"):
        hash_stability.verify_stability(path, tmp_path / "stability.json", interval_seconds=0.1)
    assert not (tmp_path / "stability.json").exists()


def test_stability_report_binds_same_bytes(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    path.write_bytes(b"stable")
    report = hash_stability.verify_stability(path, tmp_path / "stability.json", interval_seconds=0.001)
    assert report["sha256_1"] == report["sha256_2"]
    assert json.loads((tmp_path / "stability.json").read_text(encoding="utf-8"))["status"] == "STABLE"
