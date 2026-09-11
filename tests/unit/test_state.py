from __future__ import annotations

import json
from pathlib import Path

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, verify_artifact_manifest


def test_atomic_json_is_canonical(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert sha256_file(target) == "e8d38819d39f705646bfb643368eca78f7db476c16471dbc33b941b27326410d"


def test_atomic_json_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"old": True})
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_artifact_manifest_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("trusted", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    atomic_write_json(
        state / "artifact_manifest.json",
        {"artifacts": [{"path": "artifact.txt", "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}]},
    )
    paths = ProjectPaths(tmp_path)
    assert verify_artifact_manifest(paths) == []
    artifact.write_text("tampered", encoding="utf-8")
    assert verify_artifact_manifest(paths) == ["size mismatch: artifact.txt", "hash mismatch: artifact.txt"]
