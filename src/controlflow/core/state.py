from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from filelock import FileLock


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"), path.open("ab") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_artifact_manifest(paths: ProjectPaths | None = None) -> list[str]:
    """Return integrity errors for every durable artifact currently registered."""
    project = paths or ProjectPaths.discover()
    manifest = json.loads(project.artifact_manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in manifest.get("artifacts", []):
        relative = str(record.get("path", ""))
        target = (project.root / relative).resolve()
        try:
            target.relative_to(project.root)
        except ValueError:
            errors.append(f"outside workspace: {relative}")
            continue
        if not target.is_file():
            errors.append(f"missing: {relative}")
            continue
        if target.stat().st_size != record.get("bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(target) != record.get("sha256"):
            errors.append(f"hash mismatch: {relative}")
    return errors


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        candidate = (start or Path.cwd()).resolve()
        for path in (candidate, *candidate.parents):
            if (path / "PROJECT_SPEC.md").exists():
                return cls(path)
        raise RuntimeError("Could not locate ControlFlow-G repository root")

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def execution_state(self) -> Path:
        return self.state / "execution_state.json"

    @property
    def journal(self) -> Path:
        return self.state / "execution_journal.jsonl"

    @property
    def artifact_manifest(self) -> Path:
        return self.state / "artifact_manifest.json"


class PhaseRun(AbstractContextManager["PhaseRun"]):
    """Fail-safe major-phase state transition with a durable journal."""

    def __init__(self, phase: str, paths: ProjectPaths | None = None) -> None:
        self.phase = phase
        self.paths = paths or ProjectPaths.discover()
        self.started_at = utc_now()

    def _read_state(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.paths.execution_state.read_text(encoding="utf-8")))

    def _write_state(self, **changes: Any) -> None:
        with FileLock(str(self.paths.execution_state) + ".lock"):
            state = self._read_state()
            state.update(changes)
            state["updated_at"] = utc_now()
            atomic_write_json(self.paths.execution_state, state)

    def journal(self, status: str, message: str, artifacts: list[str] | None = None) -> None:
        append_jsonl(
            self.paths.journal,
            {
                "event_id": f"{self.phase}-{status}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}",
                "timestamp": utc_now(),
                "phase": self.phase,
                "status": status,
                "message": message,
                "artifacts": artifacts or [],
            },
        )

    def __enter__(self) -> PhaseRun:
        self._write_state(current_phase=self.phase, status="in_progress")
        self.journal("started", f"{self.phase} execution started or resumed")
        return self

    def register(self, path: Path, kind: str) -> dict[str, Any]:
        relative = path.resolve().relative_to(self.paths.root).as_posix()
        record = {
            "path": relative,
            "phase": self.phase,
            "kind": kind,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "recorded_at": utc_now(),
        }
        with FileLock(str(self.paths.artifact_manifest) + ".lock"):
            manifest = json.loads(self.paths.artifact_manifest.read_text(encoding="utf-8"))
            artifacts = [item for item in manifest["artifacts"] if item["path"] != relative]
            artifacts.append(record)
            manifest.update(updated_at=utc_now(), artifacts=sorted(artifacts, key=lambda item: item["path"]))
            atomic_write_json(self.paths.artifact_manifest, manifest)
        self._write_state(last_successful_artifact=relative)
        return record

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        with FileLock(str(self.paths.execution_state) + ".lock"):
            state = self._read_state()
            if exc is None:
                state["completed_phases"] = sorted(set([*state["completed_phases"], self.phase]))
                state["failed_phases"] = [phase for phase in state["failed_phases"] if phase != self.phase]
                state["status"] = "complete"
            else:
                state["failed_phases"] = sorted(set([*state["failed_phases"], self.phase]))
                state["status"] = "failed"
            state["updated_at"] = utc_now()
            atomic_write_json(self.paths.execution_state, state)
        if exc is None:
            self.journal("completed", f"{self.phase} completed")
        else:
            self.journal("failed", f"{type(exc).__name__}: {exc}")
        return False
