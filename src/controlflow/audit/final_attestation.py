from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from controlflow.core.state import ProjectPaths, atomic_write_json, canonical_json


@dataclass(frozen=True)
class VerifiedFinalArtifacts:
    attestation: dict[str, Any]
    artifacts: dict[str, bytes]


class FinalRunAttestor:
    """Local protected-root authenticator for one-shot final-run artifacts."""

    def __init__(self) -> None:
        self.trust = Path(
            os.environ.get(
                "CONTROLFLOW_AUDIT_TRUST_DIR",
                str(ProjectPaths.discover().state / "audit_trust"),
            )
        )
        self.trust.mkdir(parents=True, exist_ok=True)
        self.key_path = self.trust / "final-run-attestation.key"
        with FileLock(str(self.trust / ".final-run-attestation-key.lock")):
            if not self.key_path.exists():
                with self.key_path.open("xb") as handle:
                    handle.write(secrets.token_bytes(32))
                    handle.flush()
                    os.fsync(handle.fileno())
                self.key_path.chmod(0o400)
            self.key = self.key_path.read_bytes()
        if len(self.key) != 32:
            raise RuntimeError("invalid final-run attestation key")

    def sign(self, payload: dict[str, Any]) -> str:
        return hmac.new(self.key, canonical_json(payload), hashlib.sha256).hexdigest()

    def envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"payload": payload, "signature": self.sign(payload)}

    def verify_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        payload = cast(dict[str, Any], envelope.get("payload"))
        signature = str(envelope.get("signature", ""))
        if not isinstance(payload, dict) or not hmac.compare_digest(signature, self.sign(payload)):
            raise RuntimeError("final-run attestation signature mismatch")
        return payload

    def write_anchor(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.trust / name
        atomic_write_json(target, self.envelope(payload))
        return target

    def read_anchor(self, name: str) -> dict[str, Any]:
        import json

        target = self.trust / name
        if not target.is_file():
            raise RuntimeError(f"missing protected final-run attestation: {name}")
        return self.verify_envelope(json.loads(target.read_text(encoding="utf-8")))

    def initialize_progress(self, run_id: str, freeze_hash: str, body: dict[str, Any]) -> dict[str, Any]:
        """Create or verify the protected append-only progress journal."""
        head_name = f"final-progress-{run_id}.json"
        with FileLock(str(self.trust / f".{head_name}.lock")):
            if (self.trust / head_name).exists():
                history = self.read_progress(run_id, freeze_hash)
                return history[-1]
            entry = {
                "run_id": run_id,
                "freeze_hash": freeze_hash,
                "sequence": 0,
                "previous_entry_hash": "GENESIS",
                **body,
            }
            entry_name = f"final-progress-{run_id}-00000000.json"
            atomic_write_json(self.trust / entry_name, self.envelope(entry))
            self.write_anchor(
                head_name,
                {
                    "run_id": run_id,
                    "freeze_hash": freeze_hash,
                    "sequence": 0,
                    "entry_hash": hashlib.sha256(canonical_json(entry)).hexdigest(),
                },
            )
            return entry

    def append_progress(self, run_id: str, freeze_hash: str, body: dict[str, Any]) -> dict[str, Any]:
        head_name = f"final-progress-{run_id}.json"
        with FileLock(str(self.trust / f".{head_name}.lock")):
            history = self.read_progress(run_id, freeze_hash)
            previous = history[-1]
            sequence = int(previous["sequence"]) + 1
            entry = {
                "run_id": run_id,
                "freeze_hash": freeze_hash,
                "sequence": sequence,
                "previous_entry_hash": hashlib.sha256(canonical_json(previous)).hexdigest(),
                **body,
            }
            entry_name = f"final-progress-{run_id}-{sequence:08d}.json"
            if (self.trust / entry_name).exists():
                raise RuntimeError("protected final progress sequence replay")
            atomic_write_json(self.trust / entry_name, self.envelope(entry))
            self.write_anchor(
                head_name,
                {
                    "run_id": run_id,
                    "freeze_hash": freeze_hash,
                    "sequence": sequence,
                    "entry_hash": hashlib.sha256(canonical_json(entry)).hexdigest(),
                },
            )
            return entry

    def read_progress(self, run_id: str, freeze_hash: str) -> list[dict[str, Any]]:
        head = self.read_anchor(f"final-progress-{run_id}.json")
        if head.get("run_id") != run_id or head.get("freeze_hash") != freeze_hash:
            raise RuntimeError("protected final progress is bound to another run")
        expected_last = int(head["sequence"])
        prefix = f"final-progress-{run_id}-"
        observed_sequences = {
            int(path.stem.removeprefix(prefix))
            for path in self.trust.glob(f"{prefix}*.json")
            if path.stem.removeprefix(prefix).isdigit()
        }
        if observed_sequences != set(range(expected_last + 1)):
            raise RuntimeError("protected final progress head rollback or entry deletion")
        history: list[dict[str, Any]] = []
        previous_hash = "GENESIS"
        for sequence in range(expected_last + 1):
            entry = self.read_anchor(f"final-progress-{run_id}-{sequence:08d}.json")
            if (
                entry.get("run_id") != run_id
                or entry.get("freeze_hash") != freeze_hash
                or entry.get("sequence") != sequence
                or entry.get("previous_entry_hash") != previous_hash
            ):
                raise RuntimeError("protected final progress chain validation failed")
            previous_hash = hashlib.sha256(canonical_json(entry)).hexdigest()
            history.append(entry)
        if previous_hash != head.get("entry_hash"):
            raise RuntimeError("protected final progress head rollback or mismatch")
        return history


def load_verified_final_artifacts(paths: ProjectPaths | None = None) -> VerifiedFinalArtifacts:
    import json

    project = paths or ProjectPaths.discover()
    run = json.loads((project.state / "final_run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise RuntimeError("final run is not complete")
    attestor = FinalRunAttestor()
    payload = attestor.read_anchor(f"final-result-{run['run_id']}.json")
    if payload.get("run_id") != run["run_id"] or payload.get("freeze_hash") != run["freeze_hash"]:
        raise RuntimeError("final-result attestation is not bound to the active frozen run")
    checkpoint = attestor.read_anchor(f"final-checkpoints-{run['run_id']}.json")
    progress = attestor.read_progress(str(run["run_id"]), str(run["freeze_hash"]))[-1]
    if (
        checkpoint.get("run_id") != run["run_id"]
        or checkpoint.get("freeze_hash") != run["freeze_hash"]
        or hashlib.sha256(canonical_json(checkpoint)).hexdigest() != payload.get("checkpoint_head")
        or checkpoint.get("progress_head") != hashlib.sha256(canonical_json(progress)).hexdigest()
        or checkpoint.get("records") != progress.get("records")
        or progress.get("active_work_id") is not None
    ):
        raise RuntimeError("final checkpoint head is not bound to the attested result")
    verified: dict[str, bytes] = {}
    for relative, expected in cast(dict[str, str], payload["artifacts"]).items():
        target = project.root / relative
        if not target.is_file():
            raise RuntimeError(f"attested final artifact failed integrity verification: {relative}")
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"attested final artifact failed integrity verification: {relative}")
        verified[relative] = content
    return VerifiedFinalArtifacts(payload, verified)


def verify_final_run_outputs(paths: ProjectPaths | None = None) -> dict[str, Any]:
    return load_verified_final_artifacts(paths).attestation
