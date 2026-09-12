from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from controlflow.core.state import ProjectPaths, atomic_write_json, canonical_json, sha256_file


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


def verify_final_run_outputs(paths: ProjectPaths | None = None) -> dict[str, Any]:
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
    if (
        checkpoint.get("run_id") != run["run_id"]
        or checkpoint.get("freeze_hash") != run["freeze_hash"]
        or hashlib.sha256(canonical_json(checkpoint)).hexdigest() != payload.get("checkpoint_head")
    ):
        raise RuntimeError("final checkpoint head is not bound to the attested result")
    for relative, expected in cast(dict[str, str], payload["artifacts"]).items():
        target = project.root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise RuntimeError(f"attested final artifact failed integrity verification: {relative}")
    return payload
