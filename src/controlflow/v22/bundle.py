from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file, utc_now

REQUIRED_BINDINGS = frozenset(
    {
        "critical_model",
        "calibrator",
        "critical_threshold",
        "noncritical_model",
        "root_model",
        "novelty_model",
        "embedding_revision",
        "retrieval_config",
        "temporal_config",
        "policy_config",
        "action_registry",
        "approval_public_key",
        "qwen_model",
        "qwen_revision",
        "vllm_version",
        "structured_output_backend",
        "prompt",
        "schema",
    }
)


def bundle_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def write_bundle(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_BINDINGS - set(payload)
    if missing:
        raise ValueError(f"candidate bundle missing bindings: {sorted(missing)}")
    complete = {"schema_version": 1, "created_at": utc_now(), **payload}
    complete["bundle_sha256"] = bundle_hash(complete)
    atomic_write_json(path, complete)
    return complete


def verify_bundle(path: Path, root: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    missing = REQUIRED_BINDINGS - set(payload)
    if missing or payload.get("bundle_sha256") != bundle_hash(payload):
        raise RuntimeError("CANDIDATE_BUNDLE_MISMATCH: manifest")
    for name in REQUIRED_BINDINGS - {
        "critical_threshold",
        "embedding_revision",
        "qwen_model",
        "qwen_revision",
        "vllm_version",
        "structured_output_backend",
    }:
        binding = payload[name]
        artifact = (root / binding["path"]).resolve()
        artifact.relative_to(root.resolve())
        if not artifact.is_file() or sha256_file(artifact) != binding["sha256"]:
            raise RuntimeError(f"CANDIDATE_BUNDLE_MISMATCH: {name}")
    return payload
