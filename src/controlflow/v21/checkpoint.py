from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from controlflow.core.state import canonical_json

REQUIRED_FIELDS = frozenset(
    {
        "git_commit",
        "dirty_tree_hash",
        "dependency_lock_hash",
        "runtime_dataset_sha256",
        "candidate_bundle_sha256",
        "critical_model_sha256",
        "calibrator_sha256",
        "critical_threshold",
        "noncritical_model_sha256",
        "root_model_sha256",
        "embedding_revision",
        "qwen_model_revision",
        "prompt_sha256",
        "schema_sha256",
        "vllm_version",
        "structured_output_backend",
        "retrieval_config_sha256",
        "temporal_config_sha256",
        "policy_bundle_hash",
        "action_policy_version",
        "scorer_hash",
        "qualification_protocol_hash",
        "concurrency",
        "seed",
    }
)


def fingerprint(fields: dict[str, Any]) -> str:
    missing = REQUIRED_FIELDS - set(fields)
    if missing:
        raise ValueError(f"checkpoint fingerprint missing: {sorted(missing)}")
    return hashlib.sha256(canonical_json({key: fields[key] for key in sorted(REQUIRED_FIELDS)})).hexdigest()


def validate_resume(checkpoint_path: Path, fields: dict[str, Any]) -> None:
    stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if stored.get("fingerprint") != fingerprint(fields):
        raise RuntimeError("CHECKPOINT_INCOMPATIBLE")
