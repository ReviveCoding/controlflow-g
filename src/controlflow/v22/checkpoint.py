from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, utc_now

REQUIRED_FIELDS = frozenset(
    {
        "git_commit",
        "dirty_state_hash",
        "dependency_lock_hash",
        "runtime_dataset_hash",
        "evidence_corpus_hash",
        "authorization_state_hash",
        "candidate_bundle_hash",
        "model_hashes",
        "critical_threshold",
        "qwen_revision",
        "vllm_version",
        "structured_backend",
        "prompt_schema_hashes",
        "retrieval_config_hash",
        "temporal_config_hash",
        "pdp_action_registry_hashes",
        "evaluator_protocol_hash",
        "gate_config_hash",
        "seed",
        "concurrency",
    }
)


def fingerprint(fields: dict[str, Any]) -> str:
    missing = REQUIRED_FIELDS - set(fields)
    if missing:
        raise ValueError(f"checkpoint fingerprint missing: {sorted(missing)}")
    return hashlib.sha256(canonical_json({key: fields[key] for key in sorted(REQUIRED_FIELDS)})).hexdigest()


def git_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            ".",
            ":(exclude)data",
            ":(exclude)results",
            ":(exclude)artifacts",
            ":(exclude)state",
            ":(exclude)reports",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "scripts",
            "configs",
            "tests",
            "*.toml",
            "*.lock",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_files = [line for line in untracked_output.splitlines() if line]
    untracked_content: list[str] = []
    for relative in sorted(untracked_files):
        path = root / relative
        if path.is_file():
            untracked_content.append(f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}")
    payload = diff + "\nUNTRACKED_CONTENT\n" + "\n".join(untracked_content)
    return commit, hashlib.sha256(payload.encode()).hexdigest()


def write_checkpoint(path: Path, fields: dict[str, Any], *, completed_case_ids: list[str]) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "fingerprint": fingerprint(fields),
            "fields": fields,
            "completed_case_ids": completed_case_ids,
        },
    )


def validate_checkpoint(path: Path, fields: dict[str, Any]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint(fields):
        raise RuntimeError("CHECKPOINT_INCOMPATIBLE")
    return list(payload.get("completed_case_ids", []))
