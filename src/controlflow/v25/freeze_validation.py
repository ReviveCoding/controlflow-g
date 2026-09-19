"""Verify the committed, immutable qualification executable boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import canonical_json, sha256_file


def verify_freeze(root: Path, *, role: str = "qualification") -> dict[str, Any]:
    if role not in {"development", "qualification", "final"}:
        raise ValueError("unknown freeze role")
    manifest_path = root / (
        "state/v25_development_rehearsal_25013_protocol.json"
        if role == "development"
        else "state/v25_final_freeze_manifest.json"
        if role == "final"
        else "state/v25_freeze_manifest.json"
    )
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if (
        freeze.get("status")
        != ("DEVELOPMENT_PROTOCOL_BOUND" if role == "development" else "QUALIFICATION_PROTOCOL_FROZEN")
        or freeze.get("freeze_hash") != hashlib.sha256(canonical_json(unsigned)).hexdigest()
    ):
        raise RuntimeError("V25_QUALIFICATION_FREEZE_INVALID")
    for item in freeze["bindings"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"V25_QUALIFICATION_FREEZE_BINDING_INVALID:{item['path']}")
    if role == "development":
        return cast(dict[str, Any], freeze)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=root, text=True
    )
    if dirty.strip():
        raise RuntimeError("V25_QUALIFICATION_EXECUTABLE_DIRTY")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD"], cwd=root, check=False
    )
    if ancestor.returncode != 0:
        raise RuntimeError("V25_QUALIFICATION_SOURCE_COMMIT_INVALID")
    committed_name = "state/v25_final_freeze_manifest.json" if role == "final" else "state/v25_freeze_manifest.json"
    committed = subprocess.check_output(["git", "show", f"HEAD:{committed_name}"], cwd=root)
    if json.loads(committed) != freeze:
        raise RuntimeError("V25_QUALIFICATION_FREEZE_NOT_COMMITTED")
    return cast(dict[str, Any], freeze)
