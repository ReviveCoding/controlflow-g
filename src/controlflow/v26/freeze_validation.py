"""Verify the committed, immutable qualification executable boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import canonical_json, sha256_file


def verify_freeze(root: Path, *, role: str = "qualification", development_seed: int | None = None) -> dict[str, Any]:
    if role not in {"development", "qualification", "final"}:
        raise ValueError("unknown freeze role")
    if role == "development" and development_seed is None:
        raise ValueError("development seed required")
    manifest_path = root / (
        f"state/v26_development_rehearsal_{development_seed}_protocol.json"
        if role == "development"
        else "state/v26_final_freeze_manifest.json"
        if role == "final"
        else "state/v26_freeze_manifest.json"
    )
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if (
        freeze.get("status")
        != ("DEVELOPMENT_PROTOCOL_BOUND" if role == "development" else "QUALIFICATION_PROTOCOL_FROZEN")
        or freeze.get("freeze_hash") != hashlib.sha256(canonical_json(unsigned)).hexdigest()
    ):
        raise RuntimeError("V26_QUALIFICATION_FREEZE_INVALID")
    for item in freeze["bindings"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"V26_QUALIFICATION_FREEZE_BINDING_INVALID:{item['path']}")
    if role == "development":
        return cast(dict[str, Any], freeze)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=root, text=True
    )
    if dirty.strip():
        raise RuntimeError("V26_QUALIFICATION_EXECUTABLE_DIRTY")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD"], cwd=root, check=False
    )
    if ancestor.returncode != 0:
        raise RuntimeError("V26_QUALIFICATION_SOURCE_COMMIT_INVALID")
    committed_name = "state/v26_final_freeze_manifest.json" if role == "final" else "state/v26_freeze_manifest.json"
    committed = subprocess.check_output(["git", "show", f"HEAD:{committed_name}"], cwd=root)
    if json.loads(committed) != freeze:
        raise RuntimeError("V26_QUALIFICATION_FREEZE_NOT_COMMITTED")
    return cast(dict[str, Any], freeze)
