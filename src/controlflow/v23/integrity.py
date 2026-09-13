from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from controlflow.core.state import sha256_file
from controlflow.v22.approval import ApprovalIssuer


def load_frozen_approval_issuer(root: Path) -> ApprovalIssuer:
    """Load the V2.2 signer without rewriting any frozen V2.2 artifact."""
    private_key = serialization.load_pem_private_key(
        (root / "artifacts/v22/local_keys/approval_ed25519.private.pem").read_bytes(), password=None
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("frozen approval private key is not Ed25519")
    return ApprovalIssuer(private_key, key_id="v22-dev-reviewer")


def file_binding(root: Path, path: Path) -> dict[str, int | str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_file_bindings(root: Path, bindings: Iterable[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    resolved_root = root.resolve()
    for binding in bindings:
        relative = str(binding.get("path", ""))
        path = (root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            failures.append(f"binding_outside_root:{relative}")
            continue
        if not relative or not path.is_file():
            failures.append(f"missing_binding:{relative}")
            continue
        if binding.get("sha256") != sha256_file(path):
            failures.append(f"binding_hash_mismatch:{relative}")
        if binding.get("bytes") != path.stat().st_size:
            failures.append(f"binding_size_mismatch:{relative}")
    return failures


def verify_hash_bindings(root: Path, bindings: Iterable[dict[str, Any]]) -> list[str]:
    """Verify path/SHA bindings that predate the byte-size binding contract."""
    failures: list[str] = []
    resolved_root = root.resolve()
    for binding in bindings:
        relative = str(binding.get("path", ""))
        path = (root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            failures.append(f"hash_binding_outside_root:{relative}")
            continue
        if not relative or not path.is_file():
            failures.append(f"missing_hash_binding:{relative}")
        elif binding.get("sha256") != sha256_file(path):
            failures.append(f"hash_binding_mismatch:{relative}")
    return failures


def verify_historical_boundary(root: Path, boundary_path: Path) -> dict[str, Any]:
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    observed: list[dict[str, Any]] = []
    for expected in boundary["tags"]:
        tag = str(expected["tag"])
        object_result = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}"], cwd=root, capture_output=True, text=True
        )
        peeled_result = subprocess.run(["git", "rev-parse", f"{tag}^{{}}"], cwd=root, capture_output=True, text=True)
        if object_result.returncode or peeled_result.returncode:
            failures.append(f"missing_historical_tag:{tag}")
            continue
        item = {
            "tag": tag,
            "object_hash": object_result.stdout.strip(),
            "peeled_hash": peeled_result.stdout.strip(),
        }
        observed.append(item)
        if item["object_hash"] != expected["object_hash"] or item["peeled_hash"] != expected["peeled_hash"]:
            failures.append(f"historical_tag_moved:{tag}")
    return {"valid": not failures, "failures": failures, "observed": observed}


def verify_request_diagnostics(
    root: Path,
    diagnostics_path: Path,
    partial_path: Path,
    candidate_path: Path,
    attribution_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        diagnostics = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line]
        partial = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
        candidate = pd.read_parquet(candidate_path, columns=["case_id"])
        attribution = json.loads(attribution_path.read_text(encoding="utf-8"))["requests"]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {"valid": False, "violations": [f"unreadable:{type(exc).__name__}:{exc}"]}
    id_sets = {
        "diagnostics": [str(row.get("case_id", "")) for row in diagnostics],
        "partial": [str(row.get("case_id", "")) for row in partial],
        "candidate": [str(value) for value in candidate["case_id"].tolist()],
        "attribution": [str(row.get("case_id", "")) for row in attribution],
    }
    for name, ids in id_sets.items():
        if len(ids) != expected_count:
            failures.append(f"{name}_count:{len(ids)}")
        if len(ids) != len(set(ids)) or "" in ids:
            failures.append(f"{name}_duplicate_or_empty_case_id")
    reference = set(id_sets["partial"])
    for name, ids in id_sets.items():
        if set(ids) != reference:
            failures.append(f"{name}_case_id_set_mismatch")
    for index, row in enumerate(diagnostics):
        start = row.get("request_start")
        end = row.get("request_end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
            failures.append(f"diagnostic_timing_invalid:{index}")
    bindings = [file_binding(root, path) for path in (diagnostics_path, partial_path, candidate_path, attribution_path)]
    return {
        "valid": not failures,
        "violations": failures,
        "expected_count": expected_count,
        "record_counts": {name: len(ids) for name, ids in id_sets.items()},
        "unique_case_ids": {name: len(set(ids)) for name, ids in id_sets.items()},
        "bindings": bindings,
    }
