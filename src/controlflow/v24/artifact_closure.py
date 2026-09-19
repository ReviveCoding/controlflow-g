"""Frozen file bindings and unbound substantive artifact detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, sha256_file


def binding(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"ARTIFACT_BINDING_PATH_INVALID:{path}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def make_graph(root: Path, named_paths: dict[str, Path], output: Path, *, role: str) -> dict[str, Any]:
    if role not in {"qualification", "final"}:
        raise ValueError("artifact role invalid")
    if len(set(path.resolve() for path in named_paths.values())) != len(named_paths):
        raise RuntimeError("ARTIFACT_BINDING_DUPLICATE_PATH")
    graph = {
        "schema_version": 1,
        "role": role,
        "bindings": {name: binding(root, path) for name, path in sorted(named_paths.items())},
        "rules": binding(root, root / "configs/v24/artifact_rules.yaml"),
    }
    atomic_write_json(output, graph)
    return graph


def unbound_files(root: Path, graph: dict[str, Any]) -> list[str]:
    rules_path = root / graph["rules"]["path"]
    if sha256_file(rules_path) != graph["rules"]["sha256"]:
        raise RuntimeError("ARTIFACT_RULES_BINDING_INVALID")
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    bound = {row["path"] for row in graph["bindings"].values()}
    allowed = set(rules["allowed_ephemeral_files"]) | set(rules["deferred_receipt_files"])
    unbound: list[str] = []
    for relative_root in rules[f"{graph['role']}_roots"]:
        directory = root / relative_root
        if not directory.exists():
            unbound.append(f"MISSING_ROOT:{relative_root}")
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative not in bound and relative not in allowed:
                    unbound.append(relative)
    return sorted(unbound)


def verify_bindings(root: Path, graph: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, row in graph["bindings"].items():
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            failures.append(f"mutation:{name}")
    return failures


def scan_to_report(root: Path, graph_path: Path, output: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    unbound = unbound_files(root, graph)
    report = {
        "schema_version": 1,
        "status": "CLEAN" if not unbound else "UNBOUND_SUBSTANTIVE_ARTIFACT",
        "unbound": unbound,
    }
    atomic_write_json(output, report)
    return report
