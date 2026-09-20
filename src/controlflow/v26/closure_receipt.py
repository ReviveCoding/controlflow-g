"""Bind deferred decision artifacts, leaving only explicit terminal trust anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json
from controlflow.v26.artifact_closure import binding, rules_for_graph, unbound_files, verify_bindings


def create_receipt(root: Path, graph_path: Path, output: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    rules = rules_for_graph(root, graph)
    role = graph["role"]
    state_key = f"{role}_state_substantive_files" if role != "qualification" else "state_substantive_files"
    anchors_key = f"{role}_terminal_root_anchors" if role != "qualification" else "terminal_root_anchors"
    paths = [root / name for name in rules[state_key]]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "CLOSURE_BOUND",
        "qualification_graph": binding(root, graph_path),
        "deferred_bindings": [binding(root, path) for path in paths],
        "terminal_root_anchors": rules[anchors_key],
    }
    receipt["receipt_hash"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    atomic_write_json(output, receipt)
    return receipt


def verify_receipt(root: Path, graph_path: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    role = graph["role"]
    state_key = f"{role}_state_substantive_files" if role != "qualification" else "state_substantive_files"
    anchors_key = f"{role}_terminal_root_anchors" if role != "qualification" else "terminal_root_anchors"
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    failures: list[str] = []
    if receipt.get("receipt_hash") != hashlib.sha256(canonical_json(unsigned)).hexdigest():
        failures.append("receipt_self_hash")
    for item in [receipt["qualification_graph"], *receipt["deferred_bindings"]]:
        path = root / item["path"]
        if not path.is_file() or binding(root, path) != item:
            failures.append(f"receipt_binding:{item['path']}")
    failures.extend(verify_bindings(root, graph))
    failures.extend(f"unbound:{path}" for path in unbound_files(root, graph))
    rules = rules_for_graph(root, graph)
    declared = set(rules[state_key])
    bound = {row["path"] for row in receipt["deferred_bindings"]}
    if declared != bound or receipt.get("terminal_root_anchors") != rules[anchors_key]:
        failures.append("state_artifact_inventory_mismatch")
    permitted_state = declared | set(rules[anchors_key]) | {row["path"] for row in graph["bindings"].values()}
    pattern_key = f"{role}_state_substantive_patterns"
    patterns = rules[pattern_key]
    if not isinstance(patterns, list) or not patterns or any(not isinstance(item, str) for item in patterns):
        failures.append("state_substantive_pattern_invalid")
        return {"status": "FAIL", "failures": failures, "receipt_hash": receipt["receipt_hash"]}
    for pattern in patterns:
        for path in (root / "state").glob(pattern):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and relative not in permitted_state:
                failures.append(f"unbound_substantive_state:{relative}")
    decision_names = [name for name in declared if name.endswith("/gate_decision.json")]
    postclose_names = [name for name in declared if name.endswith("postclose_verification.json")]
    if len(decision_names) != 1 or len(postclose_names) != 1:
        failures.append("decision_or_postclose_inventory_invalid")
        return {"status": "FAIL", "failures": failures, "receipt_hash": receipt["receipt_hash"]}
    decision = json.loads((root / decision_names[0]).read_text(encoding="utf-8"))
    postclose = json.loads((root / postclose_names[0]).read_text(encoding="utf-8"))
    gate_rows = decision.get("gates", {})
    if (
        not isinstance(decision.get("all_passed"), bool)
        or not gate_rows
        or decision["all_passed"] != all(bool(row.get("passed")) for row in gate_rows.values())
        or postclose.get("status") != "PASS"
    ):
        failures.append("decision_or_postclose_invalid")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "receipt_hash": receipt["receipt_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = verify_receipt(args.root, args.graph, args.receipt)
    if result["status"] != "PASS":
        print(f"CLOSURE_RECEIPT_INVALID:{result['failures']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
