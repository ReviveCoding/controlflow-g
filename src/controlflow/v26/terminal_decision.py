"""Commit the exact prospective terminal manifest through a separate verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v26.artifact_closure import binding
from controlflow.v26.closure_receipt import verify_receipt


def create_anchor(root: Path, receipt: Path, proposed: Path, output: Path) -> dict[str, Any]:
    anchor: dict[str, Any] = {
        "schema_version": 1,
        "status": "TERMINAL_DECISION_BOUND",
        "closure_receipt": binding(root, receipt),
        "proposed_manifest": binding(root, proposed),
    }
    anchor["anchor_hash"] = hashlib.sha256(canonical_json(anchor)).hexdigest()
    atomic_write_json(output, anchor)
    return anchor


def verify_terminal(
    root: Path, graph: Path, receipt: Path, proposed: Path, anchor_path: Path, *, after: bool
) -> dict[str, Any]:
    graph_payload = json.loads(graph.read_text(encoding="utf-8"))
    role = str(graph_payload["role"])
    identity = str(graph_payload.get("identity", role))
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in anchor.items() if key != "anchor_hash"}
    failures: list[str] = []
    if anchor.get("anchor_hash") != hashlib.sha256(canonical_json(unsigned)).hexdigest():
        failures.append("anchor_self_hash")
    if binding(root, receipt) != anchor["closure_receipt"] or binding(root, proposed) != anchor["proposed_manifest"]:
        failures.append("terminal_binding_mismatch")
    receipt_result = verify_receipt(root, graph, receipt)
    failures.extend(receipt_result["failures"])
    manifest = json.loads(proposed.read_text(encoding="utf-8"))
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    decision_bindings = [
        row for row in receipt_payload["deferred_bindings"] if str(row["path"]).endswith("/gate_decision.json")
    ]
    if len(decision_bindings) != 1:
        raise RuntimeError("TERMINAL_DECISION_BINDING_MISSING")
    decision_path = root / decision_bindings[0]["path"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected_status = (
        "DEVELOPMENT_REHEARSAL_PASS"
        if role == "development" and decision["all_passed"]
        else "COMPLETE"
        if decision["all_passed"]
        else "V26_DEVELOPMENT_NO_GO"
        if role == "development"
        else "NO_PROMOTE"
    )
    if (
        manifest.get("status") != expected_status
        or manifest.get("all_passed") != decision["all_passed"]
        or manifest.get("gates") != decision
        or manifest.get("closure_receipt_sha256") != sha256_file(receipt)
        or manifest.get("gate_decision_sha256") != sha256_file(decision_path)
    ):
        failures.append("terminal_decision_mismatch")
    if (
        after
        and binding(root, root / f"state/v26_{identity}_manifest.json")["sha256"]
        != anchor["proposed_manifest"]["sha256"]
    ):
        failures.append("terminal_manifest_mutated")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("proposed", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("--after", action="store_true")
    args = parser.parse_args()
    result = verify_terminal(args.root, args.graph, args.receipt, args.proposed, args.anchor, after=args.after)
    if result["status"] != "PASS":
        print(f"TERMINAL_DECISION_INVALID:{result['failures']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
