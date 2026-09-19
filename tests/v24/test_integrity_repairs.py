from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlflow.core.state import atomic_write_json
from controlflow.v24.artifact_closure import make_graph
from controlflow.v24.closure_receipt import create_receipt, verify_receipt
from controlflow.v24.gpu_preflight import _descends, verify_gpu_ownership
from controlflow.v24.terminal_decision import create_anchor, verify_terminal


def test_gpu_parent_chain_rejects_unrelated_consumer() -> None:
    parents = {20: 10, 21: 20, 30: 1}
    assert _descends(21, 10, parents)
    assert not _descends(30, 10, parents)
    assert not _descends(20, 30, parents)


def test_gpu_lock_owner_fails_before_process_probe(tmp_path: Path) -> None:
    lock = tmp_path / "state/gpu.lock"
    lock.mkdir(parents=True)
    (lock / "owner.pid").write_text("123", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OWNER_INVALID"):
        verify_gpu_ownership(tmp_path, 124)


def test_receipt_detects_mutated_decision_artifact(tmp_path: Path) -> None:
    root = tmp_path
    for name in (
        "configs/v24",
        "data/v24/qualification/V24QUAL",
        "results/v24/qualification",
        "artifacts/v24/qualification",
        "state",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "configs/v24/artifact_rules.yaml").write_text(
        "schema_version: 1\n"
        "qualification_roots: [data/v24/qualification/V24QUAL, results/v24/qualification, "
        "artifacts/v24/qualification]\n"
        "final_roots: []\nallowed_ephemeral_files: []\n"
        "deferred_receipt_files: [results/v24/qualification/gate_decision.json]\n"
        "state_substantive_files: [state/v24_qualification_bindings.json, state/v24_unbound_artifacts.json, "
        "state/v24_postclose_verification.json, results/v24/qualification/gate_decision.json]\n"
        "state_substantive_patterns: [v24_qualification*.json, v24_postclose*.json, v24_unbound*.json]\n"
        "terminal_root_anchors: [state/v24_qualification_closure_receipt.json, state/v24_qualification_manifest.json]\n"
        "substantive_patterns: ['**/*']\n",
        encoding="utf-8",
    )
    evidence = root / "data/v24/qualification/V24QUAL/input.json"
    atomic_write_json(evidence, {"value": 1})
    graph_path = root / "state/v24_qualification_bindings.json"
    make_graph(root, {"input": evidence}, graph_path, role="qualification")
    atomic_write_json(root / "state/v24_unbound_artifacts.json", {"status": "CLEAN"})
    atomic_write_json(root / "state/v24_postclose_verification.json", {"status": "PASS"})
    decision = root / "results/v24/qualification/gate_decision.json"
    atomic_write_json(decision, {"all_passed": True, "gates": {"one": {"passed": True}}})
    receipt = root / "state/v24_qualification_closure_receipt.json"
    create_receipt(root, graph_path, receipt)
    assert verify_receipt(root, graph_path, receipt)["status"] == "PASS"
    atomic_write_json(decision, {"all_passed": False, "gates": {"one": {"passed": False}}})
    assert verify_receipt(root, graph_path, receipt)["status"] == "FAIL"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "CLOSURE_BOUND"


def test_terminal_manifest_mutation_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = tmp_path / "receipt.json"
    proposed = tmp_path / "proposed.json"
    anchor = tmp_path / "anchor.json"
    atomic_write_json(receipt, {"status": "CLOSURE_BOUND"})
    atomic_write_json(proposed, {"status": "COMPLETE", "all_passed": True})
    create_anchor(tmp_path, receipt, proposed, anchor)
    manifest = tmp_path / "state/v24_qualification_manifest.json"
    manifest.parent.mkdir()
    atomic_write_json(manifest, {"status": "COMPLETE", "all_passed": True})
    monkeypatch.setattr(
        "controlflow.v24.terminal_decision.verify_receipt",
        lambda *_arguments: {"status": "PASS", "failures": []},
    )
    decision = tmp_path / "results/v24/qualification/gate_decision.json"
    decision.parent.mkdir(parents=True)
    atomic_write_json(decision, {"all_passed": True, "gates": {"one": {"passed": True}}})
    payload = json.loads(proposed.read_text(encoding="utf-8"))
    from controlflow.core.state import sha256_file

    payload.update(
        {
            "gates": json.loads(decision.read_text(encoding="utf-8")),
            "closure_receipt_sha256": sha256_file(receipt),
            "gate_decision_sha256": sha256_file(decision),
        }
    )
    atomic_write_json(proposed, payload)
    create_anchor(tmp_path, receipt, proposed, anchor)
    atomic_write_json(manifest, payload)
    assert verify_terminal(tmp_path, tmp_path / "unused", receipt, proposed, anchor, after=True)["status"] == "PASS"
    payload["status"] = "V24_DEVELOPMENT_NO_GO"
    atomic_write_json(manifest, payload)
    assert verify_terminal(tmp_path, tmp_path / "unused", receipt, proposed, anchor, after=True)["status"] == "FAIL"
