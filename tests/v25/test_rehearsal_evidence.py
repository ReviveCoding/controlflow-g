"""Verify the real V2.5 workflow-generated development closure artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v25.closure_receipt import verify_receipt
from controlflow.v25.terminal_decision import create_anchor, verify_terminal

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "development_rehearsal_25013"


def _json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _reconstructed_rehearsal_root(tmp_path: Path) -> Path:
    """Restore the original CRLF-bound JSONL bytes without changing frozen files."""
    root = tmp_path / "historical_rehearsal"
    graph_name = f"state/v25_{IDENTITY}_bindings.json"
    receipt_name = f"state/v25_{IDENTITY}_closure_receipt.json"
    graph = _json(graph_name)
    receipt = _json(receipt_name)
    bound = {row["path"]: row for row in graph["bindings"].values()}
    names = {
        graph_name,
        receipt_name,
        f"state/v25_{IDENTITY}_manifest.json",
        f"state/v25_{IDENTITY}_manifest.proposed.json",
        f"state/v25_{IDENTITY}_terminal_anchor.json",
        graph["rules"]["path"],
        *(row["path"] for row in graph["bindings"].values()),
        *(row["path"] for row in receipt["deferred_bindings"]),
    }
    for name in sorted(names):
        content = (ROOT / name).read_bytes()
        if name.endswith(".jsonl") and name in bound:
            content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            assert hashlib.sha256(content).hexdigest() == bound[name]["sha256"]
            assert len(content) == bound[name]["size"]
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


def test_real_workflow_checkpoint_loads_all_required_accessors() -> None:
    checkpoint_path = ROOT / "artifacts/v25/development/rehearsal_25013/rehearsal.checkpoint.json"
    assert _json(f"state/v25_{IDENTITY}_manifest.json")["status"] == "DEVELOPMENT_REHEARSAL_PASS"
    view = CheckpointEnvelope.load(checkpoint_path)
    assert view.schema_version == 1
    assert len(view.completed_case_ids) == 60
    assert len(set(view.completed_case_ids)) == 60
    assert 0 <= view.critical_threshold <= 1
    assert len(view.candidate_bundle_hash) == 64
    assert len(view.gate_config_hash) == 64
    assert len(view.qualification_gate_freeze_hash) == 64
    assert len(view.runtime_dataset_hash) == 64
    assert len(view.evidence_corpus_hash) == 64
    assert len(view.authorization_state_hash) == 64
    assert view.qwen_revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert view.vllm_version == "0.29.0"
    assert view.seed == 25013
    assert view.concurrency == 2
    assert view.model_hashes and view.prompt_schema_hashes and view.pdp_action_registry_hashes


def test_stale_policy_metric_matches_independent_denominator() -> None:
    base = "results/v25/development/rehearsal_25013"
    aggregate = _json(f"{base}/aggregate_metrics.json")
    independent = _json(f"{base}/denominators.json")
    stale = "stale_policy_error_rate"
    assert independent["status"] == "VALID"
    assert aggregate[stale]["denominator"] == independent["ratios"][stale]["denominator"] == 15
    assert aggregate[stale]["numerator"] == independent["ratios"][stale]["numerator"]


def test_terminal_verifier_detects_mutated_manifest(tmp_path: Path) -> None:
    root = _reconstructed_rehearsal_root(tmp_path)
    graph = root / f"state/v25_{IDENTITY}_bindings.json"
    receipt = root / f"state/v25_{IDENTITY}_closure_receipt.json"
    proposed = root / f"state/v25_{IDENTITY}_manifest.proposed.json"
    anchor = root / f"state/v25_{IDENTITY}_terminal_anchor.json"
    assert verify_receipt(root, graph, receipt)["status"] == "PASS"
    assert verify_terminal(root, graph, receipt, proposed, anchor, after=True)["status"] == "PASS"
    altered = root / "altered_proposed.json"
    payload = json.loads(proposed.read_text(encoding="utf-8"))
    payload["actual_maximum_llm_concurrency"] = 1
    altered.write_text(json.dumps(payload), encoding="utf-8")
    altered_anchor = root / "altered_anchor.json"
    create_anchor(root, receipt, altered, altered_anchor)
    result = verify_terminal(root, graph, receipt, altered, altered_anchor, after=True)
    assert "terminal_manifest_mutated" in result["failures"]


def test_unbound_substantive_state_artifact_is_rejected(tmp_path: Path) -> None:
    root = _reconstructed_rehearsal_root(tmp_path)
    graph = root / f"state/v25_{IDENTITY}_bindings.json"
    receipt = root / f"state/v25_{IDENTITY}_closure_receipt.json"
    extra = root / f"state/v25_{IDENTITY}_unbound_test.json"
    assert not extra.exists()
    try:
        extra.write_text('{"synthetic_test":true}', encoding="utf-8")
        result = verify_receipt(root, graph, receipt)
        assert any("unbound_substantive_state" in failure for failure in result["failures"])
    finally:
        extra.unlink(missing_ok=True)
