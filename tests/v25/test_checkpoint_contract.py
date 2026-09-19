"""Mutate real producer serialization, keeping its fingerprint valid where useful."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from controlflow.v22.checkpoint import fingerprint, write_checkpoint
from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v25.checkpoint_integrity import CheckpointExpectations, verify_checkpoint

H = "a" * 64


@pytest.fixture
def produced_checkpoint(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    # This exercises the actual serialization function used by CandidateExecutionWorkflow.
    # A real workflow-generated positive fixture remains a pre-qualification requirement.
    fields: dict[str, Any] = {
        "git_commit": "b" * 40,
        "dirty_state_hash": H,
        "dependency_lock_hash": H,
        "runtime_dataset_hash": H,
        "evidence_corpus_hash": H,
        "authorization_state_hash": H,
        "candidate_bundle_hash": H,
        "model_hashes": {"critical_model": H},
        "critical_threshold": 0.25,
        "qwen_revision": "revision",
        "vllm_version": "0.29.0",
        "structured_backend": "xgrammar",
        "prompt_schema_hashes": {"prompt": H},
        "retrieval_config_hash": H,
        "temporal_config_hash": H,
        "pdp_action_registry_hashes": {"pdp": H},
        "evaluator_protocol_hash": H,
        "gate_config_hash": H,
        "qualification_gate_freeze_hash": H,
        "seed": 25001,
        "concurrency": 2,
    }
    checkpoint = tmp_path / "checkpoint.json"
    stream = tmp_path / "partial.jsonl"
    row = {"case_id": "V25DEV-00001"}
    stream.write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_checkpoint(checkpoint, fields, completed_case_ids=[row["case_id"]], completed_records=[row])
    return checkpoint, stream, fields


def _mutate(checkpoint: Path, payload: dict[str, Any]) -> None:
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")


def _refingerprint(payload: dict[str, Any]) -> None:
    payload["fingerprint"] = fingerprint(payload["fields"])


def test_producer_serializer_loads(produced_checkpoint: tuple[Path, Path, dict[str, Any]]) -> None:
    checkpoint, _, fields = produced_checkpoint
    view = CheckpointEnvelope.load(checkpoint)
    assert view.critical_threshold == fields["critical_threshold"]
    assert view.candidate_bundle_hash == fields["candidate_bundle_hash"]
    assert view.gate_config_hash == fields["gate_config_hash"]
    assert view.completed_case_ids == ["V25DEV-00001"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_fields",
        "threshold_missing",
        "threshold_wrong_level",
        "threshold_wrong_type",
        "bundle_missing",
        "gate_missing",
        "freeze_missing",
        "models_missing",
        "prompts_wrong_type",
        "ids_wrong_type",
        "records_hash_missing",
        "seed_wrong_type",
        "concurrency_wrong_type",
        "unknown_version",
        "incompatible_nested",
    ],
)
def test_schema_mutations_fail_closed(produced_checkpoint: tuple[Path, Path, dict[str, Any]], mutation: str) -> None:
    checkpoint, _, _ = produced_checkpoint
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if mutation == "missing_fields":
        del payload["fields"]
    elif mutation == "threshold_missing":
        del payload["fields"]["critical_threshold"]
    elif mutation == "threshold_wrong_level":
        payload["critical_threshold"] = payload["fields"].pop("critical_threshold")
    elif mutation == "threshold_wrong_type":
        payload["fields"]["critical_threshold"] = "0.25"
    elif mutation == "bundle_missing":
        del payload["fields"]["candidate_bundle_hash"]
    elif mutation == "gate_missing":
        del payload["fields"]["gate_config_hash"]
    elif mutation == "freeze_missing":
        del payload["fields"]["qualification_gate_freeze_hash"]
    elif mutation == "models_missing":
        del payload["fields"]["model_hashes"]
    elif mutation == "prompts_wrong_type":
        payload["fields"]["prompt_schema_hashes"] = [H]
    elif mutation == "ids_wrong_type":
        payload["completed_case_ids"] = "V25DEV-00001"
    elif mutation == "records_hash_missing":
        del payload["completed_records_sha256"]
    elif mutation == "seed_wrong_type":
        payload["fields"]["seed"] = "25001"
    elif mutation == "concurrency_wrong_type":
        payload["fields"]["concurrency"] = 2.0
    elif mutation == "unknown_version":
        payload["schema_version"] = 2
    elif mutation == "incompatible_nested":
        payload["fields"]["future_field"] = H
    _mutate(checkpoint, payload)
    with pytest.raises((ValueError, KeyError)):
        CheckpointEnvelope.load(checkpoint)


def test_truncated_json_fails_closed(produced_checkpoint: tuple[Path, Path, dict[str, Any]]) -> None:
    checkpoint, _, _ = produced_checkpoint
    checkpoint.write_text('{"fields":', encoding="utf-8")
    with pytest.raises(ValueError):
        CheckpointEnvelope.load(checkpoint)


@pytest.mark.parametrize(
    "changed,expected_failure",
    [
        ("critical_threshold", "checkpoint_critical_threshold_mismatch"),
        ("candidate_bundle_hash", "checkpoint_candidate_bundle_hash_mismatch"),
        ("gate_config_hash", "checkpoint_gate_config_hash_mismatch"),
        ("qualification_gate_freeze_hash", "checkpoint_qualification_gate_freeze_hash_mismatch"),
    ],
)
def test_postclose_rejects_frozen_value_drift(
    produced_checkpoint: tuple[Path, Path, dict[str, Any]], changed: str, expected_failure: str
) -> None:
    checkpoint, stream, fields = produced_checkpoint
    expected = CheckpointExpectations(
        critical_threshold=fields["critical_threshold"],
        candidate_bundle_hash=H,
        gate_config_hash=H,
        qualification_gate_freeze_hash=H,
        runtime_dataset_hash=H,
        evidence_corpus_hash=H,
        authorization_state_hash=H,
        qwen_revision="revision",
        vllm_version="0.29.0",
        seed=25001,
        concurrency=2,
        sample_count=1,
    )
    assert verify_checkpoint(checkpoint, stream, expected)["status"] == "PASS"
    payload = deepcopy(json.loads(checkpoint.read_text(encoding="utf-8")))
    payload["fields"][changed] = 0.5 if changed == "critical_threshold" else "c" * 64
    _refingerprint(payload)
    _mutate(checkpoint, payload)
    assert expected_failure in verify_checkpoint(checkpoint, stream, expected)["failures"]
