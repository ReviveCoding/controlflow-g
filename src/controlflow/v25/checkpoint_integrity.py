"""Post-close checkpoint checks shared by development, qualification, and final."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controlflow.core.state import sha256_file
from controlflow.v22.checkpoint import records_hash
from controlflow.v25.checkpoint_contract import CheckpointEnvelope


@dataclass(frozen=True)
class CheckpointExpectations:
    critical_threshold: float
    candidate_bundle_hash: str
    gate_config_hash: str
    qualification_gate_freeze_hash: str
    runtime_dataset_hash: str
    evidence_corpus_hash: str
    authorization_state_hash: str
    qwen_revision: str
    vllm_version: str
    seed: int
    concurrency: int
    sample_count: int


def verify_checkpoint(
    checkpoint_path: Path, durable_stream_path: Path, expected: CheckpointExpectations
) -> dict[str, Any]:
    """Fail closed on schema, freeze drift, count drift, or durable stream drift."""
    failures: list[str] = []
    try:
        checkpoint = CheckpointEnvelope.load(checkpoint_path)
        comparisons = {
            "critical_threshold": checkpoint.critical_threshold,
            "candidate_bundle_hash": checkpoint.candidate_bundle_hash,
            "gate_config_hash": checkpoint.gate_config_hash,
            "qualification_gate_freeze_hash": checkpoint.qualification_gate_freeze_hash,
            "runtime_dataset_hash": checkpoint.runtime_dataset_hash,
            "evidence_corpus_hash": checkpoint.evidence_corpus_hash,
            "authorization_state_hash": checkpoint.authorization_state_hash,
            "qwen_revision": checkpoint.qwen_revision,
            "vllm_version": checkpoint.vllm_version,
            "seed": checkpoint.seed,
            "concurrency": checkpoint.concurrency,
        }
        for name, observed in comparisons.items():
            if observed != getattr(expected, name):
                failures.append(f"checkpoint_{name}_mismatch")
        records = [json.loads(line) for line in durable_stream_path.read_text(encoding="utf-8").splitlines() if line]
        case_ids = [str(row["case_id"]) for row in records]
        if len(case_ids) != expected.sample_count or checkpoint.completed_case_ids != case_ids:
            failures.append("checkpoint_completed_case_ids_mismatch")
        if checkpoint.completed_records_sha256 != records_hash(records):
            failures.append("checkpoint_durable_stream_hash_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(f"checkpoint_contract_invalid:{type(exc).__name__}:{exc}")
    return {"schema_version": 1, "status": "PASS" if not failures else "FAIL", "failures": failures}


def expectations_from_evidence(
    *,
    candidate_threshold: float,
    candidate_bundle_hash: str,
    gate_config_path: Path,
    freeze_path: Path,
    runtime_path: Path,
    evidence_path: Path,
    authorization_path: Path,
    qwen_revision: str,
    vllm_version: str,
    seed: int,
    sample_count: int,
) -> CheckpointExpectations:
    """Derive expected hashes from sealed artifacts, outside the checkpoint."""
    return CheckpointExpectations(
        critical_threshold=candidate_threshold,
        candidate_bundle_hash=candidate_bundle_hash,
        gate_config_hash=sha256_file(gate_config_path),
        qualification_gate_freeze_hash=sha256_file(freeze_path),
        runtime_dataset_hash=sha256_file(runtime_path),
        evidence_corpus_hash=sha256_file(evidence_path),
        authorization_state_hash=sha256_file(authorization_path),
        qwen_revision=qwen_revision,
        vllm_version=vllm_version,
        seed=seed,
        concurrency=2,
        sample_count=sample_count,
    )
