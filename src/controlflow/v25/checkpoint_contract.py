"""The sole physical-schema reader for active V2.5 execution checkpoints.

The layout follows controlflow.v22.checkpoint.write_checkpoint, which is called
by CandidateExecutionWorkflow. Version 1 is explicit in that producer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from controlflow.v22.checkpoint import fingerprint

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _hash(value: str, label: str) -> str:
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label}: expected lowercase SHA256")
    return value


class CheckpointFields(BaseModel):
    """All producer fingerprint inputs, with no unclassified additions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    git_commit: StrictStr
    dirty_state_hash: StrictStr
    dependency_lock_hash: StrictStr
    runtime_dataset_hash: StrictStr
    evidence_corpus_hash: StrictStr
    authorization_state_hash: StrictStr
    candidate_bundle_hash: StrictStr
    model_hashes: dict[StrictStr, StrictStr]
    critical_threshold: StrictFloat
    qwen_revision: StrictStr
    vllm_version: StrictStr
    structured_backend: StrictStr
    prompt_schema_hashes: dict[StrictStr, StrictStr]
    retrieval_config_hash: StrictStr
    temporal_config_hash: StrictStr
    pdp_action_registry_hashes: dict[StrictStr, StrictStr]
    evaluator_protocol_hash: StrictStr
    gate_config_hash: StrictStr
    qualification_gate_freeze_hash: StrictStr
    seed: StrictInt
    concurrency: StrictInt

    @model_validator(mode="after")
    def check_values(self) -> Self:
        if not COMMIT.fullmatch(self.git_commit):
            raise ValueError("git_commit: expected lowercase commit hash")
        for name in (
            "dirty_state_hash",
            "dependency_lock_hash",
            "runtime_dataset_hash",
            "evidence_corpus_hash",
            "authorization_state_hash",
            "candidate_bundle_hash",
            "retrieval_config_hash",
            "temporal_config_hash",
            "evaluator_protocol_hash",
            "gate_config_hash",
            "qualification_gate_freeze_hash",
        ):
            _hash(getattr(self, name), name)
        for name in ("model_hashes", "prompt_schema_hashes", "pdp_action_registry_hashes"):
            values: dict[str, str] = getattr(self, name)
            if not values:
                raise ValueError(f"{name}: empty mapping")
            for key, value in values.items():
                if not key:
                    raise ValueError(f"{name}: empty key")
                _hash(value, f"{name}.{key}")
        if not 0 <= self.critical_threshold <= 1:
            raise ValueError("critical_threshold: outside probability range")
        if self.seed < 0 or self.concurrency < 1:
            raise ValueError("seed/concurrency: invalid range")
        for name in ("qwen_revision", "vllm_version", "structured_backend"):
            if not getattr(self, name):
                raise ValueError(f"{name}: empty")
        return self


class CheckpointEnvelope(BaseModel):
    """Validated view; only this class knows checkpoint nesting."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: StrictInt = Field(ge=1)
    updated_at: StrictStr
    fingerprint: StrictStr
    fields: CheckpointFields
    completed_case_ids: list[StrictStr]
    completed_records_sha256: StrictStr

    @model_validator(mode="after")
    def check_envelope(self) -> Self:
        if self.schema_version != 1:
            raise ValueError(f"unsupported checkpoint schema version: {self.schema_version}")
        if not self.updated_at:
            raise ValueError("updated_at: empty")
        if len(self.completed_case_ids) != len(set(self.completed_case_ids)) or any(
            not case_id for case_id in self.completed_case_ids
        ):
            raise ValueError("completed_case_ids: empty or duplicated ID")
        _hash(self.completed_records_sha256, "completed_records_sha256")
        _hash(self.fingerprint, "fingerprint")
        if fingerprint(self.fields.model_dump()) != self.fingerprint:
            raise ValueError("checkpoint fingerprint mismatch")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read and validate a producer checkpoint; malformed JSON fails closed."""
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(payload, strict=True)

    @property
    def critical_threshold(self) -> float:
        return self.fields.critical_threshold

    @property
    def candidate_bundle_hash(self) -> str:
        return self.fields.candidate_bundle_hash

    @property
    def gate_config_hash(self) -> str:
        return self.fields.gate_config_hash

    @property
    def qualification_gate_freeze_hash(self) -> str:
        return self.fields.qualification_gate_freeze_hash

    @property
    def runtime_dataset_hash(self) -> str:
        return self.fields.runtime_dataset_hash

    @property
    def evidence_corpus_hash(self) -> str:
        return self.fields.evidence_corpus_hash

    @property
    def authorization_state_hash(self) -> str:
        return self.fields.authorization_state_hash

    @property
    def qwen_revision(self) -> str:
        return self.fields.qwen_revision

    @property
    def vllm_version(self) -> str:
        return self.fields.vllm_version

    @property
    def concurrency(self) -> int:
        return self.fields.concurrency

    @property
    def seed(self) -> int:
        return self.fields.seed

    @property
    def model_hashes(self) -> dict[str, str]:
        return self.fields.model_hashes

    @property
    def prompt_schema_hashes(self) -> dict[str, str]:
        return self.fields.prompt_schema_hashes

    @property
    def pdp_action_registry_hashes(self) -> dict[str, str]:
        return self.fields.pdp_action_registry_hashes
