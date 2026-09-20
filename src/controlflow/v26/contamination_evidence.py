"""Strict, stable contamination semantics with separately bound run provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from controlflow.core.state import canonical_json, sha256_file
from controlflow.v22.dgp import contamination_against_prior, resolve_prior_runtime_paths
from controlflow.v25.contamination import _normalize

SCHEMA_VERSION = 1
ALGORITHM_ID = "v26-v25-prior-overlap-normalized-text"
ALGORITHM_VERSION = "1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InventoryEntry(StrictModel):
    path: str
    sha256: str
    size: int


class ExactMatch(StrictModel):
    candidate_case_id: str
    prior_path: str
    prior_case_id: str
    match_type: Literal["case_id", "entity_id", "normalized_text"]


class NearMatch(StrictModel):
    candidate_case_id: str
    prior_path: str
    prior_case_id: str
    score: str


class ContaminationSemantic(StrictModel):
    schema_version: Literal[1] = 1
    algorithm_id: Literal["v26-v25-prior-overlap-normalized-text"] = "v26-v25-prior-overlap-normalized-text"
    algorithm_version: Literal["1"] = "1"
    candidate_runtime_sha256: str
    candidate_case_count: int
    prior_inventory_digest: str
    prior_runtime_count: int
    contamination_config_digest: str
    exact_case_id_matches: list[ExactMatch]
    exact_normalized_text_matches: list[ExactMatch]
    near_duplicate_matches: list[NearMatch]
    leakage_findings: int

    @model_validator(mode="after")
    def normalize_matches(self) -> ContaminationSemantic:
        self.exact_case_id_matches.sort(
            key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.match_type)
        )
        self.exact_normalized_text_matches.sort(
            key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.match_type)
        )
        self.near_duplicate_matches.sort(
            key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.score)
        )
        return self


class ContaminationProvenance(StrictModel):
    created_at: str
    finished_at: str
    run_id: str
    pid: int
    host: str
    duration_seconds: str
    implementation_version: str


def semantic_digest(semantic: ContaminationSemantic) -> str:
    return hashlib.sha256(canonical_json(semantic.model_dump(mode="json"))).hexdigest()


class ContaminationEvidence(StrictModel):
    schema_version: Literal[1] = 1
    semantic: ContaminationSemantic
    semantic_digest: str
    provenance: ContaminationProvenance

    @model_validator(mode="after")
    def check_digest(self) -> ContaminationEvidence:
        if self.semantic_digest != semantic_digest(self.semantic):
            raise ValueError("CONTAMINATION_SEMANTIC_DIGEST_INVALID")
        return self


def relative_path(root: Path, path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError("CONTAMINATION_PATH_OUTSIDE_REPOSITORY") from exc
    if relative.startswith("/") or "\\" in relative or relative.startswith("../"):
        raise ValueError("CONTAMINATION_PATH_NOT_NORMALIZED")
    return relative


def prior_inventory(
    root: Path, candidate_path: Path, config_path: Path
) -> tuple[list[Path], list[InventoryEntry], str]:
    paths = resolve_prior_runtime_paths(root, config_path, excluded_directory=candidate_path.parent)
    entries = sorted(
        (
            InventoryEntry(path=relative_path(root, path), sha256=sha256_file(path), size=path.stat().st_size)
            for path in paths
        ),
        key=lambda entry: entry.path,
    )
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("CONTAMINATION_DUPLICATE_PRIOR_PATH")
    digest = hashlib.sha256(canonical_json([entry.model_dump() for entry in entries])).hexdigest()
    return [root / entry.path for entry in entries], entries, digest


def config_digest(root: Path, config_path: Path) -> str:
    # Bind the inventory policy, overlap algorithm, normalization, and near-match implementation.
    dependencies = (
        config_path,
        root / "src/controlflow/v22/dgp.py",
        root / "src/controlflow/v25/contamination.py",
        Path(__file__),
    )
    rows = [{"path": relative_path(root, path), "sha256": sha256_file(path)} for path in dependencies]
    return hashlib.sha256(canonical_json(sorted(rows, key=lambda row: row["path"]))).hexdigest()


def _score(value: object) -> str:
    score = float(str(value))
    if not math.isfinite(score):
        raise ValueError("CONTAMINATION_NONFINITE_MATCH_SCORE")
    return format(score, ".17g")


def generate_evidence(
    root: Path, candidate_path: Path, config_path: Path
) -> tuple[ContaminationEvidence, list[InventoryEntry]]:
    started = datetime.now(UTC)
    prior, entries, inventory_digest = prior_inventory(root, candidate_path, config_path)
    raw = contamination_against_prior(candidate_path, prior)
    candidate = pd.read_parquet(candidate_path)
    candidate_texts: dict[str, list[str]] = {}
    for case_id, text in zip(candidate.case_id, candidate.narrative, strict=True):
        candidate_texts.setdefault(_normalize(text), []).append(str(case_id))
    normalized: list[ExactMatch] = []
    for path in prior:
        frame = pd.read_parquet(path)
        if "narrative" not in frame or "case_id" not in frame:
            continue
        prior_rel = relative_path(root, path)
        for case_id, narrative in zip(frame.case_id, frame.narrative, strict=True):
            for current in candidate_texts.get(_normalize(narrative), []):
                normalized.append(
                    ExactMatch(
                        candidate_case_id=current,
                        prior_case_id=str(case_id),
                        prior_path=prior_rel,
                        match_type="normalized_text",
                    )
                )
    exact: list[ExactMatch] = []
    for match in raw["exact_matches"]:
        match_type: Literal["case_id", "entity_id"] = "case_id" if "case_id" in match else "entity_id"
        identifier = str(match.get("case_id", match.get("entity_id")))
        exact.append(
            ExactMatch(
                candidate_case_id=identifier,
                prior_case_id=identifier,
                prior_path=relative_path(root, Path(match["prior_path"])),
                match_type=match_type,
            )
        )
    near = [
        NearMatch(
            candidate_case_id=str(item["candidate_case"]),
            prior_path=relative_path(root, Path(item["prior_path"])),
            prior_case_id=str(item["prior_case"]),
            score=_score(item["similarity"]),
        )
        for item in raw["near_duplicate_matches"]
    ]
    exact.sort(key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.match_type))
    normalized.sort(key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.match_type))
    near.sort(key=lambda item: (item.candidate_case_id, item.prior_path, item.prior_case_id, item.score))
    semantic = ContaminationSemantic(
        candidate_runtime_sha256=sha256_file(candidate_path),
        candidate_case_count=len(candidate),
        prior_inventory_digest=inventory_digest,
        prior_runtime_count=len(entries),
        contamination_config_digest=config_digest(root, config_path),
        exact_case_id_matches=exact,
        exact_normalized_text_matches=normalized,
        near_duplicate_matches=near,
        leakage_findings=len(exact) + len(normalized) + len(near),
    )
    finished = datetime.now(UTC)
    evidence = ContaminationEvidence(
        semantic=semantic,
        semantic_digest=semantic_digest(semantic),
        provenance=ContaminationProvenance(
            created_at=started.isoformat(),
            finished_at=finished.isoformat(),
            run_id=str(uuid.uuid4()),
            pid=os.getpid(),
            host=socket.gethostname(),
            duration_seconds=format((finished - started).total_seconds(), ".6f"),
            implementation_version="v26.1",
        ),
    )
    return evidence, entries


def read_evidence(path: Path) -> ContaminationEvidence:
    return ContaminationEvidence.model_validate(json.loads(path.read_text(encoding="utf-8")))
