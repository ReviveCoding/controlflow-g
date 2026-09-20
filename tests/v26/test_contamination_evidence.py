"""Semantic/provenance separation and fail-closed schema mutation tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from controlflow.core.state import atomic_write_json, sha256_file
from controlflow.v26.contamination_evidence import (
    ContaminationEvidence,
    ContaminationSemantic,
    semantic_digest,
)


def _semantic() -> ContaminationSemantic:
    return ContaminationSemantic(
        candidate_runtime_sha256="a" * 64,
        candidate_case_count=60,
        prior_inventory_digest="b" * 64,
        prior_runtime_count=3,
        contamination_config_digest="c" * 64,
        exact_case_id_matches=[],
        exact_normalized_text_matches=[],
        near_duplicate_matches=[],
        leakage_findings=0,
    )


def _report() -> dict[str, object]:
    semantic = _semantic()
    return {
        "schema_version": 1,
        "semantic": semantic.model_dump(),
        "semantic_digest": semantic_digest(semantic),
        "provenance": {
            "created_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "run_id": "first",
            "pid": 1,
            "host": "host",
            "duration_seconds": "1.0",
            "implementation_version": "v26.1",
        },
    }


def test_repeated_digest() -> None:
    assert semantic_digest(_semantic()) == semantic_digest(_semantic())


@pytest.mark.parametrize(
    "field,value",
    [
        ("created_at", "2026-01-02T00:00:00Z"),
        ("finished_at", "2026-01-02T00:00:01Z"),
        ("run_id", "second"),
        ("pid", 2),
        ("duration_seconds", "2.0"),
    ],
)
def test_provenance_changes_artifact_not_semantics(tmp_path: Path, field: str, value: object) -> None:
    original = _report()
    changed = copy.deepcopy(original)
    changed["provenance"][field] = value  # type: ignore[index]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    atomic_write_json(first, original)
    atomic_write_json(second, changed)
    assert sha256_file(first) != sha256_file(second)
    assert ContaminationEvidence.model_validate(original).semantic_digest == (
        ContaminationEvidence.model_validate(changed).semantic_digest
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("leakage_findings", 1),
        ("candidate_runtime_sha256", "d" * 64),
        ("prior_inventory_digest", "e" * 64),
        ("contamination_config_digest", "f" * 64),
        ("candidate_case_count", 61),
        (
            "exact_case_id_matches",
            [{"candidate_case_id": "A", "prior_path": "data/a.parquet", "prior_case_id": "A", "match_type": "case_id"}],
        ),
        (
            "exact_normalized_text_matches",
            [
                {
                    "candidate_case_id": "A",
                    "prior_path": "data/a.parquet",
                    "prior_case_id": "B",
                    "match_type": "normalized_text",
                }
            ],
        ),
        (
            "near_duplicate_matches",
            [{"candidate_case_id": "A", "prior_path": "data/a.parquet", "prior_case_id": "B", "score": "0.95"}],
        ),
    ],
)
def test_semantic_mutation_changes_digest(field: str, value: object) -> None:
    original = _semantic()
    changed = original.model_dump()
    changed[field] = value
    assert semantic_digest(original) != semantic_digest(ContaminationSemantic.model_validate(changed))


def test_match_order_invariance() -> None:
    rows = [
        {
            "candidate_case_id": str(index),
            "prior_path": f"data/{index}.parquet",
            "prior_case_id": str(index),
            "match_type": "case_id",
        }
        for index in range(5)
    ]
    first = _semantic().model_dump()
    second = _semantic().model_dump()
    first["exact_case_id_matches"] = rows
    second["exact_case_id_matches"] = list(reversed(rows))
    assert semantic_digest(ContaminationSemantic.model_validate(first)) == semantic_digest(
        ContaminationSemantic.model_validate(second)
    )


@pytest.mark.parametrize("part", ["semantic", "provenance"])
def test_unknown_field_rejected(part: str) -> None:
    report = _report()
    report[part]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ValidationError):
        ContaminationEvidence.model_validate(report)


def test_tampered_digest_rejected() -> None:
    report = _report()
    report["semantic_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        ContaminationEvidence.model_validate(report)


def test_report_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    atomic_write_json(path, _report())
    assert (
        ContaminationEvidence.model_validate(json.loads(path.read_text())).semantic_digest
        == _report()["semantic_digest"]
    )
