from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from controlflow.v22.dgp import ADVERSARIAL_VARIANTS, OOD_VARIANTS, TEMPLATE_FAMILIES
from controlflow.v22.temporal import CandidateTemporalRetriever, load_policy_corpus
from controlflow.v23.dgp import ADVERSARIAL, OOD, TEMPLATES, generate_v23_split
from controlflow.v23.integrity import file_binding, verify_file_bindings, verify_request_diagnostics


def test_file_binding_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "bound.txt"
    path.write_text("before", encoding="utf-8")
    binding = file_binding(tmp_path, path)
    assert verify_file_bindings(tmp_path, [binding]) == []
    path.write_text("after", encoding="utf-8")
    assert verify_file_bindings(tmp_path, [binding])


def test_request_diagnostic_binding_requires_matching_unique_ids(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics.jsonl"
    partial = tmp_path / "partial.jsonl"
    candidate = tmp_path / "candidate.parquet"
    attribution = tmp_path / "attribution.json"
    diagnostics.write_text(
        json.dumps({"case_id": "CASE-1", "request_start": 1.0, "request_end": 2.0}) + "\n", encoding="utf-8"
    )
    partial.write_text(json.dumps({"case_id": "CASE-1"}) + "\n", encoding="utf-8")
    pd.DataFrame([{"case_id": "CASE-1"}]).to_parquet(candidate, index=False)
    attribution.write_text(json.dumps({"requests": [{"case_id": "CASE-1"}]}), encoding="utf-8")
    evidence = verify_request_diagnostics(tmp_path, diagnostics, partial, candidate, attribution, 1)
    assert evidence["valid"] is True
    diagnostics.write_text(diagnostics.read_text(encoding="utf-8") * 2, encoding="utf-8")
    assert verify_request_diagnostics(tmp_path, diagnostics, partial, candidate, attribution, 1)["valid"] is False


def test_v23_qualification_generator_is_distinct_from_consumed_v22_design(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    assert set(TEMPLATES["QUALIFICATION"]).isdisjoint(TEMPLATE_FAMILIES["QUALIFICATION"])
    assert ADVERSARIAL["QUALIFICATION"] != ADVERSARIAL_VARIANTS["QUALIFICATION"]
    assert OOD["QUALIFICATION"] != OOD_VARIANTS["QUALIFICATION"]
    v22_schedule = [(index + 22901) % 10 for index in range(20)]
    v23_schedule = [(index * 7 + 23907 + 3) % 10 for index in range(20)]
    assert v23_schedule != v22_schedule
    directory = tmp_path / "fresh_v23"
    manifest = generate_v23_split(
        directory, role="QUALIFICATION", count=10, seed=23907, prefix="TEST-V23QUAL", root=root
    )
    runtime = pd.read_parquet(directory / "runtime_cases.parquet")
    truth = pd.read_parquet(directory / "evaluator_truth.parquet")
    evidence = pd.read_parquet(directory / "evidence_corpus.parquet")
    assert manifest["generator_revision"] == 1
    assert all(any(template in narrative for template in TEMPLATES["QUALIFICATION"]) for narrative in runtime.narrative)
    assert all(str(scenario).startswith("v23_") for scenario in truth.temporal_scenario)
    for runtime_row, truth_row in zip(runtime.itertuples(), truth.itertuples(), strict=True):
        expected = evidence[evidence.document_id.isin(truth_row.expected_evidence_ids)]
        assert all(str(value)[:4] == str(runtime_row.event_time)[:4] for value in expected.valid_from)


def test_v23_temporal_truth_fixtures_match_frozen_candidate_oracle() -> None:
    root = Path(__file__).resolve().parents[2]
    fixtures = yaml.safe_load((root / "configs/v23/temporal_truth_fixtures.yaml").read_text(encoding="utf-8"))[
        "fixtures"
    ]
    retriever = CandidateTemporalRetriever(load_policy_corpus(root / "configs/v22/temporal_policies.yaml"))
    for fixture in fixtures:
        event_time = datetime.fromisoformat(str(fixture["event_time"]).replace("Z", "+00:00"))
        system_time = datetime.fromisoformat(str(fixture["system_time"]).replace("Z", "+00:00"))
        assert retriever.select(event_time, system_time) == fixture.get("expected_policy_id")
