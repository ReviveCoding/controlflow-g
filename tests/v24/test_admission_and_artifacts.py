from __future__ import annotations

from pathlib import Path

from controlflow.v24.admission import temporal_oracle
from controlflow.v24.artifact_closure import make_graph, unbound_files, verify_bindings


def test_temporal_oracle_rejects_future_and_late_correction() -> None:
    policies = [
        {
            "policy_id": "base",
            "business_valid_from": "2023-01-01T00:00:00Z",
            "business_valid_to": "2024-01-01T00:00:00Z",
            "system_known_from": "2022-12-01T00:00:00Z",
            "correction_rank": 0,
        },
        {
            "policy_id": "correction",
            "business_valid_from": "2023-01-01T00:00:00Z",
            "business_valid_to": "2024-01-01T00:00:00Z",
            "system_known_from": "2025-03-01T00:00:00Z",
            "correction_rank": 1,
        },
    ]
    assert temporal_oracle("2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z", policies) == "base"
    assert temporal_oracle("2023-06-01T00:00:00Z", "2025-04-01T00:00:00Z", policies) == "correction"


def test_artifact_graph_detects_unbound_and_mutation(tmp_path: Path) -> None:
    rules = tmp_path / "configs/v24/artifact_rules.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "schema_version: 1\nqualification_roots: [results/v24/qualification]\n"
        "final_roots: []\nallowed_ephemeral_files: []\ndeferred_receipt_files: []\n"
        "substantive_patterns: ['**/*']\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "results/v24/qualification/output.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("original", encoding="utf-8")
    graph = make_graph(tmp_path, {"output": artifact}, tmp_path / "graph.json", role="qualification")
    assert unbound_files(tmp_path, graph) == []
    extra = artifact.parent / "unbound.json"
    extra.write_text("substantive", encoding="utf-8")
    assert unbound_files(tmp_path, graph) == ["results/v24/qualification/unbound.json"]
    artifact.write_text("tampered", encoding="utf-8")
    assert verify_bindings(tmp_path, graph) == ["mutation:output"]
