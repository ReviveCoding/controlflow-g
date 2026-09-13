from __future__ import annotations

import time
from pathlib import Path

from controlflow.v23.latency import bootstrap_quantile_ci, parse_prometheus, percentile_summary, tail_membership
from controlflow.v23.vllm_client import AttributedVllmClient, MinimalExplanation


def test_prometheus_parser_preserves_labels_and_numeric_samples() -> None:
    text = '# HELP ignored x\nvllm:kv_cache_usage_perc{model_name="m"} 0.25\nbad text\n'
    assert parse_prometheus(text) == [
        {"name": "vllm:kv_cache_usage_perc", "labels": {"model_name": "m"}, "value": 0.25}
    ]


def test_latency_summaries_are_deterministic() -> None:
    assert percentile_summary([1, 2, 3, 4])["p95"] == 3.8499999999999996
    assert bootstrap_quantile_ci([1, 2, 3], samples=20) == bootstrap_quantile_ci([1, 2, 3], samples=20)
    assert tail_membership([{"total_latency_seconds": value} for value in range(100)])[0]["count"] == 1


def test_minimal_explanation_rejects_extra_safety_fields() -> None:
    valid = MinimalExplanation.model_validate({"summary": "brief", "claims": ["evidence_count=2"]})
    assert valid.summary == "brief"
    try:
        MinimalExplanation.model_validate({"summary": "brief", "claims": ["evidence_count=2"], "severity": "LOW"})
    except ValueError:
        pass
    else:
        raise AssertionError("extra deterministic safety field must be rejected")


def test_request_diagnostics_are_durable_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.jsonl"
    arguments = {
        "endpoint": "http://unused",
        "model": "model",
        "schema": {},
        "prompt_template": "{{INPUT_JSON}}",
        "max_tokens": 128,
        "contract": "minimal",
        "diagnostics_path": path,
    }
    client = AttributedVllmClient(**arguments)
    client._finish({"case_id": "CASE-1"}, time.perf_counter(), "summary", True, ("evidence_count=1",))
    reloaded = AttributedVllmClient(**arguments)
    assert [row["case_id"] for row in reloaded.diagnostics] == ["CASE-1"]
    reloaded.clear_diagnostics()
    assert path.read_text(encoding="utf-8") == ""
