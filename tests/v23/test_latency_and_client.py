from __future__ import annotations

import time
from pathlib import Path

import pytest

from controlflow.v23.integrity import load_frozen_approval_issuer
from controlflow.v23.latency import (
    block_bootstrap_quantile_ci,
    bootstrap_quantile_ci,
    parse_prometheus,
    percentile_summary,
    tail_membership,
)
from controlflow.v23.vllm_client import AttributedVllmClient, MinimalExplanation


def test_prometheus_parser_preserves_labels_and_numeric_samples() -> None:
    text = '# HELP ignored x\nvllm:kv_cache_usage_perc{model_name="m"} 0.25\nbad text\n'
    assert parse_prometheus(text) == [
        {"name": "vllm:kv_cache_usage_perc", "labels": {"model_name": "m"}, "value": 0.25}
    ]


def test_latency_summaries_are_deterministic() -> None:
    assert percentile_summary([1, 2, 3, 4])["p95"] == 3.8499999999999996
    assert bootstrap_quantile_ci([1, 2, 3], samples=20) == bootstrap_quantile_ci([1, 2, 3], samples=20)
    assert block_bootstrap_quantile_ci([1, 2, 3], samples=20) == block_bootstrap_quantile_ci([1, 2, 3], samples=20)
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


def test_development_benchmark_loads_frozen_signer_without_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    from controlflow.v22.approval import ApprovalIssuer

    private_path = root / "artifacts/v22/local_keys/approval_ed25519.private.pem"
    public_path = root / "artifacts/v22/approval_public_key.pem"
    before = (private_path.read_bytes(), public_path.read_bytes())

    def prohibited(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2.3 must not call the V2.2 artifact-writing key helper")

    monkeypatch.setattr(ApprovalIssuer, "create_ephemeral", prohibited)
    issuer = load_frozen_approval_issuer(root)

    assert issuer.key_id == "v22-dev-reviewer"
    assert (private_path.read_bytes(), public_path.read_bytes()) == before
