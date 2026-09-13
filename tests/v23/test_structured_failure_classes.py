from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from controlflow.v23.vllm_client import AttributedVllmClient


def _client(tmp_path: Path) -> AttributedVllmClient:
    return AttributedVllmClient(
        endpoint="http://development.invalid/v1/chat/completions",
        model="development-model",
        schema={},
        prompt_template="{{INPUT_JSON}}",
        max_tokens=128,
        contract="minimal",
        diagnostics_path=tmp_path / "diagnostics.jsonl",
    )


def _inputs() -> dict[str, Any]:
    return {
        "case_id": "V23-ANALOG-001",
        "deterministic_summary": "Development-only synthetic explanation.",
        "evidence_ids": ["DEV-EVIDENCE-1", "DEV-EVIDENCE-2"],
        "policy_id": "development-policy",
    }


def _response(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"x-request-id": "development-request"},
        request=httpx.Request("POST", "http://development.invalid"),
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "metrics": {},
        },
    )


@pytest.mark.parametrize(
    ("content", "finish_reason", "expected_field"),
    (
        ("{", "stop", "json_parse_failure"),
        ('{"summary":"brief","claims":["evidence_count=2"],"severity":"LOW"}', "stop", "schema_failure"),
        (
            json.dumps(
                {
                    "summary": "Development-only synthetic explanation.",
                    "claims": ["evidence_count=2"],
                }
            ),
            "length",
            "truncation_failure",
        ),
        ('{"summary":"wrong","claims":["evidence_count=2"]}', "stop", "semantic_reference_failure"),
    ),
)
def test_new_development_analogues_classify_structured_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: str,
    finish_reason: str,
    expected_field: str,
) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: _response(content, finish_reason=finish_reason))
    client = _client(tmp_path)
    _summary, valid, _latency, _claims = client(_inputs())
    assert not valid
    assert client.diagnostics[-1][expected_field] is True
    assert len((tmp_path / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_new_development_analogue_classifies_http_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("development-only synthetic connection failure")

    monkeypatch.setattr(httpx, "post", fail)
    client = _client(tmp_path)
    _summary, valid, _latency, _claims = client(_inputs())
    assert not valid
    assert client.diagnostics[-1]["http_failure"] is True
