from __future__ import annotations

import json

import pytest

from controlflow.v22.vllm_client import StructuredVllmClient


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": json.dumps(self.payload)}}]}


@pytest.mark.parametrize(
    ("returned_evidence", "expected_valid"),
    [(["DOC-B", "DOC-A"], True), (["DOC-A", "DOC-A"], False), (["DOC-X", "DOC-A"], False)],
)
def test_semantic_verifier_treats_evidence_as_an_exact_unique_set(
    monkeypatch: pytest.MonkeyPatch, returned_evidence: list[str], expected_valid: bool
) -> None:
    response = _Response(
        {
            "case_id": "CASE-1",
            "summary": "supported",
            "claims": [],
            "evidence_ids": returned_evidence,
            "policy_id": "policy-2026",
        }
    )
    monkeypatch.setattr("controlflow.v22.vllm_client.httpx.post", lambda *_args, **_kwargs: response)
    client = StructuredVllmClient(
        endpoint="http://test",
        model="test",
        schema={},
        prompt_template="{{INPUT_JSON}}",
    )
    _, valid, _, _ = client(
        {
            "case_id": "CASE-1",
            "evidence_ids": ["DOC-A", "DOC-B"],
            "policy_id": "policy-2026",
            "deterministic_summary": "fallback",
        }
    )
    assert valid is expected_valid
    assert client.diagnostics[0]["semantic_reference_failure"] is (not expected_valid)
