from __future__ import annotations

import json
import time
from typing import Any

import httpx
from pydantic import ValidationError

from controlflow.v22.schemas import ExplanationOutput


class StructuredVllmClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        schema: dict[str, Any],
        prompt_template: str,
        constrained: bool = True,
        semantic_verify: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.schema = schema
        self.prompt_template = prompt_template
        self.constrained = constrained
        self.semantic_verify = semantic_verify
        self.diagnostics: list[dict[str, Any]] = []

    def __call__(self, inputs: dict[str, Any]) -> tuple[str, bool, float, tuple[str, ...]]:
        prompt = self.prompt_template.replace("{{INPUT_JSON}}", json.dumps(inputs, sort_keys=True))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 220,
            "seed": 22022,
        }
        if self.constrained:
            payload["structured_outputs"] = {"json": self.schema}
        started = time.perf_counter()
        diagnostic: dict[str, Any] = {
            "case_id": inputs["case_id"],
            "http_failure": False,
            "json_parse_failure": False,
            "schema_failure": False,
            "semantic_reference_failure": False,
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=90)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            elapsed = time.perf_counter() - started
            diagnostic["http_failure"] = True
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["latency_seconds"] = elapsed
            self.diagnostics.append(diagnostic)
            return inputs["deterministic_summary"], False, elapsed, ()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            elapsed = time.perf_counter() - started
            diagnostic["json_parse_failure"] = True
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["latency_seconds"] = elapsed
            self.diagnostics.append(diagnostic)
            return content, False, elapsed, ()
        try:
            explanation = ExplanationOutput.model_validate(parsed)
        except ValidationError as exc:
            elapsed = time.perf_counter() - started
            diagnostic["schema_failure"] = True
            diagnostic["error"] = str(exc)
            diagnostic["latency_seconds"] = elapsed
            self.diagnostics.append(diagnostic)
            return content, False, elapsed, ()
        semantic_valid = (not self.semantic_verify) or (
            explanation.case_id == inputs["case_id"]
            and tuple(explanation.evidence_ids) == tuple(inputs["evidence_ids"])
            and explanation.policy_id == inputs["policy_id"]
        )
        elapsed = time.perf_counter() - started
        diagnostic["semantic_reference_failure"] = not semantic_valid
        diagnostic["latency_seconds"] = elapsed
        self.diagnostics.append(diagnostic)
        return explanation.summary, semantic_valid, elapsed, explanation.claims
