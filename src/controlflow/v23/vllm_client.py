from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MinimalExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(max_length=72)
    claims: tuple[str, ...] = Field(min_length=1, max_length=1)


class AttributedVllmClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        schema: dict[str, Any],
        prompt_template: str,
        max_tokens: int,
        contract: str,
        seed: int = 23023,
        diagnostics_path: Path | None = None,
    ) -> None:
        if contract not in {"current", "minimal"}:
            raise ValueError("contract must be current or minimal")
        self.endpoint = endpoint
        self.model = model
        self.schema = schema
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.contract = contract
        self.seed = seed
        self.diagnostics_path = diagnostics_path
        self._lock = threading.Lock()
        self.diagnostics: list[dict[str, Any]] = []
        if diagnostics_path is not None and diagnostics_path.is_file():
            self.diagnostics = [
                json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line
            ]

    def clear_diagnostics(self) -> None:
        with self._lock:
            self.diagnostics.clear()
            if self.diagnostics_path is not None:
                self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with self.diagnostics_path.open("w", encoding="utf-8") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())

    def __call__(self, inputs: dict[str, Any]) -> tuple[str, bool, float, tuple[str, ...]]:
        prompt = self.prompt_template.replace("{{INPUT_JSON}}", json.dumps(inputs, sort_keys=True))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "structured_outputs": {"json": self.schema},
        }
        wall_started = time.time()
        perf_started = time.perf_counter()
        diagnostic: dict[str, Any] = {
            "case_id": str(inputs["case_id"]),
            "request_start": wall_started,
            "max_tokens": self.max_tokens,
            "contract": self.contract,
            "http_failure": False,
            "json_parse_failure": False,
            "schema_failure": False,
            "truncation_failure": False,
            "semantic_reference_failure": False,
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=90)
            response.raise_for_status()
            response_payload = response.json()
            choice = response_payload["choices"][0]
            content = choice["message"]["content"]
            usage = response_payload.get("usage", {})
            diagnostic.update(
                {
                    "request_id": response.headers.get("x-request-id"),
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "server_metrics": response_payload.get("metrics", {}),
                }
            )
            diagnostic["truncation_failure"] = choice.get("finish_reason") == "length"
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            diagnostic["http_failure"] = True
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            return self._finish(diagnostic, perf_started, inputs["deterministic_summary"], False, ())
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            diagnostic["json_parse_failure"] = True
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            return self._finish(diagnostic, perf_started, content, False, ())
        try:
            if self.contract == "minimal":
                minimal = MinimalExplanation.model_validate(parsed)
                summary, claims = minimal.summary, minimal.claims
                expected_claim = f"evidence_count={len(inputs['evidence_ids'])}"
                references_valid = summary == inputs["deterministic_summary"] and claims == (expected_claim,)
                diagnostic["semantic_reference_failure"] = not references_valid
            else:
                from controlflow.v22.schemas import ExplanationOutput

                current = ExplanationOutput.model_validate(parsed)
                expected_evidence = tuple(str(item) for item in inputs["evidence_ids"])
                returned_evidence = tuple(current.evidence_ids)
                references_valid = (
                    current.case_id == inputs["case_id"]
                    and len(returned_evidence) == len(set(returned_evidence))
                    and set(returned_evidence) == set(expected_evidence)
                    and current.policy_id == inputs["policy_id"]
                )
                diagnostic["semantic_reference_failure"] = not references_valid
                summary, claims = current.summary, current.claims
        except ValidationError as exc:
            diagnostic["schema_failure"] = True
            diagnostic["error"] = str(exc)
            return self._finish(diagnostic, perf_started, content, False, ())
        valid = not any(
            diagnostic[key]
            for key in (
                "http_failure",
                "json_parse_failure",
                "schema_failure",
                "truncation_failure",
                "semantic_reference_failure",
            )
        )
        return self._finish(diagnostic, perf_started, summary, valid, tuple(claims))

    def _finish(
        self,
        diagnostic: dict[str, Any],
        perf_started: float,
        summary: str,
        valid: bool,
        claims: tuple[str, ...],
    ) -> tuple[str, bool, float, tuple[str, ...]]:
        elapsed = time.perf_counter() - perf_started
        diagnostic["request_end"] = time.time()
        diagnostic["llm_latency_seconds"] = elapsed
        with self._lock:
            self.diagnostics.append(diagnostic)
            if self.diagnostics_path is not None:
                self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with self.diagnostics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        return summary, valid, elapsed, claims
