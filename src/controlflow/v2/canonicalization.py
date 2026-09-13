from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split
from controlflow.v2.decomposed import SEVERITIES, predict_root_causes
from controlflow.v2.evidence import build_evidence_packet
from controlflow.v2.generation import VllmDecisionClient
from controlflow.v2.policy import verify_generated_decision
from controlflow.v2.schemas import RiskProbabilities, RootCauseCode
from controlflow.v2.tournament import _ordered, _score


def _risk(values: np.ndarray[Any, Any]) -> RiskProbabilities:
    normalized = values / values.sum()
    return RiskProbabilities(low=normalized[0], medium=normalized[1], high=normalized[2], critical=normalized[3])


def _raw_tool_dump(packet: Any) -> str:
    base = packet.model_dump(mode="json")
    return json.dumps(
        {
            "tool_messages": [
                {"tool": "case_lookup", "response": {"case": base["case_facts"], "narrative": base["narrative"]}},
                {
                    "tool": "risk_services",
                    "response": {
                        "probability_output": base["risk_probabilities"],
                        "candidate_root_cause": base["root_cause_candidate"],
                        "case_identifier": base["case_id"],
                    },
                },
                {
                    "tool": "retrieval_and_policy",
                    "response": {
                        "controls": base["applicable_controls"],
                        "regulations": base["applicable_regulations"],
                        "retrieved_records": base["evidence_records"],
                        "authorization": base["authorization_outcome"],
                        "actions_allowlist": base["allowed_actions"],
                        "evidence_sufficient": base["evidence_sufficient"],
                    },
                },
            ],
            "transport_metadata": {"format": "heterogeneous_tool_responses", "case_id": base["case_id"]},
        },
        sort_keys=True,
    )


def run_canonicalization_benchmark(dataset: Path, sample_size: int = 8) -> Path:
    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    validation_ids = set(load_v2_split("validation"))
    sample = frame[frame["case_id"].isin(validation_ids)].groupby("case_type", sort=True).head(1).head(sample_size)
    flat = joblib.load(paths.root / "artifacts/v2/models/severity_flat.joblib")
    root = joblib.load(paths.root / "artifacts/v2/models/root_linear.joblib")
    probability = _ordered(flat, sample, SEVERITIES)
    root_prediction = predict_root_causes(root, sample)
    client = VllmDecisionClient()

    tasks: list[tuple[int, str]] = [
        (position, mode) for position in range(len(sample)) for mode in ("raw", "canonical")
    ]

    def evaluate(task: tuple[int, str]) -> dict[str, Any]:
        position, mode = task
        row = sample.iloc[position]
        packet = build_evidence_packet(
            row,
            risk_probabilities=_risk(probability[position]),
            root_cause_candidate=RootCauseCode(str(root_prediction[position])),
        )
        generated = client.generate(
            packet,
            constrained=True,
            prompt_override=_raw_tool_dump(packet) if mode == "raw" else None,
            enforce_packet_constraints=True,
        )
        verification = verify_generated_decision(packet, generated.decision) if generated.decision is not None else None
        return {
            "representation": mode,
            "case_id": str(row["case_id"]),
            **_score(
                row,
                generated,
                generated.decision,
                semantic_valid=bool(verification and verification.semantic_valid),
            ),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        traces = pd.DataFrame(executor.map(evaluate, tasks))
    summaries = []
    for representation, group in traces.groupby("representation", sort=True):
        summaries.append(
            {
                "representation": representation,
                "cases": len(group),
                "mean_prompt_tokens": float(group["prompt_tokens"].mean()),
                "p50_latency_seconds": float(group["latency_seconds"].quantile(0.5)),
                "p95_latency_seconds": float(group["latency_seconds"].quantile(0.95)),
                "structured_output_valid_rate": float(group["structured_output_valid"].mean()),
                "semantic_valid_rate": float(group["semantic_valid"].mean()),
                "safe_task_completion": float(group["safe_task_completion"].mean()),
            }
        )
    result = pd.DataFrame(summaries)
    target = paths.root / "results/v2/canonicalization.parquet"
    result.to_parquet(target, index=False)
    traces.to_parquet(paths.root / "results/v2/canonicalization_traces.parquet", index=False)
    atomic_write_json(
        paths.state / "v2_canonicalization_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "split": "validation",
            "sample_case_ids": sample["case_id"].astype(str).tolist(),
            "results": json.loads(result.to_json(orient="records")),
        },
    )
    return target
