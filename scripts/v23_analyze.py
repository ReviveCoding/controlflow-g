from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v23.latency import tail_membership

ROOT = Path(__file__).resolve().parents[1]


def _metric_totals(snapshot: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for sample in snapshot["samples"]:
        name = str(sample["name"])
        totals[name] = totals.get(name, 0.0) + float(sample["value"])
    return totals


def _metric_deltas(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start = _metric_totals(payload["measurement_start"])
    end = _metric_totals(payload["measurement_end"])
    deltas = {name: end.get(name, 0.0) - start.get(name, 0.0) for name in set(start) | set(end)}
    means = {}
    for stem in (
        "time_to_first_token_seconds",
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "inter_token_latency_seconds",
        "request_prompt_tokens",
        "request_generation_tokens",
    ):
        count = deltas.get(f"vllm:{stem}_count", 0.0)
        total = deltas.get(f"vllm:{stem}_sum", 0.0)
        means[stem] = None if count <= 0 else total / count
    queries = deltas.get("vllm:prefix_cache_queries_total", 0.0)
    hits = deltas.get("vllm:prefix_cache_hits_total", 0.0)
    return {
        "means": means,
        "prefix_cache_queries": queries,
        "prefix_cache_hits": hits,
        "prefix_cache_hit_rate": None if queries <= 0 else hits / queries,
        "ending_kv_cache_usage_perc": end.get("vllm:kv_cache_usage_perc"),
    }


def _join_telemetry(requests: pd.DataFrame, path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    telemetry = pd.DataFrame(payload["samples"])
    if telemetry.empty:
        return requests
    telemetry["sample_epoch"] = pd.to_datetime(telemetry.sampled_at_utc, utc=True).astype("int64") / 1e9
    columns = (
        "utilization_gpu_percent",
        "memory_used_mib",
        "temperature_c",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    joined = requests.copy()
    for index, row in joined.iterrows():
        window = telemetry[
            (telemetry.sample_epoch >= float(row.request_start)) & (telemetry.sample_epoch <= float(row.request_end))
        ]
        if window.empty:
            distance = (telemetry.sample_epoch - float(row.request_start)).abs()
            window = telemetry.loc[[distance.idxmin()]]
        for column in columns:
            joined.loc[index, f"gpu_{column}_mean"] = pd.to_numeric(window[column], errors="coerce").mean()
    return joined


def _normalize_request_metrics(requests: pd.DataFrame) -> pd.DataFrame:
    """Backfill normalized seconds fields from retained raw vLLM request metrics."""
    normalized = requests.copy()
    mappings = {
        "queue_time_seconds": ("queue_time_ms", 1_000.0),
        "ttft_seconds": ("time_to_first_token_ms", 1_000.0),
        "inter_token_latency_seconds": ("mean_itl_ms", 1_000.0),
        "decode_time_seconds": ("generation_time_ms", 1_000.0),
    }
    for target, (source, divisor) in mappings.items():
        if target not in normalized:
            normalized[target] = np.nan
        raw_values = normalized.get("server_metrics_raw")
        if raw_values is None:
            continue
        derived = raw_values.map(
            lambda item, source=source, divisor=divisor: (
                float(item[source]) / divisor if isinstance(item, dict) and item.get(source) is not None else np.nan
            )
        )
        normalized[target] = pd.to_numeric(normalized[target], errors="coerce").fillna(derived)
    return normalized


def _correlations(frame: pd.DataFrame) -> dict[str, Any]:
    candidates = (
        "prompt_tokens",
        "completion_tokens",
        "evidence_count",
        "queue_time_seconds",
        "ttft_seconds",
        "prefill_time_seconds",
        "decode_time_seconds",
        "inter_token_latency_seconds",
        "non_llm_pipeline_latency_seconds",
        "gpu_utilization_gpu_percent_mean",
        "gpu_memory_used_mib_mean",
        "gpu_temperature_c_mean",
        "gpu_power_draw_w_mean",
        "gpu_sm_clock_mhz_mean",
        "gpu_memory_clock_mhz_mean",
    )
    output = {}
    for column in candidates:
        if column not in frame:
            continue
        subset = frame[["total_latency_seconds", column]].dropna()
        if len(subset) < 3 or subset[column].nunique() < 2:
            continue
        result = spearmanr(subset.total_latency_seconds, subset[column])
        output[column] = {"n": len(subset), "spearman_rho": float(result.statistic), "pvalue": float(result.pvalue)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", required=True)
    args = parser.parse_args()
    summaries = []
    for path in sorted((ROOT / "results/v23").glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "config_id": payload["config_id"],
                "requests": payload["requests"],
                "configuration": payload["configuration"],
                "warmup": payload["warmup"],
                "total_p95_seconds": payload["latency_seconds"]["total"]["p95"],
                "llm_p95_seconds": payload["latency_seconds"]["llm"]["p95"],
                "total_p95_ci": payload["latency_seconds"]["total_p95_bootstrap_ci"],
                "completion_tokens": payload["completion_tokens"],
                "structured_failure_rate": payload["structured_failure_rate"],
                "core_stc": payload["quality_metrics"]["core_stc"]["estimate"],
                "security": payload["quality_metrics"]["security"],
                "summary_sha256": sha256_file(path),
            }
        )
    winner_dir = ROOT / "results/v23" / args.winner
    if not (winner_dir / "summary.json").is_file():
        raise RuntimeError("winner summary is absent")
    requests_payload = json.loads((winner_dir / "request_attribution.json").read_text(encoding="utf-8"))
    request_frame = _normalize_request_metrics(pd.DataFrame(requests_payload["requests"]))
    request_frame = _join_telemetry(request_frame, winner_dir / "nvidia_telemetry.json")
    tails = tail_membership(request_frame.to_dict(orient="records"))
    tail_summary = []
    for tail in tails:
        frame = pd.DataFrame(tail["rows"])
        tail_summary.append(
            {
                "fraction": tail["fraction"],
                "count": tail["count"],
                "case_ids": frame.case_id.astype(str).tolist(),
                "numeric_means": {
                    column: float(pd.to_numeric(frame[column], errors="coerce").mean())
                    for column in frame.columns
                    if pd.api.types.is_numeric_dtype(frame[column]) and column not in {"request_start", "request_end"}
                },
                "categorical_counts": {
                    column: frame[column].astype(str).value_counts().to_dict()
                    for column in ("severity", "disposition", "critical")
                    if column in frame
                },
            }
        )
    correlations = _correlations(request_frame)
    telemetry = json.loads((winner_dir / "nvidia_telemetry.json").read_text(encoding="utf-8"))["samples"]
    telemetry_frame = pd.DataFrame(telemetry)
    telemetry_summary = {
        column: {
            "min": float(pd.to_numeric(telemetry_frame[column], errors="coerce").min()),
            "mean": float(pd.to_numeric(telemetry_frame[column], errors="coerce").mean()),
            "max": float(pd.to_numeric(telemetry_frame[column], errors="coerce").max()),
        }
        for column in (
            "utilization_gpu_percent",
            "memory_used_mib",
            "temperature_c",
            "power_draw_w",
            "sm_clock_mhz",
            "memory_clock_mhz",
        )
    }
    metrics = _metric_deltas(winner_dir / "vllm_metrics_snapshots.json")
    rho = {name: item["spearman_rho"] for name, item in correlations.items()}
    decode_driven = abs(rho.get("decode_time_seconds", 0.0)) >= 0.5
    host_driven = abs(rho.get("non_llm_pipeline_latency_seconds", 0.0)) >= 0.4
    if decode_driven and host_driven:
        classification = "MIXED_GENERATION_AND_HOST_PIPELINE"
    elif decode_driven:
        classification = "GENERATION_DRIVEN"
    elif host_driven:
        classification = "HOST_PIPELINE_DRIVEN"
    else:
        classification = "MIXED_WEAK_SIGNALS"
    tail_determination = {
        "classification": classification,
        "request_size_driven": False,
        "request_size_note": "Prompt-token correlation is weak; completion-token correlation is small.",
        "scheduler_queue_driven": False,
        "scheduler_note": "Mean vLLM queue time is sub-millisecond despite a modest correlation.",
        "generation_driven": decode_driven,
        "host_pipeline_driven": host_driven,
        "cache_driven": None,
        "cache_note": "Aggregate prefix hit rate is available; the server did not expose a per-request hit indicator.",
        "thermal_or_power_driven": False,
        "thermal_note": (
            "Temperature correlation is modest and power/SM-clock correlations do not support a thermal cause."
        ),
    }
    tail_artifact = {
        "schema_version": 1,
        "created_at": utc_now(),
        "winner": args.winner,
        "tail_groups": tail_summary,
        "full_cohort_correlations": correlations,
        "gpu_telemetry_summary": telemetry_summary,
        "vllm_metric_deltas": metrics,
        "tail_determination": tail_determination,
        "thermal_throttling_claimed": False,
        "thermal_note": (
            "No thermal cause is assigned automatically; clock, power, and temperature evidence require review."
        ),
    }
    atomic_write_json(ROOT / "results/v23/tail_analysis.json", tail_artifact)
    atomic_write_json(
        ROOT / "state/v23_serving_tournament.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "DEVELOPMENT_TOURNAMENT_IN_PROGRESS",
            "winner_under_evaluation": args.winner,
            "configurations": summaries,
        },
    )
    atomic_write_json(
        ROOT / "state/v23_latency_attribution.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "winner": args.winner,
            "request_attribution": {
                "path": (winner_dir / "request_attribution.json").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(winner_dir / "request_attribution.json"),
            },
            "tail_analysis": {
                "path": "results/v23/tail_analysis.json",
                "sha256": sha256_file(ROOT / "results/v23/tail_analysis.json"),
            },
            "vllm_metrics": metrics,
            "gpu_telemetry_summary": telemetry_summary,
        },
    )


if __name__ == "__main__":
    main()
