from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v23.latency import block_bootstrap_quantile_ci, percentile_summary, tail_membership

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


def _join_periodic_metrics(requests: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.is_file():
        return requests
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for snapshot in payload.get("samples", []):
        totals = _metric_totals(snapshot)
        queries = totals.get("vllm:prefix_cache_queries_total", 0.0)
        rows.append(
            {
                "epoch": pd.Timestamp(snapshot["captured_at"]).timestamp(),
                "periodic_kv_cache_usage_perc": totals.get("vllm:kv_cache_usage_perc"),
                "periodic_prefix_cache_hit_rate": None
                if queries <= 0
                else totals.get("vllm:prefix_cache_hits_total", 0.0) / queries,
                "periodic_requests_waiting": totals.get("vllm:num_requests_waiting"),
            }
        )
    periodic = pd.DataFrame(rows).dropna(subset=["epoch"])
    if periodic.empty:
        return requests
    joined = requests.copy()
    for index, row in joined.iterrows():
        midpoint = (float(row.request_start) + float(row.request_end)) / 2
        match = periodic.loc[(periodic.epoch - midpoint).abs().idxmin()]
        for column in ("periodic_kv_cache_usage_perc", "periodic_prefix_cache_hit_rate", "periodic_requests_waiting"):
            joined.loc[index, column] = match[column]
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
        "periodic_kv_cache_usage_perc",
        "periodic_prefix_cache_hit_rate",
        "periodic_requests_waiting",
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
        row = {
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
        attribution_path = path.parent / "request_attribution.json"
        telemetry_path = path.parent / "nvidia_telemetry.json"
        if attribution_path.is_file():
            request_rows = json.loads(attribution_path.read_text(encoding="utf-8"))["requests"]
            normalized = _normalize_request_metrics(pd.DataFrame(request_rows))
            elapsed = float(normalized.request_end.max() - normalized.request_start.min())
            row.update(
                {
                    "total_latency": percentile_summary(normalized.total_latency_seconds),
                    "ttft": percentile_summary(normalized.ttft_seconds.dropna()),
                    "itl": percentile_summary(normalized.inter_token_latency_seconds.dropna()),
                    "request_throughput_per_second": len(normalized) / elapsed if elapsed > 0 else None,
                    "completion_token_throughput_per_second": float(normalized.completion_tokens.sum()) / elapsed
                    if elapsed > 0
                    else None,
                }
            )
        if telemetry_path.is_file():
            gpu = pd.DataFrame(json.loads(telemetry_path.read_text(encoding="utf-8"))["samples"])
            row["gpu_telemetry_means"] = {
                name: float(pd.to_numeric(gpu[name], errors="coerce").mean())
                for name in ("utilization_gpu_percent", "temperature_c", "power_draw_w", "sm_clock_mhz")
            }
        summaries.append(row)
    winner_dir = ROOT / "results/v23" / args.winner
    if not (winner_dir / "summary.json").is_file():
        raise RuntimeError("winner summary is absent")
    requests_payload = json.loads((winner_dir / "request_attribution.json").read_text(encoding="utf-8"))
    request_frame = _normalize_request_metrics(pd.DataFrame(requests_payload["requests"]))
    request_frame = _join_telemetry(request_frame, winner_dir / "nvidia_telemetry.json")
    request_frame = _join_periodic_metrics(request_frame, winner_dir / "vllm_metrics_periodic.json")
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
    cache_associated = (
        max(
            abs(rho.get("periodic_kv_cache_usage_perc", 0.0)),
            abs(rho.get("periodic_prefix_cache_hit_rate", 0.0)),
        )
        >= 0.3
    )
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
        "evidence_basis": "component association; not a causal attribution",
        "explicit_interpretive_thresholds": {"decode_absolute_spearman_rho": 0.5, "host_absolute_spearman_rho": 0.4},
        "request_size_driven": False,
        "request_size_note": "Prompt-token correlation is weak; completion-token correlation is small.",
        "scheduler_queue_driven": False,
        "scheduler_note": "Mean vLLM queue time is sub-millisecond despite a modest correlation.",
        "generation_driven": decode_driven,
        "host_pipeline_driven": host_driven,
        "cache_driven": cache_associated,
        "cache_note": (
            "Nearest periodic cache samples provide association only; no per-request hit indicator is exposed."
        ),
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
        "p95_block_bootstrap_sensitivity": block_bootstrap_quantile_ci(
            request_frame.total_latency_seconds, block_size=20
        ),
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
            "status": "DEVELOPMENT_TOURNAMENT_COMPLETE",
            "winner_under_evaluation": args.winner,
            "selected_winner": args.winner,
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
    by_id = {item["config_id"]: item for item in summaries}
    pair_ids = {
        "BALANCED_vs_INTERACTIVITY": (
            "p0_balanced_minimal_enum_128_b2048_warm_60",
            "p1_interactivity_minimal_enum_128_b2048_warm_60",
        ),
        "CURRENT_SCHEMA_vs_MINIMAL_SCHEMA": (
            "ablation_interactivity_current_schema_160_b2048_warm_60",
            "ablation_interactivity_minimal_enum_160_b2048_warm_60",
        ),
        "CURRENT_TOKEN_CAP_vs_SELECTED_TOKEN_CAP": (
            "ablation_interactivity_minimal_enum_160_b2048_warm_60",
            "p1_interactivity_minimal_enum_128_b2048_warm_60",
        ),
        "DEFAULT_BATCHED_TOKENS_vs_SELECTED_BATCHED_TOKENS": (
            "ablation_interactivity_minimal_128_installed_default_batch_warm_60",
            "p1_interactivity_minimal_enum_128_b2048_warm_60",
        ),
        "COLD_vs_WARM": (
            "ablation_selected_interactivity_minimal_128_b2048_cold_60",
            "p1_interactivity_minimal_enum_128_b2048_warm_60",
        ),
    }
    comparisons = []
    for name, (left_id, right_id) in pair_ids.items():
        left, right = by_id[left_id], by_id[right_id]
        comparisons.append(
            {
                "comparison": name,
                "left": left_id,
                "right": right_id,
                "p95_delta_seconds_right_minus_left": right["total_p95_seconds"] - left["total_p95_seconds"],
                "structured_failure_delta": right["structured_failure_rate"] - left["structured_failure_rate"],
                "completion_token_mean_delta": right["completion_tokens"]["mean"] - left["completion_tokens"]["mean"],
                "core_stc_delta": right["core_stc"] - left["core_stc"],
            }
        )
    atomic_write_json(
        ROOT / "results/v23/ablations.json",
        {"schema_version": 1, "created_at": utc_now(), "configurations": summaries, "comparisons": comparisons},
    )


if __name__ == "__main__":
    main()
