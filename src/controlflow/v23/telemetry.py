from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from controlflow.core.state import atomic_write_json
from controlflow.v23.latency import parse_prometheus

PERIODIC_METRICS = frozenset(
    {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:request_prompt_tokens_sum",
        "vllm:request_prompt_tokens_count",
        "vllm:request_generation_tokens_sum",
        "vllm:request_generation_tokens_count",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:inter_token_latency_seconds_sum",
        "vllm:inter_token_latency_seconds_count",
    }
)

FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "name",
    "memory_total_mib",
    "memory_used_mib",
    "utilization_gpu_percent",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
    "sm_clock_mhz",
    "memory_clock_mhz",
)
QUERY = (
    "timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,"
    "power.draw,power.limit,clocks.current.sm,clocks.current.memory"
)


def sample_nvidia() -> dict[str, Any]:
    process = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    values = next(csv.reader([process.stdout.strip()], skipinitialspace=True))
    row: dict[str, Any] = dict(zip(FIELDS, values, strict=True))
    for field in FIELDS[1:]:
        if field in {"uuid", "name"}:
            continue
        try:
            row[field] = float(row[field])
        except ValueError:
            row[field] = None
    row["sampled_at_utc"] = datetime.now(UTC).isoformat()
    return row


class NvidiaSampler:
    def __init__(self, output: Path, interval_seconds: float = 1.0) -> None:
        self.output = output
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.partial = output.with_suffix(output.suffix + ".partial.jsonl")

    def __enter__(self) -> NvidiaSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = sample_nvidia()
                self.samples.append(sample)
                self._append_durable({"kind": "sample", **sample})
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.errors.append(error)
                self._append_durable(
                    {"kind": "error", "message": error, "sampled_at_utc": datetime.now(UTC).isoformat()}
                )
            self._stop.wait(self.interval_seconds)

    def _append_durable(self, record: dict[str, Any]) -> None:
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        with self.partial.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        atomic_write_json(self.output, {"schema_version": 1, "samples": self.samples, "errors": self.errors})


class PrometheusSampler:
    def __init__(self, endpoint_root: str, output: Path, interval_seconds: float = 5.0) -> None:
        self.endpoint_root = endpoint_root
        self.output = output
        self.partial = output.with_suffix(output.suffix + ".partial.jsonl")
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PrometheusSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _append(self, record: dict[str, Any]) -> None:
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        with self.partial.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _run(self) -> None:
        while not self._stop.is_set():
            captured_at = datetime.now(UTC).isoformat()
            try:
                response = httpx.get(f"{self.endpoint_root}/metrics", timeout=20)
                response.raise_for_status()
                record = {
                    "captured_at": captured_at,
                    "samples": [item for item in parse_prometheus(response.text) if item["name"] in PERIODIC_METRICS],
                }
                self.samples.append(record)
                self._append({"kind": "sample", **record})
            except (httpx.HTTPError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.errors.append(error)
                self._append({"kind": "error", "captured_at": captured_at, "message": error})
            self._stop.wait(self.interval_seconds)

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        atomic_write_json(self.output, {"schema_version": 1, "samples": self.samples, "errors": self.errors})
