from __future__ import annotations

import csv
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json

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

    def __enter__(self) -> NvidiaSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(sample_nvidia())
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        atomic_write_json(self.output, {"schema_version": 1, "samples": self.samples, "errors": self.errors})
