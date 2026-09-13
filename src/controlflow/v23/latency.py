from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)$"
)
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse_prometheus(text: str, names: set[str] | None = None) -> list[dict[str, Any]]:
    """Parse numeric Prometheus samples without executing or trusting endpoint text."""
    samples: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None or (names is not None and match["name"] not in names):
            continue
        value = float(match["value"])
        labels = {
            key: bytes(value_, "utf-8").decode("unicode_escape")
            for key, value_ in _LABEL.findall(match["labels"] or "")
        }
        samples.append({"name": match["name"], "labels": labels, "value": value})
    return samples


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = np.asarray([float(item) for item in values if math.isfinite(float(item))], dtype=float)
    if not len(clean):
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "mean": None}
    return {
        "count": len(clean),
        "p50": float(np.quantile(clean, 0.50)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "mean": float(np.mean(clean)),
    }


def bootstrap_quantile_ci(
    values: Iterable[float], *, quantile: float = 0.95, seed: int = 23023, samples: int = 2000
) -> dict[str, float | int | None]:
    clean = np.asarray([float(item) for item in values if math.isfinite(float(item))], dtype=float)
    if not len(clean):
        return {"count": 0, "estimate": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    boot = np.quantile(rng.choice(clean, size=(samples, len(clean)), replace=True), quantile, axis=1)
    return {
        "count": len(clean),
        "estimate": float(np.quantile(clean, quantile)),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
    }


def tail_membership(rows: list[dict[str, Any]], latency_key: str = "total_latency_seconds") -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: float(item[latency_key]), reverse=True)
    result: list[dict[str, Any]] = []
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, math.ceil(len(ordered) * fraction))
        result.append({"fraction": fraction, "count": count, "rows": ordered[:count]})
    return result
