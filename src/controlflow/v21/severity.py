from __future__ import annotations

from collections.abc import Mapping

from controlflow.schemas import Severity

NONCRITICAL = (Severity.LOW, Severity.MEDIUM, Severity.HIGH)


def hard_route_severity(
    calibrated_critical_probability: float,
    critical_threshold: float,
    noncritical_probabilities: Mapping[str | Severity, float],
) -> Severity:
    """Critical threshold is authoritative; stage two cannot downgrade it."""
    if calibrated_critical_probability >= critical_threshold:
        return Severity.CRITICAL
    scores = {Severity(str(key)): float(value) for key, value in noncritical_probabilities.items()}
    if Severity.CRITICAL in scores:
        raise ValueError("noncritical classifier must not output CRITICAL")
    missing = set(NONCRITICAL) - set(scores)
    if missing:
        raise ValueError(f"missing noncritical probabilities: {sorted(item.value for item in missing)}")
    return max(NONCRITICAL, key=lambda item: (scores[item], -NONCRITICAL.index(item)))
