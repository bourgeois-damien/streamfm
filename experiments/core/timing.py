"""Latency-sample summaries (mean and percentiles) for benchmark timings."""
from __future__ import annotations

import math


def summarize_ms(times_ms: list[float]) -> dict[str, float]:
    """Summarize latency samples with mean and percentile metrics."""
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": values[len(values) // 2],
        "p90_ms": values[int(0.90 * (len(values) - 1))],
        "p99_ms": values[int(0.99 * (len(values) - 1))],
    }


def summarize_prefixed_ms(times_ms: list[float], prefix: str) -> dict[str, float]:
    """Summarize latency samples using namespaced metric keys."""
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    p90_idx = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    p99_idx = min(len(values) - 1, math.ceil(0.99 * len(values)) - 1)
    return {
        f"{prefix}_mean_ms": sum(values) / len(values),
        f"{prefix}_p50_ms": values[len(values) // 2],
        f"{prefix}_p90_ms": values[p90_idx],
        f"{prefix}_p99_ms": values[p99_idx],
    }
