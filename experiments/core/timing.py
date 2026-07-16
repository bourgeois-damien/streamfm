"""Latency-sample summaries (mean and percentiles) for benchmark timings."""
from __future__ import annotations

import math


def summarize_ms(times_ms: list[float]) -> dict[str, float]:
    """Reduce per-frame latency samples (milliseconds) to mean/p50/p90/p99.

    In: one wall-clock duration per measured frame. Out: a dict merged into a
    benchmark result row. Raises on an empty list (a run with zero measured
    frames is a bug, not a zero).
    """
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    # Percentiles use nearest-rank on the sorted samples; exact interpolation
    # is not worth it at the 100+ sample sizes benchmarks run with.
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": values[len(values) // 2],
        "p90_ms": values[int(0.90 * (len(values) - 1))],
        "p99_ms": values[int(0.99 * (len(values) - 1))],
    }


def summarize_prefixed_ms(times_ms: list[float], prefix: str) -> dict[str, float]:
    """Same reduction as summarize_ms but with ``{prefix}_``-namespaced keys.

    Used when one result row carries several timing series (for example the
    TensorRT per-stage timings) that must not collide with the main metrics.
    """
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    # Ceil-based rank so small sample counts round the percentile index up
    # instead of collapsing p90/p99 onto the same sample.
    p90_idx = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    p99_idx = min(len(values) - 1, math.ceil(0.99 * len(values)) - 1)
    return {
        f"{prefix}_mean_ms": sum(values) / len(values),
        f"{prefix}_p50_ms": values[len(values) // 2],
        f"{prefix}_p90_ms": values[p90_idx],
        f"{prefix}_p99_ms": values[p99_idx],
    }
