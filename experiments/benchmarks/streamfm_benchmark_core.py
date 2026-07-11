from __future__ import annotations

"""Compatibility imports for the split benchmark implementation.

New code should import from the focused modules in this package:
`paths`, `options`, `loading`, `model_loops`, `cuda_graph`, `runner`,
and `results`.
"""

from experiments.common import (
    device_label,
    ensure_repo_importable as _ensure_repo_importable_path,
    find_repo_root,
    forward_step as _forward_step,
    prepare_streaming_state as _prepare_streaming_state,
    summarize_ms as _summarize_ms,
    sync_device,
    zero_streaming_state as _zero_streaming_state,
)
from experiments.benchmarks.cuda_graph import (
    benchmark_flow_steps_cuda_graph as _benchmark_flow_steps_cuda_graph,
    benchmark_se_flow_cuda_graph as _benchmark_se_flow_cuda_graph,
    benchmark_se_full_cuda_graph as _benchmark_se_full_cuda_graph,
    benchmark_se_predictor_cuda_graph as _benchmark_se_predictor_cuda_graph,
)
from experiments.benchmarks.loading import (
    load_backbone_from_checkpoint as _load_backbone_from_checkpoint,
    load_flow_model as _load_flow_model,
    load_se_flow as _load_se_flow,
    load_se_full as _load_se_full,
    load_se_predictor as _load_se_predictor,
)
from experiments.benchmarks.model_loops import (
    benchmark_flow_steps as _benchmark_flow_steps,
    benchmark_se_flow as _benchmark_se_flow,
    benchmark_se_full as _benchmark_se_full,
    benchmark_se_predictor as _benchmark_se_predictor,
)
from experiments.benchmarks.options import (
    normalize_cli_options,
    parse_model_dtype,
    parse_steps,
    resolve_execution,
)
from experiments.benchmarks.paths import (
    BenchmarkPaths,
    checkpoint_path as _checkpoint_path,
    make_benchmark_paths,
)
from experiments.benchmarks.results import (
    DEFAULT_HISTORY_JSON,
    HISTORY_SUMMARY_COLUMNS,
    compact_history_value as _compact_history_value,
    history_file_lock as _history_file_lock,
    record_benchmark_results,
    write_history_summaries,
    write_json_atomic as _write_json_atomic,
)
from experiments.benchmarks.runner import (
    run_benchmark,
    run_internal_benchmark as _run_internal_benchmark,
)


def _ensure_repo_importable(paths) -> None:
    """Compatibility wrapper for the old BenchmarkPaths-based helper."""
    repo_root = paths.repo_root if hasattr(paths, "repo_root") else paths
    _ensure_repo_importable_path(repo_root)


__all__ = [
    "BenchmarkPaths",
    "DEFAULT_HISTORY_JSON",
    "HISTORY_SUMMARY_COLUMNS",
    "device_label",
    "find_repo_root",
    "make_benchmark_paths",
    "normalize_cli_options",
    "parse_model_dtype",
    "parse_steps",
    "record_benchmark_results",
    "resolve_execution",
    "run_benchmark",
    "sync_device",
    "write_history_summaries",
]
