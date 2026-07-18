"""Benchmark result shaping, history summaries and W&B logging.

Compacts raw benchmark records into JSON history rows and builds the matching
Weights & Biases payloads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.core.history import history_file_lock, write_json_atomic


DEFAULT_HISTORY_JSON = "outputs/streamfm_benchmark_history.json"
DEFAULT_WANDB_PROJECT = "streamfm-benchmarks"
DEFAULT_SWEEP_WANDB_PROJECT = "streamfm-benchmark-sweeps"
# Columns kept in the compact *_summary.json — the fields worth eyeballing
# when comparing runs; the full history keeps everything.
HISTORY_SUMMARY_COLUMNS = (
    "run_started_at",
    "run_id",
    "backend",
    "hardware",
    "device",
    "gpu_name",
    "requested_task",
    "requested_part",
    "requested_pipeline",
    "requested_execution",
    "execution",
    "task",
    "pipeline",
    "steps",
    "requested_model_dtype",
    "num_threads",
    "num_interop_threads",
    "model_memory_format",
    "preallocate_model_buffers",
    "ptq_int8",
    "ptq_engine",
    "ptq_calib_steps",
    "ptq_fallback_dtype",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "total_mean_ms",
    "total_p50_ms",
    "total_p90_ms",
    "total_p99_ms",
    "budget_ratio_mean",
    "cuda_graph",
    "compiled",
    "predictor_calls_per_frame",
    "flow_calls_per_frame",
    "total_dnn_calls_per_frame",
    "iterations",
    "warmup",
    "benchmark_s",
    "total_s",
)

# W&B split: config = what identifies/configures the run (filterable),
# metrics = numbers to plot, excluded = bulky or non-serializable payloads.
# Numeric keys missing from both sets fall back to _looks_like_metric below.
WANDB_CONFIG_KEYS = {
    "audio_sample_rate",
    "backend",
    "benchmark_row_id",
    "benchmark_row_index",
    "benchmark_run_id",
    "compiled",
    "cuda_graph",
    "cuda_graph_model",
    "device",
    "execution",
    "flow_calls_per_frame",
    "frame_budget_ms",
    "gpu_name",
    "hardware",
    "input_audio_path",
    "internal_pipeline",
    "internal_task",
    "iterations",
    "mode",
    "model_dtype",
    "model_memory_format",
    "num_interop_threads",
    "num_threads",
    "pipeline",
    "pre_generated_noise",
    "preallocate_model_buffers",
    "ptq_int8",
    "ptq_engine",
    "ptq_calib_steps",
    "ptq_fallback_dtype",
    "predictor_calls_per_frame",
    "profile",
    "profile_file",
    "requested_execution",
    "requested_model_dtype",
    "requested_part",
    "requested_pipeline",
    "requested_task",
    "run_id",
    "run_started_at",
    "saved_audio_path",
    "steps",
    "task",
    "torch_version",
    "total_dnn_calls_per_frame",
    "warmup",
}

WANDB_METRIC_KEYS = {
    "benchmark_s",
    "budget_ratio_mean",
    "graph_eager_max_abs_diff",
    "graph_eager_mean_abs_diff",
    "graph_eager_ref_mean_abs",
    "istft_mean_ms",
    "istft_p50_ms",
    "istft_p90_ms",
    "istft_p99_ms",
    "mean_ms",
    "measured_wall_s",
    "model_load_s",
    "model_mean_ms",
    "model_p50_ms",
    "model_p90_ms",
    "model_p99_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "stft_mean_ms",
    "stft_p50_ms",
    "stft_p90_ms",
    "stft_p99_ms",
    "total_mean_ms",
    "total_p50_ms",
    "total_p90_ms",
    "total_p99_ms",
    "total_s",
}

WANDB_EXCLUDED_KEYS = {
    "audio",
    "command",
    "profile_summary",
}


def compact_history_value(row: dict, key: str):
    """Return compact values for comparison summaries."""
    value = row.get(key, "")
    if isinstance(value, float):
        if key.endswith("_ms") or key in {"budget_ratio_mean", "benchmark_s", "total_s"}:
            return round(value, 4)
        return round(value, 6)
    return value


def write_history_summaries(history_path: Path, rows: list[dict]) -> None:
    """Write compact JSON summary next to the full history."""
    summary_rows = [
        {key: compact_history_value(row, key) for key in HISTORY_SUMMARY_COLUMNS}
        for row in rows
    ]
    summary_json_path = history_path.with_name(f"{history_path.stem}_summary.json")
    write_json_atomic(summary_json_path, summary_rows)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _json_scalar(value: Any) -> Any:
    if _is_scalar(value):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _looks_like_metric(key: str, value: Any) -> bool:
    """Heuristic for numeric keys not in WANDB_METRIC_KEYS: rely on the naming conventions (_ms/_s suffixes, ratio)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return (
        key in WANDB_METRIC_KEYS
        or key.endswith("_ms")
        or key.endswith("_s")
        or key.startswith("remote_")
        or "ratio" in key
    )


def _wandb_tags_for_config(config: dict[str, Any], extra_tags: tuple[str, ...]) -> list[str]:
    tags = [
        f"backend:{config.get('backend')}",
        f"device:{config.get('device')}",
        f"pipeline:{config.get('pipeline')}",
        f"execution:{config.get('execution')}",
    ]
    if config.get("hardware"):
        tags.append(f"hardware:{config['hardware']}")
    if config.get("requested_model_dtype"):
        tags.append(f"dtype:{config['requested_model_dtype']}")
    return [tag for tag in (*tags, *extra_tags) if tag and not tag.endswith(":None")]


def _wandb_row_name(config: dict[str, Any]) -> str:
    parts = [
        config.get("task") or config.get("requested_task") or "benchmark",
        config.get("pipeline") or config.get("requested_pipeline") or "pipeline",
        config.get("execution") or config.get("requested_execution") or "execution",
        f"steps{config.get('steps', 'na')}",
        str(config.get("benchmark_row_id", ""))[-4:],
    ]
    return "-".join(str(part).replace(" ", "_") for part in parts if part)


def build_benchmark_records(
    *,
    results: list[dict],
    command: dict[str, Any],
    run_id: str,
    run_started_at: str,
    extra_tags: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Build row-oriented benchmark records for exports such as W&B.

    Each result row is split into config/metrics/metadata using the explicit
    key sets first, then the metric-name heuristic, with remaining scalars
    kept as metadata so nothing quietly disappears.
    """
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(results):
        benchmark_row_id = f"{run_id}-{row_idx:03d}"
        enriched = {
            "benchmark_run_id": run_id,
            "benchmark_row_id": benchmark_row_id,
            "benchmark_row_index": row_idx,
            "run_id": run_id,
            "run_started_at": run_started_at,
            "cuda_graph": False,
            "cuda_graph_model": False,
            **row,
        }

        config: dict[str, Any] = {}
        metrics: dict[str, float | int] = {}
        metadata: dict[str, Any] = {}

        for key, value in enriched.items():
            if key in WANDB_EXCLUDED_KEYS:
                continue
            if key in WANDB_METRIC_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
            elif key in WANDB_CONFIG_KEYS:
                config[key] = _json_scalar(value)
            elif _looks_like_metric(key, value):
                metrics[key] = value
            elif _is_scalar(value):
                metadata[key] = value

        for key, value in command.items():
            if key in WANDB_EXCLUDED_KEYS:
                continue
            config[f"command_{key}"] = _json_scalar(value)

        records.append(
            {
                "name": _wandb_row_name(config),
                "group": run_id,
                "config": config,
                "metrics": metrics,
                "metadata": metadata,
                "tags": _wandb_tags_for_config(config, extra_tags),
            }
        )
    return records


def build_wandb_log_payload(*, results: list[dict], command: dict[str, Any]) -> dict[str, Any]:
    """Build the legacy flattened W&B payload from benchmark rows."""
    payload: dict[str, Any] = {}
    for key, value in command.items():
        payload[f"command_{key}"] = value

    for row_idx, row in enumerate(results):
        for key, value in row.items():
            if key == "audio":
                continue
            if isinstance(value, (dict, list)):
                payload[f"results_{row_idx}_{key}"] = json.dumps(value, sort_keys=True)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                payload[f"results_{row_idx}_{key}"] = value

    return payload


def _log_to_wandb(*, results: list[dict], command: dict[str, Any], enabled: bool) -> None:
    _log_records_to_wandb(
        results=results,
        command=command,
        enabled=enabled,
        run_id=uuid4().hex[:12],
        run_started_at=datetime.now(timezone.utc).isoformat(),
    )


def _log_records_to_wandb(
    *,
    results: list[dict],
    command: dict[str, Any],
    enabled: bool,
    run_id: str,
    run_started_at: str,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str = "",
    group: str = "",
    mode: str = "",
    tags: tuple[str, ...] = (),
) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping W&B logging.")
        return

    records = build_benchmark_records(
        results=results,
        command=command,
        run_id=run_id,
        run_started_at=run_started_at,
        extra_tags=tags,
    )
    if not records:
        return

    # One W&B run per result row (a benchmark can emit several rows, e.g. one
    # per steps value); the shared group ties them back to the same launch.
    for record in records:
        init_kwargs = {
            "project": project,
            "name": record["name"],
            "group": group or record["group"],
            "job_type": "benchmark",
            "config": record["config"],
            "tags": record["tags"],
            "reinit": True,
        }
        if entity:
            init_kwargs["entity"] = entity
        if mode:
            init_kwargs["mode"] = mode

        run = wandb.init(**init_kwargs)
        try:
            if record["metrics"]:
                wandb.log(record["metrics"])
            if record["metadata"]:
                run.summary.update({f"metadata/{key}": value for key, value in record["metadata"].items()})
        finally:
            wandb.finish()


def log_benchmark_results_to_wandb(
    *,
    results: list[dict],
    command: dict[str, Any],
    run_id: str,
    run_started_at: str,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str = "",
    group: str = "",
    mode: str = "",
    tags: tuple[str, ...] = (),
) -> None:
    """Log already-recorded benchmark rows to W&B."""
    _log_records_to_wandb(
        results=results,
        command=command,
        enabled=True,
        run_id=run_id,
        run_started_at=run_started_at,
        project=project,
        entity=entity,
        group=group,
        mode=mode,
        tags=tags,
    )


def record_benchmark_results(
    *,
    results: list[dict],
    command: dict,
    output_json: str = "",
    history_json: str = "",
    wandb_enabled: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = "",
    wandb_group: str = "",
    wandb_mode: str = "",
    wandb_tags: tuple[str, ...] = (),
) -> None:
    """Write optional per-run JSON plus shared full/summary histories."""
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote results to {output_path}")

    run_id = uuid4().hex[:12]
    run_started_at = datetime.now(timezone.utc).isoformat()
    history_rows = [
        {
            "run_id": run_id,
            "run_started_at": run_started_at,
            "command": command,
            **row,
        }
        for row in results
    ]

    # Always append to the shared default history; --history-json adds a
    # second copy (e.g. a per-sweep file) without replacing the shared one.
    history_paths = [Path(DEFAULT_HISTORY_JSON)]
    if history_json:
        extra_history_path = Path(history_json)
        if extra_history_path not in history_paths:
            history_paths.append(extra_history_path)
    for history_path in history_paths:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        # The whole read-append-write cycle sits under the file lock so
        # concurrent runs (sweep workers) cannot drop each other's rows.
        with history_file_lock(history_path):
            if history_path.exists():
                all_history_rows = json.loads(history_path.read_text(encoding="utf-8"))
                if not isinstance(all_history_rows, list):
                    raise ValueError(f"History file must contain a JSON list: {history_path}")
            else:
                all_history_rows = []
            all_history_rows.extend(history_rows)
            write_json_atomic(history_path, all_history_rows)
            write_history_summaries(history_path, all_history_rows)
        print(f"Appended {len(results)} result(s) to {history_path}")

    _log_records_to_wandb(
        results=results,
        command=command,
        enabled=wandb_enabled,
        run_id=run_id,
        run_started_at=run_started_at,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        mode=wandb_mode,
        tags=wandb_tags,
    )
