from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.benchmarks.results import history_file_lock, write_json_atomic


DEFAULT_EVAL_HISTORY_JSON = "outputs/streamfm_eval_history.json"
DEFAULT_EVAL_WANDB_PROJECT = "streamfm-evals"
EVAL_SUMMARY_COLUMNS = (
    "run_started_at",
    "run_id",
    "backend",
    "hardware",
    "device",
    "gpu_name",
    "task",
    "config_name",
    "config_overrides",
    "split",
    "part",
    "pipeline",
    "execution",
    "solver",
    "steps",
    "selection",
    "selection_seed",
    "model_dtype",
    "model_memory_format",
    "num_files",
    "num_errors",
    "elapsed_s",
    "mean_file_s",
    "output_dir",
    "manifest_path",
)

EVAL_CONFIG_KEYS = {
    "backend",
    "checkpoint_path",
    "ckpt",
    "config_name",
    "crop_mode",
    "data_format",
    "data_path",
    "device",
    "execution",
    "gpu_name",
    "hardware",
    "limit",
    "mode",
    "model_dtype",
    "model_memory_format",
    "num_interop_threads",
    "num_threads",
    "offset",
    "part",
    "pipeline",
    "run_id",
    "run_name",
    "run_started_at",
    "seed",
    "selection",
    "selection_seed",
    "solver",
    "split",
    "steps",
    "task",
    "torch_version",
}

EVAL_METRIC_KEYS = {
    "elapsed_s",
    "mean_file_s",
    "num_available",
    "num_errors",
    "num_files",
}

EVAL_EXCLUDED_KEYS = {
    "command",
    "errors",
    "files",
    "selected_indices",
}


def _compact_value(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_eval_history_summaries(history_path: Path, rows: list[dict]) -> None:
    """Write a compact sidecar summary for evaluation history."""
    summary_rows = [
        {key: _compact_value(row.get(key, "")) for key in EVAL_SUMMARY_COLUMNS}
        for row in rows
    ]
    summary_path = history_path.with_name(f"{history_path.stem}_summary.json")
    write_json_atomic(summary_path, summary_rows)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _json_scalar(value: Any) -> Any:
    if _is_scalar(value):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _parse_tags(tags: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(tags, tuple):
        return tags
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _eval_tags(config: dict[str, Any], extra_tags: tuple[str, ...]) -> list[str]:
    tags = [
        f"backend:{config.get('backend')}",
        f"device:{config.get('device')}",
        f"task:{config.get('task')}",
        f"split:{config.get('split')}",
        f"execution:{config.get('execution')}",
    ]
    if config.get("hardware"):
        tags.append(f"hardware:{config['hardware']}")
    if config.get("model_dtype"):
        tags.append(f"dtype:{config['model_dtype']}")
    return [tag for tag in (*tags, *extra_tags) if tag and not tag.endswith(":None")]


def _eval_run_name(config: dict[str, Any]) -> str:
    parts = [
        config.get("task") or "eval",
        config.get("split") or "split",
        config.get("solver") or "solver",
        f"steps{config.get('steps', 'na')}",
        str(config.get("run_id", ""))[-6:],
    ]
    return "-".join(str(part).replace(" ", "_") for part in parts if part)


def _flatten_score_metrics(score_result: dict[str, Any] | None) -> dict[str, float | int]:
    if not score_result:
        return {}
    metrics: dict[str, float | int] = {}
    if isinstance(score_result.get("num_files"), int):
        metrics["score_num_files"] = score_result["num_files"]
    for section in ("enhanced", "noisy", "delta_vs_noisy"):
        values = score_result.get(section)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"score_{section}_{name}"] = value
    for section in ("enhanced_stats", "noisy_stats", "delta_vs_noisy_stats"):
        values = score_result.get(section)
        if not isinstance(values, dict):
            continue
        for metric_name, stats in values.items():
            if not isinstance(stats, dict):
                continue
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"score_{section}_{metric_name}_{stat_name}"] = value
    return metrics


def build_eval_wandb_record(
    *,
    result: dict[str, Any],
    command: dict[str, Any],
    score_result: dict[str, Any] | None = None,
    extra_tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build one W&B record for a complete evaluation run."""
    config: dict[str, Any] = {}
    metrics: dict[str, float | int] = {}
    metadata: dict[str, Any] = {}

    for key, value in result.items():
        if key in EVAL_EXCLUDED_KEYS:
            continue
        if key in EVAL_METRIC_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
        elif key in EVAL_CONFIG_KEYS:
            config[key] = _json_scalar(value)
        elif key.endswith("_path") or key.endswith("_dir"):
            metadata[key] = _json_scalar(value)
        elif _is_scalar(value):
            metadata[key] = value

    for key, value in command.items():
        if key.startswith("wandb_"):
            continue
        config[f"command_{key}"] = _json_scalar(value)

    metrics.update(_flatten_score_metrics(score_result))
    if score_result:
        config["score_target"] = score_result.get("target", command.get("score_target", ""))
        metadata["score_manifest_path"] = score_result.get("manifest_path", "")

    return {
        "name": _eval_run_name(config),
        "group": str(result.get("run_id") or command.get("run_name") or ""),
        "config": config,
        "metrics": metrics,
        "metadata": metadata,
        "tags": _eval_tags(config, extra_tags),
    }


def log_eval_result_to_wandb(
    *,
    result: dict[str, Any],
    command: dict[str, Any],
    score_result: dict[str, Any] | None = None,
    project: str = DEFAULT_EVAL_WANDB_PROJECT,
    entity: str = "",
    group: str = "",
    mode: str = "",
    tags: str | tuple[str, ...] = (),
) -> None:
    """Log one complete evaluation run to W&B."""
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping W&B logging.")
        return

    record = build_eval_wandb_record(
        result=result,
        command=command,
        score_result=score_result,
        extra_tags=_parse_tags(tags),
    )
    init_kwargs = {
        "project": project,
        "name": record["name"],
        "group": group or record["group"],
        "job_type": "eval",
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


def record_eval_result(
    *,
    result: dict,
    command: dict,
    history_json: str = "",
) -> None:
    """Append one evaluation result to the shared full/summary histories."""
    history_paths = [Path(DEFAULT_EVAL_HISTORY_JSON)]
    if history_json:
        extra_history_path = Path(history_json)
        if extra_history_path not in history_paths:
            history_paths.append(extra_history_path)

    row = {
        "command": command,
        **result,
    }
    for history_path in history_paths:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_file_lock(history_path):
            if history_path.exists():
                rows = json.loads(history_path.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError(f"History file must contain a JSON list: {history_path}")
            else:
                rows = []
            rows.append(row)
            write_json_atomic(history_path, rows)
            write_eval_history_summaries(history_path, rows)
        print(f"Appended evaluation result to {history_path}")
