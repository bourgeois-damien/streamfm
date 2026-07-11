from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.sweep_grid import load_sweep_metadata, load_sweep_trials
from experiments.evaluation.results import DEFAULT_EVAL_WANDB_PROJECT


VALUE_FLAGS = {
    "backend": "--backend",
    "hardware": "--hardware",
    "task": "--task",
    "variant": "--variant",
    "config_name": "--config-name",
    "ckpt": "--ckpt",
    "split": "--split",
    "data_path": "--data-path",
    "data_format": "--data-format",
    "part": "--part",
    "pipeline": "--pipeline",
    "execution": "--execution",
    "solver": "--solver",
    "steps": "--steps",
    "limit": "--limit",
    "offset": "--offset",
    "selection": "--selection",
    "selection_seed": "--selection-seed",
    "seed": "--seed",
    "dtype": "--dtype",
    "crop_mode": "--crop-mode",
    "memory_format": "--memory-format",
    "num_threads": "--num-threads",
    "num_interop_threads": "--num-interop-threads",
    "output_dir": "--output-dir",
    "history_json": "--history-json",
    "local_log_dir": "--local-log-dir",
    "score_target": "--score-target",
    "wandb_project": "--wandb-project",
    "wandb_entity": "--wandb-entity",
    "wandb_group": "--wandb-group",
    "wandb_mode": "--wandb-mode",
    "wandb_tags": "--wandb-tags",
}

BOOL_FLAGS = {
    "no_local_log": "--no-local-log",
    "overwrite": "--overwrite",
    "save_inputs": "--save-inputs",
    "continue_on_error": "--continue-on-error",
    "score_after_run": "--score-after-run",
    "score_with_distillmos": "--score-with-distillmos",
    "score_include_stats": "--score-include-stats",
    "score_include_per_file": "--score-include-per-file",
    "wandb": "--wandb",
}

SPECIAL_KEYS = frozenset({"config_overrides"})


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load evaluation sweep YAML files.") from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep YAML must contain a mapping: {path}")
    return payload


def load_eval_sweep_trials(path: str | Path) -> list[dict[str, Any]]:
    """Load filtered grid trials, then apply optional named variant presets."""
    payload = _load_yaml(path)
    presets = payload.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("Evaluation sweep 'presets' must be a mapping.")

    resolved: list[dict[str, Any]] = []
    for trial in load_sweep_trials(path):
        item = dict(trial)
        variant = str(item.get("variant", "baseline"))
        if presets and variant not in presets:
            raise ValueError(f"Variant '{variant}' has no matching entry in presets.")
        preset = presets.get(variant, {})
        if not isinstance(preset, dict):
            raise ValueError(f"Preset '{variant}' must be a mapping.")
        item.update(preset)
        item["variant"] = variant
        resolved.append(item)
    return resolved


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "eval-sweep"


def build_eval_trial_command(
    trial: dict[str, Any],
    *,
    run_name: str,
    default_project: str = "",
    default_entity: str = "",
    default_group: str = "",
) -> list[str]:
    """Translate one evaluation trial into the regular streamfm_eval CLI."""
    unknown = set(trial) - set(VALUE_FLAGS) - set(BOOL_FLAGS) - SPECIAL_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported evaluation sweep parameter(s): {names}")

    effective = dict(trial)
    effective.setdefault("wandb_project", default_project)
    effective.setdefault("wandb_entity", default_entity)
    effective.setdefault("wandb_group", default_group)

    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "evaluation" / "streamfm_eval.py"),
        "--run-name",
        run_name,
    ]
    for key, flag in VALUE_FLAGS.items():
        value = effective.get(key)
        if value is not None and value != "":
            command.extend([flag, str(value)])
    overrides = effective.get("config_overrides", ())
    if isinstance(overrides, str):
        overrides = (overrides,)
    if not isinstance(overrides, (list, tuple)):
        raise ValueError("config_overrides must be a string or a list of Hydra override strings.")
    for override in overrides:
        command.extend(["--config-override", str(override)])
    for key, flag in BOOL_FLAGS.items():
        if bool(effective.get(key, False)):
            command.append(flag)
    return command


def run_eval_sweep(
    *,
    sweep_yaml: str | Path,
    group: str = "",
    dry_run: bool = False,
    resume: bool = False,
    continue_on_trial_error: bool = False,
) -> tuple[int, int]:
    """Run a filtered evaluation grid sequentially from the local orchestrator."""
    trials = load_eval_sweep_trials(sweep_yaml)
    if not trials:
        raise ValueError("Evaluation sweep produced zero trials after exclusions.")

    payload = _load_yaml(sweep_yaml)
    metadata = load_sweep_metadata(sweep_yaml)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_group = _safe_label(group or str(payload.get("name") or f"eval-sweep-{timestamp}"))
    project = str(metadata.get("project") or DEFAULT_EVAL_WANDB_PROJECT)
    entity = str(metadata.get("entity") or "")
    local_log_root = Path(str(payload.get("local_log_dir") or REPO_ROOT / "outputs" / "evaluation_logs"))

    completed = 0
    failed = 0
    for index, trial in enumerate(trials, start=1):
        trial.setdefault("local_log_dir", str(local_log_root))
        variant = _safe_label(str(trial.get("variant", "baseline")))
        run_name = f"{sweep_group}-{index:03d}-{variant}"
        metrics_path = local_log_root / run_name / "metrics_all.json"
        if resume and metrics_path.exists():
            print(f"[{index}/{len(trials)}] Skip completed trial: {run_name}", flush=True)
            completed += 1
            continue

        command = build_eval_trial_command(
            trial,
            run_name=run_name,
            default_project=project,
            default_entity=entity,
            default_group=sweep_group,
        )
        print(f"[{index}/{len(trials)}] {shlex.join(command)}", flush=True)
        if dry_run:
            continue
        try:
            subprocess.run(command, check=True, cwd=REPO_ROOT)
            completed += 1
        except subprocess.CalledProcessError:
            failed += 1
            if not continue_on_trial_error:
                raise

    if dry_run:
        print(f"Would run {len(trials)} evaluation trial(s) in group '{sweep_group}'.")
    else:
        print(f"Evaluation sweep finished: {completed} completed, {failed} failed.")
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a filtered Stream.FM evaluation grid locally or on Modal, with scoring and W&B logging.",
    )
    parser.add_argument("--sweep-yaml", default="experiments/evaluation/sweep.yaml")
    parser.add_argument("--group", default="", help="Stable run/group prefix; useful together with --resume.")
    parser.add_argument("--dry-run", action="store_true", help="Print every trial without launching it.")
    parser.add_argument("--resume", action="store_true", help="Skip trials whose metrics_all.json already exists.")
    parser.add_argument("--continue-on-trial-error", action="store_true")
    args = parser.parse_args()
    run_eval_sweep(
        sweep_yaml=args.sweep_yaml,
        group=args.group,
        dry_run=args.dry_run,
        resume=args.resume,
        continue_on_trial_error=args.continue_on_trial_error,
    )


if __name__ == "__main__":
    main()
