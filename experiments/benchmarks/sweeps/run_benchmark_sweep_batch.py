"""Run a whole sweep grid in one pass.

Expands the grid, shards the trials and runs each one locally or on Modal,
logging every trial to W&B. The batch alternative to a hosted sweep agent.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.results import DEFAULT_SWEEP_WANDB_PROJECT, record_benchmark_results
from experiments.benchmarks.sweeps.run_benchmark_sweep import (
    _run_benchmark_local,
    build_sweep_command,
    log_sweep_results_to_run,
    normalize_modal_hardware,
    resolve_sweep_iterations,
)
from experiments.benchmarks.sweeps.sweep_grid import (
    expand_parameter_grid,
    load_sweep_metadata,
    load_sweep_parameters,
    load_sweep_trials,
)

DEFAULT_INPUT_AUDIO = "inputs/test_clips/audio_43m28_10s.wav"
GPU_HARDWARE = frozenset({"T4", "L4", "L40S", "A100"})
GPU_WORKERS_CONFIRM_THRESHOLD = 4


def _resolve_input_audio_path(input_audio: str, *, pipeline: str) -> str:
    """Twin of run_benchmark_sweep._resolve_input_audio_path (see there)."""
    if pipeline.lower().replace("-", "_") != "audio":
        return ""
    requested = input_audio.strip()
    if not requested:
        return ""
    path = Path(requested).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        return str(path)
    if requested == DEFAULT_INPUT_AUDIO:
        return ""
    raise FileNotFoundError(f"Input audio not found: {path}")


def shard_trials(trials: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Split trials into up to `workers` contiguous shards for parallel Modal containers."""
    if not trials:
        return []
    worker_count = max(1, min(int(workers), len(trials)))
    if worker_count == 1:
        return [list(trials)]

    base, remainder = divmod(len(trials), worker_count)
    shards: list[list[dict[str, Any]]] = []
    start = 0
    for worker_idx in range(worker_count):
        size = base + (1 if worker_idx < remainder else 0)
        shards.append(list(trials[start : start + size]))
        start += size
    return shards


def needs_gpu_workers_confirmation(hardware_names: list[str] | set[str], workers: int) -> bool:
    """Return True when a GPU sweep would launch more than the safe worker threshold."""
    return int(workers) > GPU_WORKERS_CONFIRM_THRESHOLD and any(
        str(name).upper() in GPU_HARDWARE for name in hardware_names
    )


def confirm_gpu_workers(*, hardware_names: list[str], workers: int, trial_count: int) -> bool:
    """Ask for interactive confirmation before launching many parallel GPU containers."""
    gpu_names = sorted({name.upper() for name in hardware_names if name.upper() in GPU_HARDWARE})
    print(
        f"Attention: tu vas lancer jusqu'a {workers} containers GPU Modal "
        f"({', '.join(gpu_names)}) pour {trial_count} trial(s).\n"
        f"Ca peut couter cher rapidement. Seuil de confirmation: >{GPU_WORKERS_CONFIRM_THRESHOLD} workers GPU.",
        flush=True,
    )
    try:
        answer = input("Es-tu sur de vouloir continuer ? [y/N] ").strip().lower()
    except EOFError:
        print("Pas d'input interactif — annulation.", flush=True)
        return False
    return answer in {"y", "yes", "o", "oui"}


def _prepare_modal_trials(
    trials: list[dict[str, Any]],
    *,
    input_audio_path: str,
) -> list[dict[str, Any]]:
    """Resolve per-trial iteration counts before sending them to Modal.

    The -1/duration modes need the local clip's duration; the remote container
    cannot probe the file before receiving it, so send resolved counts.
    """
    prepared: list[dict[str, Any]] = []
    for trial in trials:
        trial_copy = dict(trial)
        command = build_sweep_command(trial_copy)
        trial_copy["iterations"] = resolve_sweep_iterations(command, input_audio_path=input_audio_path)
        prepared.append(trial_copy)
    return prepared


def run_modal_benchmark_batch(
    *,
    hardware: str,
    trials: list[dict[str, Any]],
    input_audio_path: str,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run a hardware group on one or more Modal containers and return per-trial results."""
    from experiments.benchmarks.modal_streamfm_benchmark import (
        MODAL_BATCH_FUNCTIONS,
        _read_input_audio_bytes,
        app,
    )

    selected_hardware = normalize_modal_hardware(hardware)
    if selected_hardware not in MODAL_BATCH_FUNCTIONS:
        supported = ", ".join(sorted(MODAL_BATCH_FUNCTIONS))
        raise ValueError(f"Unsupported Modal hardware '{hardware}'. Supported values: {supported}.")

    prepared_trials = _prepare_modal_trials(trials, input_audio_path=input_audio_path)
    shards = shard_trials(prepared_trials, workers)
    input_audio_bytes, input_audio_name = _read_input_audio_bytes(input_audio_path)
    batch_fn = MODAL_BATCH_FUNCTIONS[selected_hardware]

    print(
        f"Launching {len(shards)} Modal {selected_hardware} container(s) "
        f"for {len(prepared_trials)} trial(s)...",
        flush=True,
    )
    with app.run():
        if len(shards) == 1:
            return batch_fn.remote(shards[0], input_audio_bytes, input_audio_name)

        # Spawn all shards first so the containers run concurrently; .get()
        # below just collects them in order.
        handles = [
            batch_fn.spawn(shard, input_audio_bytes, input_audio_name)
            for shard in shards
        ]
        outputs: list[dict[str, Any]] = []
        for shard_idx, handle in enumerate(handles):
            shard_outputs = handle.get()
            print(
                f"Container {shard_idx + 1}/{len(handles)} finished "
                f"({len(shard_outputs)} trial(s)).",
                flush=True,
            )
            outputs.extend(shard_outputs)
        return outputs


def run_local_benchmark_batch(
    *,
    trials: list[dict[str, Any]],
    input_audio_path: str,
) -> list[dict[str, Any]]:
    """Run all trials sequentially on local hardware and return per-trial results."""
    prepared_trials = _prepare_modal_trials(trials, input_audio_path=input_audio_path)
    outputs: list[dict[str, Any]] = []
    total = len(prepared_trials)
    for trial_index, trial in enumerate(prepared_trials):
        command = build_sweep_command(trial)
        command["iterations"] = int(trial["iterations"])
        command["backend"] = "local"
        print(
            f"[{trial_index + 1}/{total}] local "
            f"execution={command.get('execution')} dtype={command.get('model_dtype')} "
            f"memory_format={command.get('memory_format')} "
            f"preallocate={command.get('preallocate_model_buffers')}",
            flush=True,
        )
        results = _run_benchmark_local(command, input_audio_path=input_audio_path)
        mean_ms = None
        if results and isinstance(results[0], dict):
            mean_ms = results[0].get("total_mean_ms", results[0].get("mean_ms"))
        if mean_ms is not None:
            print(f"[{trial_index + 1}/{total}] done — mean={mean_ms:.2f} ms", flush=True)
        else:
            print(f"[{trial_index + 1}/{total}] done", flush=True)
        outputs.append(
            {
                "trial_index": trial_index,
                "trial": trial,
                "results": results,
            }
        )
    return outputs


def log_batch_trials_to_wandb(
    *,
    batch_outputs: list[dict[str, Any]],
    project: str,
    entity: str,
    group: str,
    input_audio_path: str,
    history_json: str = "",
    backend: str = "modal",
    wandb_enabled: bool = True,
) -> int:
    """Create one W&B run per batch trial and return the number of runs logged."""
    wandb = None
    if wandb_enabled:
        import wandb as wandb_module

        wandb = wandb_module

    logged_runs = 0
    total = len(batch_outputs)
    for batch_item in batch_outputs:
        trial = dict(batch_item["trial"])
        results = batch_item["results"]
        command = build_sweep_command(trial)
        command["iterations"] = resolve_sweep_iterations(command, input_audio_path=input_audio_path)
        command["backend"] = backend
        if backend == "modal":
            command["hardware"] = normalize_modal_hardware(str(command["hardware"]))
        else:
            command["hardware"] = str(command["hardware"]).lower()

        run_id = uuid4().hex[:12]
        run_started_at = datetime.now(timezone.utc).isoformat()
        init_kwargs = {
            "project": project,
            "job_type": "benchmark-sweep-batch",
            "config": command,
            # One process creates many runs back to back; reinit allows the
            # next wandb.init after the previous run finishes.
            "reinit": True,
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group

        if wandb_enabled:
            print(
                f"Logging W&B run {logged_runs + 1}/{total} "
                f"({command.get('execution')}, {command.get('model_dtype')}, "
                f"{command.get('memory_format')}) → project={project}",
                flush=True,
            )
            assert wandb is not None
            wandb.init(**init_kwargs)
        try:
            if wandb_enabled:
                assert wandb is not None
                log_sweep_results_to_run(
                    results=results,
                    command=command,
                    run_id=run_id,
                    run_started_at=run_started_at,
                )
            record_benchmark_results(
                results=results,
                command=command,
                history_json=history_json,
                wandb_enabled=False,
            )
            logged_runs += 1
        finally:
            if wandb_enabled:
                assert wandb is not None
                wandb.finish()
    return logged_runs


def run_sweep_batch(
    *,
    sweep_yaml: str | Path,
    wandb_project: str,
    wandb_entity: str = "",
    wandb_group: str = "",
    history_json: str = "",
    workers: int = 1,
    dry_run: bool = False,
    assume_yes: bool = False,
    wandb_enabled: bool = True,
) -> tuple[int, int]:
    """Expand a sweep YAML grid, run it locally or on Modal, and log to W&B."""
    # 1) Expand the grid and validate the backend.
    trials = load_sweep_trials(sweep_yaml)
    if not trials:
        raise ValueError("Sweep YAML produced zero trials after exclusions.")

    raw_count = len(expand_parameter_grid(load_sweep_parameters(sweep_yaml)))
    excluded_count = raw_count - len(trials)
    worker_count = max(1, int(workers))

    backend = str(trials[0].get("backend", "modal")).lower().replace("-", "_")
    if backend not in {"modal", "local"}:
        raise ValueError("Batch mode supports only backend=modal or backend=local.")

    first_trial = trials[0]
    pipeline = str(first_trial.get("pipeline", "audio"))
    input_audio_path = _resolve_input_audio_path(
        str(first_trial.get("input_audio", DEFAULT_INPUT_AUDIO)),
        pipeline=pipeline,
    )

    # 2) Group trials by hardware tier: each Modal batch function is bound to
    # one GPU type, so a mixed sweep runs one batch per tier.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        if backend == "modal":
            hardware = normalize_modal_hardware(str(trial.get("hardware", "L4")))
        else:
            hardware = str(trial.get("hardware", "cpu")).lower()
        trial["hardware"] = hardware
        trial["backend"] = backend
        grouped[hardware].append(trial)

    # 3) Dry run: print the shard plan without launching any compute.
    total_trials = len(trials)
    if dry_run:
        if backend == "local":
            print(f"Would run {total_trials} local trial(s) sequentially (workers ignored).")
        else:
            print(f"Would run {total_trials} trial(s) with up to {worker_count} worker(s) per hardware.")
        if excluded_count:
            print(f"Excluded {excluded_count} redundant trial(s) via exclude rules.")
        for hardware, hardware_trials in grouped.items():
            if backend == "local":
                print(f"- local/{hardware}: {len(hardware_trials)} trial(s)")
            else:
                shards = shard_trials(hardware_trials, worker_count)
                sizes = ", ".join(str(len(shard)) for shard in shards)
                print(f"- {hardware}: {len(hardware_trials)} trial(s) → {len(shards)} container(s) [{sizes}]")
        if backend == "modal" and needs_gpu_workers_confirmation(grouped.keys(), worker_count):
            print(
                f"Note: this GPU plan would ask for confirmation "
                f"(workers={worker_count} > {GPU_WORKERS_CONFIRM_THRESHOLD})."
            )
        return len(grouped), total_trials

    # 4) Many parallel GPU containers cost real money — ask first.
    if backend == "modal" and needs_gpu_workers_confirmation(grouped.keys(), worker_count) and not assume_yes:
        if not confirm_gpu_workers(
            hardware_names=list(grouped.keys()),
            workers=worker_count,
            trial_count=total_trials,
        ):
            print("Annule.")
            return 0, 0

    if backend == "local" and worker_count > 1:
        print("Note: --workers is ignored for backend=local (runs sequentially on this machine).", flush=True)

    # 5) Run each hardware group, then create the W&B runs from the local
    # side once the results are back.
    logged_runs = 0
    for hardware, hardware_trials in grouped.items():
        if backend == "local":
            print(
                f"Running {len(hardware_trials)} trial(s) locally on {hardware}...",
                flush=True,
            )
            batch_outputs = run_local_benchmark_batch(
                trials=hardware_trials,
                input_audio_path=input_audio_path,
            )
        else:
            shards = shard_trials(hardware_trials, worker_count)
            print(
                f"Running {len(hardware_trials)} trial(s) on Modal {hardware} "
                f"with {len(shards)} container(s)...",
                flush=True,
            )
            batch_outputs = run_modal_benchmark_batch(
                hardware=hardware,
                trials=hardware_trials,
                input_audio_path=input_audio_path,
                workers=worker_count,
            )
        logged_runs += log_batch_trials_to_wandb(
            batch_outputs=batch_outputs,
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            input_audio_path=input_audio_path,
            history_json=history_json,
            backend=backend,
            wandb_enabled=wandb_enabled,
        )

    action = "Logged to W&B" if wandb_enabled else "Recorded locally"
    print(f"{action}: {logged_runs}/{total_trials} batch trial(s).")
    return len(grouped), logged_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an entire sweep grid locally or on Modal and log each trial to W&B.",
    )
    parser.add_argument("--sweep-yaml", default="experiments/benchmarks/sweeps/configs/sweep_l4_steps1.yaml")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="", help="Optional W&B group for all batch runs.")
    parser.add_argument("--history-json", default="", help="Optional extra benchmark history JSON path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Modal containers per hardware (ignored for backend=local).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for >4 parallel GPU workers.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the batch plan without launching compute.")
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Write local history files without importing or logging to W&B.",
    )
    args = parser.parse_args()

    metadata = load_sweep_metadata(args.sweep_yaml)
    project = args.wandb_project or str(metadata.get("project") or DEFAULT_SWEEP_WANDB_PROJECT)
    entity = args.wandb_entity or str(metadata.get("entity") or "")

    run_sweep_batch(
        sweep_yaml=args.sweep_yaml,
        wandb_project=project,
        wandb_entity=entity,
        wandb_group=args.wandb_group,
        history_json=args.history_json,
        workers=args.workers,
        dry_run=args.dry_run,
        assume_yes=args.yes,
        wandb_enabled=not args.skip_wandb,
    )


if __name__ == "__main__":
    main()
