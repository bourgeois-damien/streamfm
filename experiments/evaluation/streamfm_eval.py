"""Command-line launcher for test-set inference.

Parses the common eval options and runs inference locally or on Modal, writing
a manifest that ``score_manifest.py`` then scores. The user-facing eval CLI on
top of ``runner.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core.paths import make_benchmark_paths
from experiments.core.repo import find_repo_root
from experiments.core.devices import select_torch_device
from experiments.evaluation.modal_defaults import resolve_modal_data_path
from experiments.evaluation.results import DEFAULT_EVAL_WANDB_PROJECT, log_eval_result_to_wandb, record_eval_result
from experiments.evaluation.runner import run_test_set_inference

MODAL_VOLUME_NAME = "streamfm-cache"


def _modal_executable() -> str:
    """Return the Modal CLI path usable from this Python process."""
    modal = shutil.which("modal")
    if modal:
        return modal
    repo_modal = REPO_ROOT / ".venv" / "bin" / "modal"
    if repo_modal.exists():
        return str(repo_modal)
    return "modal"


def _safe_label(value: str) -> str:
    """Convert a run label into a filesystem-safe label."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "run"


def _generated_run_name(args: argparse.Namespace) -> str:
    """Generate a stable run name when the user did not pass --run-name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    limit_label = f"limit{args.limit}" if args.limit > 0 else "all"
    parts = [args.task, args.solver, str(args.steps), args.dtype, args.execution, limit_label, timestamp]
    return _safe_label("_".join(parts))


def _local_log_root(args: argparse.Namespace) -> Path:
    """Resolve the local evaluation log root."""
    return Path(args.local_log_dir) if args.local_log_dir else REPO_ROOT / "outputs" / "evaluation_logs"


def _write_local_json(path: Path, payload: dict) -> None:
    """Write a local JSON sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_wandb_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _read_json_if_exists(path: Path) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return {}


def _remote_run_dir(output_dir: str, run_name: str) -> str | None:
    """Return the Modal volume run directory, when it is retrievable."""
    if not output_dir:
        return f"/outputs/eval_runs/{run_name}"
    output_path = Path(output_dir)
    if output_path.is_absolute() and str(output_path).startswith("/data/"):
        return "/" + str(output_path.relative_to("/data") / run_name)
    if str(output_path) == "/data":
        return f"/{run_name}"
    return None


def _download_modal_file(remote_path: str, local_path: Path) -> bool:
    """Download one file from the Modal volume into the local log dir."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _modal_executable(),
        "volume",
        "get",
        "--force",
        MODAL_VOLUME_NAME,
        remote_path,
        str(local_path),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode == 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add test-set inference options shared by local and Modal backends."""
    parser.add_argument("--backend", choices=("modal", "local"), default="modal")
    parser.add_argument("--local", action="store_true", help="Shortcut for --backend local.")
    parser.add_argument("--hardware", default="", help="Modal: cpu/t4/l4/l40s/a100. Local: auto/cpu/mps/cuda.")
    parser.add_argument("--task", default="se", help="Task: se, stftpr, bwe, derev, lyra, or melflow.")
    parser.add_argument("--variant", default="baseline", help="Experiment variant label recorded in logs/W&B.")
    parser.add_argument("--config-name", default="", help="Override the Hydra config name.")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Hydra override applied before loading the checkpoint. Repeat for multiple overrides.",
    )
    parser.add_argument("--ckpt", default="", help="Override checkpoint path/name.")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--data-path", default="", help="Override the selected split path/csv from the config.")
    parser.add_argument("--data-format", default="", help="Override cfg.model.data_module.format.")
    parser.add_argument("--part", default="model", choices=("model", "predictor"))
    parser.add_argument("--pipeline", default="offline", choices=("offline",))
    parser.add_argument("--execution", default="eager", help="Execution: eager, compiled, or cuda_graph.")
    parser.add_argument("--solver", default="euler", help="ODE solver, or compact official form like 5xeuler.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 means all files after offset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--selection",
        default="random",
        choices=("first", "random"),
        help="random selects a reproducible subset; first selects files after offset in dataset order.",
    )
    parser.add_argument("--selection-seed", type=int, default=42, help="Seed used when --selection random.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision mode used for float32 matmul kernels.",
    )
    parser.add_argument(
        "--crop-mode",
        default="full",
        choices=("config", "full"),
        help="full evaluates complete files; config uses cfg.model.data_module.target_duration.",
    )
    parser.add_argument(
        "--memory-format",
        default="contiguous",
        choices=("contiguous", "channels_last"),
        help="4D model tensor memory format.",
    )
    parser.add_argument("--num-threads", type=int, default=0, help="CPU only. 0 leaves PyTorch default unchanged.")
    parser.add_argument("--num-interop-threads", type=int, default=0, help="CPU only. 0 leaves PyTorch default unchanged.")
    parser.add_argument("--output-dir", default="", help="Base output dir. A run_id subdirectory is created inside it.")
    parser.add_argument("--run-name", default="", help="Optional stable run directory name instead of a random run_id.")
    parser.add_argument("--history-json", default="")
    parser.add_argument(
        "--local-log-dir",
        default=str(REPO_ROOT / "outputs" / "evaluation_logs"),
        help="Local directory where command/config/manifest summaries are copied.",
    )
    parser.add_argument("--no-local-log", action="store_true", help="Disable local run metadata copies.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-inputs", action="store_true", help="Also save clean/noisy WAV copies into the run dir.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--score-after-run", action="store_true", help="Run score_manifest.py after a successful eval run.")
    parser.add_argument("--score-with-distillmos", action="store_true", help="Include DistillMOS when using --score-after-run.")
    parser.add_argument("--score-include-stats", action="store_true", help="Include mean/min/median/max stats in score output.")
    parser.add_argument("--score-include-per-file", action="store_true", help="Include per-file rows in score output.")
    parser.add_argument(
        "--score-target",
        choices=("enhanced", "noisy"),
        default="enhanced",
        help="Target passed to score_manifest.py when using --score-after-run.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log this evaluation run to Weights & Biases.")
    parser.add_argument("--wandb-project", default=DEFAULT_EVAL_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="", help="Override the W&B group. Defaults to the eval run id.")
    parser.add_argument("--wandb-mode", default="", help="Optional W&B mode, for example online, offline, or disabled.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags.")


def _run_local(args: argparse.Namespace, hardware: str) -> tuple[dict, str, str]:
    """Run test-set inference locally."""
    device = select_torch_device(hardware)
    repo_root = find_repo_root(Path(__file__))
    paths = make_benchmark_paths(
        repo_root=repo_root,
        config_dir=repo_root / "config",
        checkpoint_roots=(repo_root / "checkpoints",),
    )
    result = run_test_set_inference(
        task=args.task,
        config_name=args.config_name,
        ckpt=args.ckpt,
        split=args.split,
        data_path=args.data_path,
        data_format=args.data_format,
        part=args.part,
        pipeline=args.pipeline,
        execution=args.execution,
        solver=args.solver,
        steps=args.steps,
        limit=args.limit,
        offset=args.offset,
        selection=args.selection,
        selection_seed=args.selection_seed,
        seed=args.seed,
        model_dtype_name=args.dtype,
        float32_matmul_precision=args.matmul_precision,
        model_memory_format=args.memory_format,
        crop_mode=args.crop_mode,
        device=device,
        paths=paths,
        backend="local",
        hardware=hardware,
        output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        save_inputs=args.save_inputs,
        continue_on_error=args.continue_on_error,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
        config_overrides=args.config_override,
    )
    record_eval_result(result=result, command=_command_dict(args, backend="local", hardware=hardware), history_json=args.history_json)
    if not args.no_local_log:
        run_name = result["run_id"]
        log_dir = _local_log_root(args) / run_name
        command_payload = _command_dict(args, backend="local", hardware=hardware)
        command_payload["run_name"] = run_name
        _write_local_json(log_dir / "command.json", command_payload)
        for key, filename in (
            ("summary_path", "summary.json"),
            ("manifest_path", "manifest.json"),
            ("config_path", "config.yaml"),
        ):
            source = Path(result.get(key, ""))
            if source.exists():
                destination = log_dir / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    return result, str(result["run_id"]), str(result["manifest_path"])


def _command_dict(args: argparse.Namespace, *, backend: str, hardware: str) -> dict:
    """Record the user-visible command options."""
    return {
        "backend": backend,
        "hardware": hardware,
        "task": args.task,
        "variant": args.variant,
        "config_name": args.config_name,
        "config_overrides": list(args.config_override),
        "ckpt": args.ckpt,
        "split": args.split,
        "data_path": args.data_path,
        "data_format": args.data_format,
        "part": args.part,
        "pipeline": args.pipeline,
        "execution": args.execution,
        "solver": args.solver,
        "steps": args.steps,
        "limit": args.limit,
        "offset": args.offset,
        "selection": args.selection,
        "selection_seed": args.selection_seed,
        "seed": args.seed,
        "dtype": args.dtype,
        "matmul_precision": args.matmul_precision,
        "crop_mode": args.crop_mode,
        "memory_format": args.memory_format,
        "num_threads": args.num_threads,
        "num_interop_threads": args.num_interop_threads,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "local_log_dir": args.local_log_dir,
        "no_local_log": args.no_local_log,
        "overwrite": args.overwrite,
        "save_inputs": args.save_inputs,
        "continue_on_error": args.continue_on_error,
        "score_after_run": args.score_after_run,
        "score_with_distillmos": args.score_with_distillmos,
        "score_include_stats": args.score_include_stats,
        "score_include_per_file": args.score_include_per_file,
        "score_target": args.score_target,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
        "wandb_mode": args.wandb_mode,
        "wandb_tags": args.wandb_tags,
    }


def _run_modal(args: argparse.Namespace, hardware: str) -> tuple[dict, str, str]:
    """Delegate Modal execution to the Modal evaluation wrapper."""
    modal_script = Path("experiments/evaluation/modal_streamfm_eval.py")
    run_name = args.run_name or _generated_run_name(args)
    data_path = resolve_modal_data_path(args.data_path, task=args.task, split=args.split)
    command_payload = _command_dict(args, backend="modal", hardware=hardware.upper())
    command_payload["run_name"] = run_name
    command_payload["data_path"] = data_path
    log_dir = _local_log_root(args) / run_name
    if not args.no_local_log:
        _write_local_json(log_dir / "command.json", command_payload)
    command = [
        _modal_executable(),
        "run",
        str(modal_script),
        "--hardware",
        hardware.upper(),
        "--task",
        args.task,
        "--split",
        args.split,
        "--data-path",
        data_path,
        "--part",
        args.part,
        "--pipeline",
        args.pipeline,
        "--execution",
        args.execution,
        "--solver",
        args.solver,
        "--steps",
        str(args.steps),
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
        "--selection",
        args.selection,
        "--selection-seed",
        str(args.selection_seed),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--matmul-precision",
        args.matmul_precision,
        "--crop-mode",
        args.crop_mode,
        "--memory-format",
        args.memory_format,
    ]
    optional_pairs = {
        "--config-name": args.config_name,
        "--ckpt": args.ckpt,
        "--data-format": args.data_format,
        "--output-dir": args.output_dir,
        "--run-name": run_name,
        "--history-json": args.history_json,
    }
    for flag, value in optional_pairs.items():
        if value:
            command.extend([flag, value])
    if args.config_override:
        command.extend(["--config-override", "\n".join(args.config_override)])
    if args.num_threads:
        command.extend(["--num-threads", str(args.num_threads)])
    if args.num_interop_threads:
        command.extend(["--num-interop-threads", str(args.num_interop_threads)])
    if args.overwrite:
        command.append("--overwrite")
    if args.save_inputs:
        command.append("--save-inputs")
    if args.continue_on_error:
        command.append("--continue-on-error")
    subprocess.run(command, check=True)
    result: dict = {"run_id": run_name, "backend": "modal", "hardware": hardware.upper()}
    if args.no_local_log:
        return result, run_name, ""

    remote_dir = _remote_run_dir(args.output_dir, run_name)
    if remote_dir is None:
        _write_local_json(
            log_dir / "modal_volume_paths.json",
            {
                "warning": "Automatic download skipped because --output-dir is not inside /data.",
                "output_dir": args.output_dir,
                "run_name": run_name,
            },
        )
        return result, run_name, ""

    downloads = {
        "summary": _download_modal_file(f"{remote_dir}/summary.json", log_dir / "summary.json"),
        "manifest": _download_modal_file(f"{remote_dir}/manifest.json", log_dir / "manifest.json"),
        "config": _download_modal_file(f"{remote_dir}/config.yaml", log_dir / "config.yaml"),
    }
    _write_local_json(
        log_dir / "modal_volume_paths.json",
        {
            "volume": MODAL_VOLUME_NAME,
            "remote_run_dir": remote_dir,
            "local_log_dir": str(log_dir),
            "downloads": downloads,
        },
    )
    print(f"Local run metadata saved to {log_dir}")
    if downloads["summary"]:
        result = _read_json_if_exists(log_dir / "summary.json") or result
    manifest_path = str(log_dir / "manifest.json") if downloads["manifest"] else ""
    return result, run_name, manifest_path


def _score_after_run(args: argparse.Namespace, *, backend: str, run_name: str, manifest_path: str) -> dict:
    """Run score_manifest.py with options matching the completed eval run."""
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "evaluation" / "scoring" / "score_manifest.py"),
        "--backend",
        backend,
        "--limit",
        "0",
        "--score-target",
        args.score_target,
        "--local-log-dir",
        args.local_log_dir,
        "--run-name",
        run_name,
    ]
    if backend != "modal":
        command.append(manifest_path)
    if args.no_local_log:
        command.append("--no-local-log")
    if args.score_with_distillmos:
        command.append("--with-distillmos")
    if args.score_include_stats:
        command.append("--include-stats")
    if args.score_include_per_file:
        command.append("--include-per-file")
    print(f"Scoring eval run with score_manifest.py for run '{run_name}'...")
    subprocess.run(command, check=True)
    metrics_path = _local_log_root(args) / run_name / "metrics_all.json"
    return _read_json_if_exists(metrics_path)


def main() -> None:
    """Run Stream.FM test-set inference locally or on Modal."""
    parser = argparse.ArgumentParser(description="Stream.FM test-set inference runner.")
    _add_common_args(parser)
    args = parser.parse_args()

    backend = "local" if args.local else args.backend
    hardware = args.hardware
    if not hardware:
        hardware = "auto" if backend == "local" else "L4"

    if backend == "local":
        result, run_name, manifest_path = _run_local(args, hardware)
    else:
        result, run_name, manifest_path = _run_modal(args, hardware)

    score_result = {}
    if args.score_after_run:
        score_result = _score_after_run(args, backend=backend, run_name=run_name, manifest_path=manifest_path)

    if args.wandb:
        log_eval_result_to_wandb(
            result=result,
            command=_command_dict(args, backend=backend, hardware=hardware.upper() if backend == "modal" else hardware),
            score_result=score_result,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode=args.wandb_mode,
            tags=_parse_wandb_tags(args.wandb_tags),
        )


if __name__ == "__main__":
    main()
