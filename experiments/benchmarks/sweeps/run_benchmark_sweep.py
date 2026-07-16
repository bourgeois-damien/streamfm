"""Run a single benchmark trial for a W&B sweep.

Reads one trial's parameters from the sweep controller, builds the benchmark
command, runs it locally or on Modal, and logs the metrics back to the run.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core.paths import make_benchmark_paths
from experiments.benchmarks.results import build_benchmark_records, record_benchmark_results
from experiments.benchmarks.runner import run_benchmark
from experiments.core.repo import find_repo_root
from experiments.core.devices import select_torch_device

DEFAULT_INPUT_AUDIO = "inputs/test_clips/audio_43m28_10s.wav"
MODAL_HARDWARE = frozenset({"CPU", "T4", "L4", "L40S", "A100"})


def sweep_config_get(config: Any, key: str, default: Any) -> Any:
    """Read a sweep parameter from wandb.config with a fallback default."""
    try:
        value = config[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _default_audio_hop_s(task: str) -> float:
    """Frame hop in seconds (lyra 20 ms, everything else 16 ms); twin of streamfm_benchmark's."""
    return 0.020 if task.lower().replace("-", "_") == "lyra" else 0.016


def _input_audio_duration_s(input_audio_path: str) -> float:
    """Clip duration in seconds from torchaudio metadata (no decode)."""
    import torchaudio

    info = torchaudio.info(input_audio_path)
    if info.sample_rate <= 0:
        raise ValueError(f"Could not determine sample rate for {input_audio_path}.")
    return float(info.num_frames) / float(info.sample_rate)


def resolve_sweep_iterations(command: dict[str, Any], *, input_audio_path: str) -> int:
    """Resolve measured frame count for audio or model-only sweep trials."""
    pipeline = str(command["pipeline"]).lower().replace("-", "_")
    iterations = int(command["iterations"])
    audio_duration_s = float(command["audio_duration_s"])
    task = str(command["task"])

    if pipeline != "audio":
        if audio_duration_s > 0:
            raise ValueError("audio_duration_s is only supported with pipeline=audio.")
        if iterations == -1:
            raise ValueError("iterations=-1 is only supported with pipeline=audio.")
        return iterations
    if audio_duration_s > 0:
        return max(1, math.ceil(audio_duration_s / _default_audio_hop_s(task)))
    if iterations == -1:
        # Whole-file mode: warmup frames consume the head of the same file,
        # so subtract them to keep measured frames within the audio that exists.
        duration_s = _input_audio_duration_s(input_audio_path) if input_audio_path else 10.0
        warmup = int(command["warmup"])
        return max(1, math.ceil(duration_s / _default_audio_hop_s(task)) - max(warmup, 0))
    return iterations


def normalize_modal_hardware(hardware: str) -> str:
    """Map sweep hardware names to a supported Modal GPU tier."""
    normalized = hardware.strip().upper()
    if not normalized or normalized in {"AUTO", "CUDA"}:
        return "L4"
    if normalized not in MODAL_HARDWARE:
        supported = ", ".join(sorted(MODAL_HARDWARE))
        raise ValueError(f"Unsupported Modal hardware '{hardware}'. Supported values: {supported}.")
    return normalized


def build_sweep_command(
    config: Any,
    *,
    hardware_override: str = "",
    backend_override: str = "",
) -> dict[str, Any]:
    """Map a wandb sweep config to the benchmark command dict."""
    backend = backend_override or str(sweep_config_get(config, "backend", "local"))
    hardware = hardware_override or str(sweep_config_get(config, "hardware", "auto"))
    steps = sweep_config_get(config, "steps", 1)
    if isinstance(steps, (list, tuple)):
        steps_value = ",".join(str(step) for step in steps)
    else:
        steps_value = str(steps)

    return {
        "backend": backend.lower().replace("-", "_"),
        "hardware": hardware,
        "task": str(sweep_config_get(config, "task", "stftpr")),
        "part": str(sweep_config_get(config, "part", "model")),
        "pipeline": str(sweep_config_get(config, "pipeline", "audio")),
        "execution": str(sweep_config_get(config, "execution", "auto")),
        "steps": steps_value,
        "iterations": int(sweep_config_get(config, "iterations", 100)),
        "warmup": int(sweep_config_get(config, "warmup", 10)),
        "audio_duration_s": float(sweep_config_get(config, "audio_duration_s", 0.0)),
        "model_dtype": str(sweep_config_get(config, "dtype", "fp32")),
        "matmul_precision": str(sweep_config_get(config, "matmul_precision", "high")),
        "num_threads": int(sweep_config_get(config, "num_threads", 0)),
        "num_interop_threads": int(sweep_config_get(config, "num_interop_threads", 0)),
        "memory_format": str(sweep_config_get(config, "memory_format", "contiguous")),
        "preallocate_model_buffers": bool(sweep_config_get(config, "preallocate_model_buffers", False)),
        "ptq_int8": str(sweep_config_get(config, "ptq_int8", "")),
        "ptq_calib_steps": int(sweep_config_get(config, "ptq_calib_steps", 32)),
        "save_audio": bool(sweep_config_get(config, "save_audio", False)),
        "audio_output_dir": str(sweep_config_get(config, "audio_output_dir", "")),
        "input_audio": str(sweep_config_get(config, "input_audio", DEFAULT_INPUT_AUDIO)),
        "profile": bool(sweep_config_get(config, "profile", False)),
        "profile_all": bool(sweep_config_get(config, "profile_all", False)),
        "profile_file": str(sweep_config_get(config, "profile_file", "")),
        "sweep": True,
    }


def _resolve_input_audio_path(input_audio: str, *, pipeline: str) -> str:
    """Resolve the clip path for audio trials: a missing default falls back to synthetic input (""); an explicit path must exist."""
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


def log_sweep_results_to_run(
    *,
    results: list[dict],
    command: dict[str, Any],
    run_id: str,
    run_started_at: str,
) -> None:
    """Log benchmark rows to the active wandb sweep run."""
    import wandb

    records = build_benchmark_records(
        results=results,
        command=command,
        run_id=run_id,
        run_started_at=run_started_at,
    )
    if not records:
        return

    if wandb.run is not None:
        primary = records[0]
        wandb.run.name = primary["name"]
        wandb.run.tags = tuple(dict.fromkeys((*wandb.run.tags, *primary["tags"], "sweep")))

    for row_idx, record in enumerate(records):
        if record["metrics"]:
            wandb.log(record["metrics"], step=row_idx)
        # The sweep controller optimizes on summary values, so mirror the
        # primary row's metrics there.
        if wandb.run is not None and row_idx == 0:
            wandb.run.summary.update(
                {f"summary/{key}": value for key, value in record["metrics"].items()}
            )
            if record["metadata"]:
                wandb.run.summary.update(
                    {f"metadata/{key}": value for key, value in record["metadata"].items()}
                )


def _run_benchmark_local(command: dict[str, Any], *, input_audio_path: str) -> list[dict]:
    """Run the trial in-process through run_benchmark on local hardware."""
    hardware = str(command["hardware"])
    device = select_torch_device(hardware)
    if command["execution"].replace("-", "_").lower() == "cuda_graph" and device.type != "cuda":
        raise ValueError("execution=cuda_graph requires local CUDA hardware.")

    repo_root = find_repo_root(Path(__file__))
    paths = make_benchmark_paths(
        repo_root=repo_root,
        config_dir=repo_root / "config",
        checkpoint_roots=(repo_root / "checkpoints",),
    )
    return run_benchmark(
        task=str(command["task"]),
        part=str(command["part"]),
        pipeline=str(command["pipeline"]),
        execution=str(command["execution"]),
        steps=str(command["steps"]),
        iterations=int(command["iterations"]),
        warmup=int(command["warmup"]),
        model_dtype_name=str(command["model_dtype"]),
        float32_matmul_precision=str(command["matmul_precision"]),
        device=device,
        paths=paths,
        backend="local",
        hardware=hardware,
        num_threads=int(command["num_threads"]),
        num_interop_threads=int(command["num_interop_threads"]),
        preallocate_model_buffers=bool(command["preallocate_model_buffers"]),
        model_memory_format=str(command["memory_format"]),
        save_audio=bool(command["save_audio"]),
        input_audio_path=input_audio_path,
        profile=bool(command["profile"]),
        profile_all=bool(command["profile_all"]),
        profile_file=str(command["profile_file"]),
        ptq_int8=str(command.get("ptq_int8", "")),
        ptq_calib_steps=int(command.get("ptq_calib_steps", 32)),
    )


def _run_benchmark_modal(command: dict[str, Any], *, input_audio_path: str, output_json: Path) -> list[dict]:
    """Run the trial through a `modal run` subprocess and read results back from output_json.

    Same round-trip as streamfm_benchmark._run_modal: the Modal CLI owns app
    setup, and the JSON file carries the rows back without importing torch here.
    """
    modal_script = REPO_ROOT / "experiments/benchmarks/modal_streamfm_benchmark.py"
    hardware = normalize_modal_hardware(str(command["hardware"]))
    modal_command = [
        "modal",
        "run",
        str(modal_script),
        "--hardware",
        hardware,
        "--task",
        str(command["task"]),
        "--part",
        str(command["part"]),
        "--pipeline",
        str(command["pipeline"]),
        "--execution",
        str(command["execution"]),
        "--steps",
        str(command["steps"]),
        "--iterations",
        str(command["iterations"]),
        "--warmup",
        str(command["warmup"]),
        "--dtype",
        str(command["model_dtype"]),
        "--matmul-precision",
        str(command["matmul_precision"]),
        "--audio-duration-s",
        str(command["audio_duration_s"]),
        "--output-json",
        str(output_json),
    ]
    if int(command["num_threads"]):
        modal_command.extend(["--num-threads", str(command["num_threads"])])
    if int(command["num_interop_threads"]):
        modal_command.extend(["--num-interop-threads", str(command["num_interop_threads"])])
    if str(command["memory_format"]) != "contiguous":
        modal_command.extend(["--memory-format", str(command["memory_format"])])
    if bool(command["preallocate_model_buffers"]):
        modal_command.append("--preallocate-model-buffers")
    if str(command.get("ptq_int8", "")):
        modal_command.extend(["--ptq-int8", str(command["ptq_int8"])])
        modal_command.extend(["--ptq-calib-steps", str(int(command.get("ptq_calib_steps", 32)))])
    if bool(command["save_audio"]):
        modal_command.append("--save-audio")
    if str(command["audio_output_dir"]):
        modal_command.extend(["--audio-output-dir", str(command["audio_output_dir"])])
    if input_audio_path:
        modal_command.extend(["--input-audio", input_audio_path])
    if bool(command["profile"]):
        modal_command.append("--profile")
    if str(command["profile_file"]):
        modal_command.extend(["--profile-file", str(command["profile_file"])])

    subprocess.run(modal_command, check=True, cwd=REPO_ROOT)
    if not output_json.exists():
        raise RuntimeError(f"Modal benchmark did not write results to {output_json}.")
    results = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError(f"Expected Modal benchmark output to be a JSON list: {output_json}")
    # Record the normalized GPU tier and backend so downstream logging matches
    # what actually ran, not what the sweep config requested.
    command["hardware"] = hardware
    command["backend"] = "modal"
    return results


def run_sweep_trial(
    config: Any,
    *,
    hardware_override: str = "",
    backend_override: str = "",
    history_json: str = "",
    output_json: str = "",
) -> list[dict]:
    """Execute one wandb sweep trial on local or Modal hardware and log metrics to the active run."""
    command = build_sweep_command(
        config,
        hardware_override=hardware_override,
        backend_override=backend_override,
    )
    input_audio_path = _resolve_input_audio_path(
        str(command["input_audio"]),
        pipeline=str(command["pipeline"]),
    )
    command["iterations"] = resolve_sweep_iterations(command, input_audio_path=input_audio_path)

    backend = str(command["backend"])
    if backend == "modal":
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="streamfm_sweep_modal_",
            delete=False,
        ) as tmp_file:
            modal_output_json = Path(tmp_file.name)
        try:
            results = _run_benchmark_modal(
                command,
                input_audio_path=input_audio_path,
                output_json=modal_output_json,
            )
        finally:
            modal_output_json.unlink(missing_ok=True)
    elif backend == "local":
        results = _run_benchmark_local(command, input_audio_path=input_audio_path)
    else:
        raise ValueError("Unsupported backend. Use 'local' or 'modal'.")

    run_id = uuid4().hex[:12]
    run_started_at = datetime.now(timezone.utc).isoformat()
    log_sweep_results_to_run(
        results=results,
        command=command,
        run_id=run_id,
        run_started_at=run_started_at,
    )
    # History still gets the rows; wandb_enabled=False because the metrics were
    # already logged onto the live sweep run above — a fresh run per row would
    # detach them from the sweep.
    record_benchmark_results(
        results=results,
        command=command,
        output_json=output_json,
        history_json=history_json,
        wandb_enabled=False,
    )
    return results


def main() -> None:
    """Entry point for `wandb agent` sweep trials."""
    parser = argparse.ArgumentParser(description="Run one Stream.FM benchmark trial for a wandb sweep.")
    parser.add_argument(
        "--hardware",
        default="",
        help="Override sweep hardware on this agent. Local: auto/cuda/cpu/mps. Modal: L4/T4/A100/...",
    )
    parser.add_argument(
        "--backend",
        default="",
        choices=("local", "modal"),
        help="Override sweep backend. The wandb agent still runs locally; Modal only changes where compute runs.",
    )
    parser.add_argument("--history-json", default="", help="Optional extra benchmark history JSON path.")
    parser.add_argument("--output-json", default="", help="Optional per-trial JSON output path.")
    args = parser.parse_args()

    import wandb

    wandb.init(job_type="benchmark-sweep")
    try:
        run_sweep_trial(
            wandb.config,
            hardware_override=args.hardware,
            backend_override=args.backend,
            history_json=args.history_json,
            output_json=args.output_json,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
