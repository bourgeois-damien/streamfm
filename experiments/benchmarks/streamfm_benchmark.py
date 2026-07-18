"""Command-line launcher for the benchmark.

Parses the common options and runs the benchmark either locally or on Modal.
The user-facing CLI on top of ``runner.py``.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core.repo import find_repo_root
from experiments.core.devices import select_torch_device
from experiments.core.paths import make_benchmark_paths
from experiments.benchmarks.results import DEFAULT_WANDB_PROJECT, record_benchmark_results
from experiments.benchmarks.runner import run_benchmark

DEFAULT_INPUT_AUDIO = "inputs/test_clips/audio_43m28_10s.wav"


def _safe_name(value: str) -> str:
    """Sanitize a value for use inside an output filename."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "unknown"


def _save_audio_results(results: list[dict], args: argparse.Namespace, *, backend: str, hardware: str) -> None:
    """Save optional returned benchmark audio locally and strip tensors from JSON rows."""
    if not args.save_audio:
        return

    import torch
    import torchaudio

    output_dir = Path(args.audio_output_dir or "outputs/benchmark_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for row_idx, row in enumerate(results):
        audio = row.pop("audio", None)
        if audio is None:
            continue
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio = audio.detach().cpu().float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        sample_rate = int(row.get("audio_sample_rate", 16000))
        stem = "_".join(
            _safe_name(str(part))
            for part in (
                timestamp,
                backend,
                hardware,
                row.get("task", args.task),
                row.get("pipeline", args.pipeline),
                row.get("execution", args.execution),
                f"steps{row.get('steps', args.steps)}",
                args.dtype,
                row_idx,
            )
        )
        audio_path = output_dir / f"{stem}.wav"
        torchaudio.save(str(audio_path), audio.clamp(-1, 1), sample_rate=sample_rate)
        row["saved_audio_path"] = str(audio_path)
        print(f"Saved benchmark audio to {audio_path}")


def _default_audio_hop_s(task: str) -> float:
    """Frame hop in seconds, used to convert an audio duration into a frame count.

    Lyra uses a 20 ms hop (320 samples at 16 kHz); every other task hops
    16 ms (256 samples).
    """
    return 0.020 if task.lower().replace("-", "_") == "lyra" else 0.016


def _resolve_input_audio_path(args: argparse.Namespace) -> str:
    """Resolve --input-audio to an absolute path, or "" to fall back to synthetic audio.

    The default clip is best-effort: if it is missing the pipeline generates
    audio instead. A path the user typed explicitly must exist.
    """
    if args.pipeline.lower().replace("-", "_") != "audio":
        return ""
    requested = args.input_audio.strip()
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


def _default_profile_file(args: argparse.Namespace, hardware: str) -> str:
    output_dir = Path("outputs/benchmark_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(
        _safe_name(str(part))
        for part in (
            timestamp,
            hardware,
            args.task,
            args.pipeline,
            args.execution,
            f"steps{args.steps}",
            f"dtype{args.dtype}",
            f"iters{args.iterations}",
        )
    )
    return str(output_dir / f"{stem}.txt")


def _parse_wandb_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _input_audio_duration_s(input_audio_path: str) -> float:
    import torchaudio

    info = torchaudio.info(input_audio_path)
    if info.sample_rate <= 0:
        raise ValueError(f"Could not determine sample rate for {input_audio_path}.")
    return float(info.num_frames) / float(info.sample_rate)


def _resolve_iterations(args: argparse.Namespace) -> int:
    """Turn the three iteration modes into a concrete measured-frame count.

    Modes: an explicit --iterations count; --audio-duration-s converted to
    frames via the task hop; or --iterations -1 meaning "the whole input
    file" (audio pipeline only, since only there do frames map to audio).
    """
    if args.pipeline.lower().replace("-", "_") != "audio":
        if args.audio_duration_s > 0:
            raise ValueError("--audio-duration-s is only supported with --pipeline audio.")
        if args.iterations == -1:
            raise ValueError("--iterations -1 is only supported with --pipeline audio.")
        return args.iterations
    if args.audio_duration_s > 0:
        return max(1, math.ceil(args.audio_duration_s / _default_audio_hop_s(args.task)))
    if args.iterations == -1:
        # Whole-file mode: warmup frames come out of the same file, so subtract
        # them to keep measured frames within the audio that actually exists.
        duration_s = _input_audio_duration_s(args.input_audio_path) if args.input_audio_path else 10.0
        return max(1, math.ceil(duration_s / _default_audio_hop_s(args.task)) - max(args.warmup, 0))
    return args.iterations


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmark options shared by local and Modal backends."""
    parser.add_argument("--backend", choices=("modal", "local"), default="modal")
    parser.add_argument("--local", action="store_true", help="Shortcut for --backend local.")
    parser.add_argument("--hardware", default="", help="Modal: cpu/t4/l4/l40s/a100. Local: auto/cpu/mps/cuda.")
    parser.add_argument("--task", default="stftpr", help="Task: stftpr, bwe, derev, lyra, or se.")
    parser.add_argument("--part", default="model", help="Part: model, predictor, or flow.")
    parser.add_argument("--pipeline", default="audio", help="Pipeline: model_only or audio.")
    parser.add_argument(
        "--execution",
        default="auto",
        help=(
            "Execution: auto, eager, compiled, cuda_graph, tensorrt, or "
            "tensorrt_cuda_graph (TensorRT is FP32/FP16; add --ptq-int8 for INT8)."
        ),
    )
    parser.add_argument("--steps", default="1", help="Comma-separated flow step counts.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured frame count. Use --audio-duration-s for duration-based audio runs.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--audio-duration-s", type=float, default=0.0, help="If >0 with --pipeline audio, override --iterations and process this duration.")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help=(
            "Floating dtype. For TensorRT INT8, fp32/fp16 selects the allowed "
            "floating-point fallback; calibration and engine I/O remain FP32."
        ),
    )
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision mode used for float32 matmul kernels.",
    )
    parser.add_argument(
        "--tf32",
        default="auto",
        choices=("auto", "on", "off"),
        help=(
            "Explicit TF32 policy. Controls cuDNN convolutions in PyTorch and "
            "TensorRT's disable_tf32 builder setting; matmul precision remains "
            "controlled separately by --matmul-precision."
        ),
    )
    parser.add_argument("--ckpt", default="", help="Optional full checkpoint; compressed checkpoints are detected automatically.")
    parser.add_argument("--num-threads", type=int, default=0, help="CPU only. 0 leaves PyTorch default unchanged.")
    parser.add_argument("--num-interop-threads", type=int, default=0, help="CPU only. 0 leaves PyTorch default unchanged.")
    parser.add_argument(
        "--memory-format",
        default="contiguous",
        choices=("contiguous", "channels_last"),
        help="4D model tensor memory format.",
    )
    parser.add_argument(
        "--preallocate-model-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse model input/state buffers where possible.",
    )
    parser.add_argument(
        "--ptq-int8",
        nargs="?",
        const="tensorrt",
        default="",
        help=(
            "INT8 PTQ. With --execution tensorrt, use --ptq-int8 tensorrt "
            "(or the bare flag locally) for TensorRT INT8. Otherwise accepts "
            "CPU components: linear, conv, causal_conv, all."
        ),
    )
    parser.add_argument(
        "--ptq-calib-steps",
        type=int,
        default=32,
        help="Calibration mini-batches for static PTQ components.",
    )
    parser.add_argument(
        "--trt-optimization-level",
        type=int,
        default=3,
        choices=range(0, 6),
        metavar="{0..5}",
        help="TensorRT builder search level. Higher can improve the engine but increases build time.",
    )
    parser.add_argument(
        "--trt-avg-timing-iters",
        type=int,
        default=1,
        help="TensorRT tactic timing repetitions during engine build (default: 1).",
    )
    parser.add_argument(
        "--trt-workspace-size-mib",
        type=int,
        default=0,
        help=(
            "TensorRT temporary workspace cap in MiB. 0 keeps TensorRT's automatic, "
            "effectively unbounded-by-the-runner policy."
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--history-json", default="")
    parser.add_argument("--save-audio", action="store_true", help="Save returned audio for --pipeline audio runs.")
    parser.add_argument("--audio-output-dir", default="", help="Local directory for --save-audio WAV files.")
    parser.add_argument(
        "--input-audio",
        default=DEFAULT_INPUT_AUDIO,
        help="Optional real audio file for --pipeline audio. Defaults to inputs/test_clips/audio_43m28_10s.wav if present.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable the generic PyTorch profiler. It is useful for eager/compiled "
            "operator attribution but includes model construction; use the TensorRT "
            "runtime stage metrics and an Nsight trace for CUDA Graph analysis."
        ),
    )
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="Enable PyTorch profiler for the complete benchmark run including load and compilation.",
    )
    parser.add_argument(
        "--profile-file",
        default="",
        help="Save the profiler summary to a local text file. Defaults to outputs/benchmark_profiles/ when profiling is enabled.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log benchmark results to Weights & Biases for this run.",
    )
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="", help="Override the W&B group. Defaults to the generated benchmark run id.")
    parser.add_argument("--wandb-mode", default="", help="Optional W&B mode, for example online, offline, or disabled.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags.")


def _run_local(args: argparse.Namespace, hardware: str) -> None:
    """Run the benchmark directly on local hardware."""
    device = select_torch_device(hardware)
    if args.execution.replace("-", "_").lower() == "cuda_graph" and device.type != "cuda":
        raise ValueError("execution=cuda_graph requires local CUDA hardware.")

    if args.profile and not args.profile_file:
        args.profile_file = _default_profile_file(args, hardware)

    repo_root = find_repo_root(Path(__file__))
    paths = make_benchmark_paths(
        repo_root=repo_root,
        config_dir=repo_root / "config",
        checkpoint_roots=(repo_root / "checkpoints",),
    )
    results = run_benchmark(
        task=args.task,
        part=args.part,
        pipeline=args.pipeline,
        execution=args.execution,
        steps=args.steps,
        iterations=args.iterations,
        warmup=args.warmup,
        model_dtype_name=args.dtype,
        float32_matmul_precision=args.matmul_precision,
        tf32_mode=args.tf32,
        device=device,
        paths=paths,
        backend="local",
        hardware=hardware,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
        preallocate_model_buffers=args.preallocate_model_buffers,
        model_memory_format=args.memory_format,
        save_audio=args.save_audio,
        input_audio_path=args.input_audio_path,
        profile=args.profile,
        profile_all=args.profile_all,
        profile_file=args.profile_file,
        checkpoint_name=args.ckpt,
        ptq_int8=args.ptq_int8,
        ptq_calib_steps=args.ptq_calib_steps,
        tensorrt_optimization_level=args.trt_optimization_level,
        tensorrt_num_avg_timing_iters=args.trt_avg_timing_iters,
        tensorrt_workspace_size_mib=args.trt_workspace_size_mib,
    )
    _save_audio_results(results, args, backend="local", hardware=hardware)
    record_benchmark_results(
        results=results,
        output_json=args.output_json,
        history_json=args.history_json,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=_parse_wandb_tags(args.wandb_tags),
        command={
            "backend": "local",
            "hardware": hardware,
            "task": args.task,
            "part": args.part,
            "pipeline": args.pipeline,
            "execution": args.execution,
            "steps": args.steps,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "audio_duration_s": args.audio_duration_s,
            "model_dtype": args.dtype,
            "matmul_precision": args.matmul_precision,
            "tf32": args.tf32,
            "ckpt": args.ckpt,
            "num_threads": args.num_threads,
            "num_interop_threads": args.num_interop_threads,
            "memory_format": args.memory_format,
            "preallocate_model_buffers": args.preallocate_model_buffers,
            "ptq_int8": args.ptq_int8,
            "ptq_calib_steps": args.ptq_calib_steps,
            "trt_optimization_level": args.trt_optimization_level,
            "trt_avg_timing_iters": args.trt_avg_timing_iters,
            "trt_workspace_size_mib": args.trt_workspace_size_mib,
            "save_audio": args.save_audio,
            "audio_output_dir": args.audio_output_dir,
            "input_audio": args.input_audio_path,
            "profile": args.profile,
            "profile_all": args.profile_all,
            "profile_file": args.profile_file,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_mode": args.wandb_mode,
            "wandb_tags": args.wandb_tags,
        },
    )


def _run_modal(args: argparse.Namespace, hardware: str) -> None:
    """Delegate Modal execution to the Modal wrapper.

    Re-invokes ``modal run`` as a subprocess because the Modal CLI owns app
    setup and remote deployment; options are forwarded only when they differ
    from their defaults to keep the remote command line readable.
    """
    modal_script = Path("experiments/benchmarks/modal_streamfm_benchmark.py")
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(modal_script),
        "--hardware",
        hardware.upper(),
        "--task",
        args.task,
        "--part",
        args.part,
        "--pipeline",
        args.pipeline,
        "--execution",
        args.execution,
        "--steps",
        args.steps,
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--matmul-precision",
        args.matmul_precision,
        "--tf32",
        args.tf32,
        "--trt-optimization-level",
        str(args.trt_optimization_level),
        "--trt-avg-timing-iters",
        str(args.trt_avg_timing_iters),
        "--trt-workspace-size-mib",
        str(args.trt_workspace_size_mib),
    ]
    command.extend(["--audio-duration-s", str(args.audio_duration_s)])
    if args.ckpt:
        command.extend(["--ckpt", args.ckpt])
    if args.num_threads:
        command.extend(["--num-threads", str(args.num_threads)])
    if args.num_interop_threads:
        command.extend(["--num-interop-threads", str(args.num_interop_threads)])
    if args.memory_format != "contiguous":
        command.extend(["--memory-format", args.memory_format])
    if args.preallocate_model_buffers:
        command.append("--preallocate-model-buffers")
    if args.ptq_int8:
        command.extend(["--ptq-int8", args.ptq_int8])
        command.extend(["--ptq-calib-steps", str(args.ptq_calib_steps)])
    if args.save_audio:
        command.append("--save-audio")
    if args.profile:
        command.append("--profile")
    if args.profile and not args.profile_file:
        args.profile_file = _default_profile_file(args, hardware)
    if args.profile_file:
        command.extend(["--profile-file", args.profile_file])
    if args.audio_output_dir:
        command.extend(["--audio-output-dir", args.audio_output_dir])
    if args.input_audio_path:
        command.extend(["--input-audio", args.input_audio_path])
    if args.output_json:
        command.extend(["--output-json", args.output_json])
    if args.history_json:
        command.extend(["--history-json", args.history_json])
    if args.wandb:
        command.append("--wandb")
    if args.wandb_project != DEFAULT_WANDB_PROJECT:
        command.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_group:
        command.extend(["--wandb-group", args.wandb_group])
    if args.wandb_mode:
        command.extend(["--wandb-mode", args.wandb_mode])
    if args.wandb_tags:
        command.extend(["--wandb-tags", args.wandb_tags])
    subprocess.run(command, check=True)


def main() -> None:
    """Run Stream.FM benchmark on local or Modal hardware."""
    parser = argparse.ArgumentParser(description="Unified Stream.FM benchmark launcher.")
    _add_common_args(parser)
    args = parser.parse_args()
    args.input_audio_path = _resolve_input_audio_path(args)
    args.iterations = _resolve_iterations(args)

    backend = "local" if args.local else args.backend
    hardware = args.hardware
    if not hardware:
        hardware = "auto" if backend == "local" else "L4"

    if backend == "local":
        _run_local(args, hardware)
    else:
        _run_modal(args, hardware)


if __name__ == "__main__":
    main()
