"""Modal entrypoints that run the benchmark on remote GPUs.

Ships the input audio to a Modal function, runs the benchmark there on the
requested hardware, and brings the results and profiles back. Invoked via
``modal run``, not imported.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable

import modal

REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

from experiments.core.paths import make_benchmark_paths
from experiments.benchmarks.results import DEFAULT_WANDB_PROJECT, record_benchmark_results
from experiments.core.modal_cache import configure_shared_modal_cache


CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_repo_root() -> Path:
    """Find the local repo root before Modal copies files into the image."""
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current_file.parent


LOCAL_ROOT = _find_repo_root()


image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .apt_install("libsndfile1")
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        # Used only by quality mode, which builds the real dataset through
        # sgmse.data_module -- that module imports lightning/audiomentations/
        # pandas at module scope.  Pins mirror the evaluation image so a quality
        # run reads the splits exactly the way the offline references did.
        "audiomentations==0.41.0",
        "pandas==2.2.3",
        "pytorch-lightning==2.5.1.post0",
        "scipy==1.15.2",
        # Used only by --execution tensorrt.  Pin the known-compatible stack
        # so the ordinary PyTorch benchmark modes retain their current setup.
        "tensorrt==10.9.0.34",
        "torch-tensorrt==2.7.0",
        "requests",
        "nvidia-modelopt[torch]==0.17.0",
    )
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    # Exclude __pycache__/*.pyc: they are rewritten on first import and, if another
    # sweep runs concurrently, Modal aborts the build ("modified during build process").
    .add_local_dir(
        str(LOCAL_ROOT / "experiments"),
        remote_path=f"{REMOTE_ROOT}/experiments",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(LOCAL_ROOT / "flow_autoparams"), remote_path=f"{REMOTE_ROOT}/flow_autoparams")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse", ignore=["**/__pycache__/**", "**/*.pyc"])
)

# Only the small DNN-only exports are baked into the image; full training
# checkpoints come from the shared cache volume instead (see _remote_paths).
for checkpoint_name in (
    "streamfm_stftpr_dnn_only.pt",
    "streamfm_bwe_dnn_only.pt",
    "streamfm_derev_dnn_only.pt",
    "streamfm_lyra_dnn_only.pt",
    "streamfm_se_predictor_dnn_only.pt",
    "streamfm_se_predgen_dnn_only.pt",
    "streamfm_se_predgen_initial_predictor_dnn_only.pt",
):
    local_checkpoint = LOCAL_ROOT / "checkpoints" / checkpoint_name
    if local_checkpoint.exists():
        image = image.add_local_file(
            str(local_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{checkpoint_name}",
        )

# Compressed (decoupled-SVD) checkpoints live in checkpoints/compressed/. Upload
# them flat into the remote checkpoints root so sweeps can reference them by base
# name (e.g. ckpt: streamfm_stftpr_k6.ckpt) and the checkpoint-root search finds them.
_compressed_dir = LOCAL_ROOT / "checkpoints" / "compressed"
if _compressed_dir.is_dir():
    for compressed_checkpoint in sorted(_compressed_dir.glob("*.ckpt")):
        image = image.add_local_file(
            str(compressed_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{compressed_checkpoint.name}",
        )


app = modal.App("streamfm-benchmark", image=image)

DEFAULT_INPUT_AUDIO = "inputs/test_clips/audio_43m28_10s.wav"


# The helpers below mirror their twins in streamfm_benchmark.py; `modal run`
# executes this file as the entrypoint, so it keeps its own copies.
def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "unknown"


def _default_audio_hop_s(task: str) -> float:
    return 0.020 if task.lower().replace("-", "_") == "lyra" else 0.016


def _resolve_input_audio_path(input_audio: str, *, pipeline: str) -> str:
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


def _input_audio_duration_s(input_audio_path: str) -> float:
    import torchaudio

    info = torchaudio.info(input_audio_path)
    if info.sample_rate <= 0:
        raise ValueError(f"Could not determine sample rate for {input_audio_path}.")
    return float(info.num_frames) / float(info.sample_rate)


def _resolve_iterations(iterations: int, *, task: str, pipeline: str, audio_duration_s: float) -> int:
    if pipeline.lower().replace("-", "_") != "audio":
        if audio_duration_s > 0:
            raise ValueError("--audio-duration-s is only supported with --pipeline audio.")
        if iterations == -1:
            raise ValueError("--iterations -1 is only supported with --pipeline audio.")
        return iterations
    if audio_duration_s > 0:
        return max(1, math.ceil(audio_duration_s / _default_audio_hop_s(task)))
    if iterations == -1:
        return max(1, math.ceil(10.0 / _default_audio_hop_s(task)))
    return iterations


def _read_input_audio_bytes(input_audio_path: str) -> tuple[bytes, str]:
    if not input_audio_path:
        return b"", ""
    path = Path(input_audio_path)
    return path.read_bytes(), path.name


def _write_remote_input_audio(input_audio_bytes: bytes, input_audio_name: str) -> str:
    if not input_audio_bytes:
        return ""
    input_path = Path(input_audio_name or "input.wav")
    safe_name = _safe_name(input_path.stem or "input")
    suffix = Path(input_audio_name).suffix or ".wav"
    path = Path("/tmp") / f"streamfm_input_{safe_name}{suffix}"
    path.write_bytes(input_audio_bytes)
    return str(path)


def _default_profile_file(task: str, hardware: str) -> str:
    output_dir = Path("outputs/benchmark_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(
        _safe_name(str(part))
        for part in (
            timestamp,
            hardware,
            task,
        )
    )
    return str(output_dir / f"{stem}.txt")


def _parse_wandb_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _save_audio_results(
    results: list[dict],
    *,
    backend: str,
    hardware: str,
    task: str,
    pipeline: str,
    execution: str,
    steps: str,
    dtype: str,
    output_dir: str,
) -> None:
    """Save returned remote audio locally and remove tensor payloads before JSON history writes."""
    import torch
    import torchaudio

    audio_output_dir = Path(output_dir or "outputs/benchmark_audio")
    audio_output_dir.mkdir(parents=True, exist_ok=True)
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
                row.get("task", task),
                row.get("pipeline", pipeline),
                row.get("execution", execution),
                f"steps{row.get('steps', steps)}",
                dtype,
                row_idx,
            )
        )
        audio_path = audio_output_dir / f"{stem}.wav"
        torchaudio.save(str(audio_path), audio.clamp(-1, 1), sample_rate=sample_rate)
        row["saved_audio_path"] = str(audio_path)
        print(f"Saved benchmark audio to {audio_path}")


def _configure_persistent_cache_env(hardware: str) -> dict[str, str]:
    """Point all runs on one hardware tier at shared compiler caches."""
    return configure_shared_modal_cache(volume_root=VOLUME_ROOT, hardware=hardware)


def _remote_paths():
    """Build benchmark paths inside the Modal container."""
    return make_benchmark_paths(
        repo_root=REMOTE_ROOT,
        config_dir=f"{REMOTE_ROOT}/config",
        # Search order: volume first (large full checkpoints uploaded once),
        # then the image-baked DNN-only exports.
        checkpoint_roots=(
            f"{VOLUME_ROOT}/checkpoints",
            f"{REMOTE_ROOT}/checkpoints",
        ),
    )


def _run_modal_benchmark(
    *,
    hardware: str,
    task: str,
    part: str,
    pipeline: str,
    execution: str,
    steps: str,
    iterations: int,
    warmup: int,
    model_dtype_name: str,
    num_threads: int,
    num_interop_threads: int,
    preallocate_model_buffers: bool,
    model_memory_format: str,
    save_audio: bool,
    input_audio_bytes: bytes,
    input_audio_name: str,
    profile: bool = False,
    profile_file: str = "",
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
    float32_matmul_precision: str = "high",
    tf32_mode: str = "auto",
    cudnn_benchmark: bool = False,
    cudnn_benchmark_limit: int = 10,
    tensorrt_optimization_level: int = 3,
    tensorrt_num_avg_timing_iters: int = 1,
    tensorrt_workspace_size_mib: int = 0,
    tensorrt_engine_cache: str = "off",
    quality: dict | None = None,
) -> list[dict]:
    """Run one benchmark inside Modal on CPU or CUDA."""
    # Keep the PyTorch benchmark runner inside the remote function.  The Modal
    # CLI imports this file locally merely to define the app and mounts; eager
    # importing it here would require local Torch/NumPy even though all actual
    # inference runs in the CUDA container.
    from experiments.benchmarks.quality import QualityRunOptions
    from experiments.benchmarks.runner import run_benchmark

    import torch

    hardware = hardware.upper()
    device = torch.device("cpu" if hardware == "CPU" else "cuda")
    if execution.lower().replace("-", "_") in {"cuda_graph", "cuda_graph_full"} and device.type != "cuda":
        raise ValueError("Modal CPU cannot run execution=cuda_graph.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this Modal container.")

    input_audio_path = _write_remote_input_audio(input_audio_bytes, input_audio_name)
    cache_info = _configure_persistent_cache_env(hardware)

    # A quality run's deliverable is the enhanced audio, so it must be written
    # to the shared volume (container-local disk dies with the container) and
    # committed once the run finishes.
    quality_options = None
    if quality:
        from experiments.evaluation.modal_defaults import resolve_modal_data_path

        quality = dict(quality)
        if not quality.get("output_dir"):
            quality["output_dir"] = f"{VOLUME_ROOT}/outputs/benchmark_quality"
        # The checkpoint config points at the paper authors' local disk; on the
        # volume the same splits live under /data/datasets.
        quality["data_path"] = resolve_modal_data_path(
            quality.get("data_path", ""),
            task=task,
            split=quality["split"],
            volume_root=VOLUME_ROOT,
        )
        quality_options = QualityRunOptions(**quality)

        # An INT8 quality score is a claim about which layers ran quantized, so
        # dump the TensorRT engine inspector next to the audio it produced.
        # Both step counts share one engine, hence the run root rather than a
        # per-steps directory.  Setting this also makes the builder emit
        # DETAILED profiling verbosity (see tensorrt/streaming.py), without
        # which the inspector reports layer names and no precisions.
        if ptq_int8.strip().lower() == "tensorrt":
            # Mirror the run directory quality.py derives, fallback included.
            run_dir = f"{quality['output_dir']}/{quality.get('run_id') or 'run'}"
            os.environ["STREAMFM_TRT_LAYER_INFO_PATH"] = (
                f"{run_dir}/tensorrt_engine_layers.json"
            )

    results = run_benchmark(
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        device=device,
        paths=_remote_paths(),
        backend="modal",
        hardware=hardware,
        cache_info=cache_info,
        float32_matmul_precision=float32_matmul_precision,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_path=input_audio_path,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality_options,
    )
    if quality_options is not None:
        CACHE_VOLUME.commit()

    # For normal benchmark runs the payload is metadata only.  Round-trip it
    # through JSON so a lightweight local Modal CLI need not have PyTorch just
    # to deserialize a result.  Keep audio tensors intact when explicitly
    # requested for local WAV export.
    if not save_audio:
        return json.loads(json.dumps(results))
    return results


# One Modal function per hardware tier: the GPU type must be fixed in the
# decorator at definition time, so the tiers cannot share a single function.
# MODAL_FUNCTIONS below maps the hardware name back to the right one.
@app.function(timeout=1800, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_cpu(task: str, part: str, pipeline: str, execution: str, steps: str, iterations: int, warmup: int, model_dtype_name: str, num_threads: int, num_interop_threads: int, preallocate_model_buffers: bool, model_memory_format: str, save_audio: bool, input_audio_bytes: bytes, input_audio_name: str, profile: bool = False, profile_file: str = "", checkpoint_name: str = "", ptq_int8: str = "", ptq_calib_steps: int = 32, float32_matmul_precision: str = "high", tf32_mode: str = "auto", cudnn_benchmark: bool = False, cudnn_benchmark_limit: int = 10, tensorrt_optimization_level: int = 3, tensorrt_num_avg_timing_iters: int = 1, tensorrt_workspace_size_mib: int = 0, tensorrt_engine_cache: str = "off", quality: dict | None = None):
    """Run the selected benchmark on Modal CPU."""
    return _run_modal_benchmark(
        hardware="CPU",
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality,
    )


@app.function(gpu="T4", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_t4(task: str, part: str, pipeline: str, execution: str, steps: str, iterations: int, warmup: int, model_dtype_name: str, num_threads: int, num_interop_threads: int, preallocate_model_buffers: bool, model_memory_format: str, save_audio: bool, input_audio_bytes: bytes, input_audio_name: str, profile: bool = False, profile_file: str = "", checkpoint_name: str = "", ptq_int8: str = "", ptq_calib_steps: int = 32, float32_matmul_precision: str = "high", tf32_mode: str = "auto", cudnn_benchmark: bool = False, cudnn_benchmark_limit: int = 10, tensorrt_optimization_level: int = 3, tensorrt_num_avg_timing_iters: int = 1, tensorrt_workspace_size_mib: int = 0, tensorrt_engine_cache: str = "off", quality: dict | None = None):
    """Run the selected benchmark on an NVIDIA T4."""
    return _run_modal_benchmark(
        hardware="T4",
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality,
    )


@app.function(gpu="L4", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_l4(task: str, part: str, pipeline: str, execution: str, steps: str, iterations: int, warmup: int, model_dtype_name: str, num_threads: int, num_interop_threads: int, preallocate_model_buffers: bool, model_memory_format: str, save_audio: bool, input_audio_bytes: bytes, input_audio_name: str, profile: bool = False, profile_file: str = "", checkpoint_name: str = "", ptq_int8: str = "", ptq_calib_steps: int = 32, float32_matmul_precision: str = "high", tf32_mode: str = "auto", cudnn_benchmark: bool = False, cudnn_benchmark_limit: int = 10, tensorrt_optimization_level: int = 3, tensorrt_num_avg_timing_iters: int = 1, tensorrt_workspace_size_mib: int = 0, tensorrt_engine_cache: str = "off", quality: dict | None = None):
    """Run the selected benchmark on an NVIDIA L4."""
    return _run_modal_benchmark(
        hardware="L4",
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality,
    )


@app.function(gpu="L40S", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_l40s(task: str, part: str, pipeline: str, execution: str, steps: str, iterations: int, warmup: int, model_dtype_name: str, num_threads: int, num_interop_threads: int, preallocate_model_buffers: bool, model_memory_format: str, save_audio: bool, input_audio_bytes: bytes, input_audio_name: str, profile: bool = False, profile_file: str = "", checkpoint_name: str = "", ptq_int8: str = "", ptq_calib_steps: int = 32, float32_matmul_precision: str = "high", tf32_mode: str = "auto", cudnn_benchmark: bool = False, cudnn_benchmark_limit: int = 10, tensorrt_optimization_level: int = 3, tensorrt_num_avg_timing_iters: int = 1, tensorrt_workspace_size_mib: int = 0, tensorrt_engine_cache: str = "off", quality: dict | None = None):
    """Run the selected benchmark on an NVIDIA L40S."""
    return _run_modal_benchmark(
        hardware="L40S",
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality,
    )


@app.function(gpu="A100", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_a100(task: str, part: str, pipeline: str, execution: str, steps: str, iterations: int, warmup: int, model_dtype_name: str, num_threads: int, num_interop_threads: int, preallocate_model_buffers: bool, model_memory_format: str, save_audio: bool, input_audio_bytes: bytes, input_audio_name: str, profile: bool = False, profile_file: str = "", checkpoint_name: str = "", ptq_int8: str = "", ptq_calib_steps: int = 32, float32_matmul_precision: str = "high", tf32_mode: str = "auto", cudnn_benchmark: bool = False, cudnn_benchmark_limit: int = 10, tensorrt_optimization_level: int = 3, tensorrt_num_avg_timing_iters: int = 1, tensorrt_workspace_size_mib: int = 0, tensorrt_engine_cache: str = "off", quality: dict | None = None):
    """Run the selected benchmark on an NVIDIA A100."""
    return _run_modal_benchmark(
        hardware="A100",
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=model_memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        profile_file=profile_file,
        checkpoint_name=checkpoint_name,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=tensorrt_optimization_level,
        tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        tensorrt_workspace_size_mib=tensorrt_workspace_size_mib,
        tensorrt_engine_cache=tensorrt_engine_cache,
        quality=quality,
    )


MODAL_FUNCTIONS: dict[str, Callable] = {
    "CPU": benchmark_cpu,
    "T4": benchmark_t4,
    "L4": benchmark_l4,
    "L40S": benchmark_l40s,
    "A100": benchmark_a100,
}


def _trial_to_benchmark_call(trial: dict) -> dict:
    """Normalize a sweep trial dict to `_run_modal_benchmark` keyword arguments."""
    steps = trial.get("steps", 1)
    if isinstance(steps, (list, tuple)):
        steps = ",".join(str(step) for step in steps)
    dtype = trial.get("dtype", trial.get("model_dtype", "fp32"))
    return {
        "task": str(trial["task"]),
        "part": str(trial.get("part", "model")),
        "pipeline": str(trial["pipeline"]),
        "execution": str(trial["execution"]),
        "steps": str(steps),
        "iterations": int(trial["iterations"]),
        "warmup": int(trial["warmup"]),
        "model_dtype_name": str(dtype).lower(),
        "num_threads": int(trial.get("num_threads", 0)),
        "num_interop_threads": int(trial.get("num_interop_threads", 0)),
        "preallocate_model_buffers": bool(trial.get("preallocate_model_buffers", False)),
        "model_memory_format": str(trial.get("memory_format", "contiguous")),
        "save_audio": bool(trial.get("save_audio", False)),
        "profile": bool(trial.get("profile", False)),
        "profile_file": str(trial.get("profile_file", "")),
        "checkpoint_name": str(trial.get("ckpt", trial.get("checkpoint", ""))),
        "ptq_int8": str(trial.get("ptq_int8", "")),
        "ptq_calib_steps": int(trial.get("ptq_calib_steps", 32)),
        "float32_matmul_precision": str(trial.get("matmul_precision", "high")),
        "tf32_mode": str(trial.get("tf32", "auto")),
        "cudnn_benchmark": bool(trial.get("cudnn_benchmark", False)),
        "cudnn_benchmark_limit": int(trial.get("cudnn_benchmark_limit", 10)),
        "tensorrt_optimization_level": int(trial.get("trt_optimization_level", 3)),
        "tensorrt_num_avg_timing_iters": int(trial.get("trt_avg_timing_iters", 1)),
        "tensorrt_workspace_size_mib": int(trial.get("trt_workspace_size_mib", 0)),
        "tensorrt_engine_cache": str(trial.get("trt_engine_cache", "off")),
    }


def _run_modal_benchmark_batch(
    *,
    hardware: str,
    trials: list[dict],
    input_audio_bytes: bytes,
    input_audio_name: str,
) -> list[dict]:
    """Run multiple benchmark trials inside one warm Modal container.

    Batching amortizes container boot and compiler warmup across the sweep,
    and keeps every trial on the exact same physical machine.
    """
    outputs: list[dict] = []
    total = len(trials)
    for trial_index, trial in enumerate(trials):
        execution = trial.get("execution", "?")
        dtype = trial.get("dtype", trial.get("model_dtype", "?"))
        memory_format = trial.get("memory_format", "contiguous")
        preallocate = trial.get("preallocate_model_buffers", False)
        steps = trial.get("steps", "?")
        print(
            f"[{trial_index + 1}/{total}] {hardware} "
            f"execution={execution} dtype={dtype} steps={steps} "
            f"memory_format={memory_format} preallocate={preallocate}",
            flush=True,
        )
        benchmark_kwargs = _trial_to_benchmark_call(trial)
        results = _run_modal_benchmark(
            hardware=hardware,
            input_audio_bytes=input_audio_bytes,
            input_audio_name=input_audio_name,
            **benchmark_kwargs,
        )
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


@app.function(timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_cpu_batch(trials: list[dict], input_audio_bytes: bytes, input_audio_name: str):
    """Run multiple benchmarks on Modal CPU inside one container."""
    return _run_modal_benchmark_batch(
        hardware="CPU",
        trials=trials,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
    )


@app.function(gpu="T4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_t4_batch(trials: list[dict], input_audio_bytes: bytes, input_audio_name: str):
    """Run multiple benchmarks on an NVIDIA T4 inside one container."""
    return _run_modal_benchmark_batch(
        hardware="T4",
        trials=trials,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
    )


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_l4_batch(trials: list[dict], input_audio_bytes: bytes, input_audio_name: str):
    """Run multiple benchmarks on an NVIDIA L4 inside one container."""
    return _run_modal_benchmark_batch(
        hardware="L4",
        trials=trials,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
    )


@app.function(gpu="L40S", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_l40s_batch(trials: list[dict], input_audio_bytes: bytes, input_audio_name: str):
    """Run multiple benchmarks on an NVIDIA L40S inside one container."""
    return _run_modal_benchmark_batch(
        hardware="L40S",
        trials=trials,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
    )


@app.function(gpu="A100", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_a100_batch(trials: list[dict], input_audio_bytes: bytes, input_audio_name: str):
    """Run multiple benchmarks on an NVIDIA A100 inside one container."""
    return _run_modal_benchmark_batch(
        hardware="A100",
        trials=trials,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
    )


MODAL_BATCH_FUNCTIONS: dict[str, Callable] = {
    "CPU": benchmark_cpu_batch,
    "T4": benchmark_t4_batch,
    "L4": benchmark_l4_batch,
    "L40S": benchmark_l40s_batch,
    "A100": benchmark_a100_batch,
}


@app.local_entrypoint()
def main(
    hardware: str = "L4",
    task: str = "stftpr",
    part: str = "model",
    pipeline: str = "audio",
    execution: str = "auto",
    steps: str = "1",
    iterations: int = 100,
    warmup: int = 10,
    audio_duration_s: float = 0.0,
    dtype: str = "fp32",
    matmul_precision: str = "high",
    tf32: str = "auto",
    cudnn_benchmark: bool = False,
    cudnn_benchmark_limit: int = 10,
    trt_optimization_level: int = 3,
    trt_avg_timing_iters: int = 1,
    trt_workspace_size_mib: int = 0,
    trt_engine_cache: str = "off",
    ckpt: str = "",
    num_threads: int = 0,
    num_interop_threads: int = 0,
    preallocate_model_buffers: bool = False,
    memory_format: str = "contiguous",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
    output_json: str = "",
    history_json: str = "",
    save_audio: bool = False,
    audio_output_dir: str = "",
    input_audio: str = DEFAULT_INPUT_AUDIO,
    quality_split: str = "",
    quality_limit: int = 0,
    quality_offset: int = 0,
    quality_selection: str = "random",
    quality_selection_seed: int = 42,
    quality_seed: int = 42,
    quality_crop_mode: str = "full",
    quality_output_dir: str = "",
    quality_run_id: str = "",
    quality_data_path: str = "",
    quality_data_format: str = "",
    quality_overwrite: bool = False,
    quality_continue_on_error: bool = False,
    profile: bool = False,
    profile_file: str = "",
    wandb: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = "",
    wandb_group: str = "",
    wandb_mode: str = "",
    wandb_tags: str = "",
):
    """Launch the selected Modal benchmark and record the result locally."""
    selected_hardware = hardware.upper()
    dtype = dtype.lower()
    matmul_precision = matmul_precision.lower().replace("-", "_")
    tf32 = tf32.lower().replace("-", "_")
    if selected_hardware not in MODAL_FUNCTIONS:
        supported = ", ".join(MODAL_FUNCTIONS)
        raise ValueError(f"Unsupported Modal hardware '{selected_hardware}'. Supported values: {supported}")
    if dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported model dtype. Use 'fp32', 'fp16', or 'bf16'.")
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("Unsupported matmul precision. Use 'highest', 'high', or 'medium'.")
    if tf32 not in {"auto", "on", "off"}:
        raise ValueError("Unsupported TF32 mode. Use 'auto', 'on', or 'off'.")
    if not 0 <= trt_optimization_level <= 5:
        raise ValueError("TensorRT optimization level must be between 0 and 5.")
    if trt_avg_timing_iters < 1:
        raise ValueError("TensorRT average timing iterations must be at least 1.")
    if trt_workspace_size_mib < 0:
        raise ValueError("TensorRT workspace size must be non-negative.")
    if trt_engine_cache not in {"off", "read", "write", "readwrite"}:
        raise ValueError(
            "Unsupported engine cache mode. Use 'off', 'read', 'write', or 'readwrite'."
        )
    quality_split = quality_split.strip().lower()
    if quality_split and quality_split not in {"test", "valid", "train"}:
        raise ValueError("Unsupported quality split. Use 'test', 'valid', or 'train'.")
    if quality_split and quality_selection not in {"random", "first", "sequential"}:
        raise ValueError("Unsupported quality selection. Use 'random', 'first', or 'sequential'.")

    if quality_split:
        # Quality mode takes its audio from the dataset split and derives the
        # frame count per file, so the single-clip options are not consulted.
        input_audio_path = ""
        iterations = max(iterations, 1)
    else:
        input_audio_path = _resolve_input_audio_path(input_audio, pipeline=pipeline)
        if iterations == -1 and input_audio_path and audio_duration_s <= 0:
            audio_duration_s = _input_audio_duration_s(input_audio_path)
        iterations = _resolve_iterations(
            iterations,
            task=task,
            pipeline=pipeline,
            audio_duration_s=audio_duration_s,
        )
    input_audio_bytes, input_audio_name = _read_input_audio_bytes(input_audio_path)

    # Sent as a plain dict: the remote side rebuilds the dataclass, so the
    # Modal function signatures stay short and serialization stays trivial.
    quality = (
        {
            "split": quality_split,
            "data_path": quality_data_path,
            "data_format": quality_data_format,
            "limit": quality_limit,
            "offset": quality_offset,
            "selection": quality_selection,
            "selection_seed": quality_selection_seed,
            "seed": quality_seed,
            "crop_mode": quality_crop_mode,
            "output_dir": quality_output_dir,
            "run_id": quality_run_id,
            "overwrite": quality_overwrite,
            "continue_on_error": quality_continue_on_error,
        }
        if quality_split
        else None
    )

    results = MODAL_FUNCTIONS[selected_hardware].remote(
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=dtype,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        preallocate_model_buffers=preallocate_model_buffers,
        model_memory_format=memory_format,
        save_audio=save_audio,
        input_audio_bytes=input_audio_bytes,
        input_audio_name=input_audio_name,
        profile=profile,
        # No remote profile path: the summary comes back inside the result row
        # and is written to a LOCAL file below.
        profile_file="",
        checkpoint_name=ckpt,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        float32_matmul_precision=matmul_precision,
        tf32_mode=tf32,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_benchmark_limit=cudnn_benchmark_limit,
        tensorrt_optimization_level=trt_optimization_level,
        tensorrt_num_avg_timing_iters=trt_avg_timing_iters,
        tensorrt_workspace_size_mib=trt_workspace_size_mib,
        tensorrt_engine_cache=trt_engine_cache,
        quality=quality,
    )

    if profile and profile_file and results:
        profile_summary = results[0].get("profile_summary", "")
        if profile_summary:
            local_profile_path = Path(profile_file)
            local_profile_path.parent.mkdir(parents=True, exist_ok=True)
            local_profile_path.write_text(profile_summary, encoding="utf-8")
            print(f"Saved profiler summary to {local_profile_path}")

    if save_audio:
        _save_audio_results(
            results,
            backend="modal",
            hardware=selected_hardware,
            task=task,
            pipeline=pipeline,
            execution=execution,
            steps=steps,
            dtype=dtype,
            output_dir=audio_output_dir,
        )

    record_benchmark_results(
        results=results,
        output_json=output_json,
        history_json=history_json,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_mode=wandb_mode,
        wandb_tags=_parse_wandb_tags(wandb_tags),
        command={
            "backend": "modal",
            "hardware": selected_hardware,
            "task": task,
            "part": part,
            "pipeline": pipeline,
            "execution": execution,
            "steps": steps,
            "iterations": iterations,
            "warmup": warmup,
            "audio_duration_s": audio_duration_s,
            "model_dtype": dtype,
            "matmul_precision": matmul_precision,
            "tf32": tf32,
            "cudnn_benchmark": cudnn_benchmark,
            "cudnn_benchmark_limit": cudnn_benchmark_limit,
            "trt_optimization_level": trt_optimization_level,
            "trt_avg_timing_iters": trt_avg_timing_iters,
            "trt_workspace_size_mib": trt_workspace_size_mib,
            "trt_engine_cache": trt_engine_cache,
            "ckpt": ckpt,
            "num_threads": num_threads,
            "num_interop_threads": num_interop_threads,
            "memory_format": memory_format,
            "preallocate_model_buffers": preallocate_model_buffers,
            "ptq_int8": ptq_int8,
            "ptq_calib_steps": ptq_calib_steps,
            "save_audio": save_audio,
            "audio_output_dir": audio_output_dir,
            "input_audio": input_audio_path,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_group": wandb_group,
            "wandb_mode": wandb_mode,
            "wandb_tags": wandb_tags,
        },
    )
