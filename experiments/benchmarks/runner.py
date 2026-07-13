from __future__ import annotations

import json
import os
import time

from experiments.common import (
    apply_model_memory_format,
    device_label,
    normalize_float32_matmul_precision,
    normalize_model_memory_format,
)
from experiments.benchmarks.cuda_graph import (
    benchmark_flow_steps_cuda_graph,
    benchmark_se_flow_cuda_graph,
    benchmark_se_full_cuda_graph,
    benchmark_se_predictor_cuda_graph,
)
from experiments.benchmarks.loading import (
    load_flow_model,
    load_se_flow,
    load_se_full,
    load_se_predictor,
)
from experiments.benchmarks.model_loops import (
    benchmark_flow_steps,
    benchmark_se_flow,
    benchmark_se_full,
    benchmark_se_predictor,
)
from experiments.benchmarks.options import (
    normalize_cli_options,
    parse_model_dtype,
    parse_steps,
    resolve_execution,
)
from experiments.benchmarks.paths import BenchmarkPaths


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_ptq_int8_if_requested(
    model,
    *,
    ptq_int8: str,
    ptq_calib_steps: int,
    device,
    use_compiled: bool,
    model_dtype=None,
):
    """Optionally apply INT8 PTQ. Returns (model, ptq_meta_or_none, model_dtype)."""
    from sgmse.util.ptq_int8 import apply_ptq_int8_, describe_ptq, parse_ptq_components
    import torch

    components = parse_ptq_components(ptq_int8)
    if not components:
        return model, None, model_dtype
    if device.type != "cpu":
        raise ValueError(
            "PTQ INT8 currently requires a CPU device (PyTorch quantized kernels). "
            "Use --hardware cpu / --backend local with hardware=cpu."
        )
    if use_compiled:
        raise ValueError(
            "PTQ INT8 requires --execution eager "
            "(torch.compile / cuda_graph are incompatible with quantized modules)."
        )
    model = apply_ptq_int8_(model, components, calib_steps=ptq_calib_steps)
    # Quantized modules expect float activations; keep benchmark tensors in fp32.
    if model_dtype is not None and model_dtype != torch.float32:
        print(
            f"Warning: overriding benchmark model_dtype {model_dtype} → float32 after PTQ "
            f"(quantized modules use float I/O).",
            flush=True,
        )
        model_dtype = torch.float32
    print(f"Applied PTQ INT8 components={list(components)} calib_steps={ptq_calib_steps}", flush=True)
    return model, describe_ptq(model), model_dtype


def _streaming_config_from_model_cfg(cfg):
    """Create a synthetic streaming STFT config matching a Stream.FM model config."""
    from experiments.streaming.pipeline import StreamingSTFTConfig

    feature_cfg = cfg.model.feature_extractor
    return StreamingSTFTConfig(
        sample_rate=int(cfg.get("sampling_rate", 16000)),
        n_fft=int(feature_cfg.get("n_fft", 512)),
        hop_length=int(feature_cfg.get("hop_length", 256)),
        alpha=float(feature_cfg.get("alpha", 0.5)),
        beta=float(feature_cfg.get("beta", 1.0)),
        cut_highest_freqs=int(feature_cfg.get("cut_highest_freqs", 1)),
        sigma_y=_float_or_default(cfg.model.get("sigma_y", 0.25), 0.25),
        normalized_stft=bool(feature_cfg.get("normalized_stft", True)),
    )


def _frame_budget_ms_from_config(config) -> float:
    return 1000.0 * float(config.hop_length) / float(config.sample_rate)


def _load_input_audio(input_audio_path: str, *, config, device):
    """Load an optional real audio file for audio-pipeline benchmarks."""
    if not input_audio_path:
        return None

    import torch
    import torchaudio

    audio, sample_rate = torchaudio.load(input_audio_path)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shaped [channels, samples], got {tuple(audio.shape)}.")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != config.sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, config.sample_rate)
        sample_rate = config.sample_rate
    audio = audio.to(device=device, dtype=torch.float32)
    return audio


def _benchmark_flow_task(
    *,
    task: str,
    internal_pipeline: str,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    model_dtype,
    device,
    paths: BenchmarkPaths,
    preallocate_model_buffers: bool,
    model_memory_format: str,
    save_audio: bool,
    input_audio_path: str,
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> tuple[list[dict], float, float]:
    """Load and benchmark a flow-only task."""
    load_started_at = time.perf_counter()
    model, cfg = load_flow_model(device, model_dtype, paths, task=task, checkpoint_name=checkpoint_name or None)
    model, _ptq_meta, model_dtype = _apply_ptq_int8_if_requested(
        model,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    model = apply_model_memory_format(model, model_memory_format)
    config = _streaming_config_from_model_cfg(cfg)
    freq_bins = int(getattr(model, "input_freqs", config.n_fft // 2 + 1 - config.cut_highest_freqs))
    frame_budget_ms = _frame_budget_ms_from_config(config)
    load_s = time.perf_counter() - load_started_at
    bench_started_at = time.perf_counter()

    if internal_pipeline == "model":
        results = benchmark_flow_steps(
            model,
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            freq_bins=freq_bins,
            frame_budget_ms=frame_budget_ms,
        )
    elif internal_pipeline == "graph_model":
        results = benchmark_flow_steps_cuda_graph(
            model,
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            model_memory_format=model_memory_format,
            freq_bins=freq_bins,
            frame_budget_ms=frame_budget_ms,
        )
    elif internal_pipeline == "audio":
        from experiments.streaming.pipeline import StreamingSTFTConfig, make_synthetic_audio, run_streaming_audio_pipeline

        audio = _load_input_audio(input_audio_path, config=config, device=device)
        if audio is None:
            audio = make_synthetic_audio(
                num_samples=(warmup + iterations) * config.hop_length,
                sample_rate=config.sample_rate,
                device=device,
            )
        results = []
        for step_count in steps_list:
            summary = run_streaming_audio_pipeline(
                model,
                audio,
                device=device,
                steps=step_count,
                iterations=iterations,
                warmup=warmup,
                use_compiled=use_compiled,
                config=config,
                model_dtype=model_dtype,
                preallocate_model_buffers=preallocate_model_buffers,
                model_memory_format=model_memory_format,
                return_audio=save_audio,
            )
            summary["task"] = task
            summary["pipeline"] = "audio"
            summary["device"] = device.type
            if input_audio_path:
                summary["input_audio_path"] = input_audio_path
            results.append(summary)
    elif internal_pipeline == "audio_graph_model":
        from experiments.streaming.pipeline import (
            make_synthetic_audio,
            run_streaming_audio_pipeline_with_cuda_graph_model,
        )

        audio = _load_input_audio(input_audio_path, config=config, device=device)
        if audio is None:
            audio = make_synthetic_audio(
                num_samples=(warmup + iterations) * config.hop_length,
                sample_rate=config.sample_rate,
                device=device,
            )
        results = []
        for step_count in steps_list:
            summary = run_streaming_audio_pipeline_with_cuda_graph_model(
                model,
                audio,
                device=device,
                steps=step_count,
                iterations=iterations,
                warmup=warmup,
                use_compiled=use_compiled,
                config=config,
                model_dtype=model_dtype,
                model_memory_format=model_memory_format,
                return_audio=save_audio,
            )
            summary["task"] = task
            summary["pipeline"] = "audio_graph_model"
            summary["device"] = device.type
            if input_audio_path:
                summary["input_audio_path"] = input_audio_path
            results.append(summary)
    else:
        raise ValueError("Unsupported flow-task pipeline.")

    for row in results:
        row["task"] = task
        if _ptq_meta is not None:
            row["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            row["ptq_engine"] = _ptq_meta.get("engine", "")
            row["ptq_calib_steps"] = _ptq_meta.get("calib_steps", ptq_calib_steps)

    return results, load_s, time.perf_counter() - bench_started_at


def _benchmark_se_predictor_task(
    *,
    internal_pipeline: str,
    iterations: int,
    warmup: int,
    use_compiled: bool,
    model_dtype,
    device,
    paths: BenchmarkPaths,
    model_memory_format: str,
    save_audio: bool,
    input_audio_path: str,
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> tuple[list[dict], float, float]:
    """Load and benchmark the SE predictor-only task."""
    load_started_at = time.perf_counter()
    predictor, _ = load_se_predictor(device, dtype=model_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
    predictor, _ptq_meta, model_dtype = _apply_ptq_int8_if_requested(
        predictor,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    predictor = apply_model_memory_format(predictor, model_memory_format)
    load_s = time.perf_counter() - load_started_at
    bench_started_at = time.perf_counter()

    if internal_pipeline == "graph_model":
        results = benchmark_se_predictor_cuda_graph(
            predictor,
            device,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            model_memory_format=model_memory_format,
        )
    elif internal_pipeline == "model":
        results = benchmark_se_predictor(
            predictor,
            device,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            model_memory_format=model_memory_format,
        )
    else:
        raise ValueError("SE predictor supports only model_only pipeline.")

    if _ptq_meta is not None:
        for row in results:
            row["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            row["ptq_engine"] = _ptq_meta.get("engine", "")
            row["ptq_calib_steps"] = _ptq_meta.get("calib_steps", ptq_calib_steps)

    return results, load_s, time.perf_counter() - bench_started_at


def _benchmark_se_flow_task(
    *,
    internal_pipeline: str,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    model_dtype,
    device,
    paths: BenchmarkPaths,
    preallocate_model_buffers: bool,
    model_memory_format: str,
    save_audio: bool,
    input_audio_path: str,
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> tuple[list[dict], float, float]:
    """Load and benchmark the SE flow-only task."""
    load_started_at = time.perf_counter()
    flow, flow_cfg = load_se_flow(device, dtype=model_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
    flow, _ptq_meta, model_dtype = _apply_ptq_int8_if_requested(
        flow,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    flow = apply_model_memory_format(flow, model_memory_format)
    sigma_e = float(flow_cfg.model.sigma_e)
    load_s = time.perf_counter() - load_started_at
    bench_started_at = time.perf_counter()

    if internal_pipeline == "graph_model":
        results = benchmark_se_flow_cuda_graph(
            flow,
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            sigma_e=sigma_e,
            model_memory_format=model_memory_format,
        )
    elif internal_pipeline == "model":
        results = benchmark_se_flow(
            flow,
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            sigma_e=sigma_e,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
        )
    else:
        raise ValueError("SE flow supports only model_only pipeline.")

    if _ptq_meta is not None:
        for row in results:
            row["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            row["ptq_engine"] = _ptq_meta.get("engine", "")
            row["ptq_calib_steps"] = _ptq_meta.get("calib_steps", ptq_calib_steps)

    return results, load_s, time.perf_counter() - bench_started_at


def _benchmark_se_full_task(
    *,
    internal_pipeline: str,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    model_dtype,
    device,
    paths: BenchmarkPaths,
    preallocate_model_buffers: bool,
    model_memory_format: str,
    save_audio: bool,
    input_audio_path: str,
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> tuple[list[dict], float, float]:
    """Load and benchmark the full SE task."""
    load_started_at = time.perf_counter()
    se_model = load_se_full(device, dtype=model_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
    se_model["predictor"], _ptq_meta_pred, model_dtype = _apply_ptq_int8_if_requested(
        se_model["predictor"],
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    se_model["flow"], _ptq_meta_flow, model_dtype = _apply_ptq_int8_if_requested(
        se_model["flow"],
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    _ptq_meta = _ptq_meta_flow or _ptq_meta_pred
    se_model["predictor"] = apply_model_memory_format(se_model["predictor"], model_memory_format)
    se_model["flow"] = apply_model_memory_format(se_model["flow"], model_memory_format)
    load_s = time.perf_counter() - load_started_at
    bench_started_at = time.perf_counter()

    if internal_pipeline == "graph_model":
        results = benchmark_se_full_cuda_graph(
            se_model["predictor"],
            se_model["flow"],
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            sigma_e=se_model["sigma_e"],
            model_memory_format=model_memory_format,
        )
    elif internal_pipeline == "model":
        results = benchmark_se_full(
            se_model["predictor"],
            se_model["flow"],
            device,
            steps_list,
            iterations,
            warmup,
            use_compiled,
            model_dtype,
            sigma_e=se_model["sigma_e"],
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
        )
    elif internal_pipeline == "audio":
        from experiments.streaming.pipeline import StreamingSTFTConfig, make_synthetic_audio, run_streaming_se_audio_pipeline

        config = StreamingSTFTConfig()
        audio = _load_input_audio(input_audio_path, config=config, device=device)
        if audio is None:
            audio = make_synthetic_audio(
                num_samples=(warmup + iterations) * config.hop_length,
                sample_rate=config.sample_rate,
                device=device,
            )
        results = []
        for step_count in steps_list:
            summary = run_streaming_se_audio_pipeline(
                se_model["predictor"],
                se_model["flow"],
                audio,
                device=device,
                steps=step_count,
                iterations=iterations,
                warmup=warmup,
                use_compiled=use_compiled,
                config=config,
                sigma_e=se_model["sigma_e"],
                model_dtype=model_dtype,
                preallocate_model_buffers=preallocate_model_buffers,
                model_memory_format=model_memory_format,
                return_audio=save_audio,
            )
            summary["task"] = "se_full"
            summary["pipeline"] = "audio"
            summary["device"] = device.type
            if input_audio_path:
                summary["input_audio_path"] = input_audio_path
            results.append(summary)
    elif internal_pipeline == "audio_graph_model":
        from experiments.streaming.pipeline import (
            StreamingSTFTConfig,
            make_synthetic_audio,
            run_streaming_se_audio_pipeline_with_cuda_graph_model,
        )

        config = StreamingSTFTConfig()
        audio = _load_input_audio(input_audio_path, config=config, device=device)
        if audio is None:
            audio = make_synthetic_audio(
                num_samples=(warmup + iterations) * config.hop_length,
                sample_rate=config.sample_rate,
                device=device,
            )
        results = []
        for step_count in steps_list:
            summary = run_streaming_se_audio_pipeline_with_cuda_graph_model(
                se_model["predictor"],
                se_model["flow"],
                audio,
                device=device,
                steps=step_count,
                iterations=iterations,
                warmup=warmup,
                use_compiled=use_compiled,
                config=config,
                sigma_e=se_model["sigma_e"],
                model_dtype=model_dtype,
                model_memory_format=model_memory_format,
                return_audio=save_audio,
            )
            summary["task"] = "se_full"
            summary["pipeline"] = "audio_graph_model"
            summary["device"] = device.type
            if input_audio_path:
                summary["input_audio_path"] = input_audio_path
            results.append(summary)
    else:
        raise ValueError("Unsupported SE full pipeline.")

    if _ptq_meta is not None:
        for row in results:
            row["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            row["ptq_engine"] = _ptq_meta.get("engine", "")
            row["ptq_calib_steps"] = _ptq_meta.get("calib_steps", ptq_calib_steps)

    return results, load_s, time.perf_counter() - bench_started_at


def run_internal_benchmark(
    *,
    internal_task: str,
    internal_pipeline: str,
    steps: str,
    iterations: int,
    warmup: int,
    use_compiled: bool,
    model_dtype_name: str,
    device,
    paths: BenchmarkPaths,
    preallocate_model_buffers: bool,
    model_memory_format: str,
    save_audio: bool = False,
    input_audio_path: str = "",
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> tuple[list[dict], float, float]:
    """Dispatch a benchmark run for an already-normalized task/pipeline."""
    model_dtype = parse_model_dtype(model_dtype_name)
    steps_list = parse_steps(steps)

    if internal_task in {"stftpr", "bwe", "derev", "lyra"}:
        return _benchmark_flow_task(
            task=internal_task,
            internal_pipeline=internal_pipeline,
            steps_list=steps_list,
            iterations=iterations,
            warmup=warmup,
            use_compiled=use_compiled,
            model_dtype=model_dtype,
            device=device,
            paths=paths,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            save_audio=save_audio,
            input_audio_path=input_audio_path,
            checkpoint_name=checkpoint_name,
            ptq_int8=ptq_int8,
            ptq_calib_steps=ptq_calib_steps,
        )
    if internal_task == "se_predictor":
        return _benchmark_se_predictor_task(
            internal_pipeline=internal_pipeline,
            iterations=iterations,
            warmup=warmup,
            use_compiled=use_compiled,
            model_dtype=model_dtype,
            device=device,
            paths=paths,
            model_memory_format=model_memory_format,
            save_audio=save_audio,
            input_audio_path=input_audio_path,
            checkpoint_name=checkpoint_name,
            ptq_int8=ptq_int8,
            ptq_calib_steps=ptq_calib_steps,
        )
    if internal_task == "se_flow":
        return _benchmark_se_flow_task(
            internal_pipeline=internal_pipeline,
            steps_list=steps_list,
            iterations=iterations,
            warmup=warmup,
            use_compiled=use_compiled,
            model_dtype=model_dtype,
            device=device,
            paths=paths,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            save_audio=save_audio,
            input_audio_path=input_audio_path,
            checkpoint_name=checkpoint_name,
            ptq_int8=ptq_int8,
            ptq_calib_steps=ptq_calib_steps,
        )
    if internal_task == "se_full":
        return _benchmark_se_full_task(
            internal_pipeline=internal_pipeline,
            steps_list=steps_list,
            iterations=iterations,
            warmup=warmup,
            use_compiled=use_compiled,
            model_dtype=model_dtype,
            device=device,
            paths=paths,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            save_audio=save_audio,
            input_audio_path=input_audio_path,
            checkpoint_name=checkpoint_name,
            ptq_int8=ptq_int8,
            ptq_calib_steps=ptq_calib_steps,
        )
    raise ValueError("Unsupported internal task.")


def run_benchmark(
    *,
    task: str,
    part: str,
    pipeline: str,
    execution: str,
    steps: str,
    iterations: int,
    warmup: int,
    model_dtype_name: str,
    device,
    paths: BenchmarkPaths,
    backend: str,
    hardware: str,
    cache_info: dict | None = None,
    float32_matmul_precision: str = "high",
    num_threads: int = 0,
    num_interop_threads: int = 0,
    preallocate_model_buffers: bool = False,
    model_memory_format: str = "contiguous",
    save_audio: bool = False,
    input_audio_path: str = "",
    profile: bool = False,
    profile_all: bool = False,
    profile_file: str = "",
    checkpoint_name: str = "",
    ptq_int8: str = "",
    ptq_calib_steps: int = 32,
) -> list[dict]:
    """Run one benchmark using the common local/Modal benchmark implementation."""
    import torch

    requested_execution = execution.lower().replace("-", "_")
    execution = resolve_execution(execution, device)
    model_memory_format = normalize_model_memory_format(model_memory_format)
    float32_matmul_precision = normalize_float32_matmul_precision(float32_matmul_precision)

    if execution == "cuda_graph" and device.type != "cuda":
        raise ValueError("execution=cuda_graph requires a CUDA device.")

    started_at = time.perf_counter()
    os.chdir(paths.repo_root)
    torch.set_float32_matmul_precision(float32_matmul_precision)
    if device.type == "cpu":
        # Interop threads can only be set once per process, and only before parallel work.
        # Batch sweeps reuse the same Modal/local process across trials, so ignore repeat sets.
        if num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(num_interop_threads)
            except RuntimeError as exc:
                current = torch.get_num_interop_threads()
                if current != num_interop_threads:
                    print(
                        f"Warning: could not set interop threads to {num_interop_threads} "
                        f"(current={current}): {exc}"
                    )
        if num_threads > 0:
            torch.set_num_threads(num_threads)

    resolved = normalize_cli_options(
        task=task,
        part=part,
        pipeline=pipeline,
        execution=execution,
    )
    if save_audio and resolved["internal_pipeline"] not in {"audio", "audio_graph_model"}:
        raise ValueError("--save-audio requires --pipeline audio.")
    if input_audio_path and resolved["internal_pipeline"] not in {"audio", "audio_graph_model"}:
        raise ValueError("--input-audio requires --pipeline audio.")
    
    def _benchmark_call() -> tuple[list[dict], float, float]:
        return run_internal_benchmark(
            internal_task=resolved["internal_task"],
            internal_pipeline=resolved["internal_pipeline"],
            steps=steps,
            iterations=iterations,
            warmup=warmup,
            use_compiled=resolved["use_compiled"],
            model_dtype_name=model_dtype_name,
            device=device,
            paths=paths,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            save_audio=save_audio,
            input_audio_path=input_audio_path,
            checkpoint_name=checkpoint_name,
            ptq_int8=ptq_int8,
            ptq_calib_steps=ptq_calib_steps,
        )

    profiler_output = ""
    profile_target = profile_all or profile
    if profile_target:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
        ) as profiler:
            results, load_s, benchmark_s = _benchmark_call()
        profiler_output = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
        print("\nPyTorch profiler summary:")
        print(profiler_output)
    else:
        results, load_s, benchmark_s = _benchmark_call()

    if profile_target and profile_file:
        os.makedirs(os.path.dirname(profile_file) or ".", exist_ok=True)
        with open(profile_file, "w", encoding="utf-8") as f:
            f.write(profiler_output or "")
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - started_at
    hardware_name = device_label(device)

    for row in results:
        row["backend"] = backend
        row["hardware"] = hardware
        row["device"] = device.type
        row["gpu_name"] = hardware_name if device.type == "cuda" else ""
        row["torch_version"] = torch.__version__
        row["model_load_s"] = load_s
        row["benchmark_s"] = benchmark_s
        row["total_s"] = total_s
        row["requested_model_dtype"] = model_dtype_name.lower()
        row["float32_matmul_precision"] = float32_matmul_precision
        row["requested_task"] = resolved["requested_task"]
        row["requested_part"] = resolved["requested_part"]
        row["requested_pipeline"] = resolved["requested_pipeline"]
        row["requested_execution"] = requested_execution
        row["execution"] = resolved["execution"]
        row["internal_task"] = resolved["internal_task"]
        row["internal_pipeline"] = resolved["internal_pipeline"]
        row["preallocate_model_buffers"] = preallocate_model_buffers
        row["model_memory_format"] = model_memory_format
        row["profile"] = profile
        row["profile_file"] = profile_file
        row["profile_summary"] = profiler_output if profile else ""
        row["checkpoint_name"] = checkpoint_name
        if input_audio_path:
            row["input_audio_path"] = input_audio_path
        if device.type == "cpu":
            row["num_threads"] = torch.get_num_threads()
            row["num_interop_threads"] = torch.get_num_interop_threads()
        if cache_info:
            row.update(cache_info)

    printable_results = [
        {key: value for key, value in row.items() if key != "audio"}
        for row in results
    ]
    print(json.dumps(printable_results, indent=2))
    return results
