"""In-process benchmark driver.

Resolves the streaming config, optionally applies INT8 PTQ, loads the input
audio and dispatches to the eager or CUDA Graph loops for the requested
task/part/execution. ``run_benchmark`` is the entry point shared by the local
CLI and the Modal functions.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from experiments.core.tensors import apply_model_memory_format, normalize_model_memory_format
from experiments.core.devices import (
    device_label,
    normalize_float32_matmul_precision,
    normalize_tf32_mode,
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
from experiments.core.options import (
    normalize_cli_options,
    parse_model_dtype,
    parse_steps,
    resolve_execution,
)
from experiments.core.paths import BenchmarkPaths


_DEFAULT_CUDNN_ALLOW_TF32: bool | None = None


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
    fallback_dtype = model_dtype
    if fallback_dtype not in {torch.float16, torch.float32}:
        print(
            f"Warning: CPU INT8 PTQ supports FP32 or FP16 fallback; overriding "
            f"{fallback_dtype} → float32.",
            flush=True,
        )
        fallback_dtype = torch.float32
    model = apply_ptq_int8_(
        model,
        components,
        calib_steps=ptq_calib_steps,
        fallback_dtype=fallback_dtype,
    )
    model_dtype = fallback_dtype
    print(
        f"Applied PTQ INT8 components={list(components)} "
        f"calib_steps={ptq_calib_steps} fallback={str(fallback_dtype).replace('torch.', '')}",
        flush=True,
    )
    return model, describe_ptq(model), model_dtype


def _ptq_calibration_load_dtype(ptq_int8: str, model_dtype):
    """Load pristine FP32 weights when CPU PTQ calibration is requested."""
    from sgmse.util.ptq_int8 import parse_ptq_components
    import torch

    return torch.float32 if parse_ptq_components(ptq_int8) else model_dtype


def _streaming_config_from_model_cfg(cfg):
    """Build the streaming STFT config from the model's Hydra config.

    Shared with the evaluation driver: a quality run and a latency run must
    frame the audio identically, or their numbers describe different pipelines.
    """
    from experiments.streaming.stft import streaming_config_from_model_cfg

    return streaming_config_from_model_cfg(cfg)


def _frame_budget_ms_from_config(config) -> float:
    # Real-time budget per frame: the time one hop of audio lasts
    # (256/16000 = 16 ms). budget_ratio_mean = mean_ms / this; < 1 = real-time.
    return 1000.0 * float(config.hop_length) / float(config.sample_rate)


def _load_input_audio(input_audio_path: str, *, config, device):
    """Load an optional real audio file for audio-pipeline benchmarks.

    Returns None when no path was given or the file is absent, in which case
    the caller falls back to synthetic audio. No clip ships with the repo, so a
    fresh checkout takes that path by default.
    """
    if not input_audio_path or not Path(input_audio_path).is_file():
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


def _resolve_streaming_pipeline(
    *,
    internal_pipeline: str,
    use_tensorrt: bool,
    tensorrt_cuda_graph: bool,
    use_compiled: bool,
    model_dtype,
    model_memory_format: str,
    preallocate_model_buffers: bool,
):
    """Pick the streaming pipeline for an execution mode and bind its options.

    Quality mode reuses the very pipelines the latency benchmark times, so the
    layers that run INT8 (and those that fall back) are the same ones the
    reported speed describes.  Returns (callable, kwargs, name); the caller adds
    the per-file arguments.
    """
    from experiments.streaming.pipeline import (
        run_streaming_audio_pipeline,
        run_streaming_audio_pipeline_with_cuda_graph_model,
        run_streaming_audio_pipeline_with_full_cuda_graph,
        run_streaming_audio_pipeline_with_tensorrt_cuda_graph,
    )

    shared = {"model_dtype": model_dtype, "model_memory_format": model_memory_format}
    if use_tensorrt and tensorrt_cuda_graph:
        # This variant drives the engine itself and takes no use_compiled.
        return run_streaming_audio_pipeline_with_tensorrt_cuda_graph, shared, "tensorrt_cuda_graph"
    if internal_pipeline == "audio_full_graph":
        return (
            run_streaming_audio_pipeline_with_full_cuda_graph,
            {**shared, "use_compiled": use_compiled},
            "audio_full_graph",
        )
    if internal_pipeline == "audio_graph_model":
        return (
            run_streaming_audio_pipeline_with_cuda_graph_model,
            {**shared, "use_compiled": use_compiled},
            "audio_graph_model",
        )
    return (
        run_streaming_audio_pipeline,
        {
            **shared,
            "use_compiled": use_compiled,
            "preallocate_model_buffers": preallocate_model_buffers,
        },
        "audio",
    )


def _benchmark_flow_task(
    *,
    task: str,
    internal_pipeline: str,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    use_tensorrt: bool,
    tensorrt_precision: str,
    tensorrt_cuda_graph: bool,
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
    tensorrt_allow_tf32: bool | None = None,
    tensorrt_optimization_level: int = 3,
    tensorrt_num_avg_timing_iters: int = 1,
    tensorrt_workspace_size_bytes: int = 0,
    tensorrt_engine_cache: str = "off",
    tensorrt_engine_cache_dir: str = "",
    quality=None,
    backbone_transform=None,
) -> tuple[list[dict], float, float]:
    """Load and benchmark a flow-only task (stftpr/bwe/derev/lyra).

    Returns (result rows, load_s, benchmark_s) — model loading is timed
    separately so it never pollutes the per-frame numbers.
    """
    # 1) Load the backbone, then stack the requested transformations:
    # PTQ INT8 (CPU) -> memory format -> TensorRT adapter (CUDA).
    load_started_at = time.perf_counter()
    # ModelOpt calibration is deliberately performed in FP32.  For an INT8
    # run, the user-facing --dtype selects the floating-point fallback that
    # TensorRT may use internally; the engine I/O remains FP32 so Q/DQ scales
    # and recurrent-state export keep the already validated contract.
    int8_fallback_dtype = model_dtype
    if use_tensorrt and tensorrt_precision == "int8":
        import torch

        model_dtype = torch.float32
    load_dtype = _ptq_calibration_load_dtype("" if use_tensorrt else ptq_int8, model_dtype)
    model, cfg = load_flow_model(
        device,
        load_dtype,
        paths,
        task=task,
        checkpoint_name=checkpoint_name or None,
        backbone_transform=backbone_transform,
    )
    model, _ptq_meta, model_dtype = _apply_ptq_int8_if_requested(
        model,
        # TensorRT INT8 uses ModelOpt Q/DQ below; do not apply the separate
        # CPU-native PTQ wrappers first.
        ptq_int8="" if use_tensorrt else ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    model = apply_model_memory_format(model, model_memory_format)
    trt_validation = None
    if use_tensorrt:
        from experiments.benchmarks.tensorrt.streaming import build_tensorrt_streaming_adapter

        model = build_tensorrt_streaming_adapter(
            model,
            dtype=model_dtype,
            precision=tensorrt_precision,
            calibration_steps=ptq_calib_steps,
            use_cuda_graph=tensorrt_cuda_graph,
            memory_format=model_memory_format,
            int8_fallback_dtype=int8_fallback_dtype,
            allow_tf32=tensorrt_allow_tf32,
            optimization_level=tensorrt_optimization_level,
            num_avg_timing_iters=tensorrt_num_avg_timing_iters,
            workspace_size_bytes=tensorrt_workspace_size_bytes,
            engine_cache=tensorrt_engine_cache,
            engine_cache_dir=tensorrt_engine_cache_dir or None,
        )
        trt_validation = model.validation
        trt_runtime_profile = model.runtime_profile
        trt_stage_profile = model.stage_profile
    else:
        trt_runtime_profile = None
        trt_stage_profile = None
    config = _streaming_config_from_model_cfg(cfg)
    freq_bins = int(getattr(model, "input_freqs", config.n_fft // 2 + 1 - config.cut_highest_freqs))
    frame_budget_ms = _frame_budget_ms_from_config(config)
    load_s = time.perf_counter() - load_started_at
    bench_started_at = time.perf_counter()

    # 2) Dispatch to the matching timing loop:
    #    model            -> random frames, eager/compiled (or TensorRT+graph)
    #    graph_model      -> random frames, CUDA Graph replay
    #    audio            -> real STFT pipeline over real/synthetic audio
    #    audio_graph_model-> same pipeline with the solver as one CUDA Graph
    #    audio_full_graph -> STFT + solver + ISTFT captured as one CUDA Graph
    #
    # Quality mode short-circuits all of these: it runs the same streaming
    # pipeline, but over a test-set split instead of one buffer, and writes
    # audio plus a scorer manifest rather than timing statistics.  It sits after
    # the model/engine construction above so one TensorRT build and one graph
    # capture serve every file in the split.
    if quality is not None:
        from experiments.benchmarks.quality import run_streaming_quality_sweep

        if internal_pipeline not in {"audio", "audio_graph_model", "audio_full_graph"}:
            raise ValueError(
                "Quality runs reconstruct waveforms; use '--pipeline audio'."
            )
        pipeline_fn, pipeline_kwargs, pipeline_name = _resolve_streaming_pipeline(
            internal_pipeline=internal_pipeline,
            use_tensorrt=use_tensorrt,
            tensorrt_cuda_graph=tensorrt_cuda_graph,
            use_compiled=use_compiled,
            model_dtype=model_dtype,
            model_memory_format=model_memory_format,
            preallocate_model_buffers=preallocate_model_buffers,
        )
        # The history row gets this too, but the manifest is what stays next to
        # the WAVs on the volume: a score read months later must be able to say
        # which engine produced it and which layers actually ran quantized,
        # without depending on a separate history file still being around.
        quality_manifest = {
            "task": task,
            "execution_pipeline": pipeline_name,
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "model_memory_format": model_memory_format,
            "device": device_label(device),
        }
        if trt_validation is not None:
            quality_manifest["tensorrt"] = {
                **trt_validation,
                "compilation_profile": getattr(model, "compilation_profile", {}),
            }
            if tensorrt_precision == "int8":
                quality_manifest["ptq_int8"] = "tensorrt"
                quality_manifest["ptq_calib_steps"] = ptq_calib_steps
                quality_manifest["ptq_fallback_dtype"] = str(
                    int8_fallback_dtype
                ).replace("torch.", "")
        if _ptq_meta is not None:
            quality_manifest["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            quality_manifest["ptq_engine"] = _ptq_meta.get("engine", "")
            quality_manifest["ptq_calib_steps"] = _ptq_meta.get(
                "calib_steps", ptq_calib_steps
            )
            quality_manifest["ptq_fallback_dtype"] = _ptq_meta.get(
                "fallback_dtype", "float32"
            )

        results = run_streaming_quality_sweep(
            pipeline_fn=pipeline_fn,
            pipeline_kwargs=pipeline_kwargs,
            model=model,
            cfg=cfg,
            config=config,
            device=device,
            steps_list=steps_list,
            options=quality,
            extra_manifest=quality_manifest,
        )
        for row in results:
            row["pipeline"] = pipeline_name
            row["device"] = device.type
    elif use_tensorrt and tensorrt_cuda_graph and internal_pipeline == "model":
        from experiments.benchmarks.tensorrt.streaming import (
            benchmark_tensorrt_flow_steps_cuda_graph,
        )

        results = benchmark_tensorrt_flow_steps_cuda_graph(
            model,
            device,
            steps_list,
            iterations,
            warmup,
            model_dtype,
            model_memory_format=model_memory_format,
            freq_bins=freq_bins,
            frame_budget_ms=frame_budget_ms,
        )
    elif internal_pipeline == "model":
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
        from experiments.streaming.pipeline import (
            StreamingSTFTConfig,
            make_synthetic_audio,
            run_streaming_audio_pipeline,
            run_streaming_audio_pipeline_with_tensorrt_cuda_graph,
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
            if use_tensorrt and tensorrt_cuda_graph:
                summary = run_streaming_audio_pipeline_with_tensorrt_cuda_graph(
                    model,
                    audio,
                    device=device,
                    steps=step_count,
                    iterations=iterations,
                    warmup=warmup,
                    config=config,
                    model_dtype=model_dtype,
                    model_memory_format=model_memory_format,
                    return_audio=save_audio,
                )
            else:
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
    elif internal_pipeline == "audio_full_graph":
        from experiments.streaming.pipeline import (
            make_synthetic_audio,
            run_streaming_audio_pipeline_with_full_cuda_graph,
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
            summary = run_streaming_audio_pipeline_with_full_cuda_graph(
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
            summary["pipeline"] = "audio_full_graph"
            summary["device"] = device.type
            if input_audio_path:
                summary["input_audio_path"] = input_audio_path
            results.append(summary)
    else:
        raise ValueError("Unsupported flow-task pipeline.")

    # 3) Stamp TensorRT/PTQ metadata onto every row so a result line is
    # self-describing in the history file.
    for row in results:
        row["task"] = task
        if trt_validation is not None:
            row["tensorrt_streaming"] = True
            row.update({f"tensorrt_{key}": value for key, value in trt_validation.items()})
            row.update({f"tensorrt_{key}": value for key, value in trt_runtime_profile.items()})
            row.update({f"tensorrt_{key}": value for key, value in trt_stage_profile.items()})
            row.update(
                {
                    f"tensorrt_compilation_{key}": value
                    for key, value in getattr(model, "compilation_profile", {}).items()
                }
            )
            if tensorrt_precision == "int8":
                row["ptq_int8"] = "tensorrt"
                row["ptq_calib_steps"] = ptq_calib_steps
                row["ptq_fallback_dtype"] = str(int8_fallback_dtype).replace("torch.", "")
        if _ptq_meta is not None:
            row["ptq_int8"] = ",".join(_ptq_meta.get("components", []))
            row["ptq_engine"] = _ptq_meta.get("engine", "")
            row["ptq_calib_steps"] = _ptq_meta.get("calib_steps", ptq_calib_steps)
            row["ptq_fallback_dtype"] = _ptq_meta.get("fallback_dtype", "float32")

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
    """Load the SE predictor and time it alone: one DNN call per frame, no flow."""
    load_started_at = time.perf_counter()
    load_dtype = _ptq_calibration_load_dtype(ptq_int8, model_dtype)
    predictor, _ = load_se_predictor(device, dtype=load_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
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
            row["ptq_fallback_dtype"] = _ptq_meta.get("fallback_dtype", "float32")

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
    """Load the SE flow backbone and time the Euler solver without the predictor."""
    load_started_at = time.perf_counter()
    load_dtype = _ptq_calibration_load_dtype(ptq_int8, model_dtype)
    flow, flow_cfg = load_se_flow(device, dtype=load_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
    flow, _ptq_meta, model_dtype = _apply_ptq_int8_if_requested(
        flow,
        ptq_int8=ptq_int8,
        ptq_calib_steps=ptq_calib_steps,
        device=device,
        use_compiled=use_compiled,
        model_dtype=model_dtype,
    )
    flow = apply_model_memory_format(flow, model_memory_format)
    # Noise scale for x_0 = e + sigma_e * noise, read from the checkpoint's
    # own config so the benchmark input distribution matches training.
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
            row["ptq_fallback_dtype"] = _ptq_meta.get("fallback_dtype", "float32")

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
    """Load and benchmark the full SE chain: predictor + flow, 1 + steps DNN calls per frame."""
    load_started_at = time.perf_counter()
    load_dtype = _ptq_calibration_load_dtype(ptq_int8, model_dtype)
    se_model = load_se_full(device, dtype=load_dtype, paths=paths, checkpoint_name=checkpoint_name or None)
    # PTQ must cover both DNNs: quantizing only one would leave an
    # int8/fp32 boundary in the middle of the per-frame chain.
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
            row["ptq_fallback_dtype"] = _ptq_meta.get("fallback_dtype", "float32")

    return results, load_s, time.perf_counter() - bench_started_at


def run_internal_benchmark(
    *,
    internal_task: str,
    internal_pipeline: str,
    steps: str,
    iterations: int,
    warmup: int,
    use_compiled: bool,
    use_tensorrt: bool = False,
    tensorrt_precision: str = "fp16",
    tensorrt_cuda_graph: bool = False,
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
    tensorrt_allow_tf32: bool | None = None,
    tensorrt_optimization_level: int = 3,
    tensorrt_num_avg_timing_iters: int = 1,
    tensorrt_workspace_size_bytes: int = 0,
    tensorrt_engine_cache: str = "off",
    tensorrt_engine_cache_dir: str = "",
    quality=None,
    backbone_transform=None,
) -> tuple[list[dict], float, float]:
    """Dispatch a benchmark run by task family: flow backbones vs the three SE variants.

    Expects the already-normalized names from normalize_cli_options; returns
    (result rows, load_s, benchmark_s) from the selected _benchmark_*_task.
    """
    model_dtype = parse_model_dtype(model_dtype_name)
    steps_list = parse_steps(steps)

    if use_tensorrt and internal_task not in {"stftpr", "bwe", "derev", "lyra"}:
        raise ValueError(
            "execution=tensorrt is currently integrated for the causal flow backbones only, "
            "not the SE predictor/full pipeline."
        )

    if quality is not None and internal_task not in {"stftpr", "bwe", "derev", "lyra"}:
        raise ValueError(
            "Quality runs are implemented for the causal flow backbones only, "
            "not the SE predictor/full pipeline."
        )

    if internal_task in {"stftpr", "bwe", "derev", "lyra"}:
        return _benchmark_flow_task(
            task=internal_task,
            internal_pipeline=internal_pipeline,
            backbone_transform=backbone_transform,
            steps_list=steps_list,
            iterations=iterations,
            warmup=warmup,
            use_compiled=use_compiled,
            use_tensorrt=use_tensorrt,
            tensorrt_precision=tensorrt_precision,
            tensorrt_cuda_graph=tensorrt_cuda_graph,
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
            tensorrt_allow_tf32=tensorrt_allow_tf32,
            tensorrt_optimization_level=tensorrt_optimization_level,
            tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
            tensorrt_workspace_size_bytes=tensorrt_workspace_size_bytes,
            tensorrt_engine_cache=tensorrt_engine_cache,
            tensorrt_engine_cache_dir=tensorrt_engine_cache_dir,
            quality=quality,
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
    tf32_mode: str = "auto",
    cudnn_benchmark: bool = False,
    cudnn_benchmark_limit: int = 10,
    tensorrt_optimization_level: int = 3,
    tensorrt_num_avg_timing_iters: int = 1,
    tensorrt_workspace_size_mib: int = 0,
    tensorrt_engine_cache: str = "off",
    tensorrt_engine_cache_dir: str = "",
    quality=None,
    backbone_transform=None,
) -> list[dict]:
    """Run one benchmark end to end; the single entry point shared by local CLI and Modal.

    Validates option combinations, prepares the process (cwd, matmul
    precision, CPU threads), runs the timing loops (optionally under the
    PyTorch profiler) and returns the result rows with run metadata stamped
    on. Printing/JSON output happens here; persistence is the caller's job.
    """
    import torch

    # 1) Resolve "auto" values and reject invalid execution/device/dtype
    # combinations before any expensive model loading.
    requested_execution = execution.lower().replace("-", "_")
    execution = resolve_execution(execution, device)
    model_memory_format = normalize_model_memory_format(model_memory_format)
    float32_matmul_precision = normalize_float32_matmul_precision(float32_matmul_precision)
    tf32_mode = normalize_tf32_mode(tf32_mode)

    if execution in {"cuda_graph", "tensorrt", "tensorrt_cuda_graph"} and device.type != "cuda":
        raise ValueError(f"execution={execution} requires a CUDA device.")
    is_tensorrt = execution in {"tensorrt", "tensorrt_cuda_graph"}
    if cudnn_benchmark and device.type != "cuda":
        raise ValueError("--cudnn-benchmark requires a CUDA device.")
    if cudnn_benchmark and is_tensorrt:
        raise ValueError(
            "--cudnn-benchmark tunes PyTorch cuDNN convolutions and does not apply "
            "to TensorRT engines."
        )
    if cudnn_benchmark_limit < 0:
        raise ValueError("--cudnn-benchmark-limit must be non-negative (0 means exhaustive).")
    if not cudnn_benchmark and cudnn_benchmark_limit != 10:
        raise ValueError("--cudnn-benchmark-limit only applies with --cudnn-benchmark.")
    trt_int8 = is_tensorrt and ptq_int8.strip().lower() == "tensorrt"
    if is_tensorrt and not trt_int8 and model_dtype_name.lower() not in {"fp16", "fp32"}:
        raise ValueError("TensorRT requires --dtype fp16 or --dtype fp32.")
    if trt_int8 and model_dtype_name.lower() not in {"fp32", "fp16"}:
        raise ValueError(
            "TensorRT INT8 PTQ supports --dtype fp32 or fp16; the dtype selects "
            "the floating-point fallback while calibration and engine I/O stay FP32."
        )
    if not 0 <= tensorrt_optimization_level <= 5:
        raise ValueError("TensorRT optimization level must be between 0 and 5.")
    if tensorrt_num_avg_timing_iters < 1:
        raise ValueError("TensorRT average timing iterations must be at least 1.")
    if tensorrt_workspace_size_mib < 0:
        raise ValueError("TensorRT workspace size must be non-negative.")
    tensorrt_workspace_size_bytes = tensorrt_workspace_size_mib * 1024 * 1024
    # Reject an unusable engine-cache request up front rather than after the
    # model has been loaded and calibrated.
    from experiments.benchmarks.tensorrt.engine_cache import EngineCache

    EngineCache(tensorrt_engine_cache, tensorrt_engine_cache_dir or None)

    # 2) Process-level setup. chdir to the repo root because checkpoint/config
    # loading uses relative paths; matmul precision trades fp32 accuracy for
    # TF32 speed on CUDA.
    started_at = time.perf_counter()
    os.chdir(paths.repo_root)
    torch.set_float32_matmul_precision(float32_matmul_precision)
    global _DEFAULT_CUDNN_ALLOW_TF32
    if device.type == "cuda":
        # This must be set before the first fixed-shape convolution. cuDNN then
        # benchmarks eligible execution plans on first use; benchmark warm-up
        # absorbs that one-time search before measured frames and graph capture.
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.benchmark_limit = int(cudnn_benchmark_limit)
        if _DEFAULT_CUDNN_ALLOW_TF32 is None:
            _DEFAULT_CUDNN_ALLOW_TF32 = bool(torch.backends.cudnn.allow_tf32)
        torch.backends.cudnn.allow_tf32 = (
            _DEFAULT_CUDNN_ALLOW_TF32 if tf32_mode == "auto" else tf32_mode == "on"
        )
    effective_cudnn_tf32 = (
        bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None
    )
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
    if save_audio and resolved["internal_pipeline"] not in {"audio", "audio_graph_model", "audio_full_graph"}:
        raise ValueError("--save-audio requires --pipeline audio.")
    if input_audio_path and resolved["internal_pipeline"] not in {"audio", "audio_graph_model", "audio_full_graph"}:
        raise ValueError("--input-audio requires --pipeline audio.")
    if quality is not None and input_audio_path:
        raise ValueError(
            "--quality-split reads its audio from the dataset split; drop --input-audio."
        )

    # Closure so the exact same call can run bare or wrapped in the profiler
    # below without duplicating this argument list.
    def _benchmark_call() -> tuple[list[dict], float, float]:
        return run_internal_benchmark(
            internal_task=resolved["internal_task"],
            internal_pipeline=resolved["internal_pipeline"],
            steps=steps,
            iterations=iterations,
            warmup=warmup,
            use_compiled=resolved["use_compiled"],
            use_tensorrt=resolved["use_tensorrt"],
            tensorrt_precision=(
                "int8"
                if trt_int8
                else ("fp32" if model_dtype_name.lower() == "fp32" else "fp16")
            ),
            tensorrt_cuda_graph=resolved["tensorrt_cuda_graph"],
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
            tensorrt_allow_tf32=(None if tf32_mode == "auto" else tf32_mode == "on"),
            tensorrt_optimization_level=tensorrt_optimization_level,
            tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
            tensorrt_workspace_size_bytes=tensorrt_workspace_size_bytes,
            tensorrt_engine_cache=tensorrt_engine_cache,
            tensorrt_engine_cache_dir=tensorrt_engine_cache_dir,
            quality=quality,
            backbone_transform=backbone_transform,
        )

    # 3) Run the timing loops, optionally under the PyTorch profiler. The
    # profiler wraps the WHOLE benchmark (model load + warmup + timed loop),
    # so its op table is for spotting hot operators, not per-frame numbers;
    # instrumentation overhead also inflates the ms stats of a profiled run.
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
        # key_averages() aggregates by op type across the whole run; sorting by
        # self CPU time surfaces launch/dispatch overhead, which dominates for
        # a model this small.
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
        torch.cuda.synchronize()  # flush queued GPU work so total_s covers real compute
    total_s = time.perf_counter() - started_at
    hardware_name = device_label(device)

    # 4) Stamp run-level metadata onto every row: requested vs resolved
    # options, environment versions and wall-clock phases, so a history line
    # can be interpreted long after the run without the CLI invocation.
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
        row["tf32_mode"] = tf32_mode
        row["cudnn_allow_tf32"] = effective_cudnn_tf32
        row["cudnn_benchmark"] = bool(cudnn_benchmark) if device.type == "cuda" else None
        row["cudnn_benchmark_limit"] = (
            int(cudnn_benchmark_limit) if device.type == "cuda" and cudnn_benchmark else None
        )
        row["inductor_compile_mode"] = (
            "max-autotune-no-cudagraphs" if resolved["use_compiled"] else None
        )
        row["tensorrt_optimization_level"] = tensorrt_optimization_level if is_tensorrt else None
        row["tensorrt_num_avg_timing_iters"] = tensorrt_num_avg_timing_iters if is_tensorrt else None
        row["tensorrt_workspace_size_mib"] = tensorrt_workspace_size_mib if is_tensorrt else None
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
