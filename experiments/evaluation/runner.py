"""In-process test-set inference driver.

Composes the Hydra config, instantiates and checkpoints the model, sets up the
requested split, applies the execution mode and enhances each selected file.
``run_test_set_inference`` is the entry point shared by the CLI and Modal.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from experiments.core.options import parse_model_dtype, resolve_execution
from experiments.core.paths import BenchmarkPaths, checkpoint_path
from experiments.core.tensors import apply_model_memory_format, normalize_model_memory_format
from experiments.core.devices import device_label, normalize_float32_matmul_precision
from experiments.core.repo import ensure_repo_importable
from experiments.benchmarks.tensorrt.engine_cache import normalize_cache_mode
from experiments.evaluation.options import (
    normalize_part,
    normalize_pipeline,
    normalize_split,
    normalize_task,
    parse_solver_and_steps,
    resolve_config_and_checkpoint,
)
from sgmse.util.model_compression import apply_checkpoint_compression_


def _compose_config(config_name: str, paths: BenchmarkPaths, overrides: list[str]):
    """Compose a Hydra config from the local repository config directory."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(paths.config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides)


def _instantiate_model(cfg):
    """Instantiate the configured model, including optional solver wrappers."""
    from hydra.utils import instantiate

    if hasattr(cfg, "solver_model"):
        wrapped_model = instantiate(cfg.model)
        return instantiate(cfg.solver_model, wrapped_model=wrapped_model)
    return instantiate(cfg.model)


def _resolve_checkpoint(ckpt: str, paths: BenchmarkPaths) -> str:
    """Resolve an absolute checkpoint path or search configured roots."""
    ckpt_path = Path(ckpt)
    if ckpt_path.is_absolute() or ckpt_path.parent != Path("."):
        if ckpt_path.exists():
            return str(ckpt_path)
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return checkpoint_path(ckpt, paths)


def _setup_split(data_module, split: str):
    """Prepare a DataModule split and return the underlying Dataset."""
    if split == "test":
        data_module.setup(stage="test")
        return data_module.test_set
    if split == "valid":
        data_module.setup(stage="fit")
        return data_module.valid_set
    if split == "train":
        data_module.setup(stage="_train_only")
        return data_module.train_set
    raise ValueError(f"Unsupported split: {split}")


def _override_data_module_path(cfg, split: str, data_path: str) -> None:
    """Override the selected split path in-place when requested."""
    if not data_path:
        return
    attr = f"{split}_path"
    if not hasattr(cfg.model, "data_module"):
        raise ValueError("Config has no model.data_module to override.")
    setattr(cfg.model.data_module, attr, data_path)


def _override_data_format(cfg, data_format: str) -> None:
    """Override data_module.format in-place when requested."""
    if not data_format:
        return
    if not hasattr(cfg.model, "data_module"):
        raise ValueError("Config has no model.data_module to override.")
    cfg.model.data_module.format = data_format


def _apply_execution(model, execution: str):
    """Apply optional PyTorch compilation for offline model inference."""
    import torch

    if execution == "eager":
        return model
    if execution == "cuda_graph":
        raise ValueError("execution=cuda_graph is not implemented for offline test-set inference.")
    if execution == "compiled":
        if hasattr(model, "dnn"):
            model.dnn = torch.compile(model.dnn)
        if hasattr(model, "initial_predictor") and hasattr(model.initial_predictor, "dnn"):
            model.initial_predictor.dnn = torch.compile(model.initial_predictor.dnn)
        if hasattr(model, "wrapped_model") and hasattr(model.wrapped_model, "dnn"):
            model.wrapped_model.dnn = torch.compile(model.wrapped_model.dnn)
        return model
    raise ValueError(f"Unsupported execution: {execution}")


def _streaming_backbone(model):
    """Find the causal DNN backbone inside an instantiated evaluation model.

    Deliberately taken from the model this run already loaded rather than
    re-loaded from the DNN-only export: the streaming numbers must describe the
    same weights as the offline numbers they will be compared against.
    """
    for holder in (model, getattr(model, "wrapped_model", None)):
        backbone = getattr(holder, "dnn", None)
        if backbone is not None:
            return backbone
    raise ValueError("Model exposes no 'dnn' backbone; --pipeline streaming needs one.")


def _prepare_streaming_flow(
    *,
    model,
    execution: str,
    model_dtype,
    model_memory_format: str,
    tensorrt_precision: str,
    tensorrt_calibration_steps: int,
    tensorrt_allow_tf32: bool | None,
    tensorrt_optimization_level: int,
    tensorrt_num_avg_timing_iters: int,
    tensorrt_workspace_size_bytes: int,
    tensorrt_engine_cache: str,
    tensorrt_engine_cache_dir: str,
):
    """Return (flow, use_engine, model_dtype, info) for the streaming pipeline.

    Streaming does not use autocast: the deployed engine has one fixed I/O
    dtype, so the backbone is cast the way the benchmark casts it, and the
    quality number then describes the precision that actually ships.
    """
    import torch

    from experiments.core.tensors import apply_model_memory_format

    flow = _streaming_backbone(model)
    info: dict = {"tensorrt_precision": "", "tensorrt_engine_cache": "off"}

    if execution != "tensorrt":
        flow = apply_model_memory_format(flow.to(dtype=model_dtype), model_memory_format)
        return flow, False, model_dtype, info

    # Same convention as the latency benchmark: ModelOpt calibrates in FP32 and
    # the engine keeps FP32 I/O, so for INT8 the requested dtype selects the
    # floating-point fallback TensorRT may use internally, not the I/O dtype.
    int8_fallback_dtype = model_dtype
    if tensorrt_precision == "int8":
        model_dtype = torch.float32

    from experiments.benchmarks.tensorrt.streaming import build_tensorrt_streaming_adapter

    flow = apply_model_memory_format(flow.to(dtype=model_dtype), model_memory_format)
    adapter = build_tensorrt_streaming_adapter(
        flow,
        dtype=model_dtype,
        precision=tensorrt_precision,
        calibration_steps=tensorrt_calibration_steps,
        use_cuda_graph=False,
        memory_format=model_memory_format,
        int8_fallback_dtype=int8_fallback_dtype,
        allow_tf32=tensorrt_allow_tf32,
        optimization_level=tensorrt_optimization_level,
        num_avg_timing_iters=tensorrt_num_avg_timing_iters,
        workspace_size_bytes=tensorrt_workspace_size_bytes,
        engine_cache=tensorrt_engine_cache,
        engine_cache_dir=tensorrt_engine_cache_dir or None,
    )
    info = {
        "tensorrt_precision": tensorrt_precision,
        "tensorrt_engine_cache": tensorrt_engine_cache,
        "tensorrt_int8_fallback_dtype": str(int8_fallback_dtype).replace("torch.", ""),
        "tensorrt_validation": getattr(adapter, "validation", None),
        # The precision partition is the whole point of scoring INT8: record
        # which layers ran quantized so the metric can be read as evidence
        # about this exact engine rather than about "INT8" in the abstract.
        "tensorrt_compilation_profile": getattr(adapter, "compilation_profile", None),
    }
    return adapter, True, model_dtype, info


def _enhance_one(
    *,
    model,
    y,
    sr: int,
    part: str,
    task: str,
    solver: str,
    steps: int,
    seed: int,
):
    """Run one official offline enhancement call."""
    import torch
    from sgmse.model import DiscriminativeModel

    if part == "predictor":
        if task != "se" or not hasattr(model, "initial_predictor"):
            raise ValueError("--part predictor is only available for task=se.")
        return model.initial_predictor.enhance(y, sr)

    if isinstance(model, DiscriminativeModel):
        return model.enhance(y, sr)

    try:
        return model.enhance(y, sr, solver=solver, N=steps, seed=seed, return_all=False)
    except TypeError:
        torch.manual_seed(seed)
        return model.enhance(y, sr, solver=solver, N=steps, return_all=False)


def _safe_stem(path: str, fallback: str) -> str:
    stem = Path(path).stem if path else fallback
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem) or fallback


def select_eval_indices(
    *,
    num_available: int,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
) -> list[int]:
    """Select dataset indices for an evaluation run."""
    if num_available < 0:
        raise ValueError("num_available must be non-negative.")
    start_index = max(offset, 0)
    if start_index >= num_available:
        return []

    candidates = list(range(start_index, num_available))
    if limit > 0:
        sample_size = min(limit, len(candidates))
    else:
        sample_size = len(candidates)

    selection = selection.lower().replace("-", "_")
    if selection in {"first", "sequential"}:
        return candidates[:sample_size]
    if selection == "random":
        if sample_size == len(candidates):
            return candidates
        rng = np.random.default_rng(selection_seed)
        return sorted(int(idx) for idx in rng.choice(candidates, size=sample_size, replace=False))
    raise ValueError("--selection must be 'first' or 'random'.")


def run_test_set_inference(
    *,
    task: str,
    config_name: str,
    ckpt: str,
    split: str,
    data_path: str,
    data_format: str,
    part: str,
    pipeline: str,
    execution: str,
    solver: str,
    steps: int,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    seed: int,
    model_dtype_name: str,
    model_memory_format: str,
    crop_mode: str,
    device,
    paths: BenchmarkPaths,
    backend: str,
    hardware: str,
    output_dir: str,
    run_name: str,
    overwrite: bool,
    save_inputs: bool,
    continue_on_error: bool,
    num_threads: int = 0,
    num_interop_threads: int = 0,
    cache_info: dict | None = None,
    config_overrides: list[str] | tuple[str, ...] = (),
    float32_matmul_precision: str = "high",
    tensorrt_precision: str = "fp16",
    tensorrt_calibration_steps: int = 32,
    tensorrt_allow_tf32: bool | None = None,
    tensorrt_optimization_level: int = 3,
    tensorrt_num_avg_timing_iters: int = 1,
    tensorrt_workspace_size_bytes: int = 0,
    tensorrt_engine_cache: str = "off",
    tensorrt_engine_cache_dir: str = "",
) -> dict:
    """Run official offline inference on a configured dataset split."""
    import torch
    import torchaudio

    started_at = time.perf_counter()
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = run_name or uuid4().hex[:12]

    task = normalize_task(task)
    split = normalize_split(split)
    part = normalize_part(part)
    pipeline = normalize_pipeline(pipeline)
    if pipeline == "streaming" and execution.lower().strip() == "auto":
        # 'auto' resolves to cuda_graph on CUDA, but graph capture is a latency
        # concern; the quality driver runs one file frame by frame.
        execution = "eager"
    execution = resolve_execution(execution, device)
    if pipeline == "streaming":
        if execution not in {"eager", "compiled", "tensorrt"}:
            raise ValueError(
                "Streaming evaluation supports --execution eager, compiled or tensorrt."
            )
    elif execution == "tensorrt":
        raise ValueError("--execution tensorrt requires --pipeline streaming.")
    if execution == "tensorrt":
        tensorrt_precision = tensorrt_precision.lower().strip()
        if tensorrt_precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("--trt-precision must be 'fp32', 'fp16' or 'int8'.")
        tensorrt_engine_cache = normalize_cache_mode(tensorrt_engine_cache)
    config_name, ckpt = resolve_config_and_checkpoint(
        task=task,
        config_name=config_name,
        checkpoint_name=ckpt,
    )
    solver, steps = parse_solver_and_steps(solver, steps)
    model_dtype = parse_model_dtype(model_dtype_name)
    if model_dtype == torch.bfloat16:
        # The DNN backbone runs complex-valued ops (torch.view_as_complex in the
        # NCSN++ U-Net), which have no bfloat16 CUDA kernel, so the full inference
        # pipeline cannot run in bf16. fp16 works (complex-half is supported) and
        # fp32 is the reference. Fail fast instead of burning a GPU job.
        raise ValueError(
            "bf16 is not supported for full-pipeline evaluation: the backbone uses "
            "torch.view_as_complex, which has no bfloat16 kernel. Use 'fp16' or 'fp32'."
        )
    model_memory_format = normalize_model_memory_format(model_memory_format)
    float32_matmul_precision = normalize_float32_matmul_precision(float32_matmul_precision)
    crop_mode = crop_mode.lower().replace("-", "_")
    if crop_mode not in {"config", "full"}:
        raise ValueError("--crop-mode must be 'config' or 'full'.")

    if device.type == "cpu" and num_threads > 0:
        torch.set_num_threads(num_threads)
    if device.type == "cpu" and num_interop_threads > 0:
        torch.set_num_interop_threads(num_interop_threads)

    os.chdir(paths.repo_root)
    ensure_repo_importable(paths.repo_root)
    torch.set_float32_matmul_precision(float32_matmul_precision)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config_overrides = [str(override) for override in config_overrides]
    cfg = _compose_config(config_name, paths, overrides=config_overrides)
    _override_data_module_path(cfg, split, data_path)
    _override_data_format(cfg, data_format)

    model = _instantiate_model(cfg)
    ckpt_path = _resolve_checkpoint(ckpt, paths)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # A compressed checkpoint describes its changed backbone architecture in metadata.
    # Build that architecture before loading its incompatible state_dict shapes.
    model = apply_checkpoint_compression_(model, checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(device=device)
    # For fp16/bf16 we use autocast instead of casting the weights: the STFT/iSTFT
    # front-end runs through cuFFT and torch.polar, neither of which has a CUDA
    # half-precision kernel, so a whole-model cast crashes ("cuFFT doesn't support
    # BFloat16" / "polar_cuda not implemented for 'Half'"). Autocast keeps those
    # ops in fp32 while running the DNN backbone (matmul/conv) in reduced precision,
    # which is the meaningful precision comparison here.
    # Streaming runs the backbone at a fixed dtype instead (see
    # _prepare_streaming_flow): a TensorRT engine has one I/O dtype and no
    # autocast, so autocasting here would score a pipeline nobody deploys.
    use_autocast = model_dtype != torch.float32 and pipeline == "offline"
    model = apply_model_memory_format(model, model_memory_format)
    if pipeline == "offline":
        model = _apply_execution(model, execution)

    streaming_info: dict = {}
    streaming_flow = None
    streaming_use_engine = False
    streaming_config = None
    if pipeline == "streaming":
        from experiments.streaming.enhance import streaming_algorithmic_delay, streaming_enhance
        from experiments.streaming.stft import streaming_config_from_model_cfg

        if solver != "euler":
            raise ValueError(
                f"Streaming evaluation implements the Euler solver only, got '{solver}'."
            )
        if part != "model":
            raise ValueError("--part predictor has no streaming evaluation path.")
        streaming_config = streaming_config_from_model_cfg(cfg)
        streaming_flow, streaming_use_engine, model_dtype, streaming_info = _prepare_streaming_flow(
            model=model,
            execution=execution,
            model_dtype=model_dtype,
            model_memory_format=model_memory_format,
            tensorrt_precision=tensorrt_precision,
            tensorrt_calibration_steps=tensorrt_calibration_steps,
            tensorrt_allow_tf32=tensorrt_allow_tf32,
            tensorrt_optimization_level=tensorrt_optimization_level,
            tensorrt_num_avg_timing_iters=tensorrt_num_avg_timing_iters,
            tensorrt_workspace_size_bytes=tensorrt_workspace_size_bytes,
            tensorrt_engine_cache=tensorrt_engine_cache,
            tensorrt_engine_cache_dir=tensorrt_engine_cache_dir,
        )

    if not hasattr(model, "data_module") or model.data_module is None:
        raise ValueError("Model has no data_module; cannot run split-based evaluation.")
    dataset = _setup_split(model.data_module, split)

    selection = selection.lower().replace("-", "_")
    indices = select_eval_indices(
        num_available=len(dataset),
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
    )

    base_output_dir = Path(output_dir) if output_dir else paths.repo_root / "outputs" / "eval_runs"
    run_dir = base_output_dir / run_id
    enhanced_dir = run_dir / "enhanced"
    clean_dir = run_dir / "clean"
    noisy_dir = run_dir / "noisy"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    if save_inputs:
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    try:
        from omegaconf import OmegaConf

        config_path.write_text(OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")
    except Exception as exc:
        config_path.write_text(f"# Failed to serialize Hydra config: {exc}\n", encoding="utf-8")

    files = []
    errors = []
    with torch.inference_mode():
        for idx in indices:
            item_started_at = time.perf_counter()
            try:
                x, y, info = dataset.__getitem__(idx, no_crop=(crop_mode == "full"))
                sr = int(info["sr"])
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                if y.shape[0] > 1:
                    y = y[0:1]
                if x.shape[0] > 1:
                    x = x[0:1]

                file_seed = seed + idx
                np.random.seed(file_seed)
                torch.manual_seed(file_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(file_seed)

                # Keep the input (and weights) in fp32; autocast handles the
                # reduced-precision compute so the STFT front-end stays in fp32.
                y_model = y.to(device=device)
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=model_dtype)
                    if use_autocast
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    if pipeline == "streaming":
                        x_hat = streaming_enhance(
                            streaming_flow,
                            y_model,
                            device=device,
                            steps=steps,
                            config=streaming_config,
                            seed=file_seed,
                            model_dtype=model_dtype,
                            model_memory_format=model_memory_format,
                            use_compiled=execution == "compiled",
                            use_engine=streaming_use_engine,
                        )
                    else:
                        x_hat = _enhance_one(
                            model=model,
                            y=y_model,
                            sr=sr,
                            part=part,
                            task=task,
                            solver=solver,
                            steps=steps,
                            seed=file_seed,
                        )
                if x_hat.ndim == 1:
                    x_hat = x_hat.unsqueeze(0)
                x_hat = x_hat.detach().cpu().float()

                stem = _safe_stem(info.get("y_path", ""), f"{idx:06d}")
                out_name = f"{idx:06d}_{stem}.wav"
                enhanced_path = enhanced_dir / out_name
                if enhanced_path.exists() and not overwrite:
                    raise FileExistsError(f"Output exists. Use --overwrite to replace: {enhanced_path}")
                torchaudio.save(str(enhanced_path), x_hat.clamp(-1, 1), sample_rate=sr)

                clean_path = ""
                noisy_path = ""
                if save_inputs:
                    clean_path = str(clean_dir / out_name)
                    noisy_path = str(noisy_dir / out_name)
                    torchaudio.save(clean_path, x.detach().cpu().float().clamp(-1, 1), sample_rate=sr)
                    torchaudio.save(noisy_path, y.detach().cpu().float().clamp(-1, 1), sample_rate=sr)

                elapsed_s = time.perf_counter() - item_started_at
                files.append(
                    {
                        "index": idx,
                        "clean_path": info.get("x_path", clean_path),
                        "noisy_path": info.get("y_path", noisy_path),
                        "enhanced_path": str(enhanced_path),
                        "saved_clean_path": clean_path,
                        "saved_noisy_path": noisy_path,
                        "sample_rate": sr,
                        "num_samples": int(x_hat.shape[-1]),
                        "duration_s": float(x_hat.shape[-1] / sr),
                        "elapsed_s": elapsed_s,
                    }
                )
            except Exception as exc:
                error = {
                    "index": idx,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
                errors.append(error)
                if not continue_on_error:
                    raise

    elapsed_s = time.perf_counter() - started_at
    manifest = {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "task": task,
        "config_name": config_name,
        "config_overrides": config_overrides,
        "checkpoint_path": ckpt_path,
        "split": split,
        "part": part,
        "pipeline": pipeline,
        "execution": execution,
        "solver": solver,
        "steps": steps,
        "model_dtype": model_dtype_name.lower(),
        "float32_matmul_precision": float32_matmul_precision,
        "model_memory_format": model_memory_format,
        "crop_mode": crop_mode,
        "seed": seed,
        "offset": offset,
        "limit": limit,
        "selection": selection,
        "selection_seed": selection_seed,
        "selected_indices": indices,
        "num_available": len(dataset),
        "num_files": len(files),
        "num_errors": len(errors),
        "files": files,
        "errors": errors,
    }
    if pipeline == "streaming":
        manifest["streaming"] = {
            "algorithmic_delay_samples": streaming_algorithmic_delay(streaming_config),
            "n_fft": streaming_config.n_fft,
            "hop_length": streaming_config.hop_length,
            **streaming_info,
        }
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    mean_file_s = sum(item["elapsed_s"] for item in files) / len(files) if files else 0.0
    result = {
        "mode": "test_set_inference",
        "run_id": run_id,
        "run_started_at": run_started_at,
        "backend": backend,
        "hardware": hardware,
        "device": device.type,
        "gpu_name": device_label(device) if device.type == "cuda" else "",
        "task": task,
        "config_name": config_name,
        "config_overrides": config_overrides,
        "checkpoint_path": ckpt_path,
        "split": split,
        "part": part,
        "pipeline": pipeline,
        "execution": execution,
        "solver": solver,
        "steps": steps,
        "model_dtype": model_dtype_name.lower(),
        "float32_matmul_precision": float32_matmul_precision,
        "model_memory_format": model_memory_format,
        "crop_mode": crop_mode,
        "seed": seed,
        "num_available": len(dataset),
        "num_files": len(files),
        "num_errors": len(errors),
        "selection": selection,
        "selection_seed": selection_seed,
        "elapsed_s": elapsed_s,
        "mean_file_s": mean_file_s,
        "output_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
        "torch_version": torch.__version__,
    }
    if pipeline == "streaming":
        result.update(
            {
                "streaming_algorithmic_delay_samples": streaming_algorithmic_delay(streaming_config),
                **streaming_info,
            }
        )
    if device.type == "cpu":
        result["num_threads"] = torch.get_num_threads()
        result["num_interop_threads"] = torch.get_num_interop_threads()
    if cache_info:
        result.update(cache_info)

    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
