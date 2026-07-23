"""TensorRT feasibility probes for the STFTPR backbone.

``fixed_window`` is an offline-only diagnostic.  ``streaming_fp16`` exports
the actual one-frame ``forward_step`` with every causal buffer as an explicit
input/output.  The latter is the only mode whose timing can be compared to the
streaming PyTorch benchmark (it still excludes STFT/iSTFT and CUDA Graph).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal


REMOTE_ROOT = "/root/streamfm"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_local_root() -> Path:
    """Resolve the repository before Modal imports this file at container root."""
    remote_repo = Path(REMOTE_ROOT)
    if (remote_repo / "config").is_dir() and (remote_repo / "sgmse").is_dir():
        return remote_repo
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    raise RuntimeError("Could not locate the StreamFM repository root.")


LOCAL_ROOT = _find_local_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        # torchaudio 2.7 dispatches to soundfile; without it `load` has no backend.
        "soundfile==0.12.1",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "numpy==1.26.4",
        "requests",
        # This probe uses the Torch-TensorRT 2.7 / ModelOpt 0.17 INT8 path.
        # TensorRT is a compiled extension, not a semver-compatible pure-Python
        # dependency: an unpinned ``tensorrt`` now resolves to 11.x, whereas
        # Torch-TensorRT 2.7 was built and validated against TRT 10.9.
        "tensorrt==10.9.0.34",
        "torch-tensorrt==2.7.0",
        # Torch-TensorRT 2.7 officially supports the ModelOpt 0.17 release.
        # The ``torch`` extra supplies its optional runtime dependencies (notably
        # ``pulp``), which ModelOpt imports before exposing quantization APIs.
        "nvidia-modelopt[torch]==0.17.0",
    )
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    .add_local_dir(
        str(LOCAL_ROOT / "experiments"),
        remote_path=f"{REMOTE_ROOT}/experiments",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        str(LOCAL_ROOT / "sgmse"),
        remote_path=f"{REMOTE_ROOT}/sgmse",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        str(LOCAL_ROOT / "checkpoints" / "streamfm_stftpr_dnn_only.pt"),
        remote_path=f"{REMOTE_ROOT}/checkpoints/streamfm_stftpr_dnn_only.pt",
    )
)

app = modal.App("streamfm-tensorrt-int8-probe", image=image)


def _measure_ms(fn, *, warmup: int, iterations: int) -> dict[str, float]:
    """Time fn with one CUDA event pair per call; returns mean/p50/p90/p99 in ms.

    end.synchronize() inside the loop makes each sample individually valid
    (no overlap between consecutive calls).
    """
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(0.90 * (len(samples) - 1))],
        "p99_ms": samples[int(0.99 * (len(samples) - 1))],
    }


def _preserve_modelopt_amax_buffers_for_export() -> None:
    """Keep calibrated Q/DQ scales as registered buffers during torch.export.

    ModelOpt 0.17 detaches ``_amax`` in export mode.  With torch 2.7 that turns
    the scales into unowned ``lifted_tensor`` subclasses; Torch-TensorRT 2.7
    subsequently cannot materialize them as TensorRT constants.  Returning the
    already-calibrated buffer preserves the graph ownership without changing
    any scale value or the trained model weights.
    """
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    from modelopt.torch.quantization.utils import is_torch_export_mode

    original_get_amax = TensorQuantizer._get_amax

    def _get_amax_preserving_buffer(self, inputs):
        if is_torch_export_mode():
            amax = self._buffers.get("_amax")
            if amax is not None:
                return amax if amax.device == inputs.device else amax.to(inputs.device)
        return original_get_amax(self, inputs)

    TensorQuantizer._get_amax = _get_amax_preserving_buffer


def _install_pure_torch_fake_quant() -> dict:
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import install_pure_torch_fake_quant

    return install_pure_torch_fake_quant()


def _register_causal_conv_for_quantization() -> dict:
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import (
        register_causal_conv_for_quantization,
    )

    return register_causal_conv_for_quantization()


def _route_causal_conv_step_through_quantizers() -> None:
    """No-op: ``CausalConv2d.forward_step`` now routes through its own quantizers.

    This used to monkeypatch the recurrence, which duplicated the model logic and
    had already drifted from it once.  Kept as a stub so the probe's older code
    paths still call something harmless.
    """
    return None


def _describe_quantizer(quantizer) -> dict | None:
    """Summarise one TensorQuantizer: is it live, and did calibration reach it?"""
    if quantizer is None:
        return None
    amax = getattr(quantizer, "_amax", None)
    try:
        enabled = bool(quantizer.is_enabled)
    except Exception:
        enabled = not bool(getattr(quantizer, "_disabled", False))
    return {
        "enabled": enabled,
        "num_bits": getattr(quantizer, "num_bits", None),
        "axis": getattr(quantizer, "axis", None),
        "has_amax": amax is not None,
        "amax_numel": int(amax.numel()) if amax is not None else 0,
        "amax_max": float(amax.max()) if amax is not None else None,
    }


def _audit_quantizers(model) -> dict:
    """Report, per convolution, whether both Q/DQ quantizers exist and are calibrated.

    This is the gate the TensorRT build is not allowed to bypass: a convolution
    missing either quantizer, or carrying an uncalibrated one, cannot run on an
    INT8 kernel no matter what the builder is asked for.
    """
    from torch import nn

    rows = []
    for name, module in model.named_modules():
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(module, nn.Conv2d) and input_quantizer is None:
            continue
        rows.append(
            {
                "name": name,
                "class": type(module).__name__,
                "mro": [cls.__name__ for cls in type(module).__mro__[:5]],
                "input_quantizer": _describe_quantizer(input_quantizer),
                "weight_quantizer": _describe_quantizer(weight_quantizer),
            }
        )

    def fully_quantized(row) -> bool:
        both = (row["input_quantizer"], row["weight_quantizer"])
        return all(q is not None and q["enabled"] and q["has_amax"] for q in both)

    ready = [row for row in rows if fully_quantized(row)]
    return {
        "conv_like_modules": len(rows),
        "fully_quantized_convs": len(ready),
        "missing_input_quantizer": [r["name"] for r in rows if r["input_quantizer"] is None],
        "missing_weight_quantizer": [r["name"] for r in rows if r["weight_quantizer"] is None],
        "uncalibrated": [
            r["name"]
            for r in rows
            if not fully_quantized(r) and r["input_quantizer"] is not None
        ],
        "layers": rows,
    }


def _summarize_engine_layers(module) -> dict:
    """Classify every engine layer by type and execution precision.

    A convolution counted under ``Int8`` here is the only proof that INT8 PTQ
    did anything: Q/DQ nodes alone are a lossy round-trip if the builder still
    picks a float tactic for the convolution between them.
    """
    import collections

    raw = module.get_layer_info()
    try:
        info = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"error": "engine inspector did not return JSON"}

    # The inspector exposes no ``Precision`` field: a layer's execution
    # precision has to be read from its tensor formats and its tactic name.
    layers = info.get("Layers", info if isinstance(info, list) else [])

    def tensor_dtypes(layer, key):
        return [str(t.get("Format/Datatype", "?")) for t in layer.get(key, []) or []]

    def is_int8(tactic: str, dtypes: list[str]) -> bool:
        lowered = tactic.lower()
        if "imma" in lowered or "i8i8" in lowered or "int8" in lowered:
            return True
        return any("int8" in d.lower() for d in dtypes)

    by_type: collections.Counter = collections.Counter()
    conv_precision: collections.Counter = collections.Counter()
    conv_rows = []
    int8_layer_count = 0
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_type = str(layer.get("LayerType", "?"))
        tactic = str(layer.get("TacticName", layer.get("TacticValue", "")))
        in_dtypes = tensor_dtypes(layer, "Inputs")
        out_dtypes = tensor_dtypes(layer, "Outputs")
        int8 = is_int8(tactic, in_dtypes + out_dtypes)
        int8_layer_count += int(int8)
        by_type[(layer_type, "Int8" if int8 else "Float")] += 1
        if "Convolution" in layer_type or layer_type == "gemm":
            conv_precision["Int8" if int8 else "Float"] += 1
            conv_rows.append(
                {
                    "name": layer.get("Name"),
                    "inputs": in_dtypes,
                    "outputs": out_dtypes,
                    "tactic": tactic,
                    "int8": int8,
                }
            )

    quantize_reformats = sum(
        1
        for layer in layers
        if isinstance(layer, dict)
        and any("Int8" in d for d in tensor_dtypes(layer, "Outputs"))
        and str(layer.get("LayerType")) in {"Reformat", "NoOp"}
    )

    return {
        "total_layers": len(layers),
        "convolutions_by_precision": dict(conv_precision),
        "int8_layer_count": int8_layer_count,
        "int8_quantize_reformats": quantize_reformats,
        "layer_type_histogram": {f"{k[0]}|{k[1]}": v for k, v in sorted(by_type.items())},
        "convolutions": conv_rows,
    }


def _flatten_tensor_state(value):
    """Flatten a nested streaming state, omitting its structural ``None``s."""
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in _flatten_tensor_state(child)]
    if value is None:
        return []
    raise TypeError(f"Unsupported state leaf: {type(value)!r}")


def _unflatten_tensor_state(template, values):
    """Rebuild a state tree from tensor inputs, retaining ``None`` leaves."""
    import torch

    if isinstance(template, torch.Tensor):
        return next(values)
    if isinstance(template, list):
        return [_unflatten_tensor_state(child, values) for child in template]
    if isinstance(template, tuple):
        return tuple(_unflatten_tensor_state(child, values) for child in template)
    if template is None:
        return None
    raise TypeError(f"Unsupported state leaf: {type(template)!r}")


def _make_streaming_wrapper(model, state_template):
    """Expose ``forward_step`` as a pure tensor-input/tensor-output module.

    ``forward_step`` mutates its nested causal state in normal PyTorch use.
    TensorRT cannot retain that Python state, so the wrapper makes all state
    tensors engine I/O.  The caller feeds the returned tensors into the next
    frame, exactly matching the production recurrence.
    """
    import torch
    raw_step = getattr(type(model).forward_step, "__wrapped__", None)
    if raw_step is None:
        raise RuntimeError("The raw CausalNCSNpp.forward_step is unavailable for TensorRT export.")

    class StreamingStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x, time_cond, *flat_state):
            state = _unflatten_tensor_state(state_template, iter(flat_state))
            # Bypass the decorated torch.compile entry point: TensorRT must
            # trace the original streaming implementation, not an Inductor
            # graph wrapped around it.
            y, next_state = raw_step(self.model, x, time_cond=time_cond, state=state)
            return (y, *_flatten_tensor_state(next_state))

    return StreamingStep().eval()


def _load_calibration_audio(device, *, num_files: int, max_seconds: float, seed: int, split: str):
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import load_calibration_audio

    return load_calibration_audio(
        device, num_files=num_files, max_seconds=max_seconds, seed=seed, split=split
    )


def _run_real_audio_stream(
    module,
    raw_step,
    audio,
    *,
    config,
    device,
    steps: int,
    max_frames: int,
    capture_frame_index: int | None = None,
) -> dict:
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import run_real_audio_stream

    return run_real_audio_stream(
        module,
        raw_step,
        audio,
        config=config,
        device=device,
        steps=steps,
        max_frames=max_frames,
        capture_frame_index=capture_frame_index,
    )


def _compare_on_real_audio(
    candidates: dict,
    audio,
    *,
    config,
    device,
    steps: int,
    max_frames: int,
) -> dict:
    """Score several one-frame implementations against the FP32 one, on real audio.

    Every implementation is fed the *same* trajectory - the one the FP32
    reference produces - so the numbers isolate per-step error instead of
    letting each variant drift into a different part of the input space.
    ``candidates`` maps a name to ``(callable, initial_flat_state)``; the entry
    named ``fp32`` drives the trajectory.
    """
    import torch

    from experiments.core.tensors import pack_ri_channels
    from experiments.streaming.stft import (
        complex_to_ri_frame,
        compress_complex,
        compression_norm,
        sqrt_hann_window,
    )

    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)
    hop = config.hop_length

    states = {name: tuple(t.clone() for t in state) for name, (_, state) in candidates.items()}
    totals = {name: {"abs_sum": 0.0, "max": 0.0, "count": 0} for name in candidates}
    reference_abs_sum = 0.0

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    total_frames = min(max_frames, audio.shape[-1] // hop)
    with torch.inference_mode():
        for frame_idx in range(total_frames):
            chunk = audio[:, frame_idx * hop : (frame_idx + 1) * hop]
            input_buffer = torch.cat([input_buffer[:, hop:], chunk], dim=-1)
            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, : -config.cut_highest_freqs]
            y_complex = compress_complex(spectrum, config)
            y_frame = complex_to_ri_frame(y_complex.abs().to(y_complex.dtype))
            x_t = y_frame + config.sigma_y * torch.randn_like(y_frame)

            for step_idx in range(steps):
                dnn_input = pack_ri_channels(x_t, y_frame)
                t = torch.full((1,), step_idx / steps, device=device, dtype=torch.float32)
                outputs = {}
                for name, (call, _) in candidates.items():
                    result = call(dnn_input, t, *states[name])
                    outputs[name] = result[0]
                    states[name] = tuple(result[1:])
                reference = outputs["fp32"].float()
                reference_abs_sum += float(reference.abs().mean())
                for name, output in outputs.items():
                    delta = (output.float() - reference).abs()
                    totals[name]["abs_sum"] += float(delta.mean())
                    totals[name]["max"] = max(totals[name]["max"], float(delta.max()))
                    totals[name]["count"] += 1
                # The FP32 reference drives the trajectory for everyone.
                x_t = x_t + reference / steps

    calls = max(totals["fp32"]["count"], 1)
    reference_mean_abs = reference_abs_sum / calls
    return {
        "frames": total_frames,
        "solver_steps": steps,
        "backbone_calls": calls,
        "reference_mean_abs": reference_mean_abs,
        "variants": {
            name: {
                "mean_abs_diff": total["abs_sum"] / max(total["count"], 1),
                "max_abs_diff": total["max"],
                "relative_mean_error": (total["abs_sum"] / max(total["count"], 1))
                / max(reference_mean_abs, 1e-12),
            }
            for name, total in totals.items()
        },
    }


def _measure_streaming_ms(fn, initial_state, *, warmup: int, iterations: int) -> dict[str, float]:
    """Measure a recurrent one-frame function, excluding state allocation."""
    import torch

    def run(state):
        result = fn(*state)
        return tuple(result[1:])

    with torch.inference_mode():
        state = tuple(tensor.clone() for tensor in initial_state)
        for _ in range(warmup):
            state = run(state)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            state = run(state)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(0.90 * (len(samples) - 1))],
        "p99_ms": samples[int(0.99 * (len(samples) - 1))],
    }


@app.function(gpu="L4", timeout=900)
def cuda_ext_check() -> dict:
    """Report why ModelOpt's CUDA extension is unavailable, with its real error.

    ModelOpt swallows the load failure and leaves ``cuda_ext = None``; the
    symptom only surfaces much later as an AttributeError inside Torch-TensorRT
    lowering.  This surfaces the cause directly.
    """
    import shutil
    import subprocess
    import traceback

    import torch

    report: dict = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "nvcc_on_path": shutil.which("nvcc"),
        "cuda_home": os.environ.get("CUDA_HOME"),
    }
    if report["nvcc_on_path"]:
        report["nvcc_version"] = subprocess.run(
            [report["nvcc_on_path"], "--version"], capture_output=True, text=True
        ).stdout.strip().splitlines()[-1:]

    from modelopt.torch.quantization import tensor_quant

    # ``cuda_ext`` is resolved through the function's globals, not necessarily a
    # public module attribute, so inspect the frame that actually fails.
    impl = getattr(tensor_quant, "fake_quant_impl", None)
    impl_globals = getattr(impl, "__globals__", {})
    report["ext_globals"] = {
        name: (None if value is None else type(value).__name__)
        for name, value in impl_globals.items()
        if "ext" in name.lower()
    }
    report["tensor_quant_module_file"] = getattr(tensor_quant, "__file__", None)

    try:
        from modelopt.torch.quantization import extensions as quant_extensions

        report["extensions_module"] = {
            name: (None if getattr(quant_extensions, name) is None else "loaded")
            for name in dir(quant_extensions)
            if not name.startswith("__")
        }
        for loader_name in ("get_cuda_ext", "precompiled_cuda_ext"):
            loader = getattr(quant_extensions, loader_name, None)
            if callable(loader):
                try:
                    report[f"{loader_name}_result"] = (
                        "loaded" if loader(raise_if_failed=True) is not None else "None"
                    )
                except TypeError:
                    report[f"{loader_name}_result"] = (
                        "loaded" if loader() is not None else "None"
                    )
                except Exception:
                    report[f"{loader_name}_traceback"] = traceback.format_exc()[-3000:]
    except Exception:
        report["extensions_import_traceback"] = traceback.format_exc()[-1500:]

    # The dispatch logic decides whether a pure-PyTorch path already exists or
    # whether the compiled extension is genuinely required.
    import inspect

    for symbol in ("fake_quant_impl", "_quantize_impl", "fake_tensor_quant", "scaled_e4m3"):
        target = getattr(tensor_quant, symbol, None)
        if target is not None:
            try:
                report[f"src_{symbol}"] = inspect.getsource(target)
            except Exception:
                pass

    return report


def _protect_quantize_op_from_constant_folding() -> dict:
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import (
        protect_quantize_op_from_constant_folding,
    )

    return protect_quantize_op_from_constant_folding()


def _patch_quantize_converter_for_constant_weights() -> dict:
    """Delegate to the shared implementation used by the benchmark pipeline."""
    from experiments.benchmarks.tensorrt.quantization import (
        patch_quantize_converter_for_constant_weights,
    )

    return patch_quantize_converter_for_constant_weights()


@app.function(gpu="T4", timeout=900)
def lowering_check() -> dict:
    """Ask torch-tensorrt whether its constant folder protects quantize_op.

    The engine shows float weights on every convolution, so the weight Q/DQ
    dies somewhere between export and conversion.  Constant folding is the
    prime suspect; this reads the actual installed source instead of guessing.
    """
    import inspect

    import torch  # noqa: F401
    import torch_tensorrt

    report: dict = {"torch_tensorrt": torch_tensorrt.__version__}

    from torch_tensorrt.dynamo.lowering.passes import constant_folding

    source = inspect.getsource(constant_folding)
    report["constant_folding_source"] = source
    report["mentions_quantize_op"] = "quantize_op" in source
    report["mentions_is_impure"] = "is_impure" in source

    try:
        from torch_tensorrt.dynamo.lowering.passes import ATEN_POST_LOWERING_PASSES

        report["post_lowering_passes"] = [
            getattr(p, "__name__", str(p)) for p in ATEN_POST_LOWERING_PASSES.passes
        ]
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["post_lowering_passes_error"] = repr(exc)

    # Is there a converter for quantize_op at all?  Without one the op cannot
    # become an IQuantizeLayer no matter what the folder does.
    try:
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            DYNAMO_CONVERTERS,
        )

        report["quantize_op_has_converter"] = (
            torch.ops.tensorrt.quantize_op.default in DYNAMO_CONVERTERS
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["converter_probe_error"] = repr(exc)

    # add_quantize() rejects a raw torch.Tensor, so the converter must be
    # promoting constants to ITensor somewhere -- or failing to.
    try:
        from torch_tensorrt.dynamo.conversion.impl import quantize as quantize_impl

        report["quantize_impl_source"] = inspect.getsource(quantize_impl)
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["quantize_impl_error"] = repr(exc)

    return report


@app.function(cpu=8, memory=16384, timeout=1800)
def quant_audit(
    *,
    calibration_steps: int = 4,
    input_freqs: int = 256,
    input_channels: int = 4,
    register_causal_conv: bool = True,
    route_step_through_quantizers: bool = True,
) -> dict:
    """Answer, without building any engine, whether INT8 PTQ reaches the convolutions.

    Runs on CPU: the question is which quantizers ModelOpt injects and whether
    the streaming calibration loop reaches them, which is device independent and
    orders of magnitude cheaper than a TensorRT build.
    """
    import torch

    import modelopt.torch.quantization as mtq

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    from experiments.benchmarks.loading import load_flow_model
    from experiments.core.paths import make_benchmark_paths

    torch.manual_seed(42)
    device = torch.device("cpu")

    registration = (
        _register_causal_conv_for_quantization() if register_causal_conv else {"action": "skipped"}
    )
    if route_step_through_quantizers:
        _route_causal_conv_step_through_quantizers()

    paths = make_benchmark_paths(Path(REMOTE_ROOT))
    model, _cfg = load_flow_model(device, torch.float32, paths, task="stftpr")
    model = model.eval()
    model.requires_grad_(False)

    raw_step = getattr(type(model).forward_step, "__wrapped__", None)
    if raw_step is None:
        raise RuntimeError("The raw CausalNCSNpp.forward_step is unavailable for calibration.")

    def calibrate(module):
        # Same contract as the deployed calibration loop in tensorrt/streaming.py:
        # the amax statistics must come from real one-frame streaming calls.
        with torch.inference_mode():
            for _ in range(calibration_steps):
                state = module.init_state()
                x = torch.randn(1, input_channels, input_freqs, 1, device=device)
                t = torch.rand(1, device=device)
                raw_step(module, x, time_cond=t, state=state)

    started = time.perf_counter()
    quantized = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
    audit = _audit_quantizers(quantized)

    return {
        "scope": "ModelOpt Q/DQ coverage of the streaming forward_step; no TensorRT build",
        "torch": torch.__version__,
        "calibration_steps": calibration_steps,
        "registration": registration,
        "step_routed_through_quantizers": route_step_through_quantizers,
        "elapsed_s": time.perf_counter() - started,
        **audit,
    }


@app.function(gpu="L4", timeout=5400, volumes={"/data": CACHE_VOLUME})
def probe(
    *,
    iterations: int = 100,
    warmup: int = 10,
    calibration_steps: int = 16,
    memory_format: str = "contiguous",
    mode: str = "fixed_window",
    input_freqs: int = 256,
    calibration_source: str = "audio",
    calibration_files: int = 4,
    calibration_seconds: float = 4.0,
    calibration_split: str = "train",
    calibration_solver_steps: str = "1,5",
    calibration_seed: int = 0,
) -> dict:
    """Run either the offline diagnostic or the recurrent streaming probe."""
    import copy
    import torch
    import torch_tensorrt
    import tensorrt

    # Fail before tracing StreamFM when an ABI/package mismatch has prevented
    # Torch-TensorRT from registering the ModelOpt Q/DQ custom operation.
    if not tensorrt.__version__.startswith("10.9."):
        raise RuntimeError(
            f"Incompatible TensorRT {tensorrt.__version__}; this probe requires 10.9.x "
            "with torch-tensorrt==2.7.0. Rebuild the Modal image."
        )
    if not hasattr(torch.ops.tensorrt, "quantize_op"):
        raise RuntimeError(
            "Torch-TensorRT did not register torch.ops.tensorrt.quantize_op. "
            "The TensorRT / Torch-TensorRT / ModelOpt package set is inconsistent."
        )

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    _preserve_modelopt_amax_buffers_for_export()

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    from experiments.benchmarks.loading import load_flow_model
    from experiments.core.paths import make_benchmark_paths

    torch.manual_seed(42)
    device = torch.device("cuda")
    if memory_format == "channels_last":
        torch_memory_format = torch.channels_last
    elif memory_format == "contiguous":
        torch_memory_format = torch.contiguous_format
    else:
        raise ValueError("memory_format must be 'contiguous' or 'channels_last'.")

    paths = make_benchmark_paths(Path(REMOTE_ROOT))
    model_fp32, _cfg = load_flow_model(device, torch.float32, paths, task="stftpr")
    model_fp32 = model_fp32.eval().to(memory_format=torch_memory_format)
    # [B, 4, F, T=8]: (x_t, y) real/imag pairs over an 8-frame offline window
    # (fixed_window mode only; streaming mode builds its own T=1 input below).
    x_fp32 = torch.randn(1, 4, 256, 8, device=device, dtype=torch.float32).contiguous(
        memory_format=torch_memory_format
    )
    time_cond_fp32 = torch.full((1,), 0.5, device=device, dtype=torch.float32)

    if mode == "streaming_fp16":
        # One frame, and the actual causal state contract used by StreamFM.
        # This deliberately does not reuse the 8-frame offline input below.
        model_fp16 = copy.deepcopy(model_fp32).half().eval().to(memory_format=torch_memory_format)
        # The step mutates *state inputs*.  AOT export rejects that combination
        # if any model parameter is still marked trainable, even though this is
        # inference-only.  Freezing parameters is semantically exact here.
        model_fp16.requires_grad_(False)
        # The in-place buffer writes in ``forward_step`` do not survive export
        # (``xbuf[..., -1] = ...`` traces to an empty stack); the functional
        # path builds the next buffer instead, which is what INT8 also uses.
        from sgmse.backbones.streaming_unet import CausalConv2d, CausalDecoupledConv2d

        for module in model_fp16.modules():
            if isinstance(module, (CausalConv2d, CausalDecoupledConv2d)):
                module.functional_state_updates = True
        x_step = torch.randn(1, 4, 256, 1, device=device, dtype=torch.float16).contiguous(
            memory_format=torch_memory_format
        )
        time_step = torch.full((1,), 0.5, device=device, dtype=torch.float16)
        state_template = model_fp16.init_state()
        initial_state = tuple(_flatten_tensor_state(state_template))
        streaming = _make_streaming_wrapper(model_fp16, state_template)

        # Validate one state transition from the same all-zero state before
        # timing the recurrent loop.  Output and every returned buffer must
        # agree with eager, otherwise a latency number is meaningless.
        eager_inputs = tuple(tensor.clone() for tensor in initial_state)
        eager_result = streaming(x_step, time_step, *eager_inputs)
        # ``torch_tensorrt.compile(module, ...)`` currently fabricates an
        # incompatible dynamic-shape spec for a ``*flat_state`` signature.
        # Exporting the already-validated fixed signature first avoids that
        # tracer bug, then uses the same Dynamo TensorRT compiler as INT8.
        streaming_program = torch.export.export(
            streaming, (x_step, time_step, *initial_state), strict=True
        )
        trt_streaming = torch_tensorrt.dynamo.compile(
            streaming_program,
            arg_inputs=[x_step, time_step, *initial_state],
            enabled_precisions={torch.float16},
        )
        trt_inputs = tuple(tensor.clone() for tensor in initial_state)
        trt_result = trt_streaming(x_step, time_step, *trt_inputs)
        output_delta = (trt_result[0].float() - eager_result[0].float()).abs()
        state_deltas = [
            (actual.float() - expected.float()).abs().max()
            for actual, expected in zip(trt_result[1:], eager_result[1:])
        ]

        def eager_call(*state):
            return streaming(x_step, time_step, *state)

        def trt_call(*state):
            return trt_streaming(x_step, time_step, *state)

        return {
            "scope": (
                "true CausalNCSNpp.forward_step recurrence: one frame, 63 causal "
                "state tensors fed from each frame to the next; excludes STFT/iSTFT "
                "and CUDA Graph replay"
            ),
            "mode": mode,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torch_tensorrt": torch_tensorrt.__version__,
            "tensorrt": tensorrt.__version__,
            "shape": list(x_step.shape),
            "state_tensor_count": len(initial_state),
            "memory_format": memory_format,
            "iterations": iterations,
            "warmup": warmup,
            "eager_fp16": _measure_streaming_ms(
                eager_call, initial_state, warmup=warmup, iterations=iterations
            ),
            "trt_fp16": _measure_streaming_ms(
                trt_call, initial_state, warmup=warmup, iterations=iterations
            ),
            "one_step_validation": {
                "output_mean_abs_diff": float(output_delta.mean()),
                "output_max_abs_diff": float(output_delta.max()),
                "state_max_abs_diff": float(torch.stack(state_deltas).max()),
            },
        }
    if mode == "streaming_int8":
        from sgmse.backbones.streaming_unet import CausalConv2d, CausalDecoupledConv2d

        # Both fixes under test, in the order they must happen: the registry
        # decides *whether* quantizers exist, the step routing decides whether
        # calibration ever reaches them.
        fake_quant_backend = _install_pure_torch_fake_quant()
        registration = _register_causal_conv_for_quantization()
        _route_causal_conv_step_through_quantizers()

        model = copy.deepcopy(model_fp32).eval()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (CausalConv2d, CausalDecoupledConv2d)):
                module.functional_state_updates = True

        raw_step = getattr(type(model).forward_step, "__wrapped__", None)
        if raw_step is None:
            raise RuntimeError("The raw CausalNCSNpp.forward_step is unavailable for calibration.")

        x_step = torch.randn(1, 4, input_freqs, 1, device=device, dtype=torch.float32)
        time_step = torch.full((1,), 0.5, device=device, dtype=torch.float32)

        calibration_report: dict = {"source": calibration_source}
        captured_frames: list = []
        if calibration_source == "audio":
            from experiments.streaming.stft import StreamingSTFTConfig

            stft_config = StreamingSTFTConfig()
            calibration_audio, calibration_names = _load_calibration_audio(
                device,
                num_files=calibration_files,
                max_seconds=calibration_seconds,
                seed=calibration_seed,
                split=calibration_split,
            )
            solver_steps = [int(part) for part in calibration_solver_steps.split(",") if part]
            calibration_report.update(
                {
                    "split": calibration_split,
                    "files": calibration_names,
                    "seconds_per_file": calibration_seconds,
                    "solver_steps": solver_steps,
                    "audio_samples": int(calibration_audio.shape[-1]),
                }
            )

            def calibrate(module):
                # One continuous pass per NFE setting: the deployed engine serves
                # both Euler1 and Euler5, whose t values (and therefore whose
                # activation ranges) differ.
                runs = []
                for index, steps in enumerate(solver_steps):
                    runs.append(
                        _run_real_audio_stream(
                            module,
                            raw_step,
                            calibration_audio,
                            config=stft_config,
                            device=device,
                            steps=steps,
                            max_frames=calibration_steps,
                            # Keep one steady-state frame for the single-step
                            # validation, so it is not judged on white noise.
                            capture_frame_index=(calibration_steps // 2) if index == 0 else None,
                        )
                    )
                if runs and runs[0]["captured"] is not None:
                    captured_frames.append(runs[0]["captured"])
                calibration_report["runs"] = [
                    {k: v for k, v in run.items() if k != "captured"} for run in runs
                ]
        else:

            def calibrate(module):
                # Legacy white-noise calibration, kept only as a control.
                with torch.inference_mode():
                    state = module.init_state()
                    for _ in range(calibration_steps):
                        frame = torch.randn(1, 4, input_freqs, 1, device=device)
                        t = torch.rand(1, device=device)
                        _out, state = raw_step(module, frame, time_cond=t, state=state)

        quantized = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
        if captured_frames:
            x_step, time_step = captured_frames[0]
            calibration_report["validation_frame"] = "captured from the real audio stream"
        audit = _audit_quantizers(quantized)
        if audit["fully_quantized_convs"] == 0:
            raise RuntimeError("No convolution carries calibrated Q/DQ; aborting before the build.")

        state_template = quantized.init_state()
        initial_state = tuple(_flatten_tensor_state(state_template))
        streaming = _make_streaming_wrapper(quantized, state_template)

        eager_inputs = tuple(tensor.clone() for tensor in initial_state)
        eager_result = streaming(x_step, time_step, *eager_inputs)

        # An unquantized copy of the same wrapper. Comparing the engine against
        # the fake-quant eager model alone cannot tell a TensorRT/simulation
        # mismatch from plain calibration damage; this third leg separates them.
        reference_model = copy.deepcopy(model_fp32).eval()
        reference_model.requires_grad_(False)
        for module in reference_model.modules():
            if isinstance(module, (CausalConv2d, CausalDecoupledConv2d)):
                module.functional_state_updates = True
        reference_template = reference_model.init_state()
        reference_streaming = _make_streaming_wrapper(reference_model, reference_template)
        fp32_result = reference_streaming(
            x_step, time_step, *(t.clone() for t in _flatten_tensor_state(reference_template))
        )

        constant_folding_patch = _protect_quantize_op_from_constant_folding()
        converter_patch = _patch_quantize_converter_for_constant_weights()

        build_started = time.perf_counter()
        with export_torch_mode():
            int8_program = torch.export.export(
                streaming, (x_step, time_step, *initial_state), strict=True
            )
        int8_engine = torch_tensorrt.dynamo.compile(
            int8_program,
            arg_inputs=[x_step, time_step, *initial_state],
            enabled_precisions={torch.int8, torch.float32},
            min_block_size=1,
            # DETAILED profiling verbosity is what exposes tactic names; without
            # it the inspector cannot tell an INT8 kernel from a float one.
            debug=True,
        )
        build_s = time.perf_counter() - build_started

        engine_summaries = []
        for name, submodule in int8_engine.named_children():
            if callable(getattr(submodule, "get_layer_info", None)):
                summary = _summarize_engine_layers(submodule)
                engine_summaries.append({"partition": name, **summary})

        trt_inputs = tuple(tensor.clone() for tensor in initial_state)
        trt_result = int8_engine(x_step, time_step, *trt_inputs)

        def compare(candidate, reference) -> dict:
            candidate = candidate.float()
            reference = reference.float()
            delta = (candidate - reference).abs()
            scale = reference.abs().mean().clamp_min(1e-12)
            return {
                "mean_abs_diff": float(delta.mean()),
                "max_abs_diff": float(delta.max()),
                "reference_mean_abs": float(reference.abs().mean()),
                "relative_mean_error": float(delta.mean() / scale),
            }

        def eager_call(*state):
            return streaming(x_step, time_step, *state)

        def trt_call(*state):
            return int8_engine(x_step, time_step, *state)

        # Single-frame diffs cannot show error accumulating through the causal
        # state; this replays a real stream with each variant keeping its own.
        stream_quality = None
        if calibration_source == "audio":
            reference_state = tuple(_flatten_tensor_state(reference_template))
            # Held-out files: scoring on the very audio the amax came from would
            # flatter the quantized model.
            eval_audio, eval_names = _load_calibration_audio(
                device,
                num_files=calibration_files,
                max_seconds=calibration_seconds,
                seed=calibration_seed + 1000,
                split=calibration_split,
            )
            calibration_report["eval_files"] = eval_names
            stream_quality = {}
            for steps in solver_steps:
                stream_quality[f"euler{steps}"] = _compare_on_real_audio(
                    {
                        "fp32": (reference_streaming, reference_state),
                        "fakequant": (streaming, initial_state),
                        "trt_int8": (int8_engine, initial_state),
                    },
                    eval_audio,
                    config=stft_config,
                    device=device,
                    steps=steps,
                    max_frames=calibration_steps,
                )

        return {
            "scope": (
                "one-frame CausalNCSNpp.forward_step with calibrated INT8 Q/DQ on every "
                "convolution; excludes STFT/iSTFT and CUDA Graph replay"
            ),
            "mode": mode,
            "gpu": torch.cuda.get_device_name(),
            "tensorrt": tensorrt.__version__,
            "registration": registration,
            "fake_quant_backend": fake_quant_backend,
            "constant_folding_patch": constant_folding_patch,
            "converter_patch": converter_patch,
            "calibration_steps": calibration_steps,
            "calibration": calibration_report,
            "quantizer_audit": {k: v for k, v in audit.items() if k != "layers"},
            "engine_build_s": build_s,
            "engines": engine_summaries,
            "eager_fp32": _measure_streaming_ms(
                eager_call, initial_state, warmup=warmup, iterations=iterations
            ),
            "trt_int8": _measure_streaming_ms(
                trt_call, initial_state, warmup=warmup, iterations=iterations
            ),
            "one_step_validation": {
                # trt_vs_fakequant isolates engine/simulation mismatch;
                # fakequant_vs_fp32 isolates the damage done by calibration.
                "trt_vs_fakequant": compare(trt_result[0], eager_result[0]),
                "fakequant_vs_fp32": compare(eager_result[0], fp32_result[0]),
                "trt_vs_fp32": compare(trt_result[0], fp32_result[0]),
            },
            "real_audio_quality": stream_quality,
        }
    if mode != "fixed_window":
        raise ValueError("mode must be 'fixed_window', 'streaming_fp16' or 'streaming_int8'.")

    # fixed_window mode: compile the same 8-frame forward at fp32/fp16/int8
    # and compare every engine against the eager fp32 reference (diff_stats).
    eager_fp32_out = model_fp32(x_fp32, time_cond=time_cond_fp32)
    trt_fp32_engine = torch_tensorrt.compile(
        model_fp32,
        ir="dynamo",
        arg_inputs=[x_fp32, time_cond_fp32],
        enabled_precisions={torch.float32},
    )
    trt_fp32_out = trt_fp32_engine(x_fp32, time_cond_fp32)

    model_fp16 = copy.deepcopy(model_fp32).half().eval().to(memory_format=torch_memory_format)
    x_fp16 = x_fp32.half().contiguous(memory_format=torch_memory_format)
    time_cond_fp16 = time_cond_fp32.half()
    eager_fp16_out = model_fp16(x_fp16, time_cond=time_cond_fp16)
    trt_fp16_engine = torch_tensorrt.compile(
        model_fp16,
        ir="dynamo",
        arg_inputs=[x_fp16, time_cond_fp16],
        enabled_precisions={torch.float16},
    )
    trt_fp16_out = trt_fp16_engine(x_fp16, time_cond_fp16)

    def calibrate(module):
        with torch.inference_mode():
            for _ in range(calibration_steps):
                module(torch.randn_like(x_fp32), time_cond=torch.rand_like(time_cond_fp32))

    # Keep the float reference model untouched: ModelOpt quantizes in place.
    quantized = mtq.quantize(
        copy.deepcopy(model_fp32).eval(), mtq.INT8_DEFAULT_CFG, forward_loop=calibrate
    )
    # Export ModelOpt's quantize/dequantize nodes first.  Re-tracing the wrapped
    # Python modules through TorchDynamo triggers ModelOpt callback internals
    # instead of exposing TensorRT Q/DQ operators.
    with export_torch_mode():
        # Use strict export now that the NumPy scalar calls in the backbone have
        # been removed.  Non-strict export leaves ModelOpt's calibration scales
        # as fake ``lifted_tensor`` subclasses; Torch-TensorRT cannot turn those
        # into TensorRT constants (``.numpy() is not supported``).
        int8_program = torch.export.export(quantized, (x_fp32, time_cond_fp32))
    int8_engine = torch_tensorrt.dynamo.compile(
        int8_program,
        arg_inputs=[x_fp32, time_cond_fp32],
        min_block_size=1,
    )
    int8_out = int8_engine(x_fp32, time_cond_fp32)

    def diff_stats(output):
        delta = (output.float() - eager_fp32_out.float()).abs()
        return {"mean_abs_diff": float(delta.mean()), "max_abs_diff": float(delta.max())}

    started = time.perf_counter()
    report = {
        "scope": "fixed-window backbone forward only; excludes streaming state, STFT/iSTFT, and CUDA Graph replay",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "shape": list(x_fp32.shape),
        "memory_format": memory_format,
        "iterations": iterations,
        "warmup": warmup,
        "calibration_steps": calibration_steps,
        "eager_fp32": {
            **_measure_ms(lambda: model_fp32(x_fp32, time_cond=time_cond_fp32), warmup=warmup, iterations=iterations),
            **diff_stats(eager_fp32_out),
        },
        "trt_fp32": {
            **_measure_ms(lambda: trt_fp32_engine(x_fp32, time_cond_fp32), warmup=warmup, iterations=iterations),
            **diff_stats(trt_fp32_out),
        },
        "eager_fp16": {
            **_measure_ms(lambda: model_fp16(x_fp16, time_cond=time_cond_fp16), warmup=warmup, iterations=iterations),
            **diff_stats(eager_fp16_out),
        },
        "trt_fp16": {
            **_measure_ms(lambda: trt_fp16_engine(x_fp16, time_cond_fp16), warmup=warmup, iterations=iterations),
            **diff_stats(trt_fp16_out),
        },
        "trt_int8": {
            **_measure_ms(lambda: int8_engine(x_fp32, time_cond_fp32), warmup=warmup, iterations=iterations),
            **diff_stats(int8_out),
        },
        "elapsed_s_after_compilation": time.perf_counter() - started,
    }
    return report


@app.local_entrypoint()
def ext_check():
    print(json.dumps(cuda_ext_check.remote(), indent=2))


@app.local_entrypoint()
def lowering():
    print(json.dumps(lowering_check.remote(), indent=2))


@app.local_entrypoint()
def audit(
    calibration_steps: int = 4,
    register_causal_conv: bool = True,
    route_step_through_quantizers: bool = True,
    output: str = "",
):
    report = quant_audit.remote(
        calibration_steps=calibration_steps,
        register_causal_conv=register_causal_conv,
        route_step_through_quantizers=route_step_through_quantizers,
    )
    if output:
        Path(output).write_text(json.dumps(report, indent=2))
    layers = report.pop("layers", [])
    print(json.dumps(report, indent=2))
    print(f"\n(layer detail: {len(layers)} entries{', written to ' + output if output else ''})")


@app.local_entrypoint()
def main(
    iterations: int = 100,
    warmup: int = 10,
    calibration_steps: int = 16,
    memory_format: str = "contiguous",
    mode: str = "fixed_window",
    input_freqs: int = 256,
    calibration_source: str = "audio",
    calibration_files: int = 4,
    calibration_seconds: float = 4.0,
    calibration_split: str = "train",
    calibration_solver_steps: str = "1,5",
    calibration_seed: int = 0,
    output: str = "",
):
    report = probe.remote(
        iterations=iterations,
        warmup=warmup,
        calibration_steps=calibration_steps,
        memory_format=memory_format,
        mode=mode,
        input_freqs=input_freqs,
        calibration_source=calibration_source,
        calibration_files=calibration_files,
        calibration_seconds=calibration_seconds,
        calibration_split=calibration_split,
        calibration_solver_steps=calibration_solver_steps,
        calibration_seed=calibration_seed,
    )
    if output:
        Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
