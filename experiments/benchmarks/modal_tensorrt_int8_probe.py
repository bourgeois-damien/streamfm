"""TensorRT feasibility probes for the STFTPR backbone.

``fixed_window`` is an offline-only diagnostic.  ``streaming_fp16`` exports
the actual one-frame ``forward_step`` with every causal buffer as an explicit
input/output.  The latter is the only mode whose timing can be compared to the
streaming PyTorch benchmark (it still excludes STFT/iSTFT and CUDA Graph).
"""

from __future__ import annotations

import json
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


@app.function(gpu="L4", timeout=1800, volumes={"/data": CACHE_VOLUME})
def probe(
    *,
    iterations: int = 100,
    warmup: int = 10,
    calibration_steps: int = 16,
    memory_format: str = "contiguous",
    mode: str = "fixed_window",
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
    if mode != "fixed_window":
        raise ValueError("mode must be 'fixed_window' or 'streaming_fp16'.")

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
def main(
    iterations: int = 100,
    warmup: int = 10,
    calibration_steps: int = 16,
    memory_format: str = "contiguous",
    mode: str = "fixed_window",
):
    print(
        json.dumps(
            probe.remote(
                iterations=iterations,
                warmup=warmup,
                calibration_steps=calibration_steps,
                memory_format=memory_format,
                mode=mode,
            ),
            indent=2,
        )
    )
