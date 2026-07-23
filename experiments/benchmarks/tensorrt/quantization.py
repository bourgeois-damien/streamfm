"""INT8/FP8 post-training quantization support for the streaming backbone.

Getting real INT8 convolutions out of Torch-TensorRT 2.7 requires more than
calling ``mtq.quantize``: ModelOpt does not recognise StreamFM's convolution
subclass, its fake-quant kernel needs a CUDA toolchain a pip-only image does
not have, and the Torch-TensorRT weight-quantization path has never been
exercised upstream.  Everything needed to work around that lives here so the
probe and the benchmark pipeline share one implementation.

Calibration runs on real audio: the amax statistics have to describe the
activation ranges production actually sees.  White noise gives ranges roughly
three orders of magnitude too wide, which collapses the useful signal into a
couple of INT8 levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION_ROOT = "/data/datasets/EARS-WHAM_v2_16k"


def install_pure_torch_fake_quant() -> dict:
    """Supply ModelOpt's missing fake-quant kernel when its CUDA extension is absent.

    ``modelopt_cuda_ext`` is JIT-compiled and needs ``CUDA_HOME``/nvcc, which a
    pip-only image does not provide.  ``fake_quant_impl`` then calls
    ``None.fake_tensor_quant_with_axis`` and Torch-TensorRT lowering dies on the
    first weight Q/DQ node.  This is the reference integer fake-quantization
    NVIDIA ships in ``pytorch_quantization``, expressed in plain PyTorch: same
    round-half-to-even, same clamp bounds, same zero-amax handling.  It runs
    only while building the engine, never during inference.
    """
    import torch
    from modelopt.torch.quantization import tensor_quant
    from modelopt.torch.quantization.extensions import get_cuda_ext

    if get_cuda_ext() is not None:
        return {"installed": False, "reason": "modelopt_cuda_ext is available"}

    def fake_quant_impl(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        if unsigned and inputs.min() < 0:
            raise TypeError("Negative values encountered in unsigned quantization.")
        max_bound = float(2.0 ** (num_bits - 1 + int(unsigned)) - 1.0)
        if unsigned:
            min_bound = 0.0
        elif narrow_range:
            min_bound = -max_bound
        else:
            min_bound = -max_bound - 1.0

        amax = amax.to(inputs.dtype)
        # A calibrated amax of zero means the tensor was identically zero; the
        # kernel emits zeros rather than dividing by it.
        zero_amax = amax <= (1.0 / (1 << 24))
        scale = max_bound / amax.masked_fill(zero_amax, 1.0)
        scale = scale.masked_fill(zero_amax, 0.0)

        outputs = torch.clamp((inputs * scale).round(), min_bound, max_bound)
        return outputs / scale.masked_fill(zero_amax, 1.0)

    tensor_quant.fake_quant_impl = fake_quant_impl
    return {"installed": True, "reason": "modelopt_cuda_ext unavailable (no CUDA_HOME/nvcc)"}


def register_causal_conv_for_quantization() -> dict:
    """Teach ModelOpt that ``CausalConv2d`` is a convolution it may quantize.

    ModelOpt keys its module registry on exact classes.  StreamFM's
    convolutions are ``nn.Conv2d`` *subclasses*, so an unmodified registry
    leaves them untouched and silently produces an engine with nothing
    quantized.
    """
    from torch import nn

    from sgmse.backbones.streaming_unet import CausalConv2d

    report: dict = {}
    try:
        from modelopt.torch.quantization.nn import QuantModuleRegistry
    except ImportError as error:  # pragma: no cover - reported, not raised
        return {"error": f"QuantModuleRegistry import failed: {error}"}

    try:
        report["conv2d_registered"] = nn.Conv2d in QuantModuleRegistry
        report["causal_conv2d_registered"] = CausalConv2d in QuantModuleRegistry
    except Exception as error:  # pragma: no cover - registry API drift
        return {"error": f"registry membership check failed: {error!r}"}

    if report["causal_conv2d_registered"]:
        report["action"] = "already registered"
        return report

    try:
        quant_conv_cls = QuantModuleRegistry.get(nn.Conv2d)
        QuantModuleRegistry.register({CausalConv2d: "CausalConv2d"})(quant_conv_cls)
        report["action"] = f"registered CausalConv2d as {quant_conv_cls.__name__}"
        report["causal_conv2d_registered"] = CausalConv2d in QuantModuleRegistry
    except Exception as error:  # pragma: no cover - registry API drift
        report["error"] = f"registration failed: {error!r}"
    return report


def protect_quantize_op_from_constant_folding() -> dict:
    """Stop Torch-TensorRT folding the weight Q/DQ into a float constant.

    Torch-TensorRT 2.7.0 ships ``_TorchTensorRTConstantFolder.is_impure``
    returning ``False`` unconditionally, with a TODO to revisit it "when
    quantization is added".  Weights are constants, so ``quantize_op`` on the
    weight path folds away and every convolution ends up with float weights.
    Activations are not constant, so their Q/DQ survives; that asymmetry is
    exactly what an engine dump shows when this is missing.
    """
    import inspect

    import torch
    from torch_tensorrt.dynamo.lowering.passes import constant_folding

    targets = []
    for name in ("quantize_op", "dynamic_block_quantize_op"):
        op = getattr(torch.ops.tensorrt, name, None)
        if op is not None:
            targets.append(op.default)

    folder = constant_folding._TorchTensorRTConstantFolder
    original = folder.is_impure

    def is_impure(self, node) -> bool:
        return node.target in targets

    folder.is_impure = is_impure
    return {
        "patched": True,
        "protected_ops": [str(target) for target in targets],
        "original_returned_constant_false": "return False" in inspect.getsource(original),
    }


def patch_quantize_converter_for_constant_weights() -> dict:
    """Let the quantize converter handle constant (weight) inputs.

    Upstream promotes only ``scale`` to an ITensor and passes ``input_tensor``
    through untouched.  That is fine for activations, which are already
    ITensors, but weights arrive as constants -- so ``add_quantize`` raises
    TypeError.  Upstream also never sets ``axis``, which per-channel weight
    quantization requires: amax has one entry per output channel, and TensorRT
    cannot place a vector scale without being told the axis.
    """
    import tensorrt as trt
    import torch
    from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
    from torch_tensorrt.dynamo.conversion import impl
    from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor, to_torch
    from torch_tensorrt.fx.converters.converter_utils import set_layer_name
    from torch_tensorrt.fx.types import TRTTensor

    def quantize(ctx, target, source_ir, name, input_tensor, amax, num_bits, exponent_bits):
        with unset_fake_temporarily():
            if num_bits == 8 and exponent_bits == 0:
                max_bound, dtype = 127, trt.DataType.INT8
            elif num_bits == 8 and exponent_bits == 4:
                max_bound, dtype = 448, trt.DataType.FP8
            else:
                raise ValueError(f"Unsupported quantization: {num_bits=}, {exponent_bits=}")

            if not isinstance(input_tensor, TRTTensor):
                input_tensor = get_trt_tensor(ctx, to_torch(input_tensor, None), name + "_input")

            scale = torch.divide(to_torch(amax, None), max_bound)

            axis = None
            if scale.numel() > 1:
                shape = list(scale.shape)
                axis = next(i for i, size in enumerate(shape) if size == scale.numel())
                scale = scale.flatten()
            scale = get_trt_tensor(ctx, scale, name + "_scale")

            quantize_layer = ctx.net.add_quantize(input_tensor, scale)
            if axis is not None:
                quantize_layer.axis = axis
            quantize_layer.set_output_type(0, dtype)
            set_layer_name(quantize_layer, target, name + "_quantize", source_ir)

            dequantize_layer = ctx.net.add_dequantize(quantize_layer.get_output(0), scale)
            if axis is not None:
                dequantize_layer.axis = axis
            dequantize_layer.precision = dtype
            set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
            return dequantize_layer.get_output(0)

    impl.quantize.quantize = quantize
    return {"patched": True, "sets_axis_for_per_channel": True}


def apply_torch_tensorrt_quantization_patches() -> dict:
    """Apply both Torch-TensorRT 2.7.0 fixes needed before compiling a Q/DQ graph."""
    return {
        "constant_folding": protect_quantize_op_from_constant_folding(),
        "converter": patch_quantize_converter_for_constant_weights(),
    }


def first_active_sample(waveform, *, block: int = 256, threshold: float = 0.05) -> int:
    """Index of the first block whose peak clears ``threshold`` of the file peak.

    EARS utterances open on a fraction of a second of room tone.  Calibrating on
    that stretch sets the activation amax from near-silence, so real speech then
    clips against it.  The failure is invisible in the report - the amax values
    look perfectly well-formed, they are just far too small.
    """
    import torch

    peak = float(waveform.abs().max())
    if peak <= 0.0 or waveform.shape[-1] < block:
        return 0
    blocks = waveform[0].abs().unfold(0, block, block).amax(dim=1)
    active = torch.nonzero(blocks >= threshold * peak)
    return int(active[0]) * block if active.numel() else 0


def load_calibration_audio(
    device,
    *,
    num_files: int,
    max_seconds: float,
    seed: int,
    split: str = "train",
    root: str = DEFAULT_CALIBRATION_ROOT,
):
    """Return one clean EARS excerpt per file, as separate calibration streams.

    STFTPR is phase retrieval: the model is conditioned on the magnitude of the
    *clean* spectrogram, so the clean side of the split is the right source.
    The default split is ``train``: activation amax has to cover the range of
    voices the engine will meet, and ``valid`` holds only two speakers against
    ``train``'s 99.  ``test`` is off limits either way - calibrating on it would
    leak the evaluation set into the quantization statistics.

    The excerpts stay separate rather than concatenated.  The caller replays each
    one through its own streaming state, so the frame budget is spread over every
    speaker instead of being consumed entirely by whichever file sorted first -
    which is what a concatenated stream truncated to ``max_frames`` does.
    """
    import random

    import torchaudio

    clean_root = Path(root) / split / "clean"
    files = sorted(str(path) for path in clean_root.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No calibration audio found under {clean_root}.")
    chosen = random.Random(seed).sample(files, min(num_files, len(files)))

    excerpts, names = [], []
    for path in chosen:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform[:, first_active_sample(waveform) :]
        excerpt = waveform[:, : int(max_seconds * 16000)]
        if excerpt.shape[-1] < 16000 // 4:  # under a quarter second is not worth a pass
            continue
        excerpts.append(excerpt.to(device))
        names.append(Path(path).name)
    if not excerpts:
        raise RuntimeError(f"No usable calibration audio under {clean_root}.")
    return excerpts, names


def run_real_audio_stream(
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
    """Drive ``forward_step`` over the real streaming STFTPR front-end.

    Mirrors ``experiments/streaming/eager.py``: sliding sqrt-Hann analysis,
    magnitude-only conditioning, then the Euler flow loop whose ``x_t`` depends
    on the previous model output.  The inputs the quantizers observe are
    therefore the ones production actually sees, which white noise is not.
    """
    import numpy as np
    import torch

    from experiments.core.tensors import pack_ri_channels
    from experiments.streaming.stft import (
        complex_to_ri_frame,
        compress_complex,
        compression_norm,
        frequency_bins,
        sqrt_hann_window,
    )

    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)
    hop = config.hop_length

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    flow_states = [module.init_state() for _ in range(steps)]
    t_tensors = [
        torch.full((1,), step_idx / steps, device=device, dtype=torch.float32)
        for step_idx in range(steps)
    ]

    total_frames = min(max_frames, audio.shape[-1] // hop)
    captured = None
    input_absmax = 0.0
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
                input_absmax = max(input_absmax, float(dnn_input.abs().max()))
                if capture_frame_index == frame_idx and step_idx == 0:
                    # numpy round-trips out of inference mode: an inference
                    # tensor cannot be fed to torch.export later on.
                    captured = (
                        torch.from_numpy(np.array(dnn_input.detach().cpu())).to(device),
                        torch.from_numpy(np.array(t_tensors[step_idx].detach().cpu())).to(device),
                    )
                velocity, flow_states[step_idx] = raw_step(
                    module, dnn_input, time_cond=t_tensors[step_idx], state=flow_states[step_idx]
                )
                x_t = x_t + velocity / steps

    return {
        "frames": total_frames,
        "solver_steps": steps,
        "freq_bins": frequency_bins(config),
        "backbone_calls": total_frames * steps,
        "input_absmax": input_absmax,
        "captured": captured,
    }


def measure_conv_macs(model, *, input_channels: int, input_freqs: int, device) -> list[dict]:
    """Return every causal conv's MAC count, in execution order.

    Execution order matters as much as the cost: a quantized region pays a
    reformat at each of its boundaries, so the layers we pick have to be
    adjacent in the graph, not merely expensive.
    """
    import torch

    from sgmse.backbones.streaming_unet import CausalConv2d, CausalDecoupledConv2d

    rows: list[dict] = []
    originals: list[tuple[Any, Any]] = []

    def instrument(name, module, original):
        def forward_step(x, *, state):
            out, next_state = original(x, state=state)
            batch, out_channels, out_freqs, out_frames = out.shape
            kh, kw = int(module.kernel_size[0]), int(module.kernel_size[1])
            rows.append(
                {
                    "name": name,
                    "macs": int(
                        batch
                        * out_channels
                        * out_freqs
                        * out_frames
                        * (module.in_channels // module.groups)
                        * kh
                        * kw
                    ),
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel": [kh, kw],
                    "freqs": out_freqs,
                }
            )
            return out, next_state

        return forward_step

    for name, module in model.named_modules():
        if isinstance(module, (CausalConv2d, CausalDecoupledConv2d)):
            originals.append((module, module.forward_step))
            module.forward_step = instrument(name, module, module.forward_step)

    raw_step = getattr(type(model).forward_step, "__wrapped__", type(model).forward_step)
    try:
        with torch.inference_mode():
            raw_step(
                model,
                torch.randn(1, input_channels, input_freqs, 1, device=device),
                time_cond=torch.rand(1, device=device),
                state=model.init_state(),
            )
    finally:
        for module, original in originals:
            module.forward_step = original
    return rows


def select_quantized_modules(rows: list[dict], *, scope: str, coverage: float = 0.8) -> dict:
    """Choose which convolutions to quantize, as one contiguous span.

    StreamFM's cost profile is flat -- the heaviest convolution is under 7% of
    total MACs -- so there is no hot layer to isolate.  What quantizing a subset
    can still buy is fewer INT8<->FP16 reformats, and those appear at the
    boundaries of the quantized region.  A single contiguous span therefore
    costs two boundaries whatever its length, whereas the same layers picked by
    rank alone would scatter boundaries throughout the graph.
    """
    total = sum(row["macs"] for row in rows)
    if scope == "all":
        return {
            "scope": "all",
            "selected": [row["name"] for row in rows],
            "macs_coverage": 1.0,
            "boundaries": 0,
        }

    if scope != "heavy_span":
        # Anything else is read as an explicit comma-separated pattern list.
        patterns = [part.strip() for part in scope.split(",") if part.strip()]
        selected = [row["name"] for row in rows if any(p in row["name"] for p in patterns)]
        covered = sum(row["macs"] for row in rows if row["name"] in set(selected))
        return {
            "scope": scope,
            "selected": selected,
            "macs_coverage": covered / total if total else 0.0,
        }

    # Shortest window of consecutive layers reaching the requested MAC coverage.
    target = coverage * total
    best: tuple[int, int] | None = None
    start = 0
    running = 0
    for end, row in enumerate(rows):
        running += row["macs"]
        while running - rows[start]["macs"] >= target:
            running -= rows[start]["macs"]
            start += 1
        if running >= target and (best is None or end - start < best[1] - best[0]):
            best = (start, end)
    if best is None:
        best = (0, len(rows) - 1)

    lo, hi = best
    window = rows[lo : hi + 1]
    return {
        "scope": "heavy_span",
        "requested_coverage": coverage,
        "selected": [row["name"] for row in window],
        "macs_coverage": sum(row["macs"] for row in window) / total if total else 0.0,
        "span": [lo, hi],
        "layers_total": len(rows),
        "boundaries": 2,
    }


def build_quant_cfg(base_cfg: dict, rows: list[dict], selected: list[str]) -> dict:
    """Disable the quantizers of every conv outside the selected span.

    ModelOpt applies its ``quant_cfg`` patterns in order, so appending the
    exclusions after the defaults leaves the selected layers untouched.
    """
    import copy

    cfg = copy.deepcopy(base_cfg)
    keep = set(selected)
    for row in rows:
        if row["name"] in keep:
            continue
        cfg["quant_cfg"][f"*{row['name']}*"] = {"enable": False}
    return cfg


def apply_int8_ptq(
    model,
    *,
    input_channels: int,
    input_freqs: int,
    calibration_steps: int,
    quant_format: str = "int8",
    quant_scope: str = "all",
    quant_coverage: float = 0.8,
    calibration_source: str = "audio",
    calibration_files: int = 16,
    calibration_seconds: float = 1.5,
    calibration_split: str = "train",
    calibration_solver_steps: tuple[int, ...] = (1, 5),
    calibration_seed: int = 0,
    calibration_root: str = DEFAULT_CALIBRATION_ROOT,
    stft_config: Any = None,
) -> tuple[Any, dict]:
    """Calibrate ModelOpt Q/DQ on real streaming audio and return (model, report).

    ``calibration_steps`` is a frame budget *per excerpt*, not per run: the total
    is ``files x steps x len(solver_steps)``.  Spreading the budget over many
    speakers matters more than length - the activation amax has to cover the
    loudest speaker in the split, not the first few frames of one utterance.

    ``calibration_source='noise'`` keeps the old white-noise loop as a control;
    it is not fit for a quality run - it puts the activation ranges about three
    orders of magnitude too wide.
    """
    import torch

    import modelopt.torch.quantization as mtq

    raw_step = getattr(type(model).forward_step, "__wrapped__", None)
    if raw_step is None:
        raise RuntimeError("TensorRT INT8 requires the raw CausalNCSNpp.forward_step implementation.")

    report: dict = {"source": calibration_source}
    fake_quant = install_pure_torch_fake_quant()
    registration = register_causal_conv_for_quantization()
    report["fake_quant_backend"] = fake_quant
    report["registration"] = registration

    if calibration_source == "audio":
        from experiments.streaming.stft import StreamingSTFTConfig

        config = stft_config if stft_config is not None else StreamingSTFTConfig()
        excerpts, names = load_calibration_audio(
            torch.device("cuda"),
            num_files=calibration_files,
            max_seconds=calibration_seconds,
            seed=calibration_seed,
            split=calibration_split,
            root=calibration_root,
        )
        report.update(
            {
                "split": calibration_split,
                "files": names,
                "seconds_per_file": calibration_seconds,
                "frames_per_file": calibration_steps,
                "solver_steps": list(calibration_solver_steps),
            }
        )

        def calibrate(module):
            # One pass per excerpt per NFE setting, each with its own streaming
            # state: the same engine serves both Euler1 and Euler5, whose t
            # values - and therefore whose activation ranges - differ.
            runs = []
            for steps in calibration_solver_steps:
                for name, excerpt in zip(names, excerpts):
                    run = run_real_audio_stream(
                        module,
                        raw_step,
                        excerpt,
                        config=config,
                        device=torch.device("cuda"),
                        steps=steps,
                        max_frames=calibration_steps,
                    )
                    runs.append(
                        {**{k: v for k, v in run.items() if k != "captured"}, "file": name}
                    )
            report["runs"] = runs
            # The headline diagnostic: if this sits far below what real speech
            # produces at inference, the activation ranges are clipping.
            report["input_absmax"] = max(run["input_absmax"] for run in runs)
            report["total_frames"] = sum(run["frames"] for run in runs)
            report["backbone_calls"] = sum(run["backbone_calls"] for run in runs)

    elif calibration_source == "noise":

        def calibrate(module):
            with torch.inference_mode():
                state = module.init_state()
                for _ in range(calibration_steps):
                    x = torch.randn(1, input_channels, input_freqs, 1, device="cuda")
                    t = torch.rand(1, device="cuda")
                    _out, state = raw_step(module, x, time_cond=t, state=state)

    else:
        raise ValueError("calibration_source must be 'audio' or 'noise'.")

    if quant_format == "int8":
        base_cfg = mtq.INT8_DEFAULT_CFG
    elif quant_format == "fp8":
        # FP8 needs Ada (sm89) or newer: L4/L40S yes, T4 and A100 no.
        base_cfg = mtq.FP8_DEFAULT_CFG
    else:
        raise ValueError("quant_format must be 'int8' or 'fp8'.")
    report["format"] = quant_format

    if quant_scope == "all":
        cfg = base_cfg
        report["selection"] = {"scope": "all"}
    else:
        rows = measure_conv_macs(
            model,
            input_channels=input_channels,
            input_freqs=input_freqs,
            device=torch.device("cuda"),
        )
        selection = select_quantized_modules(rows, scope=quant_scope, coverage=quant_coverage)
        cfg = build_quant_cfg(base_cfg, rows, selection["selected"])
        report["selection"] = {
            **{k: v for k, v in selection.items() if k != "selected"},
            "selected_count": len(selection["selected"]),
            "selected": selection["selected"],
        }

    quantized = mtq.quantize(model, cfg, forward_loop=calibrate)
    return quantized, report
