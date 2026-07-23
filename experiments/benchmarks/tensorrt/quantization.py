from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION_ROOT = "/data/datasets/EARS-WHAM_v2_16k"


def install_pure_torch_fake_quant() -> dict:
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
        zero_amax = amax <= (1.0 / (1 << 24))
        scale = max_bound / amax.masked_fill(zero_amax, 1.0)
        scale = scale.masked_fill(zero_amax, 0.0)

        outputs = torch.clamp((inputs * scale).round(), min_bound, max_bound)
        return outputs / scale.masked_fill(zero_amax, 1.0)

    tensor_quant.fake_quant_impl = fake_quant_impl
    return {"installed": True, "reason": "modelopt_cuda_ext unavailable (no CUDA_HOME/nvcc)"}


def register_causal_conv_for_quantization() -> dict:
    """Register CausalConv2d with ModelOpt.
    Without this, Conv2d subclasses are skipped and the engine quietly stays unquantized.
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
    """Stop Torch-TensorRT 2.7 from constant-folding weight Q/DQ into float (otherwise INT8 never sticks)."""
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
    return {
        "constant_folding": protect_quantize_op_from_constant_folding(),
        "converter": patch_quantize_converter_for_constant_weights(),
    }


def first_active_sample(waveform, *, block: int = 256, threshold: float = 0.05) -> int:
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
    total = sum(row["macs"] for row in rows)
    if scope == "all":
        return {
            "scope": "all",
            "selected": [row["name"] for row in rows],
            "macs_coverage": 1.0,
            "boundaries": 0,
        }

    if scope != "heavy_span":
        patterns = [part.strip() for part in scope.split(",") if part.strip()]
        selected = [row["name"] for row in rows if any(p in row["name"] for p in patterns)]
        covered = sum(row["macs"] for row in rows if row["name"] in set(selected))
        return {
            "scope": scope,
            "selected": selected,
            "macs_coverage": covered / total if total else 0.0,
        }

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
    """Calibrate ModelOpt Q/DQ on real streaming audio; returns (model, report).
    calibration_steps is per excerpt (total ≈ files x steps x solver lengths) - prefer many speakers over long clips.
    calibration_source='noise' is only a control; ranges end up way too wide for quality runs.
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
