"""Post-training INT8 quantization (PTQ) helpers for Stream.FM backbones.

Selectable components (comma-separated):

- ``linear``: dynamic INT8 on all ``nn.Linear`` (temb + Dense projections).
- ``conv``: static INT8 on plain ``nn.Conv2d`` only (e.g. SVD depthwise/pointwise).
- ``causal_conv``: static INT8 on ``CausalConv2d`` (streaming-aware wrapper).
- ``all``: ``linear,conv,causal_conv``.

Prefer ``execution=eager`` on CPU. ``dtype=fp32`` keeps the historical float
fallback, while ``dtype=fp16`` keeps unquantized modules and streaming state in
FP16 and inserts FP16/FP32 casts around quantized kernels. Speedups need a real
quantized backend (``qnnpack`` on ARM/macOS, ``x86`` on typical Linux x86_64).
"""

from __future__ import annotations

import platform
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PTQ_COMPONENTS = frozenset({"linear", "conv", "causal_conv"})
PTQ_METADATA_KEY = "streamfm_ptq_int8"


def parse_ptq_components(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a CLI/sweep PTQ component spec into a sorted unique tuple."""
    if spec is None or spec == "" or spec is False:
        return ()
    if isinstance(spec, bool):
        raise ValueError("ptq_int8 must be a component string, not a boolean.")
    if isinstance(spec, str):
        parts = [p.strip().lower().replace("-", "_") for p in spec.split(",")]
    else:
        parts = [str(p).strip().lower().replace("-", "_") for p in spec]
    parts = [p for p in parts if p]
    if not parts:
        return ()
    expanded: list[str] = []
    for part in parts:
        if part in {"none", "off", "false"}:
            continue
        if part == "all":
            expanded.extend(sorted(PTQ_COMPONENTS))
            continue
        if part not in PTQ_COMPONENTS:
            supported = ", ".join(sorted(PTQ_COMPONENTS | {"all", "none"}))
            raise ValueError(f"Unknown PTQ component {part!r}. Use one of: {supported}.")
        expanded.append(part)
    order = ("linear", "conv", "causal_conv")
    return tuple(c for c in order if c in set(expanded))


def default_quant_engine() -> str:
    """Pick a PyTorch quantized engine for the current machine."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if machine in {"arm64", "aarch64"} or system == "darwin":
        return "qnnpack"
    return "x86"


def set_quant_engine(engine: str | None = None) -> str:
    """Set and return ``torch.backends.quantized.engine``."""
    chosen = (engine or default_quant_engine()).lower()
    if chosen == "fbgemm":
        chosen = "x86"
    torch.backends.quantized.engine = chosen
    return chosen


def _qconfig(engine: str):
    from torch.ao.quantization import get_default_qconfig

    try:
        return get_default_qconfig(engine)
    except Exception:
        if engine == "x86":
            return get_default_qconfig("fbgemm")
        raise


def _clone_plain_conv2d(conv: nn.Conv2d) -> nn.Conv2d:
    plain = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        plain.weight.copy_(conv.weight.detach().float().cpu())
        if conv.bias is not None and plain.bias is not None:
            plain.bias.copy_(conv.bias.detach().float().cpu())
    return plain.eval()


def _static_quantize_conv_float_io(
    float_conv: nn.Conv2d,
    calibration_inputs: Sequence[torch.Tensor],
    *,
    engine: str,
) -> nn.Module:
    """Quantize a conv and keep a float-in / float-out wrapper around it.

    Eager-mode ``prepare``/``convert`` on a bare ``nn.Conv2d`` silently fails to
    convert on some PyTorch builds. QuantStub/DeQuantStub are required, and the
    resulting quantized conv expects quantized tensors — so we keep the stubs.
    """
    from torch.ao.quantization import DeQuantStub, QuantStub, convert, prepare

    if not calibration_inputs:
        raise ValueError("Static conv PTQ requires at least one calibration input.")

    plain = _clone_plain_conv2d(float_conv)

    class _QuantWrapper(nn.Module):
        def __init__(self, conv: nn.Conv2d):
            super().__init__()
            self.quant = QuantStub()
            self.conv = conv
            self.dequant = DeQuantStub()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dequant(self.conv(self.quant(x)))

    wrapped = _QuantWrapper(plain).eval()
    wrapped.qconfig = _qconfig(engine)
    prepared = prepare(wrapped, inplace=False)
    with torch.inference_mode():
        for sample in calibration_inputs:
            prepared(sample.detach().float().cpu().contiguous())
    converted = convert(prepared, inplace=False)
    if type(converted.conv).__module__.startswith("torch.nn.modules"):
        raise RuntimeError(
            "Static PTQ convert left a float nn.Conv2d behind; "
            "expected torch.ao.nn.quantized.Conv2d. Check quantized engine support."
        )
    return converted


class QuantizedFallbackWrapper(nn.Module):
    """Run a float-I/O quantized module behind an explicit fallback boundary.

    PyTorch eager quantized kernels consume and produce FP32 tensors at their
    public boundary.  The rest of a mixed model may nevertheless use FP16: cast
    the input to FP32 immediately before the quantized module and cast its output
    back to the requested fallback dtype.
    """

    def __init__(self, quantized_module: nn.Module, fallback_dtype: torch.dtype):
        super().__init__()
        if fallback_dtype not in {torch.float16, torch.float32}:
            raise ValueError("INT8 fallback dtype must be torch.float16 or torch.float32.")
        self.quantized_module = quantized_module
        self.fallback_dtype = fallback_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.quantized_module(x.float())
        return output.to(dtype=self.fallback_dtype)


class QuantizedCausalConv2d(nn.Module):
    """Streaming-capable float I/O wrapper around quantized conv + stubs."""

    def __init__(
        self,
        float_conv: nn.Module,
        quantized_wrapper: nn.Module,
        *,
        fallback_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if fallback_dtype not in {torch.float16, torch.float32}:
            raise ValueError("INT8 fallback dtype must be torch.float16 or torch.float32.")
        self.time_padding = tuple(float_conv.time_padding)
        self.pad_freq = int(float_conv.pad_freq)
        self.dilation = tuple(float_conv.dilation)
        self.stride = tuple(float_conv.stride)
        self.kernel_size = tuple(float_conv.kernel_size)
        self.in_channels = int(float_conv.in_channels)
        self.out_channels = int(float_conv.out_channels)
        self.depthwise_separable = bool(getattr(float_conv, "depthwise_separable", False))
        self.Tbuf = int(float_conv.Tbuf)
        self.fallback_dtype = fallback_dtype
        # Keep quant/conv/dequant: quantized conv kernels require quantized inputs.
        self.quant = quantized_wrapper.quant
        self.qconv = quantized_wrapper.conv
        self.dequant = quantized_wrapper.dequant
        if self.depthwise_separable:
            raise NotImplementedError(
                "PTQ causal_conv does not support depthwise_separable CausalConv2d; "
                "use component 'conv' after SVD compression instead."
            )
        _register_quantized_causal_as_streaming()

    def _run_qconv(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dequant(self.qconv(self.quant(x.float())))
        return output.to(dtype=self.fallback_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_padding)
        return self._run_qconv(x)

    def init_state(self, input_freqs: int) -> tuple:
        weight = self.qconv.weight() if callable(self.qconv.weight) else self.qconv.weight
        device = weight.device
        xbuf_shape = (self.in_channels, input_freqs, self.Tbuf)
        return (torch.zeros(xbuf_shape, dtype=self.fallback_dtype, device=device),)

    def forward_step(self, x: torch.Tensor, *, state: tuple) -> tuple[torch.Tensor, tuple]:
        xbuf, = state
        if xbuf.device != x.device or xbuf.dtype != self.fallback_dtype:
            xbuf = xbuf.to(device=x.device, dtype=self.fallback_dtype)
        xbuf = xbuf.clone()
        xbuf[..., :-1] = xbuf[..., 1:]
        xbuf[..., :, -1] = x[0, :, :, 0]
        xbuf_in = xbuf.view(1, x.shape[1], x.shape[2], -1)
        h = self._run_qconv(xbuf_in)
        return h, (xbuf,)


_QUANTIZED_CAUSAL_REGISTERED = False


def _register_quantized_causal_as_streaming() -> None:
    """So parent ResNet ``init_state`` keeps treating replaced convs as streaming modules."""
    global _QUANTIZED_CAUSAL_REGISTERED
    if _QUANTIZED_CAUSAL_REGISTERED:
        return
    from sgmse.backbones.streaming_unet import CausalStreamingModule

    CausalStreamingModule.register(QuantizedCausalConv2d)
    _QUANTIZED_CAUSAL_REGISTERED = True

def _eager_forward_step(module: nn.Module, x: torch.Tensor, time_cond, state):
    """Call streaming forward_step, bypassing ``@torch.compile`` when present."""
    bound = module.forward_step
    raw = getattr(bound, "__wrapped__", None)
    if raw is None:
        return bound(x, time_cond=time_cond, state=state)
    return raw(module, x, time_cond=time_cond, state=state)


def _collect_module_inputs(
    root: nn.Module,
    targets: list[nn.Module],
    run_fn: Callable[[], None],
) -> dict[int, list[torch.Tensor]]:
    buckets: dict[int, list[torch.Tensor]] = {id(m): [] for m in targets}

    def make_hook(module: nn.Module):
        def _hook(_module, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if torch.is_tensor(tensor):
                buckets[id(module)].append(tensor.detach().float().cpu().contiguous())

        return _hook

    handles = [module.register_forward_pre_hook(make_hook(module)) for module in targets]
    try:
        run_fn()
    finally:
        for handle in handles:
            handle.remove()
    return buckets


def _default_calibration_runner(module: nn.Module, *, steps: int) -> Callable[[], None]:
    """Build a calibration callable.

    Prefer the offline ``forward`` path so ``nn.Module`` pre-hooks on
    ``CausalConv2d.forward`` fire (``forward_step`` bypasses those hooks).
    Fall back to eager ``forward_step`` when ``forward`` is unavailable.
    """

    def _run() -> None:
        module.eval()
        channels = int(getattr(module, "input_channels", 4))
        freqs = int(getattr(module, "input_freqs", 256))
        cpu_module = module.cpu()
        with torch.inference_mode():
            for _ in range(max(1, int(steps))):
                t = torch.rand(1, dtype=torch.float32) * 0.99 + 0.01
                if callable(getattr(cpu_module, "forward", None)):
                    # T>1 so causal time padding / convs see a short context window.
                    x = torch.randn(1, channels, freqs, 8, dtype=torch.float32)
                    cpu_module(x, time_cond=t)
                    continue
                if hasattr(cpu_module, "forward_step") and hasattr(cpu_module, "init_state"):
                    state = cpu_module.init_state()
                    for _frame in range(8):
                        x = torch.randn(1, channels, freqs, 1, dtype=torch.float32)
                        _, state = _eager_forward_step(cpu_module, x, t, state)
                    continue
                x = torch.randn(1, channels, freqs, 8, dtype=torch.float32)
                cpu_module(x)

    return _run


def _replace_modules(root: nn.Module, mapping: dict[int, nn.Module]) -> nn.Module:
    for name, child in list(root.named_modules()):
        if id(child) not in mapping:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root if parent_name == "" else root.get_submodule(parent_name)
        setattr(parent, attr, mapping[id(child)])
    return root


def _cast_float_fallback_state_(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Cast ordinary floating state while preserving quantized subtrees."""
    if dtype == torch.float32:
        return module
    if dtype != torch.float16:
        raise ValueError("INT8 fallback dtype must be torch.float16 or torch.float32.")

    def cast_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor

    def visit(current: nn.Module) -> None:
        if isinstance(current, (QuantizedFallbackWrapper, QuantizedCausalConv2d)):
            return
        # Apply only to parameters/buffers owned by this module. Recursing with
        # Module.half() would also rewrite quantizer scale buffers and packed
        # quantized submodules.
        current._apply(cast_tensor, recurse=False)
        for child in current.children():
            visit(child)

    visit(module)
    return module


def apply_dynamic_linear_ptq_(
    module: nn.Module,
    *,
    fallback_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Replace all ``nn.Linear`` children with dynamically quantized Linears."""
    quantized = torch.ao.quantization.quantize_dynamic(
        module,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    src = dict(quantized.named_modules())
    mapping: dict[int, nn.Module] = {}
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear) or name not in src:
            continue
        src_mod = src[name]
        # PyTorch keeps the class __name__ as "Linear"; detect via module path.
        if type(src_mod).__module__.startswith("torch.ao.nn.quantized"):
            mapping[id(child)] = (
                QuantizedFallbackWrapper(src_mod, fallback_dtype)
                if fallback_dtype != torch.float32
                else src_mod
            )
    return _replace_modules(module, mapping)


def apply_static_plain_conv_ptq_(
    module: nn.Module,
    *,
    engine: str,
    calibration_runner: Callable[[], None],
    precollected_inputs: dict[int, list[torch.Tensor]] | None = None,
    fallback_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Statically quantize exact ``nn.Conv2d`` modules (not CausalConv2d subclasses)."""
    from sgmse.backbones.streaming_unet import CausalConv2d

    targets = [m for m in module.modules() if type(m) is nn.Conv2d and not isinstance(m, CausalConv2d)]
    if not targets:
        return module
    inputs = precollected_inputs if precollected_inputs is not None else _collect_module_inputs(
        module, targets, calibration_runner
    )
    replacements: dict[int, nn.Module] = {}
    for target in targets:
        samples = inputs.get(id(target), [])
        if not samples:
            c_in, _, k_h, k_w = target.weight.shape
            samples = [torch.randn(1, c_in, max(8, k_h * 4), max(4, k_w * 4))]
        quantized = _static_quantize_conv_float_io(target, samples[:64], engine=engine)
        replacements[id(target)] = (
            QuantizedFallbackWrapper(quantized, fallback_dtype)
            if fallback_dtype != torch.float32
            else quantized
        )
    return _replace_modules(module, replacements)


def apply_static_causal_conv_ptq_(
    module: nn.Module,
    *,
    engine: str,
    calibration_runner: Callable[[], None],
    precollected_inputs: dict[int, list[torch.Tensor]] | None = None,
    fallback_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Statically quantize ``CausalConv2d`` modules with a streaming wrapper."""
    from sgmse.backbones.streaming_unet import CausalConv2d

    targets = [m for m in module.modules() if isinstance(m, CausalConv2d)]
    if not targets:
        return module
    inputs = precollected_inputs if precollected_inputs is not None else _collect_module_inputs(
        module, targets, calibration_runner
    )
    replacements: dict[int, nn.Module] = {}
    for target in targets:
        samples = inputs.get(id(target), [])
        if not samples:
            samples = [torch.randn(1, int(target.in_channels), 32, max(4, int(target.Tbuf)))]
        # CausalConv2d.forward pads time before the inner conv; hooks see pre-pad inputs.
        padded = [F.pad(sample, tuple(target.time_padding)) for sample in samples[:64]]
        qwrap = _static_quantize_conv_float_io(target, padded, engine=engine)
        replacements[id(target)] = QuantizedCausalConv2d(
            target,
            qwrap,
            fallback_dtype=fallback_dtype,
        )
    return _replace_modules(module, replacements)


def apply_ptq_int8_(
    module: nn.Module,
    components: str | Sequence[str] | None,
    *,
    engine: str | None = None,
    calib_steps: int = 32,
    calibration_runner: Callable[[], None] | None = None,
    fallback_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Apply selected INT8 PTQ transforms and return the (CPU) module."""
    from sgmse.backbones.streaming_unet import CausalConv2d

    selected = parse_ptq_components(components)
    if not selected:
        return module
    if fallback_dtype not in {torch.float16, torch.float32}:
        raise ValueError("INT8 fallback dtype must be torch.float16 or torch.float32.")

    param = next(module.parameters(), None)
    if param is not None and param.dtype != torch.float32:
        print(
            f"Warning: PTQ INT8 expects float32 weights; casting from {param.dtype} "
            f"to float32 before calibration/convert (stacked approximation).",
            flush=True,
        )
        module = module.float()

    chosen_engine = set_quant_engine(engine)
    module = module.cpu().eval()
    runner = calibration_runner or _default_calibration_runner(module, steps=calib_steps)

    # Collect static-conv activations on the still-float model, then replace.
    # Re-running calibration after partial quantized replacement breaks (float into qconv).
    needs_plain = "conv" in selected
    needs_causal = "causal_conv" in selected
    plain_targets = [
        m for m in module.modules() if type(m) is nn.Conv2d and not isinstance(m, CausalConv2d)
    ] if needs_plain else []
    causal_targets = [
        m for m in module.modules() if isinstance(m, CausalConv2d)
    ] if needs_causal else []
    precollected: dict[int, list[torch.Tensor]] | None = None
    if plain_targets or causal_targets:
        precollected = _collect_module_inputs(module, plain_targets + causal_targets, runner)

    if "linear" in selected:
        module = apply_dynamic_linear_ptq_(module, fallback_dtype=fallback_dtype)
    if needs_plain:
        module = apply_static_plain_conv_ptq_(
            module,
            engine=chosen_engine,
            calibration_runner=runner,
            precollected_inputs=precollected,
            fallback_dtype=fallback_dtype,
        )
    if needs_causal:
        module = apply_static_causal_conv_ptq_(
            module,
            engine=chosen_engine,
            calibration_runner=runner,
            precollected_inputs=precollected,
            fallback_dtype=fallback_dtype,
        )

    module = _cast_float_fallback_state_(module, fallback_dtype)

    module._streamfm_ptq_int8 = {  # type: ignore[attr-defined]
        "components": list(selected),
        "engine": chosen_engine,
        "calib_steps": int(calib_steps),
        "fallback_dtype": str(fallback_dtype).replace("torch.", ""),
    }
    return module.eval()


def ptq_transform_fn(
    components: str | Sequence[str] | None,
    *,
    engine: str | None = None,
    calib_steps: int = 32,
    fallback_dtype: torch.dtype = torch.float32,
) -> Callable[[nn.Module], nn.Module]:
    """Return a ``transform_backbone_``-compatible PTQ transform."""

    def _transform(backbone: nn.Module) -> nn.Module:
        return apply_ptq_int8_(
            backbone,
            components,
            engine=engine,
            calib_steps=calib_steps,
            fallback_dtype=fallback_dtype,
        )

    return _transform


def describe_ptq(module: nn.Module) -> dict[str, Any] | None:
    meta = getattr(module, "_streamfm_ptq_int8", None)
    return dict(meta) if isinstance(meta, dict) else None
