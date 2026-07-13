from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for the Stream.FM source layout."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current.parent


def ensure_repo_importable(repo_root: Path) -> None:
    """Put the repository root on sys.path for direct script execution."""
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def select_torch_device(name: str | None = "auto") -> "torch.device":
    """Resolve a CLI device name to the best available torch device."""
    torch = _torch()
    if name is None or name.lower() == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name.lower())


def sync_device(device: "torch.device") -> None:
    """Synchronize accelerator work around wall-clock timing."""
    torch = _torch()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_label(device: "torch.device") -> str:
    """Return a stable human-readable label for benchmark rows."""
    torch = _torch()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def normalize_model_memory_format(memory_format: str) -> str:
    """Normalize the model tensor memory-format option."""
    normalized = memory_format.lower().replace("-", "_")
    if normalized not in {"contiguous", "channels_last"}:
        raise ValueError("Unsupported memory format. Use 'contiguous' or 'channels_last'.")
    return normalized


def normalize_float32_matmul_precision(precision: str) -> str:
    """Normalize torch float32 matmul precision modes."""
    normalized = precision.lower().replace("-", "_")
    if normalized not in {"highest", "high", "medium"}:
        raise ValueError("Unsupported matmul precision. Use 'highest', 'high', or 'medium'.")
    return normalized


def torch_model_memory_format(memory_format: str):
    """Return the torch memory_format object for 4D model tensors."""
    torch = _torch()
    normalized = normalize_model_memory_format(memory_format)
    return torch.channels_last if normalized == "channels_last" else torch.contiguous_format


def apply_model_memory_format(module: Any, memory_format: str):
    """Apply a 4D convolution-friendly memory format to module parameters."""
    normalized = normalize_model_memory_format(memory_format)
    if normalized == "channels_last":
        return module.to(memory_format=torch_model_memory_format(normalized))
    return module


def format_model_tensor(tensor: "torch.Tensor", memory_format: str) -> "torch.Tensor":
    """Convert 4D model inputs/buffers to the requested memory format."""
    normalized = normalize_model_memory_format(memory_format)
    if normalized == "channels_last" and tensor.dim() == 4:
        return tensor.contiguous(memory_format=torch_model_memory_format(normalized))
    return tensor


def empty_model_tensor(
    shape,
    *,
    device,
    dtype,
    memory_format: str,
) -> "torch.Tensor":
    """Allocate a 4D model tensor with the requested memory layout."""
    torch = _torch()
    if normalize_model_memory_format(memory_format) == "channels_last" and len(shape) == 4:
        return torch.empty(tuple(shape), device=device, dtype=dtype, memory_format=torch.channels_last)
    return torch.empty(tuple(shape), device=device, dtype=dtype)


def pack_ri_channels(
    *frames: "torch.Tensor",
    memory_format: str = "contiguous",
    out: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Pack [real, imag] frame tensors like CausalNCSNpp's complex wrapper.

    Each input frame is shaped [B, 2, F, T] with channel order [real, imag].
    The full model receives complex tensors concatenated as [X, E, Y] and then
    internally converts them to [X.real, E.real, Y.real, X.imag, E.imag, Y.imag].
    DNN-only benchmarks must use the same channel layout.
    """
    if not frames:
        raise ValueError("At least one frame is required.")
    for frame in frames:
        if frame.dim() != 4 or frame.shape[1] != 2:
            raise ValueError("Expected every frame to have shape [B, 2, F, T].")

    if out is None:
        packed = _torch().cat(
            [*(frame[:, 0:1] for frame in frames), *(frame[:, 1:2] for frame in frames)],
            dim=1,
        )
        return format_model_tensor(packed, memory_format)

    n = len(frames)
    if out.shape[1] != 2 * n:
        raise ValueError(f"Output buffer has {out.shape[1]} channels, expected {2 * n}.")
    for idx, frame in enumerate(frames):
        out[:, idx : idx + 1].copy_(frame[:, 0:1])
        out[:, n + idx : n + idx + 1].copy_(frame[:, 1:2])
    return out


def forward_step(
    module: Any,
    x: "torch.Tensor",
    *,
    state: Any,
    time_cond: "torch.Tensor | None" = None,
    use_compiled: bool = False,
):
    """Call a streaming module step, bypassing torch.compile when requested."""
    if use_compiled:
        return module.forward_step(x, time_cond=time_cond, state=state)

    fn = getattr(module.forward_step, "__wrapped__", None)
    if fn is None:
        return module.forward_step(x, time_cond=time_cond, state=state)
    return fn(module, x, time_cond=time_cond, state=state)


def prepare_streaming_state(module: Any):
    """Allocate graph-capturable streaming state when supported."""
    if hasattr(module, "prepare_state"):
        return module.prepare_state()
    return module.init_state()


def zero_streaming_state(module: Any, state: Any) -> None:
    """Reset a streaming state in place when supported."""
    if hasattr(module, "zero_state"):
        module.zero_state(state)


def summarize_ms(times_ms: list[float]) -> dict[str, float]:
    """Summarize latency samples with mean and percentile metrics."""
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": values[len(values) // 2],
        "p90_ms": values[int(0.90 * (len(values) - 1))],
        "p99_ms": values[int(0.99 * (len(values) - 1))],
    }


def summarize_prefixed_ms(times_ms: list[float], prefix: str) -> dict[str, float]:
    """Summarize latency samples using namespaced metric keys."""
    values = sorted(times_ms)
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    p90_idx = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    p99_idx = min(len(values) - 1, math.ceil(0.99 * len(values)) - 1)
    return {
        f"{prefix}_mean_ms": sum(values) / len(values),
        f"{prefix}_p50_ms": values[len(values) // 2],
        f"{prefix}_p90_ms": values[p90_idx],
        f"{prefix}_p99_ms": values[p99_idx],
    }
