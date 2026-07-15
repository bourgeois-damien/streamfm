"""4D model-tensor memory-format handling and real/imag channel packing.

The streaming U-Net runs on 4D convolution tensors ``[B, C, F, T]``. These
helpers keep the channels-last memory format consistent between the model
parameters and the input/state buffers, and pack complex frames into the
real/imag channel layout the model expects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def normalize_model_memory_format(memory_format: str) -> str:
    """Normalize the model tensor memory-format option."""
    normalized = memory_format.lower().replace("-", "_")
    if normalized not in {"contiguous", "channels_last"}:
        raise ValueError("Unsupported memory format. Use 'contiguous' or 'channels_last'.")
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
