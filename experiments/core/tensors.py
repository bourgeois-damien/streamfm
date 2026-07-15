from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def normalize_model_memory_format(memory_format: str) -> str:
    normalized = memory_format.lower().replace("-", "_")
    if normalized not in {"contiguous", "channels_last"}:
        raise ValueError("Unsupported memory format. Use 'contiguous' or 'channels_last'.")
    return normalized


def torch_model_memory_format(memory_format: str):
    torch = _torch()
    normalized = normalize_model_memory_format(memory_format)
    return torch.channels_last if normalized == "channels_last" else torch.contiguous_format


def apply_model_memory_format(module: Any, memory_format: str):
    normalized = normalize_model_memory_format(memory_format)
    if normalized == "channels_last":
        return module.to(memory_format=torch_model_memory_format(normalized))
    return module


def format_model_tensor(tensor: "torch.Tensor", memory_format: str) -> "torch.Tensor":
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
    torch = _torch()
    if normalize_model_memory_format(memory_format) == "channels_last" and len(shape) == 4:
        return torch.empty(tuple(shape), device=device, dtype=dtype, memory_format=torch.channels_last)
    return torch.empty(tuple(shape), device=device, dtype=dtype)


def pack_ri_channels(
    *frames: "torch.Tensor",
    memory_format: str = "contiguous",
    out: "torch.Tensor | None" = None,
) -> "torch.Tensor":
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
        out[:, idx : idx + 1].copy_(frame[:, 0:1])  # real of frame idx
        out[:, n + idx : n + idx + 1].copy_(frame[:, 1:2])  # imag of frame idx
    return out
