"""Torch device selection, synchronization, and compute-precision helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


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


def normalize_float32_matmul_precision(precision: str) -> str:
    """Normalize torch float32 matmul precision modes."""
    normalized = precision.lower().replace("-", "_")
    if normalized not in {"highest", "high", "medium"}:
        raise ValueError("Unsupported matmul precision. Use 'highest', 'high', or 'medium'.")
    return normalized
