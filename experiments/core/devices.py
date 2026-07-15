from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def select_torch_device(name: str | None = "auto") -> "torch.device":
    """Map a CLI device name to a torch.device. auto prefers MPS, then CUDA, then CPU."""
    torch = _torch()
    if name is None or name.lower() == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name.lower())


def sync_device(device: "torch.device") -> None:
    """Wait for pending GPU/MPS work.
    Bracket wall-clock timings with this or you measure queue submit, not compute. No-op on CPU.
    """
    torch = _torch()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_label(device: "torch.device") -> str:
    torch = _torch()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def normalize_float32_matmul_precision(precision: str) -> str:
    normalized = precision.lower().replace("-", "_")
    if normalized not in {"highest", "high", "medium"}:
        raise ValueError("Unsupported matmul precision. Use 'highest', 'high', or 'medium'.")
    return normalized


def normalize_tf32_mode(mode: str) -> str:
    normalized = mode.lower().replace("-", "_")
    if normalized not in {"auto", "on", "off"}:
        raise ValueError("Unsupported TF32 mode. Use 'auto', 'on', or 'off'.")
    return normalized
