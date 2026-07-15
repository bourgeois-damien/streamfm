"""Streaming STFT framing helpers.

Sqrt-Hann analysis/synthesis windowing, complex compression/decompression,
real-imag frame packing and synthetic-audio generation shared by the streaming
pipelines. Configured through ``StreamingSTFTConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamingSTFTConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    alpha: float = 0.5
    beta: float = 1.0
    cut_highest_freqs: int = 1
    sigma_y: float = 0.25
    normalized_stft: bool = True


def make_synthetic_audio(num_samples: int, sample_rate: int, device: torch.device) -> torch.Tensor:
    """Create deterministic-length mono audio for pipeline smoke tests."""
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    audio = 0.05 * torch.sin(2 * math.pi * 220 * t)
    audio += 0.03 * torch.sin(2 * math.pi * 440 * t)
    audio += 0.005 * torch.randn_like(audio)
    return audio.unsqueeze(0)


def sqrt_hann_window(config: StreamingSTFTConfig, device: torch.device) -> torch.Tensor:
    """Create the square-root Hann window used for overlap-add."""
    return torch.hann_window(config.n_fft, periodic=True, device=device).sqrt()


def compression_norm(config: StreamingSTFTConfig) -> str | None:
    """Return the torch FFT normalization mode for this streaming config."""
    if config.normalized_stft:
        return "ortho"
    return None


def frequency_bins(config: StreamingSTFTConfig) -> int:
    """Return the number of model frequency bins after optional high-bin cut."""
    return config.n_fft // 2 + 1 - config.cut_highest_freqs


def compress_complex(x: torch.Tensor, config: StreamingSTFTConfig, eps: float = 1e-8) -> torch.Tensor:
    """Apply Stream.FM-style magnitude compression to a complex spectrum."""
    if config.alpha == 1 and config.beta == 1:
        return x
    return config.beta * torch.polar(torch.abs(x).pow(config.alpha), torch.angle(x + eps))


def decompress_complex(x: torch.Tensor, config: StreamingSTFTConfig, eps: float = 1e-8) -> torch.Tensor:
    """Invert the magnitude compression before waveform reconstruction."""
    if config.alpha == 1 and config.beta == 1:
        return x
    return torch.polar((torch.abs(x) / config.beta).pow(1 / config.alpha), torch.angle(x + eps))


def complex_to_ri_frame(x: torch.Tensor) -> torch.Tensor:
    """Convert a complex frequency frame to real/imaginary channel format."""
    return torch.view_as_real(x).permute(0, 2, 1).unsqueeze(-1).contiguous()


def ri_frame_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real/imaginary channel format back to a complex frame."""
    x = x.squeeze(-1).permute(0, 2, 1).contiguous()
    return torch.view_as_complex(x)


def pad_cut_highest_freqs(x_complex: torch.Tensor, config: StreamingSTFTConfig) -> torch.Tensor:
    """Restore removed high-frequency bins before inverse FFT."""
    if not config.cut_highest_freqs:
        return x_complex
    pad_shape = (x_complex.shape[0], config.cut_highest_freqs)
    if x_complex.ndim == 3:
        pad_shape = (*pad_shape, x_complex.shape[-1])
    pad = torch.zeros(
        pad_shape,
        device=x_complex.device,
        dtype=x_complex.dtype,
    )
    return torch.cat([x_complex, pad], dim=1)
