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
    sample_rate: int = 16000  # Hz
    n_fft: int = 512  # analysis window length in samples
    hop_length: int = 256  # samples between frames -> frame budget = 1000*hop/sr ms
    alpha: float = 0.5  # magnitude compression exponent (|x|^alpha)
    beta: float = 1.0  # compression output scale
    cut_highest_freqs: int = 1  # bins dropped from the top: 257 -> 256 (power of two for the U-Net)
    sigma_y: float = 0.25  # STFTPR prior noise level around the conditioning frame
    normalized_stft: bool = True  # use "ortho" FFT normalization (matches training)


def make_synthetic_audio(num_samples: int, sample_rate: int, device: torch.device) -> torch.Tensor:
    """Create mono test audio [1, num_samples]: two tones (220/440 Hz) plus light noise.

    Used when a benchmark needs "real" pipeline input but no audio file was
    given; content only has to be plausible, not meaningful.
    """
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    audio = 0.05 * torch.sin(2 * math.pi * 220 * t)
    audio += 0.03 * torch.sin(2 * math.pi * 440 * t)
    audio += 0.005 * torch.randn_like(audio)
    return audio.unsqueeze(0)


def sqrt_hann_window(config: StreamingSTFTConfig, device: torch.device) -> torch.Tensor:
    """Create the square-root Hann window used for overlap-add.

    Sqrt because the window is applied twice (analysis and synthesis), so the
    effective window is a full Hann — which sums to a constant at 50% overlap
    and gives perfect reconstruction.
    """
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
    """Apply Stream.FM-style magnitude compression to a complex spectrum.

    beta * |x|^alpha with the phase kept: speech magnitudes are heavy-tailed,
    and alpha < 1 flattens their dynamic range into something a DNN handles
    well. ``eps`` keeps angle() defined at exact zeros. Must mirror training.
    """
    if config.alpha == 1 and config.beta == 1:
        return x
    return config.beta * torch.polar(torch.abs(x).pow(config.alpha), torch.angle(x + eps))


def decompress_complex(x: torch.Tensor, config: StreamingSTFTConfig, eps: float = 1e-8) -> torch.Tensor:
    """Invert the magnitude compression before waveform reconstruction."""
    if config.alpha == 1 and config.beta == 1:
        return x
    return torch.polar((torch.abs(x) / config.beta).pow(1 / config.alpha), torch.angle(x + eps))


def complex_to_ri_frame(x: torch.Tensor) -> torch.Tensor:
    """Convert a complex frame [B, F] to model layout [B, 2, F, T=1] (ch 0=real, 1=imag)."""
    return torch.view_as_real(x).permute(0, 2, 1).unsqueeze(-1).contiguous()


def ri_frame_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert model layout [B, 2, F, T=1] back to a complex frame [B, F]."""
    x = x.squeeze(-1).permute(0, 2, 1).contiguous()
    return torch.view_as_complex(x)


def pad_cut_highest_freqs(x_complex: torch.Tensor, config: StreamingSTFTConfig) -> torch.Tensor:
    """Re-append the dropped top bins as zeros: irfft needs all n_fft//2+1 bins.

    The model works on 256 bins (top bin cut for a power-of-two F); the lost
    content near Nyquist is negligible for speech.
    """
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
