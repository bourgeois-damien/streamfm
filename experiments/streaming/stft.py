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


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def streaming_config_from_model_cfg(cfg) -> StreamingSTFTConfig:
    feature_cfg = cfg.model.feature_extractor
    return StreamingSTFTConfig(
        sample_rate=int(cfg.get("sampling_rate", 16000)),
        n_fft=int(feature_cfg.get("n_fft", 512)),
        hop_length=int(feature_cfg.get("hop_length", 256)),
        alpha=float(feature_cfg.get("alpha", 0.5)),
        beta=float(feature_cfg.get("beta", 1.0)),
        cut_highest_freqs=int(feature_cfg.get("cut_highest_freqs", 1)),
        sigma_y=_float_or_default(cfg.model.get("sigma_y", 0.25), 0.25),
        normalized_stft=bool(feature_cfg.get("normalized_stft", True)),
    )


def make_synthetic_audio(num_samples: int, sample_rate: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    audio = 0.05 * torch.sin(2 * math.pi * 220 * t)
    audio += 0.03 * torch.sin(2 * math.pi * 440 * t)
    audio += 0.005 * torch.randn_like(audio)
    return audio.unsqueeze(0)


def sqrt_hann_window(config: StreamingSTFTConfig, device: torch.device) -> torch.Tensor:
    return torch.hann_window(config.n_fft, periodic=True, device=device).sqrt()


def compression_norm(config: StreamingSTFTConfig) -> str | None:
    if config.normalized_stft:
        return "ortho"
    return None


def frequency_bins(config: StreamingSTFTConfig) -> int:
    return config.n_fft // 2 + 1 - config.cut_highest_freqs


def streaming_algorithmic_delay(config: StreamingSTFTConfig) -> int:
    """OLA lag in samples (n_fft - hop).
    Ignore for latency benches; do not ignore when scoring (misaligns refs, nukes SI-SDR).
    """
    return config.n_fft - config.hop_length


def streaming_num_frames(num_samples: int, config: StreamingSTFTConfig) -> int:
    # need one extra frame: delay pushes the last samples past a delay-free ceil()
    span = num_samples + streaming_algorithmic_delay(config)
    return -(-span // config.hop_length) + 1


def compensate_streaming_delay(
    audio: torch.Tensor, num_samples: int, config: StreamingSTFTConfig
) -> torch.Tensor:
    """Trim streamed audio so sample i matches input sample i (pad/truncate to num_samples)."""
    delay = streaming_algorithmic_delay(config)
    aligned = audio[:, delay:delay + num_samples]
    if aligned.shape[-1] < num_samples:
        aligned = torch.nn.functional.pad(aligned, (0, num_samples - aligned.shape[-1]))
    return aligned


def compress_complex(x: torch.Tensor, config: StreamingSTFTConfig, eps: float = 1e-8) -> torch.Tensor:
    if config.alpha == 1 and config.beta == 1:
        return x
    return config.beta * torch.polar(torch.abs(x).pow(config.alpha), torch.angle(x + eps))


def decompress_complex(x: torch.Tensor, config: StreamingSTFTConfig, eps: float = 1e-8) -> torch.Tensor:
    if config.alpha == 1 and config.beta == 1:
        return x
    return torch.polar((torch.abs(x) / config.beta).pow(1 / config.alpha), torch.angle(x + eps))


def complex_to_ri_frame(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).permute(0, 2, 1).unsqueeze(-1).contiguous()


def ri_frame_to_complex(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze(-1).permute(0, 2, 1).contiguous()
    return torch.view_as_complex(x)


def pad_cut_highest_freqs(x_complex: torch.Tensor, config: StreamingSTFTConfig) -> torch.Tensor:
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
