from typing import Optional
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt, matplotlib as mpl


class ComplexAbs(torch.nn.Module):
    """
    A PyTorch module that computes the absolute value of a complex tensor, ensuring the output is complex-valued
    (imaginary part is zero).

    Can e.g. be used as a post_Y_fn for a FlowModel with a phase retrieval task.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs() + 0j


class PhaselessMelAndBack(torch.nn.Module):
    """
    Takes a complex spectrogram (produced by an STFT with given window, window length, hop length)
    and feeds it through magnitude -> Mel compression -> Mel pseudoinverse,
    to get a 'mel-compressed' STFT magnitude spectrogram.
    We use torch / torchaudio functions where possible.
    """
    def __init__(self, n_mels: int, sample_rate: int,
                 f_min: float = 0.0, f_max = None,  # sample_rate//2
                 n_stft: int = 512,
                 norm: str = 'slaney', mel_scale: str = 'slaney',
                 alpha: float = 1.0, keep_sign: bool = False,
                 atol: Optional[float] = None, rtol: Optional[float] = None,
                 cut_highest_freqs: int = 0):
        super().__init__()
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max, n_stft=n_stft,
            norm=norm, mel_scale=mel_scale)
        self.alpha = alpha
        self.keep_sign = keep_sign
        self.cut_highest_freqs = cut_highest_freqs

        pseudoinverse = torch.linalg.pinv(self.mel_scale.fb, atol=atol, rtol=rtol)
        self.register_buffer('pseudoinverse', pseudoinverse)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mel = self.only_mel(X)
        return self.back_from_mel(mel)

    def back_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        spec_pinv = torch.matmul(mel.transpose(-1, -2), self.pseudoinverse).transpose(-1, -2)

        out = spec_pinv.abs()**self.alpha  # more info preserved with abs() than if we set coefs<0 to 0
        if self.keep_sign:
            out = torch.sign(spec_pinv) * out

        if self.cut_highest_freqs:
            out = out[..., :-self.cut_highest_freqs, :]

        return out + 0j

    def only_mel(self, X: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(X.abs()**(1/self.alpha))


class CutHighestFreqs(torch.nn.Module):
    """
    Cuts the highest frequency bins from a complex spectrogram.
    """
    def __init__(self, cut_highest_freqs: int):
        super().__init__()
        self.cut_highest_freqs = cut_highest_freqs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.cut_highest_freqs:
            X = X[..., :-self.cut_highest_freqs, :]
        return X


def ccmap_img(cimg, amp_tf=lambda a: a, inv=False, mult=True, cmap='hsv', prange=(-np.pi, np.pi)):
    """
    Maps a complex array to rgba values with information about amplitude and phase, by using
    `phase_cmap` ('twilight' by default) as the colormap for the phase information and mapping
    the normalized amplitudes to:

        - mult=False: The opacity (i.e., large amplitudes map to high saturation, when the background is white).
        - mult=True: The brightness, by multiplying all RGB channels with the normalized amplitudes.
    """
    cmp = mpl.colormaps.get_cmap(cmap)
    a = np.abs(cimg)
    p = np.angle(cimg)
    pn = (p - prange[0]) / (prange[1] - prange[0])
    phase = cmp(pn)
    if amp_tf is not None:
        a = amp_tf(a)
    a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

    if mult:
        rgba = phase
        fac = (1-a) if inv else a
        fac = fac[..., None]
        rgba[...,:3] *= fac
        rgba[...,3] = 1
    else:
        rgba = phase
        rgba[...,3] = (1-a) if inv else a
    return rgba