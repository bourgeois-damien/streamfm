import typing
from typing import List

from einops import rearrange
import torch
from torch import nn
from scipy.signal import windows
import torchaudio, torchaudio.transforms

from ._pesq import PesqLoss as TorchPesqLossImpl
from .shared import Loss

EPS = 1e-12


def magnitude_clamp(x: torch.Tensor, *args, eps: float, **kwargs):
    x_abs = x.abs()
    x_abs = torch.clamp(x_abs, *args, **kwargs)
    return torch.polar(x_abs, (x + eps).angle())


class FeatureDomainLoss(Loss):
    @property
    def domain(self):
        return 'feature'

    @property
    def name(self):
        if isinstance(self.loss_fn, nn.L1Loss):
            return "feat_l1"
        elif isinstance(self.loss_fn, nn.MSELoss):
            return "feat_l2"
        else:
            return f"feat_{self.loss_fn.__class__.__name__.lower()}"

    def __init__(self, loss_fn: typing.Callable, abs_only: bool = False, amp_comp: float = 1.0, amp_log: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.abs_only = abs_only
        self.amp_comp = amp_comp
        self.amp_log = amp_log

        if self.amp_log and not self.abs_only:
            raise ValueError("amp_log=True requires abs_only=True.")

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        if self.abs_only:
            x, xhat = x.abs(), xhat.abs()
            if self.amp_comp != 1.0:
                x = x.pow(self.amp_comp)
                xhat = xhat.pow(self.amp_comp)
            if self.amp_log:
                x = torch.log(x + EPS)
                xhat = torch.log(xhat + EPS)

        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            if self.amp_comp != 1.0:
                x = torch.polar(x.abs().pow(self.amp_comp), (x + EPS).angle())
                xhat = torch.polar(xhat.abs().pow(self.amp_comp), (xhat + EPS).angle())

            x = torch.view_as_real(x)
            xhat = torch.view_as_real(xhat)

        return self.loss_fn(x, xhat)


class TimeDomainLoss(Loss):
    @property
    def domain(self):
        return 'time'

    @property
    def name(self):
        if isinstance(self.loss_fn, nn.L1Loss):
            return "time_l1"
        elif isinstance(self.loss_fn, nn.MSELoss):
            return "time_l2"
        else:
            return f"time_{self.loss_fn.__class__.__name__.lower()}"

    def __init__(self, loss_fn: typing.Callable, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        return self.loss_fn(x, xhat)


class MultiScaleSTFTLoss(Loss):
    """
    Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py

    Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-8
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation adapted from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """
    def __init__(
        self,
        window_lengths: List[int],
        loss_fn: typing.Callable,
        clamp_eps: float = 1e-8,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        match_stride: bool = False,
        window_type: str = 'hann',
        complex_weight: float = 0.0,
        complex_amplitude_compression: float = 1.0,
        pcs_weight: float = 0.0,
        log1p_weight: float = 0.0,
        n_hops: int = 4,
        sampling_rate: int = 48000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # register buffers for windows
        for w in window_lengths:
            self.register_buffer(
                f"window_{w}",
                torch.from_numpy(getattr(windows, window_type)(w)).to(torch.float32),
            )

        self.stft_params = [
            dict(
                window_length=w,
                hop_length=w // n_hops,
                match_stride=match_stride,
            )
            for w in window_lengths
        ]
        self.sampling_rate = sampling_rate
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.log1p_weight = log1p_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.pow = pow
        self.complex_weight = complex_weight
        self.complex_amplitude_compression = complex_amplitude_compression
        self.pcs_weight = pcs_weight

        if self.pcs_weight > 0:
            for i, s in enumerate(self.stft_params):
                win_dur = s['window_length'] / self.sampling_rate
                hop_dur = s['hop_length'] / self.sampling_rate
                if not 28e-3 <= win_dur <= 36e-3:
                    raise ValueError(f"Window duration {win_dur*1000:.2f}ms is outside the range (28-36ms) for which"
                                    f"our PCS is approximately well defined. Choose something close to 32ms, ideally.")

                PCS = torch.ones((s['window_length']//2+1, 1))      # Perceptual Contrast Stretching
                PCS[0:3] = 1
                PCS[3:6] = 1.070175439
                PCS[6:9] = 1.182456140
                PCS[9:12] = 1.287719298
                PCS[12:138] = 1.4       # Pre Set
                PCS[138:166] = 1.322807018
                PCS[166:200] = 1.238596491
                PCS[200:241] = 1.161403509
                PCS[241:257] = 1.077192982
                PCS[257:] = 1.077192982  # extrapolate last band (-9500Hz) all the way to Nyquist w/ zero-order hold
                self.register_buffer(f'PCS_{i}', PCS)

    @property
    def domain(self):
        return 'time'

    @property
    def name(self):
        if isinstance(self.loss_fn, nn.L1Loss):
            return "multiscale_stft_l1"
        elif isinstance(self.loss_fn, nn.MSELoss):
            return "multiscale_stft_l2"
        else:
            return f"multiscale_stft_{self.loss_fn.__class__.__name__.lower()}"

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Reference signal
        xhat : torch.Tensor
            Estimate signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for i, s in enumerate(self.stft_params):
            stftkw = dict(
                n_fft=s['window_length'], hop_length=s['hop_length'],
                window=self.__getattr__(f"window_{s['window_length']}"),
                return_complex=True)
            X = torch.stft(rearrange(x, "b c t -> (b c t)"), **stftkw)
            Xhat = torch.stft(rearrange(xhat, "b c t -> (b c t)"), **stftkw)

            if self.log_weight > 0:
                loss += self.log_weight * self.loss_fn(
                    X.abs().clamp(min=self.clamp_eps).pow(self.pow).log10(),
                    Xhat.abs().clamp(min=self.clamp_eps).pow(self.pow).log10()
                )
            if self.log1p_weight > 0:
                loss += self.log1p_weight * self.loss_fn(
                    (1 + X.abs()).log10(),
                    (1 + Xhat.abs()).log10()
                )
            if self.mag_weight > 0:
                loss += self.mag_weight * self.loss_fn(X.abs(), Xhat.abs())
            if self.complex_weight > 0:
                # this addition avoids unstable backprop through angle()
                Xc = X + 1e-12
                Xhatc = Xhat + 1e-12
                # apply complex amplitude compression (may be 1.0, as the default)
                Xc = Xc.abs()**self.complex_amplitude_compression * torch.exp(1j * Xc.angle())
                Xhatc = Xhatc.abs()**self.complex_amplitude_compression * torch.exp(1j * Xhatc.angle())
                # compare real and imaginary part separately -- torch doesn't like complex for MSE for some reason
                loss += self.complex_weight * (self.loss_fn(Xc.real, Xhatc.real) + self.loss_fn(Xc.imag, Xhatc.imag))
            if self.pcs_weight > 0:
                Xc = X + 1e-12
                Xhatc = Xhat + 1e-12
                PCS = self.__getattr__(f'PCS_{i}')
                Xc = torch.polar(PCS * torch.log1p(Xc.abs()), Xc.angle())
                Xhatc = torch.polar(PCS * torch.log1p(Xhatc.abs()), Xhatc.angle())
                loss += self.pcs_weight * (self.loss_fn(X.real, Xhat.real) + self.loss_fn(X.imag, Xhat.imag))
        return loss / len(self.stft_params)


class MultiScaleMelSpectrogramLoss(Loss):
    """
    Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py

    Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Uses a Hann window.

    Parameters
    ----------
    sampling_rate : int
        Sampling rate of the audio signals
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    mel_fmin : List[float], optional
        Minimum frequency for each mel spectrogram, by default [0.0, 0.0]
    mel_fmax : List[float], optional
        Maximum frequency for each mel spectrogram, by default [None, None]

    Implementation adapted from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        power: float = 2.0,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(n_mels) == len(window_lengths)
        assert len(mel_fmin) == len(window_lengths)
        assert len(mel_fmax) == len(window_lengths)
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.power = power

        self.mel_spec_modules = torch.nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=w,
                win_length=w,
                hop_length=w // 4,
                n_mels=n_mels[i],
                f_min=mel_fmin[i],
                f_max=mel_fmax[i],
                power=power,
                norm='slaney',  # default in descriptinc audiotools (through librosa)
                # hann window is already the torchaudio default
            )
            for i, w in enumerate(window_lengths)
        ])

    @property
    def domain(self):
        return 'time'

    @property
    def name(self):
        if isinstance(self.loss_fn, nn.L1Loss):
            return "multiscale_mel_l1"
        elif isinstance(self.loss_fn, nn.MSELoss):
            return "multiscale_mel_l2"
        else:
            return f"multiscale_mel_{self.loss_fn.__class__.__name__.lower()}"

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for mel_spec_module in self.mel_spec_modules:
            X_mels = mel_spec_module(x)
            Xhat_mels = mel_spec_module(xhat)

            if self.log_weight > 0:
                loss += self.log_weight * self.loss_fn(
                    X_mels.clamp(self.clamp_eps).log10(),
                    Xhat_mels.clamp(self.clamp_eps).log10(),
                )
            if self.mag_weight > 0:
                loss += self.mag_weight * self.loss_fn(X_mels, Xhat_mels)
        return loss / len(self.mel_spec_modules)


class PESQLoss(Loss):
    """
    Based on https://github.com/audiolabs/torch-pesq/blob/main/torch_pesq/

    MIT License to audiolabs, 2022.

    Computes a differentiable PESQ approximation to be used as a loss function.
    """

    @property
    def domain(self):
        return 'time'

    @property
    def name(self):
        return "pesq"

    def __init__(
        self,
        sampling_rate: int,  # input sampling rate, will resample to 16 kHz if needed
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        # torch_pesq.PesqLoss uses their own resampler but with the terrible torchaudio default
        # (lowpass_filter_width=6) which can lead to significant aliasing. So we use our own resampler
        # and tell PesqLoss that its inputs are already at 16 kHz (which they will be).
        self.resampler = torchaudio.transforms.Resample(self.sampling_rate, 16000, lowpass_filter_width=64)
        self.pesq_loss_impl = TorchPesqLossImpl(1.0, sample_rate=16000)

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        assert x.shape == xhat.shape
        if x.ndim == 3:
            x = x.squeeze(1)
            assert x.ndim == 2, "expected single-channel input"
        if xhat.ndim == 3:
            xhat = xhat.squeeze(1)
            assert xhat.ndim == 2, "expected single-channel input"

        x, xhat = self.resampler(x), self.resampler(xhat)
        # expected input order: (reference, degraded) -- matches (x, xhat)
        loss = self.pesq_loss_impl(x, xhat).nanmean()  # ignore NaN entries
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=x.device)
        return loss

