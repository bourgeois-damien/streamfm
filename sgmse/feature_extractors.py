from typing import Optional, List
import abc
import scipy
import torch
import einops


class InvertibleFeatureExtractor(torch.nn.Module, abc.ABC):
    """
    Invertible feature extractor, should fulfill `extractor.invert(extractor(x)) == x`.
    """
    @abc.abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abc.abstractmethod
    def invert(self, X, **kwargs):
        pass

    def matches(self, other: 'InvertibleFeatureExtractor', ignore=None) -> bool:
        """
        Check if this extractor matches another extractor.
        This is used to determine if two extractors can be used interchangeably.
        """
        if not self.__class__ == other.__class__:
            return False
        return compare_equal(
            self.__dict__, other.__dict__, key='self.__dict__',
            ignore=['self.__dict__["training"]', *(ignore or [])],
        )


def compare_equal(x, y, key: str = '', ignore: Optional[List] = None) -> bool:
    if ignore is not None and key in ignore:
        return True

    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        if torch.allclose(x, y):
            return True
        else:
            print(f"mismatch in torch.equal {x=} vs {y=} at {key=}")
            return False
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y):
            print(f"mismatch in {len(x)=} vs {len(y)=} at {key=}")
            return False
        return all(compare_equal(xi, yi, key=f'{key}[{idx}]', ignore=ignore) for idx, (xi, yi) in enumerate(zip(x, y)))
    elif isinstance(x, dict):
        if set(x.keys()) != set(y.keys()):
            print(f"mismatch in {x.keys()=} vs {y.keys()=} at {key=}")
            return False
        return all(compare_equal(x[k], y[k], key=f'{key}["{k}"]', ignore=ignore) for k in x.keys())
    else:
        if x == y:
            return True
        else:
            print(f"mismatch in {x=} vs {y=} at {key=}")
            return False


class CompressedAmplitudeComplexSTFT(InvertibleFeatureExtractor):
    def __init__(
        self, window, n_fft, sampling_rate, hop_length, alpha, beta,
        compression_is_learnable=False, normalized_stft=True,
        cut_highest_freqs: Optional[int] = None, sqrt_window: bool = True,
        pad_highest_freqs_on_inverse: Optional[int] = None,
        amplitude_only: bool = False,
    ):
        super().__init__()
        self.window = window
        self.sqrt_window = sqrt_window
        window_fn = getattr(scipy.signal.windows, self.window)
        window = torch.from_numpy(window_fn(n_fft, sym=False)).to(torch.float32)
        if sqrt_window:
            window = window.abs()**0.5 * window.sign()

        self.window_tensor = torch.nn.Parameter(window, requires_grad=False)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.compression_is_learnable = compression_is_learnable
        self.normalized_stft = normalized_stft
        self.cut_highest_freqs = cut_highest_freqs
        self.pad_highest_freqs_on_inverse = pad_highest_freqs_on_inverse
        self.amplitude_only = amplitude_only

        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=compression_is_learnable)
        self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=compression_is_learnable)

        win_dur = self.n_fft / self.sampling_rate
        hop_dur = self.hop_length / self.sampling_rate
        print(f"CompressedAmplitudeComplexSTFT initialized with {win_dur*1000:.2f}ms window, "
              f"{hop_dur*1000:.2f}ms hop. {alpha=}, {beta=}, {compression_is_learnable=}, "
              f"{normalized_stft=}, {amplitude_only=}.")

    def _stft(self, x: torch.Tensor):
        assert x.ndim == 3, "Input tensor must be of shape [B, C, T]"
        X = torch.stft(
            einops.rearrange(x, f"b c t -> (b c) t"),
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window_tensor, center=True,
            onesided=True, return_complex=True,
            normalized=self.normalized_stft,
        )
        return einops.rearrange(X, f"(b c) f t -> b c f t", b=x.shape[0])

    def _istft(self, X: torch.Tensor, T_orig: Optional[int] = None):
        assert X.ndim == 4, "Input tensor must be of shape [B, C, F, T]"
        x = torch.istft(
            einops.rearrange(X, f"b c f t -> (b c) f t"),
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window_tensor, center=True,
            onesided=True, return_complex=False,
            length=T_orig,
            normalized=self.normalized_stft,
        )
        x = einops.rearrange(x, f"(b c) t -> b c t", b=X.shape[0])
        return x

    def forward(self, x: torch.Tensor, comp_eps: Optional[float] = 1e-12,
                alpha_override: Optional[float] = None, beta_override: Optional[float] = None):
        """Assumes x is an audio tensor of shape [B, C, T]"""
        assert x.ndim == 3
        alpha = self.alpha if alpha_override is None else alpha_override
        beta = self.beta if beta_override is None else beta_override

        X = self._stft(x)

        if self.amplitude_only:
            X = torch.abs(X)

        if alpha != 1 or beta != 1:
            if self.amplitude_only:
                X = beta * X**alpha
            else:
                X = beta * torch.polar(torch.abs(X)**alpha, torch.angle(X + comp_eps))

        if self.cut_highest_freqs:
            X = X[..., :-self.cut_highest_freqs, :]

        return X

    def invert(self, X: torch.Tensor, T_orig: Optional[int] = None,
               alpha_override: Optional[float] = None, beta_override: Optional[float] = None,
               comp_eps: float = 1e-12):
        """Assumes X is a (complex) spectrogram tensor of shape [B, C, F, T]"""
        assert X.ndim == 4
        alpha = self.alpha if alpha_override is None else alpha_override
        beta = self.beta if beta_override is None else beta_override

        if alpha != 1 or beta != 1:
            if self.amplitude_only:
                X = (X / beta)**(1/alpha)
            else:
                X = torch.polar((torch.abs(X) / beta)**(1/alpha), torch.angle(X + comp_eps))

        if self.cut_highest_freqs:
            X = torch.nn.functional.pad(X, (0, 0, 0, self.cut_highest_freqs), mode='constant', value=0.0)
        if self.pad_highest_freqs_on_inverse:
            X = torch.nn.functional.pad(X, (0, 0, 0, self.pad_highest_freqs_on_inverse), mode='constant', value=0.0)

        x = self._istft(X, T_orig=T_orig)
        return x



# test for CompressedAmplitudeComplexSTFT
# the inverted output should be close to the input
# we test for a few different values of alpha and beta
if __name__ == "__main__":
    # Create a random audio tensor
    B, C, T = 2, 1, 16000
    x = torch.randn(B, C, T)

    for alpha in torch.linspace(0.1, 3.0, 10):
        for beta in torch.linspace(0.1, 3.0, 10):
            for n_fft in [510, 512, 1024]:
                # Test with different alpha and beta values
                alpha = alpha
                beta = beta

                # Initialize the feature extractor
                extractor = CompressedAmplitudeComplexSTFT(
                    window='hann', n_fft=n_fft, sampling_rate=16000,
                    hop_length=128, alpha=alpha, beta=beta,
                    compression_is_learnable=False,
                    normalized_stft=True,
                )

                # Forward pass
                X = extractor(x)

                # Inversion
                x_inv = extractor.invert(X)

                # Check if the inverted output is close to the input
                assert torch.allclose(x, x_inv, atol=1e-5), "Inverted output is not close to the input"