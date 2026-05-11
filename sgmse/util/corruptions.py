from typing import Optional, List
import torch
from librosa import resample

import numpy as np


class RandomDownsampleUpsample(torch.nn.Module):
    def __init__(
        self,
        target_fs: int = 16000,
        possible_fs: Optional[List[int]] = None,
        fs_probs: Optional[List[float]] = None,
        res_types: Optional[List[str]] = None,
        res_probs: Optional[List[float]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.target_fs = target_fs
        self.possible_fs = possible_fs or [4000, 6000, 8000, 10000, 12000, 14000, 16000]
        self.fs_probs = fs_probs or [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25]
        self.res_types = res_types or ['soxr_hq', 'soxr_mq', 'soxr_lq',
                                       'kaiser_fast', 'kaiser_best', 'scipy', 'polyphase']
        self.res_probs = res_probs or [0.25, 0.1, 0.1, 0.1, 0.25, 0.1, 0.1]
        self.rng = np.random.default_rng(seed=seed)

    def forward(self, speech: torch.Tensor):
        res_type = self.rng.choice(self.res_types, p=self.res_probs)
        limited_fs = self.rng.choice(self.possible_fs)
        if limited_fs == self.target_fs:
            return speech

        speech_down = resample(speech.cpu().numpy(), orig_sr=self.target_fs, target_sr=limited_fs, res_type=res_type)
        speech_up = resample(speech_down, orig_sr=limited_fs, target_sr=self.target_fs, res_type=res_type)
        return torch.from_numpy(speech_up).to(speech.device)
