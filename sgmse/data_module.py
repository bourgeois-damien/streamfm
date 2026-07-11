from typing import Optional
from os.path import join
import os
import warnings
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import audiomentations as am
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as FA
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, path: str, format: str, random_crop: bool,
                 target_duration: float | int, sampling_rate: int, whichset: str,
                 spatial_channels: int = 1, dummy: bool = False,
                 crop_paired_to_shorter: bool = False,
                 peak_normalize_clean: bool = False,
                 random_neg_gain_noisy: Optional[float] = None,
                 return_x_as_y: bool = False,
                 y_corruption: Optional[torch.nn.Module] = None,
                 ):
        self.path = path
        self.format = format
        self.spatial_channels = spatial_channels
        self.crop_paired_to_shorter = crop_paired_to_shorter
        self.sampling_rate = sampling_rate
        self.random_crop = random_crop
        self.peak_normalize_clean = peak_normalize_clean
        self.random_neg_gain_noisy = random_neg_gain_noisy
        self.whichset = whichset
        self.dummy = dummy
        self.return_x_as_y = return_x_as_y  # FIXME would be better to use a separate dataset for this...

        self.target_duration = target_duration
        self.target_samples = int(target_duration * sampling_rate) if isinstance(target_duration, float) else target_duration
        print(f"AudioDataset ({whichset}): target samples = {self.target_samples} "
              f"({target_duration} {'seconds' if isinstance(target_duration, float) else 'samples'} at {sampling_rate} Hz).")

        self.x_augmentation = None
        self.y_augmentation = None
        if peak_normalize_clean:
            self.x_augmentation = am.Normalize(apply_to='all', p=1.0)
        if random_neg_gain_noisy is not None:
            assert random_neg_gain_noisy < 0
            self.y_augmentation = am.Compose([
                am.Normalize(apply_to='all', p=1.0),
                am.Gain(min_gain_db=random_neg_gain_noisy, max_gain_db=0, p=1.0),
            ])

        self.y_corruption = y_corruption

        if format == "urgent2025_pairs":
            c1 = list(sorted(glob(join(path) + '/clean1/**/*.flac')))
            c2 = list(sorted(glob(join(path) + '/clean2/**/*.flac')))
            n1 = list(sorted(glob(join(path) + '/noisy1/**/*.flac')))
            n2 = list(sorted(glob(join(path) + '/noisy2/**/*.flac')))
            assert all(os.path.basename(c1[i]) == os.path.basename(n1[i]) for i in range(len(c1)))
            assert all(os.path.basename(c2[i]) == os.path.basename(n2[i]) for i in range(len(c2)))
            self.clean_files = c1 + c2
            self.noisy_files = n1 + n2
        elif format == "paired_dirs":
            cs = list(sorted(glob(join(path, 'clean') + '/**/*.wav')))
            ns = list(sorted(glob(join(path, 'noisy') + '/**/*.wav')))
            assert all(os.path.basename(cs[i]) == os.path.basename(ns[i]) for i in range(len(cs)))
            self.clean_files = cs
            self.noisy_files = ns
        elif format == 'csv':
            csv = pd.read_csv(path)
            self.clean_files = csv['clean_path'].tolist()
            self.noisy_files = csv['noisy_path'].tolist()
        elif format in ('ears_reverb', 'ears_wham', 'ears_other'):
            filekey = {'ears_reverb': 'rt60', 'ears_wham': 'snr_dB', 'ears_other': ''}[format]
            filesuffix = {'ears_reverb': '', 'ears_wham': 'dB', 'ears_other': ''}[format]
            noisydir = {'ears_reverb': 'reverberant', 'ears_wham': 'noisy', 'ears_other': 'noisy'}[format]

            csv = pd.read_csv(path, dtype={'id': str, 'speaker': str, **({filekey: str} if filekey else {})})
            if format == 'ears_wham' and whichset == 'test' and not str(csv.iloc[0]['speaker']).startswith('p'):
                # The official ears_benchmark test.csv header has 9 columns, but rows contain
                # the same 12 fields as save_files() writes for train/valid. Re-read with the
                # actual schema so path reconstruction keeps speaker/id/snr aligned.
                csv = pd.read_csv(
                    path,
                    skiprows=1,
                    header=None,
                    names=[
                        'id', 'speaker', 'speech_file', 'speech_start', 'speech_end',
                        'noise_file', 'noise_start', 'noise_end',
                        'speech_dB', 'noise_dB', 'mixture_dB', 'snr_dB',
                    ],
                    dtype={'id': str, 'speaker': str, **({filekey: str} if filekey else {})},
                )
            basedir = os.path.dirname(path)
            self.clean_files = [
                join(
                    basedir, whichset, 'clean',
                    csv.iloc[i]['speaker'], f"{csv.iloc[i]['id']}.wav"
                )
                for i in range(len(csv))
            ]

            filekeys = [csv.iloc[i][filekey] for i in range(len(csv))] if filekey else ['']*len(csv)
            self.noisy_files = [
                join(
                    basedir, whichset, noisydir,
                    csv.iloc[i]['speaker'], f"{csv.iloc[i]['id']}_{filekeys[i]}{filesuffix}.wav"
                )
                for i in range(len(csv))
            ]
        elif format == 'csv_enhanced_pred':
            csv = pd.read_csv(path)
            self.clean_files = csv['clean_path'].tolist()
            self.noisy_files = csv['noisy_path'].tolist()
            self.enhanced_pred_files = csv['enhanced_pred_path'].tolist()
            assert len(self.enhanced_pred_files) > 0, f"No enhanced_pred files found"
        else:
            raise NotImplementedError(f"Format {format} not implemented!")

        assert len(self.clean_files) > 0, f"No clean files found"
        assert len(self.noisy_files) > 0, f"No noisy files found"

    def _croppad_multiple(self, xs):
        "Crops/pads multiple signals to the same random slice"
        target_len = self.target_samples
        current_len = xs[0].size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.random_crop:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            xs = [x[..., start:start + target_len] for x in xs]
        else:
            # pad audio if the length T is smaller than num_frames
            xs = [F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant') for x in xs]
        return xs

    def _prep_spatial_channels(self, x):
        if x.ndimension() == 2 and self.spatial_channels == 1:
            x = x[0].unsqueeze(0) # Select first channel
        # Select channels
        assert self.spatial_channels <= x.size(0), \
            f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
        return x[..., :self.spatial_channels, :]

    def _resample(self, x, sr):
        return FA.resample(x, sr, self.sampling_rate, lowpass_filter_width=64)

    def __getitem__(self, i, no_crop=False):
        # regular paired formats (x,y) i.e. (clean,noisy)
        x, sr = load(self.clean_files[i])
        y, sr_y = load(self.noisy_files[i])
        assert sr_y == sr

        if self.crop_paired_to_shorter:
            # crop to the shorter of the two signals
            if x.shape[-1] < y.shape[-1]:
                y = y[..., :x.shape[-1]]
            elif x.shape[-1] > y.shape[-1]:
                x = x[..., :y.shape[-1]]
        else:
            assert x.shape[-1] == y.shape[-1]

        x, y = [self._prep_spatial_channels(v) for v in (x, y)]
        x, y = [self._resample(v, sr) for v in (x, y)]

        # Apply augmentations (esp. normalizations) before cropping, otherwise we may blow up close-to-silent random segments
        if self.x_augmentation is not None:
            x = torch.from_numpy(self.x_augmentation(x.cpu().numpy(), sr)).to(x.device)
        if self.y_augmentation is not None:
            y = torch.from_numpy(self.y_augmentation(y.cpu().numpy(), sr)).to(y.device)

        if self.y_corruption is not None:
            y = self.y_corruption(y)

        # do the cropping / padding again after augmentations...
        if self.crop_paired_to_shorter:
            # crop to the shorter of the two signals
            if x.shape[-1] < y.shape[-1]:
                y = y[..., :x.shape[-1]]
            elif x.shape[-1] > y.shape[-1]:
                x = x[..., :y.shape[-1]]
        else:
            assert x.shape[-1] == y.shape[-1]

        if not no_crop:
            x, y = self._croppad_multiple((x, y))

        if self.return_x_as_y:
            # return x as y, i.e. (x,x) instead of (x,y)
            y = x

        return x, y, {"sr": self.sampling_rate, "x_path": self.clean_files[i], "y_path": self.noisy_files[i]}

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return min(10000, len(self.clean_files))
        else:
            return len(self.clean_files)


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        format: str,
        sampling_rate: int,
        target_duration: float | int,
        crop_paired_to_shorter: bool = False,
        train_path: Optional[str] = None,
        valid_path: Optional[str] = None,
        test_path: Optional[str] = None,
        spatial_channels: int = 1,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        peak_normalize_clean: bool = False,
        random_neg_gain_noisy: float = None,
        dummy: bool = False,
        return_x_as_y: bool = False,  # if True, return x as y, i.e. (x,x) instead of (x,y)
        y_corruption: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.target_duration = target_duration
        self.crop_paired_to_shorter = crop_paired_to_shorter
        self.sampling_rate = sampling_rate
        self.format = format
        self.spatial_channels = spatial_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.peak_normalize_clean = peak_normalize_clean
        self.random_neg_gain_noisy = random_neg_gain_noisy
        self.dummy = dummy
        self.return_x_as_y = return_x_as_y
        self.y_corruption = y_corruption
        print("Initialized AudioDataModule with batch size", self.batch_size)

    def setup(self, stage=None):
        dataset_kwargs = dict(
            format=self.format,
            sampling_rate=self.sampling_rate,
            crop_paired_to_shorter=self.crop_paired_to_shorter,
            spatial_channels=self.spatial_channels,
            target_duration=self.target_duration,
            peak_normalize_clean=self.peak_normalize_clean,
            random_neg_gain_noisy=self.random_neg_gain_noisy,
            dummy=self.dummy,
            return_x_as_y=self.return_x_as_y,  # if True, return x as y, i.e. (x,x) instead of (x,y)
            y_corruption=self.y_corruption,
        )
        if stage == 'fit' or stage is None:
            self.train_set = AudioDataset(self.train_path, whichset='train', random_crop=True, **dataset_kwargs)
            self.valid_set = AudioDataset(self.valid_path, whichset='valid', random_crop=False, **dataset_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = AudioDataset(self.test_path, whichset='test', random_crop=False, **dataset_kwargs)
        if stage == '_train_only':
            warnings.warn("Using _train_only stage, no validation or test set will be created.")
            self.train_set = AudioDataset(self.train_path, whichset='train', random_crop=True, **dataset_kwargs)
            self.valid_set = None
            self.test_set = None

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False
        )
