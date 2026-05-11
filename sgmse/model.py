import abc
import warnings
from typing import Callable, Tuple, Optional, Sequence
import wandb
import matplotlib.pyplot as plt

import numpy as np
from einops import rearrange
import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import grad_norm

from sgmse.util.graphics import visualize_example
from sgmse.util.inference import EvaluateModel
from sgmse.util.solvers import solve_ode, get_butcher_tableau
from sgmse.data_module import AudioDataModule
from sgmse.losses import Loss
from sgmse.feature_extractors import InvertibleFeatureExtractor, CompressedAmplitudeComplexSTFT


class GenericEnhancementModel(abc.ABC, pl.LightningModule):
    def __init__(self,
        backbone: torch.nn.Module,
        sampling_rate: int,
        feature_extractor: InvertibleFeatureExtractor,
        data_module: Optional[AudioDataModule] = None,
        optimizer_constructor: Optional[Callable] = None,
        scheduler_config: dict = None,  # 'scheduler_constructor' and 'scheduler_config_dict' must be keys
        lr: float = 1e-4,
        normalize_mode: str = 'noisy',
        num_eval_files: int = 20,
        pad_feature_to_multiple_of: Optional[int] = None,
    ):
        """
        Create a new GenericEnhancementModel. This cannot actually be called since this class is abstract,
        but this docstring serves as a reference for the arguments.

        Args:
            backbone: The backbone DNN (nn.Module) to use for the enhancement task.
            data_module: The LightningDataModule to use for the enhancement task. Can be None, e.g. for inference.
            sampling_rate: The sampling rate of the audio data this model will use. Must match the data_module.
            feature_extractor: The InvertibleFeatureExtractor to use for the enhancement task.
            optimizer_constructor: A callable that constructs an optimizer from the model's parameters
                and the learning rate `lr`. Can be None, e.g. for inference.
            lr: The learning rate of the optimizer.
            ema_mode: Which parameters to use for the EMA. Can be 'backbone_only' or 'all'.
            normalize_mode: The mode to use for normalization. Either 'noisy' or 'none'.
            num_eval_files: The number of files to use for evaluation during validation.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.lr = lr
        self.num_eval_files = num_eval_files
        self.normalize_mode = normalize_mode
        self.pad_feature_to_multiple_of = pad_feature_to_multiple_of
        assert self.normalize_mode in ['noisy', 'none'], f"Unknown normalize_mode: {self.normalize_mode}"

        self.dnn = backbone
        self.feature_extractor = feature_extractor
        self.data_module = data_module
        if self.data_module is not None:
            assert hasattr(self.data_module, "sampling_rate") and self.data_module.sampling_rate == self.sampling_rate, \
                f"Data module sampling rate nonexistent or does not match model sampling rate {self.sampling_rate}"

        self._optimizer_constructor = optimizer_constructor
        self._scheduler_config = scheduler_config

        self.save_hyperparameters(ignore=['backbone', 'feature_extractor', 'data_module', 'drift_feature_extractor'])

        # state flags
        self._logged_xy = False

        self.evaluate_model = EvaluateModel(
            self, num_eval_files=self.num_eval_files, spec=True, audio=True)

        # add custom logging axes for validation to w&b, if self.logger is a WandbLogger
        # see https://docs.wandb.ai/guides/track/log/customize-logging-axes/
        if isinstance(self.logger, WandbLogger):
            wb_run = self.logger.experiment
            wb_run.define_metric("ValidationPESQ", step_metric="val_step")
            wb_run.define_metric("ValidationSISDR", step_metric="val_step")
            wb_run.define_metric("ValidationESTOI", step_metric="val_step")
            wb_run.define_metric("ValidationDistillMOS", step_metric="val_step")

    def transform_backbone_(self, transform_fn):
        """
        Apply a transformation function to the backbone model in-place.
        The function should take a torch.nn.Module as input and return a transformed torch.nn.Module.
        This can be used for model compression, e.g. pruning or quantization.

        Args:
            transform_fn: A function that takes a torch.nn.Module and returns a transformed torch.nn.Module.
        """
        self.dnn = transform_fn(self.dnn)
        return self

    def _instantiate_optimizer_and_scheduler(self):
        """
        To be called by each subclass during __init__ at the correct spot,
        after initializing all submodules and `get_trainable_parameters` is ready.
        """
        optimizer_constructor = self._optimizer_constructor
        scheduler_config = self._scheduler_config

        optimizer = optimizer_constructor(self.get_trainable_parameters(), lr=self.lr)
        if scheduler_config is not None:
            scheduler_constructor = scheduler_config['scheduler_constructor']
            scheduler_config_dict = scheduler_config['scheduler_config_dict']
            scheduler = scheduler_constructor(optimizer)
            scheduler_config_dict = scheduler_config_dict
        else:
            scheduler = None
            scheduler_config_dict = None

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config_dict = scheduler_config_dict

    def get_trainable_parameters(self):
        """
        Get the parameters to be optimized by the optimizer.
        Override this method if you want to freeze some parameters.
        """
        return self.parameters()

    def configure_optimizers(self):
        if self.scheduler is None:
            # Just an optimizer
            return self.optimizer
        else:
            # Optimizer and the scheduler as lightning expects it
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {'scheduler': self.scheduler, **self.scheduler_config_dict}
            }

    ### Methods related to training and forward pass

    @abc.abstractmethod
    def _step(self, batch, batch_idx, stage: str):
        pass

    @abc.abstractmethod
    def forward(self, Y, *args, **kwargs):
        pass

    def setup(self, stage=None):
        super().setup(stage=stage)
        if self.data_module is not None:
            self.data_module.setup(stage=stage)
        return self

    # Methods related to enhancement

    @abc.abstractmethod
    def enhance(self, y, y_sr, *args, **kwargs):
        """
        One-call speech enhancement of noisy time-domain speech signal(s) `y` with sampling rate `y_sr`.
        It's recommended to use the methods self.preprocess()/self.postprocess() inside your implementation.
        """
        pass

    # Methods related to feature extraction

    def _normalize_inputs(self, y, x):
        if self.normalize_mode == 'noisy':
            norm_factors = (y.abs().amax(dim=-1, keepdim=True))
            norm_factors = torch.where(norm_factors < 1e-6, 1, norm_factors)
            y = y / norm_factors
            x = x / norm_factors if x is not None else None
        elif self.normalize_mode in (None, 'none'):
            norm_factors = 1.0
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

        return y, x, norm_factors

    def _resample_inputs(self, y, x, y_sr):
        """
        Resamples y/e/x to the model's sampling rate.
        Uses an increased lowpass_filter_width over the torchaudio default (6!!) to avoid aliasing.
        """
        y = torchaudio.functional.resample(
            y, y_sr, self.sampling_rate, lowpass_filter_width=64)
        x = torchaudio.functional.resample(
            x, y_sr, self.sampling_rate, lowpass_filter_width=64) if x is not None else None
        return y, x

    def preprocess(
        self,
        y: torch.Tensor,
        y_sr: int,
        x: Optional[torch.Tensor] = None,
        pad_signal_right: int = 0,
        pad_signal_left: int = 0,
        pad_mode: str = 'constant',
        feature_extractor_kwargs: Optional[dict] = None,
    ):
        """
        Preprocess the input audio signals y and x (if given) and extract features.

        Args:
            y: Noisy input audio signal(s), shape (B, C, T).
            y_sr: Sampling rate of y. Audio will be resampled to self.sampling_rate if different.
            x: (Optional) Clean input audio signal(s) corresponding to y, shape (B, C, T). Only required for training or analysis.
            pad_signal_right: Amount of padding to add to the right of the signal before feature extraction. Default 0.
            pad_signal_left: Amount of padding to add to the left of the signal before feature extraction. Default 0.
            pad_mode: The mode to use for padding. Passed to torch.nn.functional.pad. Default 'constant' (zero-padding).
            feature_extractor_kwargs: Optional dictionary of additional keyword arguments to pass to the feature extractor.
                This can be used to pass things like n_fft, hop_length etc. if the feature extractor supports it and you want to override the defaults.

        Returns:
            * If x is not given: (Y, y_pre, preprocess_info), where Y is the extracted features from y_pre,
                y_pre is the preprocessed audio signal (after resampling, normalization and padding) that was fed into the feature extractor,
                and preprocess_info is a dictionary containing information about the preprocessing steps (e.g. original and resampled lengths, normalization factors, etc.)
                that are useful for postprocessing.
            * If x is given: (Y, X, y_pre, x_pre, preprocess_info) where Y and X are the extracted features of y_pre and x_pre,
                y_pre and x_pre are the preprocessed audio signals, otherwise see above.
        """
        if feature_extractor_kwargs is None:
            feature_extractor_kwargs = {}
        device_orig = y.device
        assert x is None or x.shape == y.shape

        y = y.to(self.device)
        x = x.to(self.device) if x is not None else None

        # Normalization
        y, x, norm_factors = self._normalize_inputs(y, x)

        # Resampling
        T_orig = y.shape[-1]
        y, x = self._resample_inputs(y, x, self.sampling_rate)
        T_orig_resampled = y.shape[-1]

        # Input padding (if configured. otherwise these are no-ops)
        y = torch.nn.functional.pad(y, (pad_signal_left, pad_signal_right), mode=pad_mode)
        x = torch.nn.functional.pad(x, (pad_signal_left, pad_signal_right), mode=pad_mode) if x is not None else None

        preprocess_info = {
            'T_orig': T_orig, 'T_orig_resampled': T_orig_resampled,
            'device_orig': device_orig, 'sr_orig': y_sr,
            'norm_factors': norm_factors,
            'pad_right': pad_signal_right, 'pad_left': pad_signal_left,
        }
        Y = self.feature_extractor(y, **feature_extractor_kwargs)
        if self.pad_feature_to_multiple_of is not None:
            Y = torch.nn.functional.pad(
                Y, (0, self.pad_feature_to_multiple_of - Y.shape[-1] % self.pad_feature_to_multiple_of),
                mode='constant', value=0.0)

        if x is not None:
            X = self.feature_extractor(x, **feature_extractor_kwargs)
            if self.pad_feature_to_multiple_of is not None:
             X = torch.nn.functional.pad(
                    X, (0, self.pad_feature_to_multiple_of - X.shape[-1] % self.pad_feature_to_multiple_of),
                    mode='constant', value=0.0)

            return Y, X, y, x, preprocess_info
        else:
            return Y, y, preprocess_info

    def postprocess(
        self, X_hat, preprocess_info,
        resample_to_orig=True,
        move_to_orig_device=True,
        undo_pad=True,
    ):
        T_orig_resampled = preprocess_info['T_orig_resampled']
        T_orig = preprocess_info['T_orig']
        norm_factors = preprocess_info['norm_factors']
        device_orig = preprocess_info['device_orig']
        sr_orig = preprocess_info['sr_orig']
        pad_left = preprocess_info['pad_left']
        pad_right = preprocess_info['pad_right']

        # no need to explicitly undo pad_right here -- the T_orig_resampled info should take care of it
        # (at least when using a CompressedAmplitudeComplexSTFT feature extractor)
        pad_total = pad_left + pad_right if not undo_pad else 0
        x_hat = self.feature_extractor.invert(X_hat, T_orig=T_orig_resampled + pad_total)

        if resample_to_orig:
            x_hat = torchaudio.functional.resample(
                x_hat, self.sampling_rate, sr_orig, lowpass_filter_width=64)
            if undo_pad:
                x_hat = x_hat[..., :T_orig]  # crop to original length

        if norm_factors is not None:
            x_hat = x_hat * norm_factors

        if move_to_orig_device:
            x_hat = x_hat.to(device_orig)
        return x_hat

    ### Methods related to training and validation

    def training_step(self, batch, batch_idx):
        bs = self.data_module.batch_size
        loss = self._step(batch, batch_idx, stage='train')
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=bs)

        if isinstance(self.feature_extractor, CompressedAmplitudeComplexSTFT):
            self.log('feat_alpha', self.feature_extractor.alpha, on_step=True, on_epoch=False, batch_size=1)
            self.log('feat_beta', self.feature_extractor.beta, on_step=True, on_epoch=False, batch_size=1)

        return loss

    def on_validation_epoch_end(self):
        # Evaluate speech enhancement performance
        if self.trainer.is_global_zero and self.logger is not None:
            pesq_est, si_sdr_est, estoi_est, distillmos_est, spec, audio = self.evaluate_model()
            if hasattr(self.logger.experiment, "add_scalar"):
                # Tensorboard
                self.logger.experiment.add_scalar("ValidationPESQ", pesq_est, self.trainer.global_step)
                self.logger.experiment.add_scalar("ValidationSISDR", si_sdr_est, self.trainer.global_step)
                self.logger.experiment.add_scalar("ValidationESTOI", estoi_est, self.trainer.global_step)
                self.logger.experiment.add_scalar("ValidationDistillMOS", distillmos_est, self.trainer.global_step)
            else:
                # W&B
                self.logger.experiment.log(
                    {
                        "ValidationPESQ": pesq_est,
                        "ValidationSISDR": si_sdr_est,
                        "ValidationESTOI": estoi_est,
                        "ValidationDistillMOS": distillmos_est,
                        "val_step": self.trainer.current_epoch
                    }
                )

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                aulog_kw = dict(sampling_rate=self.sampling_rate, val_step=self.trainer.current_epoch)
                for idx, x_hat in enumerate(x_hat_list):
                    self.log_audio(f"Estimate/{idx}", x_hat.unsqueeze(0), **aulog_kw)
                if not self._logged_xy:
                    for idx, (x, y) in enumerate(zip(x_list, y_list)):
                        self.log_audio(f"Mix/{idx}", y.unsqueeze(0), **aulog_kw)
                        self.log_audio(f"Clean/{idx}", x.unsqueeze(0), **aulog_kw)
                    self._logged_xy = True

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                            torch.abs(y_stft),
                            torch.abs(x_hat_stft),
                            torch.abs(x_stft)))
                self.log_figure("Spec", figures, val_step=self.trainer.current_epoch)
                for figure in figures:
                    plt.close(figure)

    def validation_step(self, batch, batch_idx):
        bs = self.data_module.batch_size
        loss = self._step(batch, batch_idx, stage='val')
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    ### Dataloader related overrides

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    ### Gradient norm logging

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        if self.global_step % 50 == 0:  # every 50 steps only
            self.log_dict(norms)

    ### Logging helpers

    def log_audio(self, name, audio_tensor, sampling_rate, val_step):
        assert audio_tensor.ndim in (1,2)
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.experiment.log({name: wandb.Audio(audio_tensor.T, sample_rate=sampling_rate), 'val_step': val_step})
        elif isinstance(logger, TensorBoardLogger):
            self.logger.experiment.add_audio(
                name, audio_tensor.unsqueeze(0) if audio_tensor.ndim == 1 else audio_tensor,
                sample_rate=sampling_rate, global_step=val_step)

    def log_figure(self, name, figure, val_step):
        logger = self.logger
        if isinstance(logger, WandbLogger):
            if isinstance(figure, list):
                logged = {f'{name}_{i}': wandb.Image(figure[i]) for i in range(len(figure))}
            else:
                logged = {name: wandb.Image(figure)}
            logger.experiment.log({**logged, 'val_step': val_step})
        elif isinstance(logger, TensorBoardLogger):
            self.logger.experiment.add_figure(name, figure, global_step=val_step)


class DiscriminativeModel(GenericEnhancementModel):
    def __init__(
        self, *args,
        losses: Sequence[Loss], discriminative_mode: str,
        extra_weight_silent_frames: float = 0.0, extra_weight_silent_frames_exponent: float = 2.0,
        extra_weight_silent_frames_log: bool = False,
        **kwargs
    ):
        """
        Create a new DiscriminativeModel.

        Args:
            **See all args of GenericEnhancementModel as well as the added args below:**

            losses: A list of loss functions to use for training. Should each be a dictionary with a 'kind' key
                and a 'weight' key. 'kind' can be one of: {'feature_domain_l2', 'time_domain_l1'}.
            discriminative_mode: The mode to use for the discriminative model. Can be 'direct' or 'residual'.
        """
        # Initialize losses (in eval mode)
        losses = torch.nn.ModuleList(losses)
        assert all(isinstance(loss, Loss) for loss in losses)
        for loss in losses:
            assert loss.domain in ('time', 'feature'), f"Unknown loss domain: {loss.domain}"
            loss.eval()
            for name, param in losses.named_parameters():
                param.requires_grad_(False)
        losses.eval()

        self.extra_weight_silent_frames = extra_weight_silent_frames
        self.extra_weight_silent_frames_exponent = extra_weight_silent_frames_exponent
        self.extra_weight_silent_frames_log = extra_weight_silent_frames_log
        if isinstance(self.extra_weight_silent_frames, list):
            assert len(self.extra_weight_silent_frames_exponent) == len(self.extra_weight_silent_frames)
            assert len(self.extra_weight_silent_frames_log) == len(self.extra_weight_silent_frames)
        elif isinstance(self.extra_weight_silent_frames, (float, int)):
            self.extra_weight_silent_frames = [self.extra_weight_silent_frames]
            self.extra_weight_silent_frames_exponent = [self.extra_weight_silent_frames_exponent]
            self.extra_weight_silent_frames_log = [self.extra_weight_silent_frames_log]
        else:
            self.extra_weight_silent_frames = list(self.extra_weight_silent_frames)
            self.extra_weight_silent_frames_exponent = list(self.extra_weight_silent_frames_exponent)
            self.extra_weight_silent_frames_log = list(self.extra_weight_silent_frames_log)
            assert len(self.extra_weight_silent_frames_exponent) == len(self.extra_weight_silent_frames)
            assert len(self.extra_weight_silent_frames_log) == len(self.extra_weight_silent_frames)

        super().__init__(*args, **kwargs)
        self.discriminative_mode = discriminative_mode
        assert self.discriminative_mode in ['direct', 'residual', 'complex_mask_unconstrained', 'magnitude_mask_unconstrained']
        self.losses = losses

        self._instantiate_optimizer_and_scheduler()

    def forward(self, Y):
        """
        Y is the feature representation of the noisy audio. Returns enhanced output in the same feature domain.
        """
        if self.discriminative_mode == 'direct':
            X_hat = self.dnn(Y)
        elif self.discriminative_mode == 'residual':
            X_hat = Y - self.dnn(Y)
        elif self.discriminative_mode == 'complex_mask_unconstrained':
            X_hat = Y * self.dnn(Y)
        elif self.discriminative_mode == 'magnitude_mask_unconstrained':
            mask = self.dnn(Y.abs())
            X_hat = Y.abs() * mask * torch.exp(1j * Y.angle())
        else:
            raise ValueError(f"Unknown discriminative mode: {self.discriminative_mode}")

        return X_hat

    def _loss(self, X, X_hat, preprocess_info, stage, return_log_as_dict=False):
        loss_dict, log_dict = self._loss_orig(
            X, X_hat, preprocess_info=preprocess_info, stage=stage, return_log_as_dict=return_log_as_dict)
        total_loss = torch.tensor(0., device=X.device)
        total_loss += sum(loss_dict.values())
        return total_loss, log_dict

    def _loss_orig(self, X, X_hat, preprocess_info, stage, return_log_as_dict=False):
        x_post = self.postprocess(X, preprocess_info, resample_to_orig=False)
        x_hat_post = self.postprocess(X_hat, preprocess_info, resample_to_orig=False)
        loss_dict = {}
        log_dict = {}

        for loss_impl in self.losses:
            if loss_impl.domain == 'feature':
                loss_val = loss_impl(X, X_hat)
            else:  # time domain loss
                # FIXME we assume here that all loss instances are working at the model's sampling rate,
                # or that they do resampling from the model's sampling rate to their own
                loss_val = loss_impl(x_post, x_hat_post)
            assert loss_val.ndim == 0, \
                f"Loss {loss_impl.name} returned a valued of ndim={loss_val.ndim} but it should be scalar-valued."

            if loss_val.isnan():
                warnings.warn(f"Loss {loss_impl.name} returned NaN. Skipping.")
            else:
                loss_dict[loss_impl.name] = loss_impl.weight * loss_val
            log_dict[f'{stage}_{loss_impl.name}_loss'] = loss_val
            log_dict[f'{stage}_{loss_impl.name}_loss_weighted'] = loss_impl.weight * loss_val

        if return_log_as_dict:
            return loss_dict, log_dict
        else:
            for key, val in log_dict.items():
                self.log(key, val, on_step=True, on_epoch=False, sync_dist=True)
            return loss_dict, None

    def _step(self, batch, batch_idx, stage):
        x, y, info = batch
        assert (info['sr'] == self.sampling_rate).all()

        Y, X, y_pre, x_pre, preprocess_info = self.preprocess(y, self.sampling_rate, x=x)
        X_hat = self(Y)
        loss, _ = self._loss(X, X_hat, preprocess_info=preprocess_info, stage=stage, return_log_as_dict=False)

        if self.extra_weight_silent_frames is not None and self.extra_weight_silent_frames != 0.0:
            for l, (w, exponent, use_log) in enumerate(
                zip(self.extra_weight_silent_frames, self.extra_weight_silent_frames_exponent, self.extra_weight_silent_frames_log)
            ):
                if w == 0.0:
                    continue

                # (B, C, F, T)
                cutoff = 0.1
                silent_frame_mask = (X.abs().sum(dim=(1, 2), keepdim=True) < cutoff).float()
                X1pred = X_hat

                if use_log:
                    x1pred_loss = ((X1pred.abs() + 1e-8)**exponent).clamp(min=1e-12).log()
                    silent_frame_losses = silent_frame_mask * x1pred_loss
                    silent_frame_loss = silent_frame_losses.mean()
                else:
                    x1pred_loss = (X1pred.abs() + 1e-8)**exponent
                    silent_frame_losses = silent_frame_mask * x1pred_loss
                    silent_frame_loss = silent_frame_losses.flatten(start_dim=1).sum(dim=1).mean()

                if l == 0:
                    self.log(f'{stage}_silent_frame_proportion',
                        silent_frame_mask.float().mean(), on_step=True, on_epoch=False, sync_dist=True)

                self.log(f'{stage}_silent_frame_loss_exp{exponent}{"_log" if use_log else ""}',
                    silent_frame_loss, on_step=True, on_epoch=False, sync_dist=True)
                loss += w * silent_frame_loss

        return loss

    def enhance(
        self, y, y_sr,
        return_all: bool = False,
        grad: bool = False,
        pad_signal_right: int = 0,
        pad_signal_left: int = 0,
        pad_mode: str = 'constant',  # pad with zeros
        postprocess_kwargs: Optional[dict] = None,
    ):
        with torch.set_grad_enabled(grad):
            squeeze_counter = 0

            # if y.ndim > 1 and y.shape[-2] > 1:
            #     warnings.warn("Input y has more than one channel. We will only process the first channel and ignore the rest.")
            #     y = y[..., [0], :]

            while y.ndim < 3:
                y = y.unsqueeze(0)
                squeeze_counter += 1

            # Map to feature domain
            Y, y_pre, preprocess_info = self.preprocess(
                y, y_sr, pad_signal_right=pad_signal_right, pad_signal_left=pad_signal_left, pad_mode=pad_mode)

            X_hat = self(Y)

            x_hat = self.postprocess(X_hat, preprocess_info, **(postprocess_kwargs if postprocess_kwargs is not None else {}))
            for _ in range(squeeze_counter):
                X_hat = X_hat.squeeze(0)
                x_hat = x_hat.squeeze(0)

            if return_all:
                y = self.postprocess(Y, preprocess_info)
                y = y.squeeze(0) if y.ndim == 3 else y
                return x_hat, X_hat, y, Y, preprocess_info
            return x_hat


class FlowModel(GenericEnhancementModel):
    def __init__(self, sigma_y: float | torch.Tensor,
                 post_Y_fn: Optional[Callable] = None, post_X_fn: Optional[Callable] = None,
                 direct_pred_losses: Optional[Sequence[torch.nn.Module]] = None,
                 fm_loss_weight: float = 1.0, direct_pred_loss_weight: float = 1.0, sigma_x: float = 0.0,
                 *args, **kwargs):
        """
        Create a new FlowModel.

        Args:
            See GenericEnhancementModel for the common arguments.
            sigma_y: The noise stdev added to y to get x_0: (x_0 := y + sigma_y*z) with i.i.d. standard Gaussian z.
                If a float or 1-D tensor, it is the same for all T-F bins.
                If it is of shape (F, 1) it can model a diagonal covariance dependent on the frequency.
                Other shape combinations (e.g. per-channel-per-requency sigma like (C, F, 1)) should also work.
            post_Y_fn: A function to apply to Y before passing it to the DNN. Useful for training
                tasks where Y is simulated from X with a fixed or random mapping, e.g. STFT phase retrieval
            direct_pred_loss: If set, this is a loss function that will be used to compute the direct prediction loss
                replacing the standard flow matching loss.
        """
        super().__init__(*args, **kwargs)

        if isinstance(sigma_y, torch.Tensor):
            self.register_buffer('sigma_y', sigma_y)
        else:
            assert isinstance(sigma_y, (float, int))
            self.sigma_y = float(sigma_y)

        if isinstance(sigma_x, torch.Tensor):
            self.register_buffer('sigma_x', sigma_x)
        else:
            assert isinstance(sigma_x, (float, int))
            self.sigma_x = float(sigma_x)

        self.post_Y_fn = post_Y_fn
        self.post_X_fn = post_X_fn
        self.fm_loss_weight = fm_loss_weight
        self.direct_pred_losses = (
            torch.nn.ModuleList(direct_pred_losses) if direct_pred_losses is not None
            else torch.nn.ModuleList()
        )
        self.direct_pred_loss_weight = direct_pred_loss_weight

        self._instantiate_optimizer_and_scheduler()

    def forward(self, X_t, Y, t):
        """
        Trained to approximate the term (X_0 - X_1) given a X_t, Y and t.
        Note that X_t and Y must be in the feature domain of this instance's feature extractor.
        """
        dnn_input = torch.cat([X_t, Y], dim=1) # b,2*d,f,t
        return self.dnn(dnn_input, time_cond=t)

    def _step(self, batch, batch_idx, stage):
        x, y, info = batch
        assert (info['sr'] == self.sampling_rate).all()

        # Map to feature domain
        # We ignore preprocess_info here for now until we add data prediction losses
        Y, X, y_pre, x_pre, preprocess_info = self.preprocess(y, self.sampling_rate, x=x)

        # If post_Y_fn is set, apply it to Y
        if self.post_Y_fn is not None:
            Y = self.post_Y_fn(Y)
        # same for X
        if self.post_X_fn is not None:
            X = self.post_X_fn(X)

        # Generate sample along the probability path
        t = torch.rand(X.shape[0], device=x.device)
        t_bc = rearrange(t, "b -> b 1 1 1")  # broadcasted along C, F, T
        Z = torch.randn_like(Y)
        X_0 = Y + self.sigma_y * Z
        if self.sigma_x is not None:
            X_1 = X + self.sigma_x * Z
        else:
            X_1 = X
        X_t = (1 - t_bc)*X_0 + t_bc*X_1

        # Forward pass
        v_pred = self(X_t, Y, t)
        v_target = (X_1 - X_0)

        # L2 loss on the v-prediction. Sum along all but batch, mean along batch.
        errs = (v_target - v_pred).abs()
        fm_loss = 0.5*errs.square().flatten(start_dim=1).sum(dim=1).mean()
        loss = self.fm_loss_weight * fm_loss

        if self.direct_pred_losses:
            # Use a single-step Euler prediction
            X_1_hat = X_t + (1-t_bc)*v_pred  # single-step prediction
            if any(direct_pred_loss.domain == 'time' for direct_pred_loss in self.direct_pred_losses):
                # If any direct_pred_loss is in time domain, we need to postprocess X_1_hat into x_1_hat
                x_1_hat = self.postprocess(X_1_hat, preprocess_info, resample_to_orig=True)
                x_target = self.postprocess(X, preprocess_info, resample_to_orig=True)

            log_dict = {}
            total_pred_loss = 0.0
            for loss_impl in self.direct_pred_losses:
                if loss_impl.domain == 'feature':
                    loss_val = loss_impl(X, X_1_hat)
                else:  # time domain loss
                    # FIXME we assume here that all loss instances are working at the model's sampling rate,
                    # or that they do resampling from the model's sampling rate to their own
                    loss_val = loss_impl(x_target, x_1_hat)

                if loss_val.ndim > 1:
                    loss_val = loss_val.flatten(start_dim=1).sum(dim=1)
                if loss_val.ndim >= 1:
                    loss_val = loss_val.mean()

                assert loss_val.ndim == 0, \
                    f"Loss {loss_impl.name} returned a valued of ndim={loss_val.ndim} but it should be scalar-valued."

                if loss_val.isnan():
                    warnings.warn(f"Loss {loss_impl.name} returned NaN. Skipping.")
                else:
                    total_pred_loss += loss_impl.weight * loss_val
                log_dict[f'{stage}_{loss_impl.name}_loss'] = loss_val
                log_dict[f'{stage}_{loss_impl.name}_loss_weighted'] = loss_impl.weight * loss_val
            for key, val in log_dict.items():
                self.log(key, val, on_step=True, on_epoch=False, sync_dist=True)
            loss += self.direct_pred_loss_weight * total_pred_loss

        assert loss.ndim == 0
        return loss

    def enhance_from_features(self, Y, solver: str = 'euler', N: int = 5, return_all: bool = False):
        """
        Enhance directly from feature domain input Y.
        """
        Z = torch.randn_like(Y)
        X_0 = Y + self.sigma_y * Z
        X_t = solve_ode(
            X_0, v_theta=lambda X_t, t: self(X_t, Y, t),
            solver=solver, N=N)

        x_t = self.feature_extractor.invert(X_t)
        if return_all:
            return x_t, X_t
        return x_t

    def enhance(
        self, y, y_sr,
        solver: str ='euler',
        N: int = 5,
        grad_for_steps: Sequence[int] = (),
        return_all: bool = False,
        seed: int = None,
        solver_args: Optional[dict] = None,
    ):
        assert isinstance(N, int) and N > 0
        assert all(isinstance(step, int) for step in grad_for_steps)
        # Map all grad_for_steps to the range [0, N-1], particularly for -1, -2 etc
        grad_for_steps = [step % N for step in grad_for_steps]

        # if y.ndim > 1 and y.shape[-2] > 1:
        #     warnings.warn("Input y has more than one channel. We will only process the first channel and ignore the rest.")
        #     y = y[..., [0], :]

        squeeze_counter = 0
        while y.ndim < 3:
            y = y.unsqueeze(0)
            squeeze_counter += 1

        # Map to feature domain
        Y, y_pre, preprocess_info = self.preprocess(y, y_sr)

        # If post_Y_fn is set, apply it to Y
        if self.post_Y_fn is not None:
            Y = self.post_Y_fn(Y)

        if seed is not None:
            torch.random.manual_seed(seed)

        # Prepare variables for solver, then run it
        Z = torch.randn_like(Y)
        X_0 = Y + self.sigma_y * Z
        X_t = solve_ode(
            X_0, v_theta=lambda X_t, t: self(X_t, Y, t),
            solver=solver, N=N, solver_args=solver_args,
            grad_for_steps=grad_for_steps)

        x_t = self.postprocess(X_t, preprocess_info)
        for _ in range(squeeze_counter):
            X_t = X_t.squeeze(0)
            x_t = x_t.squeeze(0)

        if return_all:
            y = self.postprocess(Y, preprocess_info)
            y = y.squeeze(0) if y.ndim == 3 else y
            return x_t, X_t, y, Y, preprocess_info
        return x_t


class StoRMOnTheFlyFlowModel(GenericEnhancementModel):
    def __init__(self, sigma_e: float,
                 initial_predictor_config_path: str, initial_predictor_ckpt_path: Optional[str] = None,
                 dnn_input_mode: str = 'e_and_y', co_train_predictor: bool = False, sigma_x: float = 0.0,
                 extra_weight_silent_frames: float = 0.0, extra_weight_silent_frames_exponent: float = 2.0,
                 extra_weight_silent_frames_log: bool = False,
                 direct_pred_losses: Optional[Sequence[torch.nn.Module]] = None,
                 *args, **kwargs):
        """
        Create a new StoRMOnTheFlyFlowModel.

        Args:
            See GenericEnhancementModel for the common arguments.
            sigma_e: The noise stdev added to E to get X_0 (X_0 := E + sigma_e*Z) with i.i.d. standard Gaussian Z.
                If a float or 1-D tensor, it is the same for all T-F bins.
                If it is of shape (F, 1) it can model a diagonal covariance dependent on the frequency.
            initial_predictor_config_path: The path to the Hydra config file for the initial predictor.
            initial_predictor_ckpt_path: The path to the checkpoint file for the initial predictor.
            dnn_input_mode: The mode to use for the DNN input. Can be 'e_and_y' or 'e'.
            co_train_predictor: Whether to co-train the initial predictor or not.
            extra_weight_silent_frames: If > 0, the loss on frames where the clean signal is silent
                will effectively be weighted by (1 + extra_weight_silent_frames).
            extra_weight_silent_frames_log: If True, use log-magnitude for weighting silent frames.
        """
        super().__init__(*args, **kwargs)

        self.co_train_predictor = co_train_predictor
        self.dnn_input_mode = dnn_input_mode
        assert self.dnn_input_mode in ('e_and_y', 'e')

        if isinstance(sigma_e, torch.Tensor):
            self.register_buffer('sigma_e', sigma_e)
        else:
            assert isinstance(sigma_e, (float, int))
            self.sigma_e = float(sigma_e)
        if isinstance(sigma_x, torch.Tensor):
            self.register_buffer('sigma_x', sigma_x)
        else:
            assert isinstance(sigma_x, (float, int))
            self.sigma_x = float(sigma_x)

        self.extra_weight_silent_frames = extra_weight_silent_frames
        self.extra_weight_silent_frames_exponent = extra_weight_silent_frames_exponent
        self.extra_weight_silent_frames_log = extra_weight_silent_frames_log
        if isinstance(self.extra_weight_silent_frames, list):
            assert len(self.extra_weight_silent_frames_exponent) == len(self.extra_weight_silent_frames)
            assert len(self.extra_weight_silent_frames_log) == len(self.extra_weight_silent_frames)
        elif isinstance(self.extra_weight_silent_frames, (float, int)):
            self.extra_weight_silent_frames = [self.extra_weight_silent_frames]
            self.extra_weight_silent_frames_exponent = [self.extra_weight_silent_frames_exponent]
            self.extra_weight_silent_frames_log = [self.extra_weight_silent_frames_log]
        else:
            self.extra_weight_silent_frames = list(self.extra_weight_silent_frames)
            self.extra_weight_silent_frames_exponent = list(self.extra_weight_silent_frames_exponent)
            self.extra_weight_silent_frames_log = list(self.extra_weight_silent_frames_log)
            assert len(self.extra_weight_silent_frames_exponent) == len(self.extra_weight_silent_frames)
            assert len(self.extra_weight_silent_frames_log) == len(self.extra_weight_silent_frames)

        self.direct_pred_losses = (
            torch.nn.ModuleList(direct_pred_losses) if direct_pred_losses is not None
            else torch.nn.ModuleList()
        )

        # Load the initial predictor
        # Instantiate the object with Hydra config first.
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from pathlib import Path
        path = Path(initial_predictor_config_path)
        if GlobalHydra.instance().is_initialized():
            # e.g. when launched from train.py
            cfg = compose(config_name=path.stem)
        else:
            # e.g. when model loaded from most other code besides train.py
            with initialize_config_dir(config_dir=str(path.parent.absolute()), version_base="1.3"):
                cfg = compose(config_name=path.stem)
        self.initial_predictor = instantiate(cfg.model)

        assert isinstance(self.initial_predictor, DiscriminativeModel)
        # Set up the initial predictor
        if self.co_train_predictor:
            self.initial_predictor.train()
        else:
            self.initial_predictor.eval()
            # avoid nested submodules having grad enabled
            for param in self.initial_predictor.parameters():
                param.requires_grad = False
        # Load initial predictor's weights from the checkpoint, if provided (otherwise will train from scratch too)
        if initial_predictor_ckpt_path is not None:
            ckpt = torch.load(initial_predictor_ckpt_path, map_location='cpu', weights_only=False)
            self.initial_predictor.load_state_dict(ckpt['state_dict'])

        assert self.feature_extractor.matches(self.initial_predictor.feature_extractor), \
            "Initial predictor's feature extractor does not match this model's feature extractor."
        assert self.normalize_mode == self.initial_predictor.normalize_mode, \
            "Initial predictor's normalize_mode does not match this model's normalize_mode."
        assert self.sampling_rate == self.initial_predictor.sampling_rate, \
            "Initial predictor's sampling rate does not match this model's sampling rate."

        self._instantiate_optimizer_and_scheduler()

    def preprocess(self, *args, **kwargs):
        return self.initial_predictor.preprocess(*args, **kwargs)

    def get_trainable_parameters(self) -> Sequence[torch.nn.Parameter]:
        if self.co_train_predictor:
            return list(self.parameters()) + list(self.initial_predictor.parameters())
        else:
            return list(self.parameters())

    def forward(self, X_t, E, Y, t):
        """
        Trained to approximate the term (X_0 - X_1) given a X_t, Y and t.
        Note that X_t and Y must be in the feature domain of this instance's feature extractor.
        """
        if self.dnn_input_mode == 'e_and_y':
            dnn_input = torch.cat([X_t, E, Y], dim=1) # b,3*d,f,t
        else:
            assert self.dnn_input_mode == 'e'
            dnn_input = torch.cat([X_t, E], dim=1) # b,2*d,f,t

        return self.dnn(dnn_input, time_cond=t)

    def _step(self, batch, batch_idx, stage):
        x, y, info = batch
        assert (info['sr'] == self.sampling_rate).all()

        loss = torch.tensor(0.0, device=x.device)

        # Initial predictor forward pass, optionally loss and log
        Y, X, y_pre, x_pre, preprocess_info = self.preprocess(y, self.sampling_rate, x=x)

        # Run initial predictor
        if self.co_train_predictor:
            E = self.initial_predictor(Y)
            loss_IP, log_IP = self.initial_predictor._loss(
                X, E, preprocess_info=preprocess_info, stage=stage, return_log_as_dict=True)
            loss += loss_IP
            for key, val in log_IP.items():
                self.log(f'IP_{key}', val, on_step=True, on_epoch=False, sync_dist=True)
        else:
            with torch.no_grad():
                E = self.initial_predictor(Y)

        # Generate sample along the probability path
        t = torch.rand(X.shape[0], device=x.device)
        t_bc = rearrange(t, "b -> b 1 1 1")  # broadcasted along C, F, T
        Z = torch.randn_like(E)
        X_0 = E + self.sigma_e * Z
        if self.sigma_x is not None:
            X_1 = X + self.sigma_x * Z
        else:
            X_1 = X
        X_t = (1 - t_bc)*X_0 + t_bc*X_1

        # Forward pass
        v_pred = self(X_t, E, Y, t)
        v_target = (X_1 - X_0)

        # L2 loss on the v-prediction. Sum along all but batch, mean along batch.
        errs = (v_target - v_pred).abs()
        loss += 0.5*errs.square().flatten(start_dim=1).sum(dim=1).mean()

        if self.extra_weight_silent_frames is not None and self.extra_weight_silent_frames != 0.0:
            for l, (w, exponent, use_log) in enumerate(
                zip(self.extra_weight_silent_frames, self.extra_weight_silent_frames_exponent, self.extra_weight_silent_frames_log)
            ):
                if w == 0.0:
                    continue

                # (B, C, F, T)
                cutoff = 0.1
                silent_frame_mask = (X.abs().sum(dim=(1, 2), keepdim=True) < cutoff).float()
                X1pred = X_t + (1-t_bc)*v_pred

                if use_log:
                    x1pred_loss = ((X1pred.abs() + 1e-8)**exponent).clamp(min=1e-12).log()
                    silent_frame_losses = silent_frame_mask * x1pred_loss
                    silent_frame_loss = silent_frame_losses.mean()
                else:
                    x1pred_loss = (X1pred.abs() + 1e-8)**exponent
                    silent_frame_losses = silent_frame_mask * x1pred_loss
                    silent_frame_loss = silent_frame_losses.flatten(start_dim=1).sum(dim=1).mean()

                if l == 0:
                    self.log(f'{stage}_silent_frame_proportion',
                        silent_frame_mask.float().mean(), on_step=True, on_epoch=False, sync_dist=True)

                self.log(f'{stage}_silent_frame_loss_exp{exponent}{"_log" if use_log else ""}',
                    silent_frame_loss, on_step=True, on_epoch=False, sync_dist=True)
                loss += w * silent_frame_loss

        if self.direct_pred_losses:
            # Use a single-step Euler prediction
            X_1_hat = X_t + (1-t_bc)*v_pred  # single-step prediction
            if any(direct_pred_loss.domain == 'time' for direct_pred_loss in self.direct_pred_losses):
                # If any direct_pred_loss is in time domain, we need to postprocess X_1_hat into x_1_hat
                x_1_hat = self.postprocess(X_1_hat, preprocess_info, resample_to_orig=True)
                x_target = self.postprocess(X, preprocess_info, resample_to_orig=True)

            log_dict = {}
            total_pred_loss = 0.0
            for loss_impl in self.direct_pred_losses:
                if loss_impl.domain == 'feature':
                    loss_val = loss_impl(X, X_1_hat)
                else:  # time domain loss
                    # FIXME we assume here that all loss instances are working at the model's sampling rate,
                    # or that they do resampling from the model's sampling rate to their own
                    loss_val = loss_impl(x_target, x_1_hat)

                if loss_val.ndim > 1:
                    loss_val = loss_val.flatten(start_dim=1).sum(dim=1)
                if loss_val.ndim >= 1:
                    loss_val = loss_val.mean()

                assert loss_val.ndim == 0, \
                    f"Loss {loss_impl.name} returned a valued of ndim={loss_val.ndim} but it should be scalar-valued."

                if loss_val.isnan():
                    warnings.warn(f"Loss {loss_impl.name} returned NaN. Skipping.")
                else:
                    total_pred_loss += loss_impl.weight * loss_val
                log_dict[f'{stage}_{loss_impl.name}_loss'] = loss_val
                log_dict[f'{stage}_{loss_impl.name}_loss_weighted'] = loss_impl.weight * loss_val
            for key, val in log_dict.items():
                self.log(key, val, on_step=True, on_epoch=False, sync_dist=True)
            loss += total_pred_loss

        assert loss.ndim == 0
        return loss

    def enhance(
        self, y, y_sr,
        solver: str = 'euler',
        N: int = 5,
        grad_for_steps: Sequence[int] = (),
        return_all: bool = False,
        seed: int = None,
        solver_args: Optional[dict] = None,
    ):
        assert isinstance(N, int) and N > 0
        assert all(isinstance(step, int) for step in grad_for_steps)
        # Map all grad_for_steps to the range [0, N-1], particularly for -1, -2 etc
        grad_for_steps = [step % N for step in grad_for_steps]

        #if y.ndim > 1 and y.shape[-2] > 1:
        #    warnings.warn("Input y has more than one channel. We will only process the first channel and ignore the rest.")
        #    y = y[..., [0], :]

        squeeze_counter = 0
        while y.ndim < 3:
            y = y.unsqueeze(0)
            squeeze_counter += 1

        # Map to feature domain
        Y, y_pre, preprocess_info = self.initial_predictor.preprocess(y, y_sr)

        # Apply initial predictor
        with torch.set_grad_enabled(self.co_train_predictor):
            E = self.initial_predictor(Y)

        if seed is not None:
            torch.random.manual_seed(seed)

        # Prepare variables for solver, then run it
        Z = torch.randn_like(E)
        X_0 = E + self.sigma_e * Z
        X_t = solve_ode(
            X_0, v_theta=lambda X_t, t: self(X_t, E, Y, t),
            solver=solver, N=N, solver_args=solver_args,
            grad_for_steps=grad_for_steps)

        x_t = self.postprocess(X_t, preprocess_info)
        for _ in range(squeeze_counter):
            X_t = X_t.squeeze(0)
            x_t = x_t.squeeze(0)

        if return_all:
            y = self.postprocess(Y, preprocess_info)
            y = y.squeeze(0) if y.ndim == 3 else y
            e = self.postprocess(E, preprocess_info)
            e = e.squeeze(0) if e.ndim == 3 else e
            return x_t, X_t, y, Y, e, E, preprocess_info
        return x_t


class CustomRKSolverEnhancementModel(GenericEnhancementModel):
    def __init__(self,
                 wrapped_model: FlowModel | StoRMOnTheFlyFlowModel,
                 losses: Sequence[Loss],
                 optimizer_constructor: Callable,
                 base_solver: str,
                 N: int = 1,  # number of steps the solver is run with
                 b_min: float = 0.0,  # minimum value for all b_i, i.e. the minimum 'weight' for any gradient estimated along the RK step
                 c_max: float = 1.0,  # maximum value for all c_i, i.e. the maximum 'flow process time' we can use for sampling
                 a_penalty: Optional[Tuple[float,float,float]] = None,
                 co_train_wrapped_model: bool = False,
                 **kwargs):
        super().__init__(
            backbone=wrapped_model.dnn,
            sampling_rate=wrapped_model.sampling_rate,
            feature_extractor=wrapped_model.feature_extractor,
            normalize_mode=wrapped_model.normalize_mode,
            scheduler_config=None,
            optimizer_constructor=optimizer_constructor,
            **kwargs,
        )
        self.wrapped_model = wrapped_model

        self.co_train_wrapped_model = co_train_wrapped_model
        if co_train_wrapped_model:
            self.wrapped_model.train()
        else:
            self.wrapped_model.eval()
            for param in self.wrapped_model.parameters():
                param.requires_grad = False

        self.N = N
        self.b_min = b_min
        self.c_max = c_max
        self.base_solver = base_solver
        a, b, c = get_butcher_tableau(base_solver)
        self.a_raw = torch.nn.Parameter(a, requires_grad=True)  # no inverse mapping used here
        self.b_raw = torch.nn.Parameter(self.b_raw_from_b(b), requires_grad=True)
        self.c_raw = torch.nn.Parameter(self.c_raw_from_c(c), requires_grad=True)
        self.q = len(b)  # number of stages
        self.a_penalty = a_penalty

        self._instantiate_optimizer_and_scheduler()

        # Initialize losses (in eval mode)
        losses = torch.nn.ModuleList(losses)
        assert all(isinstance(loss, Loss) for loss in losses), \
            f"All losses must be instances of Loss but found {[type(loss) for loss in losses]}"
        for loss in losses:
            assert loss.domain in ('time', 'feature'), f"Unknown loss domain: {loss.domain}"
            loss.eval()
            for name, param in losses.named_parameters():
                param.requires_grad_(False)
        losses.eval()
        self.losses = losses

    def get_trainable_parameters(self):
        """Overwritten to only train a,b,c and freeze the wrapped model parameters"""
        if self.co_train_wrapped_model:
            return list(self.wrapped_model.parameters()) + [self.a_raw, self.b_raw, self.c_raw]
        else:
            return [self.a_raw, self.b_raw, self.c_raw]

    def b_raw_from_b(self, b):
        """
        inverse of the mapping from b_raw -> b, useful for initialization of b_raw
        """
        n = b.numel()
        residual = 1.0 - self.b_min * n
        if not torch.allclose(b.sum(), torch.tensor(1.0, device=b.device), atol=1e-4):
            raise ValueError("b must sum to 1.0")
        if (b < self.b_min).any():
            raise ValueError(f"b must have entries ≥ {self.b_min=}")
        s = torch.clamp((b - self.b_min) / residual, min=1e-8)  # avoid log(0)
        log_s = torch.log(s)
        log_s_centered = log_s - log_s.mean()
        b_raw = log_s_centered
        return b_raw

    def c_raw_from_c(self, c):
        # logit inverts the sigmoid used for mapping c_raw -> c
        diffs = torch.diff(c.clamp(max=self.c_max), dim=0)
        return torch.cat([
            torch.tensor([-np.inf], dtype=c.dtype, device=c.device),  # the first entry is not optimized over so just set it to -infty
            torch.logit(diffs.clamp(min=1e-8))  # avoid logit(0)
        ])

    @property
    def a(self):
        eps = 1e-8
        tril_mask = torch.tril(torch.ones_like(self.a_raw), diagonal=-1)  # strictly lower triangular mask
        a_tilde = self.a_raw * tril_mask  # zero out diagonal and upper
        # ensure the usual (but not necessary!) constraint that the rows in a sum to the corresponding c
        sum_lower = a_tilde.sum(dim=1, keepdim=True)  # sum over j < i, shape (q,1)
        a = a_tilde * (self.c.unsqueeze(1) / (sum_lower + eps))
        return a

    @property
    def b(self):
        n = self.b_raw.shape[0]
        eps_mass = self.b_min * n
        if eps_mass >= 1.0:
            raise ValueError("b_min too large for number of elements")
        # Softmax of raw logits scaled to fill the remaining mass
        soft = torch.softmax(self.b_raw, dim=0)
        b = (1.0 - eps_mass) * soft + self.b_min
        return b

    @property
    def c(self):
        device = self.c_raw.device
        c_tilde = torch.cat([torch.zeros(1, device=device), torch.sigmoid(self.c_raw[1:])], dim=0)  # (q,)
        c_raw_ascending = torch.cumsum(c_tilde, dim=0)
        # Normalize to [0, c_max] but only if the rightmost point is > c_max
        c = c_raw_ascending * (self.c_max / c_raw_ascending[-1]).clamp(max=1.0)
        return c

    def forward(self, y, y_sr=None, return_all=False):
        # call enhance on wrapped model with (A, b, c) solver args
        y_sr = y_sr if y_sr is not None else self.sampling_rate

        result = self.wrapped_model.enhance(
            y, y_sr,
            grad_for_steps=range(self.q) if self.training else (),
            N=self.N,
            solver='rk',
            solver_args={'a': self.a, 'b': self.b, 'c': self.c},
            return_all=return_all,
        )
        return result

    def enhance(self, y, y_sr, *args, return_all=False, **kwargs):
        return self.forward(y, y_sr=y_sr, return_all=return_all)

    def get_a_penalty_loss(self):
        if self.a_penalty is None:
            return 0
        a_min, a_max, penalty_weight = self.a_penalty
        a = self.a
        # (x-l)^2 when x<l, 0 when l<=x<=r, (x-r)^2 when x>r
        penalties = torch.where(a < a_min, (a - a_min)**2, torch.where(a > a_max, (a - a_max)**2, torch.tensor(0.0, device=a.device)))
        return penalty_weight * penalties.sum()

    def _loss(self, X, X_hat, preprocess_info, stage, return_log_as_dict=False):
        loss_dict, log_dict = self._loss_orig(
            X, X_hat, preprocess_info=preprocess_info, stage=stage, return_log_as_dict=return_log_as_dict)
        total_loss = torch.tensor(0., device=X.device)
        total_loss += sum(loss_dict.values())
        total_loss += self.get_a_penalty_loss()
        return total_loss, log_dict

    def _loss_orig(self, X, X_hat, preprocess_info, stage, return_log_as_dict=False):
        x_post = self.wrapped_model.postprocess(X, preprocess_info, resample_to_orig=False)
        x_hat_post = self.wrapped_model.postprocess(X_hat, preprocess_info, resample_to_orig=False)
        loss_dict = {}
        log_dict = {}

        for loss_impl in self.losses:
            if loss_impl.domain == 'feature':
                loss_val = loss_impl(X, X_hat)
            else:  # time domain loss
                # FIXME we assume here that all loss instances are working at the model's sampling rate,
                # or that they do resampling from the model's sampling rate to their own
                loss_val = loss_impl(x_post, x_hat_post)
            assert loss_val.ndim == 0, \
                f"Loss {loss_impl.name} returned a valued of ndim={loss_val.ndim} but it should be scalar-valued."

            if loss_val.isnan():
                warnings.warn(f"Loss {loss_impl.name} returned NaN. Skipping.")
            else:
                loss_dict[loss_impl.name] = loss_impl.weight * loss_val
            log_dict[f'{stage}_{loss_impl.name}_loss'] = loss_val
            log_dict[f'{stage}_{loss_impl.name}_loss_weighted'] = loss_impl.weight * loss_val

        if return_log_as_dict:
            return loss_dict, log_dict
        else:
            for key, val in log_dict.items():
                self.log(key, val, on_step=True, on_epoch=False, sync_dist=True)
            return loss_dict, None

    def _log_params(self):
        """Logs the current parameters a,b,c as a figure."""
        a, b, c = self.a.detach().cpu().numpy(), self.b.detach().cpu().numpy(), self.c.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Heatmap for A
        im = axes[0].imshow(a, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
        axes[0].set_title("Matrix A")
        plt.colorbar(im, ax=axes[0])
        # Plot the actual a[i,j] value over each lower triangular entry
        for i in range(1, self.q):
            for j in range(i):
                axes[0].text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", color="black")
        # Line plots for b and c
        axes[1].plot(b, label="b")
        axes[1].plot(c, label="c")
        axes[1].legend()
        axes[1].set_title("Vectors b and c")
        self.log_figure("abc", fig, self.global_step)
        plt.close(fig)

    def _step(self, batch, batch_idx, stage):
        x, y, info = batch
        assert (info['sr'] == self.sampling_rate).all()

        Y, X, y_pre, x_pre, preprocess_info = self.wrapped_model.preprocess(y, self.sampling_rate, x=x)
        x_hat, X_hat, *_ = self(y, return_all=True)
        loss, _ = self._loss(X, X_hat, preprocess_info=preprocess_info, stage=stage, return_log_as_dict=False)

        if self.global_step % self.trainer.log_every_n_steps == 0:
            self._log_params()
        return loss
