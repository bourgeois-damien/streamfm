import copy
import os
import warnings
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import omegaconf

import random
import numpy as np
import torch, torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sgmse.util.distributed import is_rank_zero


def is_wandb(logger):
    return isinstance(logger, pl.loggers.WandbLogger)


def model_load_from_ckpt(model, ckpt_file):
    """
    Load model weights from a checkpoint file. Here, we **only** load model weights, ignoring other states.
    Useful e.g. for loading a model for finetuning after performing model compression.

    NOTE: This is different from resume_from_ckpt=... passed to pl.Trainer.fit, which would include optimizer
    state etc., and usually does *not* work when you don't want to immediately continue training from the checkpoint.
    """
    print(f"Loading model weights from checkpoint {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


@hydra.main(config_path="./config/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    # Initialize logger, trainer, model, datamodule
    model = instantiate(cfg.model)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed + 1)
        torch.manual_seed(cfg.seed + 2)
        torch.cuda.manual_seed(cfg.seed + 3)

    config_name = getattr(cfg, 'config_name', HydraConfig.get()['job']['config_name'])
    logger_constructor = instantiate(cfg.logger)
    logger = logger_constructor(name=cfg.get('wandb_run_name', config_name))
    if is_wandb(logger):
        run = logger.experiment
        run_id = run.id
        # Store run ID in config so it's saved in each checkpoint
        if cfg.run_id is None:
            cfg.run_id = run_id() if callable(run_id) else str(run_id)

        if is_rank_zero():
            logged_cfg = {**get_loggable_config(cfg), 'config_name': config_name}
            run.log_code(os.path.dirname(__file__), include_fn=lambda path: path.endswith(".py"))
            # Nicer hparam/config logging for W&B. Using model.save_hyperparameters() would turn all vals into strings
            run.config.update(logged_cfg, allow_val_change=True)

    # Set up some global torch options
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    torchaudio.set_audio_backend("ffmpeg")

    # Initialize the Trainer and the DataModule
    trainer_constructor = instantiate(cfg.trainer_constructor)
    trainer = trainer_constructor(logger=logger, default_root_dir=cfg.get('req_ckpt_path', None))
    assert isinstance(trainer, pl.Trainer)

    # Long-running cloud jobs need a recovery point more often than once per
    # validation epoch.  Keep this opt-in so the published experiment configs
    # retain their original checkpoint policy, while launchers can request
    # durable step-based snapshots with a Hydra override.
    checkpoint_every_n_train_steps = cfg.get('checkpoint_every_n_train_steps', None)
    if checkpoint_every_n_train_steps is not None:
        checkpoint_every_n_train_steps = int(checkpoint_every_n_train_steps)
        if checkpoint_every_n_train_steps <= 0:
            raise ValueError("checkpoint_every_n_train_steps must be a positive integer.")
        checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="step={step:08d}",
                every_n_train_steps=checkpoint_every_n_train_steps,
                # No monitored quantity here -- these are recovery points, not
                # best-model selection -- and Lightning only accepts 0, 1 or -1
                # for save_top_k when monitor is None.  1 keeps the newest
                # snapshot, save_last keeps last.ckpt in step with it.
                save_top_k=1,
                save_last=True,
            )
        )
        print(
            "Enabled step-based recovery checkpoints every "
            f"{checkpoint_every_n_train_steps} training steps in {checkpoint_dir}."
        )

    # Print # of devices that trainer uses
    print("Number of devices: ", trainer.num_devices)

    # Load model from checkpoint, if specified
    # NOTE: this is different from resume_from_ckpt, see model_load_from_ckpt docstring for details
    if getattr(cfg, 'load_model_from_ckpt', None) is not None:
        model = model_load_from_ckpt(model, cfg.load_model_from_ckpt)

    # Post-processing model, if requested by config -- e.g. for model compression before finetuning
    if getattr(cfg, 'transform_model_backbone_fn', None) is not None:
        print("[...] Transforming model backbone as specified in config...")
        transform_fn = instantiate(cfg.transform_model_backbone_fn)
        model.transform_backbone_(transform_fn)
        print("[ . ] Transformed model backbone.")

    # Load model from checkpoint *post transform*, if specified
    if getattr(cfg, 'load_model_from_ckpt_after_transform', None) is not None:
        if not getattr(cfg, 'transform_model_backbone_fn', None) is not None:
            warnings.warn(
                "You are loading model weights after model transformation (`load_model_from_ckpt_after_transform`), "
                "but you did not specify any model transformation in the config!")
        model = model_load_from_ckpt(model, cfg.load_model_from_ckpt_after_transform)

    # Train model
    trainer.fit(model, ckpt_path=cfg.resume_from_ckpt)


def get_loggable_config(cfg: omegaconf.DictConfig) -> dict:
    cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = convert_list_of_dicts(cfg)
    return cfg


def convert_list_of_dicts(config):
    config = copy.deepcopy(config)
    for key, value in list(config.items()):
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            config[key] = {str(i): item for i, item in enumerate(value)}
        if isinstance(value, dict):
            config[key] = convert_list_of_dicts(value)
    return config


if __name__ == "__main__":
    main()
