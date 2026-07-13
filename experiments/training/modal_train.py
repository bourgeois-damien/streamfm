"""Launch resumable StreamFM training jobs on Modal.

The launcher deliberately executes ``train.py`` in a subprocess.  This keeps
the normal Hydra entrypoint intact and lets PyTorch Lightning manage its own
processes when a Modal function has more than one GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

import modal


REMOTE_ROOT = "/root/streamfm"
DATA_ROOT = "/data"
RUNS_ROOT = "/runs"
TRAINING_ROOT = f"{RUNS_ROOT}/training"

DATA_VOLUME = modal.Volume.from_name("streamfm-cache", create_if_missing=True)
RUNS_VOLUME = modal.Volume.from_name("streamfm-runs", create_if_missing=True)
WANDB_SECRET = modal.Secret.from_name("wandb")


def _find_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    raise RuntimeError("Could not find the StreamFM repository root.")


LOCAL_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .apt_install("build-essential", "ffmpeg", "libsndfile1")
    .pip_install(
        "audiomentations==0.41.0",
        "distillmos==0.9.1",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "matplotlib==3.10.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pesq==0.0.4",
        "pyroomacoustics==0.6.0",
        "pystoi==0.3.3",
        "pytorch-lightning==2.5.1.post0",
        "scikit-image==0.25.1",
        "scipy==1.15.2",
        "seaborn==0.13.2",
        "tensorboard==2.18.0",
        "torch==2.7.0",
        "torch-pesq==0.1.2",
        "torchaudio==2.7.0",
        "torchinfo==1.8.0",
        "tqdm==4.67.1",
        "transformers==4.51.3",
        "wandb==0.19.1",
    )
    .add_local_file(str(LOCAL_ROOT / "train.py"), remote_path=f"{REMOTE_ROOT}/train.py")
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    .add_local_dir(str(LOCAL_ROOT / "flow_autoparams"), remote_path=f"{REMOTE_ROOT}/flow_autoparams")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse")
)

# These project checkpoints are convenient seeds for the first fine-tuning
# runs.  New checkpoints should live on ``streamfm-runs`` rather than being
# baked into future images.
for checkpoint_name in (
    "streamfm_stftpr.ckpt",
    "streamfm_se_predgen.ckpt",
    "streamfm_se_predictor.ckpt",
    "streamfm_bwe.ckpt",
    "streamfm_derev.ckpt",
    "streamfm_lyra.ckpt",
    "streamfm_melflow.ckpt",
):
    local_checkpoint = LOCAL_ROOT / "checkpoints" / checkpoint_name
    if local_checkpoint.exists():
        image = image.add_local_file(
            str(local_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{checkpoint_name}",
        )


app = modal.App("streamfm-train", image=image)


MODAL_PATH_OVERRIDES = (
    f"ewv2_train_csv={DATA_ROOT}/datasets/EARS-WHAM_v2_16k/train.csv",
    f"ewv2_valid_csv={DATA_ROOT}/datasets/EARS-WHAM_v2_16k/valid.csv",
    f"ewv2_test_csv={DATA_ROOT}/datasets/EARS-WHAM_v2_16k/test.csv",
    f"erv2_train_csv={DATA_ROOT}/datasets/EARS-Reverb_v2_16k/train.csv",
    f"erv2_valid_csv={DATA_ROOT}/datasets/EARS-Reverb_v2_16k/valid.csv",
    f"erv2_test_csv={DATA_ROOT}/datasets/EARS-Reverb_v2_16k/test.csv",
    f"ewv2_bwr_train_path={DATA_ROOT}/datasets/EARS_v2_16k_BWR/train",
    f"ewv2_bwr_valid_path={DATA_ROOT}/datasets/EARS_v2_16k_BWR/valid",
    f"ewv2_bwr_test_path={DATA_ROOT}/datasets/EARS_v2_16k_BWR/test",
    f"ewv2_lyra_train_path={DATA_ROOT}/datasets/EARS_v2_16k_Lyra/dataset_highpass75/3200bit/train",
    f"ewv2_lyra_valid_path={DATA_ROOT}/datasets/EARS_v2_16k_Lyra/dataset_highpass75/3200bit/valid",
    f"ewv2_lyra_test_path={DATA_ROOT}/datasets/EARS_v2_16k_Lyra/dataset_highpass75/3200bit/test",
)


def _run_name_or_raise(run_name: str) -> str:
    if not run_name:
        raise ValueError("run_name is required so checkpoints and W&B runs are never ambiguous.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name):
        raise ValueError("run_name may contain only letters, numbers, '.', '_' and '-'.")
    return run_name


def _split_overrides(config_override: str) -> list[str]:
    return [item.strip() for item in config_override.split("\n") if item.strip()]


def _resolve_seed_checkpoint(seed_checkpoint: str) -> Path:
    if not seed_checkpoint:
        raise ValueError("mode='finetune' requires seed_checkpoint.")
    requested = Path(seed_checkpoint)
    candidates = [requested] if requested.is_absolute() else [
        Path(RUNS_ROOT) / "checkpoints" / requested,
        Path(DATA_ROOT) / "checkpoints" / requested,
        Path(REMOTE_ROOT) / "checkpoints" / requested,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find seed checkpoint '{seed_checkpoint}'. Searched: {searched}")


def _training_command(
    *,
    config_name: str,
    mode: str,
    run_dir: Path,
    seed_checkpoint: str,
    checkpoint_every_n_steps: int,
    max_steps: int,
    batch_size: int,
    precision: str,
    devices: int,
    wandb_project: str,
    config_override: str,
) -> list[str]:
    if mode not in {"from_scratch", "finetune", "resume"}:
        raise ValueError("mode must be one of: from_scratch, finetune, resume.")
    if checkpoint_every_n_steps <= 0:
        raise ValueError("checkpoint_every_n_steps must be positive.")

    overrides = [
        *MODAL_PATH_OVERRIDES,
        f"logger.project={wandb_project}",
        f"logger.save_dir={run_dir}",
        f"+wandb_run_name={run_dir.name}",
        f"+req_ckpt_path={run_dir}",
        f"+checkpoint_every_n_train_steps={checkpoint_every_n_steps}",
        f"trainer_constructor.devices={devices}",
    ]
    # One GPU does not need DDP.  For multiple GPUs, the config's DDPStrategy
    # remains in place and works because this module launches train.py as a
    # subprocess instead of invoking Lightning inside the Modal function.
    if devices == 1:
        overrides.append("trainer_constructor.strategy=auto")
    if max_steps > 0:
        overrides.extend((f"num_training_steps={max_steps}", f"trainer_constructor.max_steps={max_steps}"))
    if batch_size > 0:
        overrides.append(f"model.data_module.batch_size={batch_size}")
    if precision:
        overrides.append(f"trainer_constructor.precision={precision}")

    if mode == "finetune":
        overrides.append(f"+load_model_from_ckpt={_resolve_seed_checkpoint(seed_checkpoint)}")
    elif mode == "resume":
        last_checkpoint = run_dir / "checkpoints" / "last.ckpt"
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume: no last checkpoint at {last_checkpoint}.")
        overrides.append(f"resume_from_ckpt={last_checkpoint}")

    # User overrides come last so custom datasets and experimental settings can
    # intentionally replace the portable Modal defaults above.
    overrides.extend(_split_overrides(config_override))
    return [sys.executable, "train.py", "--config-name", config_name, *overrides]


def _run_training(
    *,
    hardware: str,
    devices: int,
    config_name: str,
    mode: str,
    run_name: str,
    seed_checkpoint: str,
    checkpoint_every_n_steps: int,
    max_steps: int,
    batch_size: int,
    precision: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_tags: str,
    config_override: str,
) -> dict:
    run_name = _run_name_or_raise(run_name)
    RUNS_VOLUME.reload()
    run_dir = Path(TRAINING_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_last_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    # A Modal retry re-invokes this function with the original arguments.  If
    # the first attempt already produced a recovery checkpoint, resume it
    # automatically instead of silently starting the experiment over.
    effective_mode = "resume" if mode != "resume" and existing_last_checkpoint.is_file() else mode

    command = _training_command(
        config_name=config_name,
        mode=effective_mode,
        run_dir=run_dir,
        seed_checkpoint=seed_checkpoint,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        max_steps=max_steps,
        batch_size=batch_size,
        precision=precision,
        devices=devices,
        wandb_project=wandb_project,
        config_override=config_override,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": REMOTE_ROOT,
            "WANDB_PROJECT": wandb_project,
            "WANDB_RUN_ID": run_name,
            "WANDB_RESUME": "allow",
            "WANDB_NAME": run_name,
        }
    )
    if wandb_entity:
        environment["WANDB_ENTITY"] = wandb_entity
    if wandb_group:
        environment["WANDB_RUN_GROUP"] = wandb_group
    if wandb_tags:
        environment["WANDB_TAGS"] = wandb_tags

    try:
        subprocess.run(command, cwd=REMOTE_ROOT, env=environment, check=True)
    finally:
        # Modal also performs background/final commits, but an explicit commit
        # makes the latest checkpoint available immediately to a resume run.
        RUNS_VOLUME.commit()

    last_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    return {
        "status": "completed",
        "hardware": hardware,
        "devices": devices,
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_exists": last_checkpoint.is_file(),
        "wandb_project": wandb_project,
        "wandb_run_id": run_name,
        "command": command,
    }


@app.function(
    gpu="L4",
    timeout=24 * 60 * 60,
    retries=2,
    volumes={DATA_ROOT: DATA_VOLUME, RUNS_ROOT: RUNS_VOLUME},
    secrets=[WANDB_SECRET],
)
def train_l4(**kwargs):
    return _run_training(hardware="L4", devices=1, **kwargs)


@app.function(
    gpu="L40S",
    timeout=24 * 60 * 60,
    retries=2,
    volumes={DATA_ROOT: DATA_VOLUME, RUNS_ROOT: RUNS_VOLUME},
    secrets=[WANDB_SECRET],
)
def train_l40s(**kwargs):
    return _run_training(hardware="L40S", devices=1, **kwargs)


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    retries=2,
    volumes={DATA_ROOT: DATA_VOLUME, RUNS_ROOT: RUNS_VOLUME},
    secrets=[WANDB_SECRET],
)
def train_a100(**kwargs):
    return _run_training(hardware="A100", devices=1, **kwargs)


@app.function(
    gpu="A100:2",
    timeout=24 * 60 * 60,
    retries=2,
    volumes={DATA_ROOT: DATA_VOLUME, RUNS_ROOT: RUNS_VOLUME},
    secrets=[WANDB_SECRET],
)
def train_a100_2x(**kwargs):
    return _run_training(hardware="A100", devices=2, **kwargs)


TRAIN_FUNCTIONS: dict[str, Callable] = {
    "L4": train_l4,
    "L40S": train_l40s,
    "A100": train_a100,
    "A100:2": train_a100_2x,
}


@app.local_entrypoint()
def main(
    hardware: str = "L4",
    config_name: str = "streamfm_bwe",
    mode: str = "from_scratch",
    run_name: str = "",
    seed_checkpoint: str = "",
    checkpoint_every_n_steps: int = 500,
    max_steps: int = 0,
    batch_size: int = 0,
    precision: str = "",
    wandb_project: str = "streamflow",
    wandb_entity: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
    config_override: str = "",
):
    """Start a Modal training job and print its durable run metadata."""
    selected_hardware = hardware.upper()
    if selected_hardware not in TRAIN_FUNCTIONS:
        supported = ", ".join(TRAIN_FUNCTIONS)
        raise ValueError(f"Unsupported hardware '{hardware}'. Supported values: {supported}")

    result = TRAIN_FUNCTIONS[selected_hardware].remote(
        config_name=config_name,
        mode=mode,
        run_name=run_name,
        seed_checkpoint=seed_checkpoint,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        max_steps=max_steps,
        batch_size=batch_size,
        precision=precision,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        config_override=config_override,
    )
    print(json.dumps(result, indent=2))
