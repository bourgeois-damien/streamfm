"""Modal entrypoints that run test-set eval on remote GPUs.

One function per hardware target (CPU/T4/L4/L40S/A100); each runs the eval on
Modal and returns the enhanced audio and metrics. Invoked via ``modal run``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import modal

REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

from experiments.core.paths import make_benchmark_paths
from experiments.evaluation.modal_defaults import resolve_modal_data_path
from experiments.evaluation.results import DEFAULT_EVAL_WANDB_PROJECT, log_eval_result_to_wandb, record_eval_result
from experiments.evaluation.runner import run_test_set_inference
from experiments.core.modal_cache import configure_shared_modal_cache


CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_repo_root() -> Path:
    """Find the local repo root before Modal copies files into the image."""
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current_file.parent


LOCAL_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .apt_install("build-essential")
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "audiomentations==0.41.0",
        "distillmos==0.9.1",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "matplotlib==3.10.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pesq==0.0.4",
        "pystoi==0.3.3",
        "pytorch-lightning==2.5.1.post0",
        "scipy==1.15.2",
        "tensorboard==2.18.0",
        "torch-pesq==0.1.2",
        "torchinfo==1.8.0",
        "tqdm==4.67.1",
        "wandb==0.19.1",
    )
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    # Exclude __pycache__/*.pyc: they are rewritten on first import and, if another
    # sweep runs concurrently, Modal aborts the build ("modified during build process").
    .add_local_dir(
        str(LOCAL_ROOT / "experiments"),
        remote_path=f"{REMOTE_ROOT}/experiments",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(LOCAL_ROOT / "flow_autoparams"), remote_path=f"{REMOTE_ROOT}/flow_autoparams")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse", ignore=["**/__pycache__/**", "**/*.pyc"])
)

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

# Compressed (decoupled-SVD) checkpoints live in checkpoints/compressed/. Upload
# them flat into the remote checkpoints root so sweeps can reference them by base
# name (e.g. ckpt: streamfm_stftpr_k6.ckpt) and the checkpoint-root search finds them.
_compressed_dir = LOCAL_ROOT / "checkpoints" / "compressed"
if _compressed_dir.is_dir():
    for compressed_checkpoint in sorted(_compressed_dir.glob("*.ckpt")):
        image = image.add_local_file(
            str(compressed_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{compressed_checkpoint.name}",
        )


app = modal.App("streamfm-eval", image=image)


def _parse_wandb_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _configure_persistent_cache_env(hardware: str) -> dict[str, str]:
    """Point all runs on one hardware tier at shared compiler caches."""
    return configure_shared_modal_cache(volume_root=VOLUME_ROOT, hardware=hardware)


def _remote_paths():
    """Build evaluation paths inside the Modal container."""
    return make_benchmark_paths(
        repo_root=REMOTE_ROOT,
        config_dir=f"{REMOTE_ROOT}/config",
        checkpoint_roots=(
            f"{VOLUME_ROOT}/checkpoints",
            f"{REMOTE_ROOT}/checkpoints",
        ),
    )


def _run_modal_eval(
    *,
    hardware: str,
    task: str,
    config_name: str,
    ckpt: str,
    split: str,
    data_path: str,
    data_format: str,
    part: str,
    pipeline: str,
    execution: str,
    solver: str,
    steps: int,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    seed: int,
    dtype: str,
    matmul_precision: str,
    crop_mode: str,
    memory_format: str,
    num_threads: int,
    num_interop_threads: int,
    output_dir: str,
    run_name: str,
    overwrite: bool,
    save_inputs: bool,
    continue_on_error: bool,
    config_overrides: list[str] | tuple[str, ...] = (),
) -> dict:
    """Run one test-set inference job inside Modal on CPU or CUDA."""
    import torch

    hardware = hardware.upper()
    device = torch.device("cpu" if hardware == "CPU" else "cuda")
    if execution.lower().replace("-", "_") == "cuda_graph":
        raise ValueError("Modal eval currently supports eager/compiled only, not cuda_graph.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this Modal container.")

    data_path = resolve_modal_data_path(data_path, task=task, split=split, volume_root=VOLUME_ROOT)
    cache_info = _configure_persistent_cache_env(hardware)
    return run_test_set_inference(
        task=task,
        config_name=config_name,
        ckpt=ckpt,
        split=split,
        data_path=data_path,
        data_format=data_format,
        part=part,
        pipeline=pipeline,
        execution=execution,
        solver=solver,
        steps=steps,
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
        seed=seed,
        model_dtype_name=dtype,
        float32_matmul_precision=matmul_precision,
        model_memory_format=memory_format,
        crop_mode=crop_mode,
        device=device,
        paths=_remote_paths(),
        backend="modal",
        hardware=hardware,
        output_dir=output_dir or f"{VOLUME_ROOT}/outputs/eval_runs",
        run_name=run_name,
        overwrite=overwrite,
        save_inputs=save_inputs,
        continue_on_error=continue_on_error,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        cache_info=cache_info,
        config_overrides=config_overrides,
    )


@app.function(timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def eval_cpu(**kwargs):
    """Run evaluation on Modal CPU."""
    return _run_modal_eval(hardware="CPU", **kwargs)


@app.function(gpu="T4", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def eval_t4(**kwargs):
    """Run evaluation on an NVIDIA T4."""
    return _run_modal_eval(hardware="T4", **kwargs)


@app.function(gpu="L4", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def eval_l4(**kwargs):
    """Run evaluation on an NVIDIA L4."""
    return _run_modal_eval(hardware="L4", **kwargs)


@app.function(gpu="L40S", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def eval_l40s(**kwargs):
    """Run evaluation on an NVIDIA L40S."""
    return _run_modal_eval(hardware="L40S", **kwargs)


@app.function(gpu="A100", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def eval_a100(**kwargs):
    """Run evaluation on an NVIDIA A100."""
    return _run_modal_eval(hardware="A100", **kwargs)


MODAL_FUNCTIONS: dict[str, Callable] = {
    "CPU": eval_cpu,
    "T4": eval_t4,
    "L4": eval_l4,
    "L40S": eval_l40s,
    "A100": eval_a100,
}


@app.local_entrypoint()
def main(
    hardware: str = "L4",
    task: str = "se",
    config_name: str = "",
    config_override: str = "",
    ckpt: str = "",
    split: str = "test",
    data_path: str = "",
    data_format: str = "",
    part: str = "model",
    pipeline: str = "offline",
    execution: str = "eager",
    solver: str = "euler",
    steps: int = 5,
    limit: int = 0,
    offset: int = 0,
    selection: str = "random",
    selection_seed: int = 42,
    seed: int = 42,
    dtype: str = "fp32",
    matmul_precision: str = "high",
    crop_mode: str = "full",
    memory_format: str = "contiguous",
    num_threads: int = 0,
    num_interop_threads: int = 0,
    output_dir: str = "",
    run_name: str = "",
    history_json: str = "",
    overwrite: bool = False,
    save_inputs: bool = False,
    continue_on_error: bool = False,
    wandb: bool = False,
    wandb_project: str = DEFAULT_EVAL_WANDB_PROJECT,
    wandb_entity: str = "",
    wandb_group: str = "",
    wandb_mode: str = "",
    wandb_tags: str = "",
):
    """Launch Modal test-set inference and record the result locally."""
    selected_hardware = hardware.upper()
    if selected_hardware not in MODAL_FUNCTIONS:
        supported = ", ".join(MODAL_FUNCTIONS)
        raise ValueError(f"Unsupported Modal hardware '{selected_hardware}'. Supported values: {supported}")

    data_path = resolve_modal_data_path(data_path, task=task, split=split, volume_root=VOLUME_ROOT)
    config_overrides = [item for item in config_override.split("\n") if item]
    result = MODAL_FUNCTIONS[selected_hardware].remote(
        task=task,
        config_name=config_name,
        config_overrides=config_overrides,
        ckpt=ckpt,
        split=split,
        data_path=data_path,
        data_format=data_format,
        part=part,
        pipeline=pipeline,
        execution=execution,
        solver=solver,
        steps=steps,
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
        seed=seed,
        dtype=dtype,
        matmul_precision=matmul_precision,
        crop_mode=crop_mode,
        memory_format=memory_format,
        num_threads=num_threads,
        num_interop_threads=num_interop_threads,
        output_dir=output_dir,
        run_name=run_name,
        overwrite=overwrite,
        save_inputs=save_inputs,
        continue_on_error=continue_on_error,
    )
    command = {
        "backend": "modal",
        "hardware": selected_hardware,
        "task": task,
        "config_name": config_name,
        "config_overrides": config_overrides,
        "ckpt": ckpt,
        "split": split,
        "data_path": data_path,
        "data_format": data_format,
        "part": part,
        "pipeline": pipeline,
        "execution": execution,
        "solver": solver,
        "steps": steps,
        "limit": limit,
        "offset": offset,
        "selection": selection,
        "selection_seed": selection_seed,
        "seed": seed,
        "dtype": dtype,
        "matmul_precision": matmul_precision,
        "crop_mode": crop_mode,
        "memory_format": memory_format,
        "num_threads": num_threads,
        "num_interop_threads": num_interop_threads,
        "output_dir": output_dir,
        "run_name": run_name,
        "overwrite": overwrite,
        "save_inputs": save_inputs,
        "continue_on_error": continue_on_error,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_mode": wandb_mode,
        "wandb_tags": wandb_tags,
    }
    record_eval_result(result=result, history_json=history_json, command=command)
    if wandb:
        log_eval_result_to_wandb(
            result=result,
            command=command,
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            mode=wandb_mode,
            tags=_parse_wandb_tags(wandb_tags),
        )
