"""Modal entrypoint: score a depth-pruned model after its healing fine-tune.

The ablation curve (``experiments/pruning/modal_ablation.py``) measures pruning
*zero-shot*: the teacher is loaded strict, then blocks are swapped for identities
with no retraining.  This entrypoint closes the loop by scoring the fine-tuned
result, so the three points are directly comparable on the same subset:

    teacher (k=0)  ->  pruned zero-shot (k)  ->  pruned + healed (k)

The healed checkpoint was *trained* in the pruned shape, so its state_dict has no
entries for the dropped blocks.  It therefore needs the transform applied BEFORE
the load (``backbone_pre_transform``), which is the mirror image of the zero-shot
path.  Everything else -- pipeline, solver, subset selection, seed -- is kept
identical to the ablation defaults so the numbers line up.

    modal run experiments/pruning/modal_eval_healed.py \
        --ckpt /runs/training/derev-prune-k3-ft/checkpoints/last.ckpt --k 3
"""

from __future__ import annotations

import json
import sys

import modal

from experiments.evaluation.modal_streamfm_eval import (
    CACHE_VOLUME,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image,
)
from experiments.pruning.modal_ablation import DEREV_RANKING, METRIC_NAMES

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

# Healing checkpoints are written by experiments/training/modal_train.py onto the
# runs volume, which the evaluation image does not normally mount.
RUNS_VOLUME = modal.Volume.from_name("streamfm-runs", create_if_missing=True)
RUNS_ROOT = "/runs"

app = modal.App("streamfm-prune-eval-healed", image=image)


@app.function(
    gpu="L4",
    timeout=14400,
    volumes={VOLUME_ROOT: CACHE_VOLUME, RUNS_ROOT: RUNS_VOLUME},
)
def evaluate(
    task: str,
    config_name: str,
    ckpt: str,
    drop: list[str],
    data_path: str,
    data_format: str,
    split: str,
    pipeline: str,
    execution: str,
    solver: str,
    steps: int,
    limit: int,
    selection: str,
    selection_seed: int,
    seed: int,
    crop_mode: str,
    with_distillmos: bool,
    run_name: str,
) -> dict:
    from functools import partial
    from pathlib import Path

    import torch

    from experiments.evaluation.modal_defaults import resolve_modal_data_path
    from experiments.evaluation.modal_streamfm_eval import _configure_persistent_cache_env, _remote_paths
    from experiments.evaluation.runner import run_test_set_inference
    from experiments.evaluation.scoring.score_manifest import score_manifest
    from sgmse.backbones.streaming_unet import prune_resblocks_

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this Modal container.")

    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Healed checkpoint not found: {ckpt}")

    device = torch.device("cuda")
    data_path = resolve_modal_data_path(data_path, task=task, split=split, volume_root=VOLUME_ROOT)
    cache_info = _configure_persistent_cache_env("L4")
    paths = _remote_paths()

    print(f"=== healed k={len(drop)}: dropped {drop or '[baseline]'} ===")
    print(f"checkpoint: {ckpt}")

    result = run_test_set_inference(
        task=task,
        config_name=config_name,
        ckpt=ckpt,
        split=split,
        data_path=data_path,
        data_format=data_format,
        part="model",
        pipeline=pipeline,
        execution=execution,
        solver=solver,
        steps=steps,
        limit=limit,
        offset=0,
        selection=selection,
        selection_seed=selection_seed,
        seed=seed,
        model_dtype_name="fp32",
        float32_matmul_precision="high",
        model_memory_format="contiguous",
        crop_mode=crop_mode,
        device=device,
        paths=paths,
        backend="modal",
        hardware="L4",
        output_dir=f"{VOLUME_ROOT}/outputs/eval_runs",
        run_name=run_name,
        overwrite=True,
        save_inputs=False,
        continue_on_error=False,
        cache_info=cache_info,
        backbone_pre_transform=partial(prune_resblocks_, drop=drop) if drop else None,
    )

    manifest_path = Path(result["manifest_path"])
    metrics_path = manifest_path.parent / "metrics.json"
    summary = score_manifest(
        manifest_path,
        limit=0,
        with_distillmos=with_distillmos,
        output_json=metrics_path,
        score_target="enhanced",
    )
    enhanced = summary.get("enhanced", {})
    row = {
        "k": len(drop),
        "dropped": drop,
        "ckpt": ckpt,
        "num_files": summary.get("num_files"),
        "mean_file_s": result.get("mean_file_s"),
        "metrics": {name: enhanced.get(name) for name in METRIC_NAMES},
    }
    print("healed: " + "  ".join(
        f"{name}={row['metrics'][name]:.4f}" if row["metrics"][name] is not None else f"{name}=NA"
        for name in METRIC_NAMES
    ))
    return row


@app.local_entrypoint()
def main(
    ckpt: str = "/runs/training/derev-prune-k3-ft/checkpoints/last.ckpt",
    k: int = 3,
    task: str = "derev",
    config_name: str = "streamfm_derev",
    ranking: str = "",
    # NOT the default /data/datasets/EARS-Reverb_v2_16k/test.csv.  Regenerating the
    # dataset re-drew the test split -- generate_ears_reverb.py draws train and test
    # from one global numpy stream, so repairing the train RIR pool shifted every
    # test row.  The ablation curve was measured on the pre-regeneration split, kept
    # under EARS-Reverb_v2_16k_ablation, and that is what the healed point must be
    # scored against.
    data_path: str = "/data/datasets/EARS-Reverb_v2_16k_ablation/test.csv",
    data_format: str = "",
    split: str = "test",
    # Same offline configuration as the ablation curve, so the healed point can be
    # read straight against the zero-shot one.
    pipeline: str = "offline",
    execution: str = "eager",
    solver: str = "euler",
    steps: int = 1,
    limit: int = 50,
    selection: str = "random",
    selection_seed: int = 42,
    seed: int = 42,
    crop_mode: str = "full",
    with_distillmos: bool = True,
    run_name: str = "prune_healed_derev_k3",
    out: str = "prune_healed_derev.json",
):
    ranking_list = [name.strip() for name in ranking.split(",") if name.strip()] or DEREV_RANKING
    drop = list(ranking_list[:k])

    row = evaluate.remote(
        task=task,
        config_name=config_name,
        ckpt=ckpt,
        drop=drop,
        data_path=data_path,
        data_format=data_format,
        split=split,
        pipeline=pipeline,
        execution=execution,
        solver=solver,
        steps=steps,
        limit=limit,
        selection=selection,
        selection_seed=selection_seed,
        seed=seed,
        crop_mode=crop_mode,
        with_distillmos=with_distillmos,
        run_name=run_name,
    )

    print("\nHealed pruned model:")
    print("  " + "  ".join(f"{name:>9}" for name in METRIC_NAMES))
    print("  " + "  ".join(
        f"{row['metrics'][name]:>9.4f}" if row["metrics"][name] is not None else f"{'NA':>9}"
        for name in METRIC_NAMES
    ))

    if out:
        with open(out, "w") as handle:
            json.dump(row, handle, indent=2)
        print(f"\nWrote {out}")
