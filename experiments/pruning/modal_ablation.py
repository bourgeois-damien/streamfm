"""Modal entrypoint: cumulative zero-shot depth-pruning ablation curve.

For k = 0, 1, 2, ... this drops the k least-influential residual blocks (ranked
by ``experiments/pruning/block_influence.py``), warm-started from the teacher
checkpoint with **no fine-tune**, then runs test-set inference on a subset and
scores the enhanced audio.  The result is a quality-vs-blocks-removed curve that
answers "how much precision do we lose if we drop rank 0, then 0+1, then
0+1+2, ...".

    modal run experiments/pruning/modal_ablation.py \
        --task derev --config-name streamfm_derev --ckpt streamfm_derev.ckpt \
        --max-k 8 --limit 50 --hardware L4

k=0 is the unpruned baseline (empty drop list), so every curve is anchored to
the teacher's own subset score under identical settings.
"""

from __future__ import annotations

import json
import sys

import modal

# Reuse the eval image (bundles config/, sgmse/, experiments/ and the
# streamfm_*.ckpt files) and the shared cache volume (datasets + checkpoints).
from experiments.evaluation.modal_streamfm_eval import (
    CACHE_VOLUME,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image,
)

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

app = modal.App("streamfm-prune-ablation", image=image)

# Block-Influence ranking on the EARS-Reverb test set (least influential first),
# from experiments/pruning/block_influence.py.  Cumulative drop k uses the first
# k names.  Override with --ranking "name1,name2,..." for another model.
DEREV_RANKING = [
    "up_modules.lvl2_rnb1",     # BI 0.137
    "up_modules.lvl1_rnb1",     # BI 0.173
    "up_modules.lvl0_rnb1",     # BI 0.187
    "up_modules.lvl1_rnb0",     # BI 0.191
    "up_modules.lvl2_rnb0",     # BI 0.196
    "up_modules.lvl3_rnb1",     # BI 0.199
    "down_modules.lvl3_rnb1",   # BI 0.210
    "up_modules.lvl3_rnb0",     # BI 0.220
    "down_modules.lvl3_rnb0",   # BI 0.238
    "down_modules.lvl2_rnb0",   # BI 0.243
    "down_modules.lvl2_rnb1",   # BI 0.258
    "bottleneck_modules.rnb1",  # BI 0.266
    "bottleneck_modules.rnb2",  # BI 0.290
    "up_modules.lvl0_rnb2",     # BI 0.346
    "down_modules.lvl1_rnb1",   # BI 0.408
    "down_modules.lvl0_rnb0",   # BI 0.639
    "down_modules.lvl0_rnb1",   # BI 0.754
]

METRIC_NAMES = ("pesq", "estoi", "si_sdr", "lsd", "distillmos")


@app.function(gpu="L4", timeout=14400, volumes={VOLUME_ROOT: CACHE_VOLUME})
def ablate(
    task: str,
    config_name: str,
    ckpt: str,
    ranking: list[str],
    max_k: int,
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
    run_prefix: str,
) -> list[dict]:
    from functools import partial

    import torch

    from experiments.evaluation.modal_defaults import resolve_modal_data_path
    from experiments.evaluation.modal_streamfm_eval import _configure_persistent_cache_env, _remote_paths
    from experiments.evaluation.runner import run_test_set_inference
    from experiments.evaluation.scoring.score_manifest import score_manifest
    from sgmse.backbones.streaming_unet import prune_resblocks_

    from pathlib import Path

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this Modal container.")

    device = torch.device("cuda")
    data_path = resolve_modal_data_path(data_path, task=task, split=split, volume_root=VOLUME_ROOT)
    cache_info = _configure_persistent_cache_env("L4")
    paths = _remote_paths()

    max_k = min(max_k, len(ranking))
    rows: list[dict] = []
    for k in range(0, max_k + 1):
        drop = list(ranking[:k])
        transform = partial(prune_resblocks_, drop=drop) if drop else None
        run_name = f"{run_prefix}_k{k}"
        print(f"\n=== k={k}: dropping {drop or '[baseline]'} ===")

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
            backbone_transform=transform,
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
            "k": k,
            "dropped": drop,
            "num_files": summary.get("num_files"),
            "mean_file_s": result.get("mean_file_s"),
            "metrics": {name: enhanced.get(name) for name in METRIC_NAMES},
        }
        # k=0 baseline metrics travel with every row so deltas are trivial to read.
        if k == 0:
            baseline = dict(row["metrics"])
        row["delta_vs_baseline"] = {
            name: (
                None
                if row["metrics"].get(name) is None or baseline.get(name) is None
                else float(row["metrics"][name] - baseline[name])
            )
            for name in METRIC_NAMES
        }
        rows.append(row)
        print(f"k={k}: " + "  ".join(
            f"{name}={row['metrics'][name]:.4f}" if row["metrics"][name] is not None else f"{name}=NA"
            for name in METRIC_NAMES
        ))

    return rows


@app.local_entrypoint()
def main(
    task: str = "derev",
    config_name: str = "streamfm_derev",
    ckpt: str = "streamfm_derev.ckpt",
    ranking: str = "",
    max_k: int = 8,
    data_path: str = "",
    data_format: str = "",
    split: str = "test",
    # Offline pipeline: this is the config that reproduces the paper's derev
    # quality (experiments/reproduction_metrics.md, Euler1). The streaming path
    # adds an algorithmic delay the scorer does not compensate, which tanks all
    # metrics on the *unpruned* baseline too. Pruning is structural (it changes
    # the eager forward and the streaming forward_step identically), so the
    # offline quality drop is a valid proxy for the deployed model's drop.
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
    run_prefix: str = "prune_ablation_derev",
    out: str = "results/pruning/prune_ablation_derev.json",
):
    ranking_list = [name.strip() for name in ranking.split(",") if name.strip()] or DEREV_RANKING

    rows = ablate.remote(
        task=task,
        config_name=config_name,
        ckpt=ckpt,
        ranking=ranking_list,
        max_k=max_k,
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
        run_prefix=run_prefix,
    )

    print("\nCumulative zero-shot pruning ablation (no fine-tune):")
    header = f"{'k':>2}  " + "  ".join(f"{name:>9}" for name in METRIC_NAMES) + "   dropped"
    print(header)
    for r in rows:
        cells = "  ".join(
            f"{r['metrics'][name]:>9.4f}" if r["metrics"][name] is not None else f"{'NA':>9}"
            for name in METRIC_NAMES
        )
        last = r["dropped"][-1] if r["dropped"] else "[baseline]"
        print(f"{r['k']:>2}  {cells}   +{last}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "task": task,
                    "config": config_name,
                    "ckpt": ckpt,
                    "ranking": ranking_list,
                    "limit": limit,
                    "pipeline": pipeline,
                    "execution": execution,
                    "steps": steps,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote ablation curve to {out}")
