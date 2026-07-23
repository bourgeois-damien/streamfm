"""Modal entrypoint: run Block-Influence scoring on the mounted validation set.

Reuses the evaluation image and shared data/checkpoint volume, so the EARS
checkpoints and datasets are already available inside the container.  Scoring is
a plain forward pass, so this runs on Modal CPU (cheap) rather than a GPU.

    modal run experiments/pruning/modal_block_influence.py \
        --config-name streamfm_derev --num-batches 8
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

# Reuse the eval image (already bundles config/, sgmse/, experiments/ and the
# streamfm_*.ckpt files) and the shared cache volume (datasets + checkpoints).
from experiments.evaluation.modal_streamfm_eval import (
    CACHE_VOLUME,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image,
)

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

app = modal.App("streamfm-block-influence", image=image)


@app.function(timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def score(config_name: str, ckpt: str, data_csv: str, num_batches: int, split: str) -> list[dict]:
    import os

    import torch

    from experiments.pruning.block_influence import (
        BlockInfluenceMeter,
        eligible_resblocks,
        load_model,
        make_backbone_input,
        real_batches,
    )

    # Checkpoints are baked into the image under {REMOTE_ROOT}/checkpoints/.
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(REMOTE_ROOT, "checkpoints", os.path.basename(ckpt))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {config_name} from {ckpt} on {device} ...")
    model, _cfg = load_model(config_name, ckpt, device)

    blocks = eligible_resblocks(model.dnn)
    print(f"Found {len(blocks)} iso-channel (identity-replaceable) residual blocks.")
    meter = BlockInfluenceMeter(blocks)

    n = 0
    for x, y, info in real_batches(model, num_batches, data_csv=data_csv, split=split):
        X_t, Y, t = make_backbone_input(model, x, y, info, device)
        model(X_t, Y, t)
        n += 1
        print(f"  processed batch {n}")
    meter.remove()
    return meter.results()


@app.local_entrypoint()
def main(
    config_name: str = "streamfm_derev",
    ckpt: str = "streamfm_derev.ckpt",
    data_csv: str = "/data/datasets/EARS-Reverb_v2_16k/test.csv",
    num_batches: int = 8,
    split: str = "test",
    out: str = "results/pruning/block_influence_derev.json",
):
    rows = score.remote(config_name, ckpt, data_csv, num_batches, split)

    print("\nBlock influence ranking (least influential first -> best drop candidates):")
    print(f"{'rank':>4}  {'bi':>10}  {'res_ratio':>10}  block")
    for i, r in enumerate(rows):
        print(f"{i:>4}  {r['bi']:>10.5f}  {r['res_ratio']:>10.5f}  {r['block']}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"config": config_name, "ckpt": ckpt, "ranking": rows}, f, indent=2)
        print(f"\nWrote ranking to {out}")
