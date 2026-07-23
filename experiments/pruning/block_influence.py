"""
Block-Influence scoring for structured depth pruning of the streaming U-Net.

Ranks the residual blocks of a trained CausalNCSNpp by how little they change
their input, i.e. how close they are to an identity map.  Blocks with the
lowest influence are the safest candidates to drop (replace by StreamingIdentity)
before a short heal / fine-tune.

Two metrics are reported per block, both averaged over every (batch, freq, time)
position of a handful of validation batches:

  * bi  = 1 - cos(x_in, x_out)     (Block Influence, Men et al. 2024 "ShortGPT";
                                    Gromov et al. 2024). Low bi  -> near identity.
  * res = ||h|| / ||x_in||         relative magnitude of the residual branch.
                                    Since these blocks compute out=(x+h)/sqrt(2)
                                    with an identity skip, h = sqrt(2)*out - x.
                                    Low res -> the block adds little.

Only *iso-channel* residual blocks (in_ch == out_ch, no freq up/down-sampling)
are scored, because those are exactly the blocks whose skip connection is an
identity and can therefore be replaced by StreamingIdentity without breaking
the channel bookkeeping of the U-Net.

Usage (on a machine with the data + a GPU, e.g. the training server):

    python -m experiments.pruning.block_influence \
        --config streamfm_derev \
        --ckpt checkpoints/streamfm_derev.ckpt \
        --num-batches 8 --device cuda --out results/pruning/block_influence_derev.json

Smoke test (no data, CPU, synthetic audio -- only checks the pipeline runs;
the ranking it prints is meaningless):

    python -m experiments.pruning.block_influence \
        --config streamfm_derev --ckpt checkpoints/streamfm_derev.ckpt \
        --smoke --num-batches 2 --device cpu
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import OrderedDict

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from sgmse.backbones.streaming_unet import CausalResnetBlockBigGANpp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def load_model(config_name: str, ckpt_path: str, device: str):
    """Instantiate the model from its Hydra config and load checkpoint weights."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name=config_name)
    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return model, cfg


def eligible_resblocks(dnn) -> "OrderedDict[str, torch.nn.Module]":
    """Iso-channel residual blocks: the identity-replaceable pruning candidates."""
    blocks = OrderedDict()
    for name, module in dnn.named_modules():
        if isinstance(module, CausalResnetBlockBigGANpp):
            iso = (module.in_ch == module.out_ch) and not module.freq_up and not module.freq_down
            if iso:
                blocks[name] = module
    return blocks


class BlockInfluenceMeter:
    """Accumulates BI and residual-ratio statistics via forward hooks."""

    def __init__(self, blocks):
        self.blocks = blocks
        # per block: [sum(1-cos), sum(res_ratio), count] over all positions
        self.acc = {name: [0.0, 0.0, 0] for name in blocks}
        self._handles = []
        for name, module in blocks.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        sqrt2 = math.sqrt(2.0)

        def hook(_module, inputs, output):
            x = inputs[0]                       # (B, C, F, T)
            y = output                          # (B, C, F, T)
            if x.shape != y.shape:              # safety: only iso-channel blocks
                return
            C = x.shape[1]
            # collapse to (positions, channels): each channel-vector is one "token"
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            y_flat = y.permute(0, 2, 3, 1).reshape(-1, C)
            cos = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=1, eps=1e-8)
            bi = (1.0 - cos)
            # residual branch h = sqrt(2)*out - x  (skip_rescale=True, identity skip)
            h_norm = (sqrt2 * y_flat - x_flat).norm(dim=1)
            x_norm = x_flat.norm(dim=1)
            res = h_norm / (x_norm + 1e-8)
            a = self.acc[name]
            a[0] += bi.double().sum().item()
            a[1] += res.double().sum().item()
            a[2] += bi.numel()

        return hook

    def results(self):
        rows = []
        for name in self.blocks:
            s_bi, s_res, n = self.acc[name]
            n = max(n, 1)
            rows.append({
                "block": name,
                "bi": s_bi / n,
                "res_ratio": s_res / n,
                "in_ch": self.blocks[name].in_ch,
                "in_freqs": self.blocks[name].in_freqs,
            })
        rows.sort(key=lambda r: r["bi"])   # least influential first
        return rows

    def remove(self):
        for h in self._handles:
            h.remove()


def synthetic_batches(num_batches, batch_size, duration, sr, device):
    """Fake (x, y, info) batches for a pipeline smoke test (no real data needed)."""
    for _ in range(num_batches):
        x = torch.randn(batch_size, 1, duration, device=device) * 0.1
        y = x + 0.05 * torch.randn_like(x)
        info = {"sr": torch.full((batch_size,), sr, device=device)}
        yield x, y, info


def real_batches(model, num_batches, data_csv=None, split="valid"):
    """First `num_batches` batches from the config's data module.

    `split` selects "valid" or "test".  The split matters beyond which manifest
    is read: for the ears_* formats the audio paths are reconstructed relative to
    the CSV directory using the split name (``<basedir>/<split>/clean/...``), so
    the on-disk folder must match.  If `data_csv` is given it overrides the
    manifest path for the chosen split (useful when the config's default path is
    not mounted, e.g. on Modal only /data/datasets/.../test{,.csv} exists).
    """
    dm = model.data_module
    dm.num_workers = 0            # a handful of batches: avoid worker spawn overhead
    if split == "test":
        if data_csv is not None:
            dm.test_path = data_csv
        dm.setup(stage="test")
        loader = dm.test_dataloader()
    else:
        if data_csv is not None:
            # point train+valid at the same manifest so setup('fit') never touches
            # a non-existent train path; we only consume the validation loader.
            dm.valid_path = data_csv
            dm.train_path = data_csv
        dm.setup(stage="fit")     # the datamodule only builds valid_set on 'fit'/None
        loader = dm.val_dataloader()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        yield batch


@torch.inference_mode()
def make_backbone_input(model, x, y, info, device):
    """Reproduce FlowModel._step's construction of (X_t, Y, t) for a batch."""
    sr = model.sampling_rate
    Y, X, _y_pre, _x_pre, _info = model.preprocess(y.to(device), sr, x=x.to(device))
    if getattr(model, "post_Y_fn", None) is not None:
        Y = model.post_Y_fn(Y)
    if getattr(model, "post_X_fn", None) is not None:
        X = model.post_X_fn(X)
    t = torch.rand(X.shape[0], device=device)
    t_bc = t.view(-1, 1, 1, 1)
    Z = torch.randn_like(Y)
    X_0 = Y + model.sigma_y * Z
    X_1 = X + model.sigma_x * Z if model.sigma_x is not None else X
    X_t = (1.0 - t_bc) * X_0 + t_bc * X_1
    return X_t, Y, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="streamfm_derev", help="Hydra config name under config/")
    ap.add_argument("--ckpt", required=True, help="Path to the teacher checkpoint (.ckpt)")
    ap.add_argument("--num-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-csv", default=None, help="Override the manifest CSV path for the chosen split")
    ap.add_argument("--split", default="valid", choices=["valid", "test"], help="Dataset split to score on")
    ap.add_argument("--smoke", action="store_true", help="Use synthetic audio (pipeline test only)")
    ap.add_argument("--batch-size", type=int, default=2, help="Batch size for --smoke mode")
    ap.add_argument("--duration", type=int, default=32512, help="Samples per clip for --smoke mode")
    ap.add_argument("--out", default=None, help="Optional path to write the ranking as JSON")
    args = ap.parse_args()

    print(f"Loading {args.config} from {args.ckpt} on {args.device} ...")
    model, _cfg = load_model(args.config, args.ckpt, args.device)

    blocks = eligible_resblocks(model.dnn)
    print(f"Found {len(blocks)} iso-channel (identity-replaceable) residual blocks:")
    for name in blocks:
        print(f"  - {name}  (C={blocks[name].in_ch}, F={blocks[name].in_freqs})")

    meter = BlockInfluenceMeter(blocks)

    if args.smoke:
        print("\n[SMOKE] synthetic audio -- the ranking below is NOT meaningful.")
        batches = synthetic_batches(args.num_batches, args.batch_size, args.duration,
                                    int(model.sampling_rate), args.device)
    else:
        batches = real_batches(model, args.num_batches, data_csv=args.data_csv, split=args.split)

    n = 0
    for x, y, info in batches:
        X_t, Y, t = make_backbone_input(model, x, y, info, args.device)
        model(X_t, Y, t)          # triggers the hooks
        n += 1
        print(f"  processed batch {n}")
    meter.remove()

    rows = meter.results()
    print("\nBlock influence ranking (least influential first -> best drop candidates):")
    print(f"{'rank':>4}  {'bi':>10}  {'res_ratio':>10}  block")
    for i, r in enumerate(rows):
        print(f"{i:>4}  {r['bi']:>10.5f}  {r['res_ratio']:>10.5f}  {r['block']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": args.config, "ckpt": args.ckpt,
                       "num_batches": n, "smoke": args.smoke, "ranking": rows}, f, indent=2)
        print(f"\nWrote ranking to {args.out}")


if __name__ == "__main__":
    main()
