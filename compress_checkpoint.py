"""Create a reusable decoupled-SVD Stream.FM checkpoint from a full checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from sgmse.backbones.streaming_unet import compress_decoupled_
from sgmse.util.model_compression import COMPRESSION_METADATA_KEY, make_decoupled_svd_metadata


def _instantiate_model(cfg):
    if hasattr(cfg, "solver_model"):
        wrapped_model = instantiate(cfg.model)
        return instantiate(cfg.solver_model, wrapped_model=wrapped_model)
    return instantiate(cfg.model)


def _resolve_module(root, dotted_path: str):
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Model has no module at '{dotted_path}' (missing '{part}').")
        current = getattr(current, part)
    return current


def compress_checkpoint(
    *,
    config_name: str,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    rank: int,
    backbone_paths: tuple[str, ...] = ("dnn",),
    config_dir: str | Path = "config",
    overwrite: bool = False,
) -> dict:
    """Compress selected model backbones and save a self-describing inference checkpoint."""
    source_path = Path(source_checkpoint)
    output_path = Path(output_checkpoint)
    config_path = Path(config_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_path}. Use --overwrite to replace it.")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")

    with initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
        cfg = compose(config_name=config_name)
    model = _instantiate_model(cfg)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or "state_dict" not in source:
        raise ValueError("Source must be a full checkpoint containing a 'state_dict'.")
    model.load_state_dict(source["state_dict"], strict=True)

    paths = tuple(backbone_paths)
    for path in paths:
        compress_decoupled_(_resolve_module(model, path), rank)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "state_dict": model.state_dict(),
        COMPRESSION_METADATA_KEY: make_decoupled_svd_metadata(rank=rank, backbone_paths=list(paths)),
        "source_checkpoint": str(source_path),
    }
    torch.save(output, output_path)
    return {
        "output_checkpoint": str(output_path),
        "rank": rank,
        "backbone_paths": list(paths),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Hydra config matching the source checkpoint.")
    parser.add_argument("--source", required=True, help="Full, uncompressed source checkpoint.")
    parser.add_argument("--output", required=True, help="Output compressed checkpoint (.ckpt).")
    parser.add_argument("--rank", type=int, required=True, help="SVD/decomposition rank K (1..9 for 3x3 convolutions).")
    parser.add_argument(
        "--backbone-path",
        action="append",
        default=[],
        help="Model module to compress; repeat if needed. Defaults to dnn.",
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = compress_checkpoint(
        config_name=args.config_name,
        source_checkpoint=args.source,
        output_checkpoint=args.output,
        rank=args.rank,
        backbone_paths=tuple(args.backbone_path or ["dnn"]),
        config_dir=args.config_dir,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
