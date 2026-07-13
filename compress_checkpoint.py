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


def _instantiate_backbone(cfg, dotted_path: str):
    """Instantiate the configured streaming backbone without importing Lightning.

    Compression only changes the ``dnn`` backbone.  Building the full model pulls
    in the training stack (including optional text-model dependencies), although
    none of it is needed to rewrite inference weights.
    """
    if dotted_path not in {"dnn", "wrapped_model.dnn"}:
        raise ValueError(
            "Backbone-only checkpoint compression currently supports 'dnn' "
            f"or 'wrapped_model.dnn', got {dotted_path!r}."
        )
    return instantiate(cfg.model.backbone)


def _resolve_module(root, dotted_path: str):
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Model has no module at '{dotted_path}' (missing '{part}').")
        current = getattr(current, part)
    return current


def _state_dict_prefix(state_dict: dict, dotted_path: str) -> str:
    """Find the state-dict prefix used for a configured backbone path."""
    for prefix in (dotted_path, f"model.{dotted_path}"):
        if any(key.startswith(f"{prefix}.") for key in state_dict):
            return prefix
    raise KeyError(f"Checkpoint has no weights under '{dotted_path}.' or 'model.{dotted_path}.'.")


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
    # Full training checkpoints also include optimizer state.  Memory-map their
    # tensor storages so a compression sweep does not need to materialize all of
    # that state in RAM before selecting the model weights.
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(source, dict) or "state_dict" not in source:
        raise ValueError("Source must be a full checkpoint containing a 'state_dict'.")
    paths = tuple(backbone_paths)
    source_state = source["state_dict"]
    compressed_state = dict(source_state)
    parameter_count = 0
    for path in paths:
        prefix = _state_dict_prefix(source_state, path)
        backbone = _instantiate_backbone(cfg, path)
        backbone_state = {
            key[len(prefix) + 1 :]: value
            for key, value in source_state.items()
            if key.startswith(f"{prefix}.")
        }
        backbone.load_state_dict(backbone_state, strict=True)
        compress_decoupled_(backbone, rank)
        for key in backbone_state:
            compressed_state.pop(f"{prefix}.{key}")
        compressed_state.update({f"{prefix}.{key}": value for key, value in backbone.state_dict().items()})
        parameter_count += sum(parameter.numel() for parameter in backbone.parameters())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "state_dict": compressed_state,
        COMPRESSION_METADATA_KEY: make_decoupled_svd_metadata(rank=rank, backbone_paths=list(paths)),
        "source_checkpoint": str(source_path),
    }
    torch.save(output, output_path)
    return {
        "output_checkpoint": str(output_path),
        "rank": rank,
        "backbone_paths": list(paths),
        "parameters": parameter_count,
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
