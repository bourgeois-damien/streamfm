"""Checkpoint metadata and architecture setup for decoupled-SVD compression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sgmse.backbones.streaming_unet import compress_decoupled_


COMPRESSION_METADATA_KEY = "streamfm_compression"
COMPRESSION_METHOD = "decoupled_svd"


def make_decoupled_svd_metadata(
    *,
    rank: int,
    backbone_paths: list[str],
    module_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe the architecture change stored in a compressed checkpoint."""
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")
    if not backbone_paths:
        raise ValueError("At least one backbone path must be provided.")
    metadata = {
        "format_version": 2 if module_names is not None else 1,
        "method": COMPRESSION_METHOD,
        "rank": rank,
        "backbone_paths": list(backbone_paths),
    }
    if module_names is not None:
        names = list(module_names)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError("module_names must contain at least one non-empty module name.")
        if len(names) != len(set(names)):
            raise ValueError("module_names must not contain duplicates.")
        metadata["module_names"] = names
    return metadata


def get_compression_metadata(checkpoint: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    """Return validated Stream.FM compression metadata, if present."""
    if not isinstance(checkpoint, Mapping):
        return None
    metadata = checkpoint.get(COMPRESSION_METADATA_KEY)
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{COMPRESSION_METADATA_KEY} must be a mapping.")
    if metadata.get("method") != COMPRESSION_METHOD:
        raise ValueError(f"Unsupported compression method: {metadata.get('method')!r}")
    rank = metadata.get("rank")
    paths = metadata.get("backbone_paths")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"Invalid compressed-checkpoint rank: {rank!r}")
    if not isinstance(paths, list) or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("Compressed checkpoint metadata has invalid backbone_paths.")
    module_names = metadata.get("module_names")
    if module_names is not None:
        if (
            not isinstance(module_names, list)
            or not module_names
            or not all(isinstance(name, str) and name for name in module_names)
            or len(module_names) != len(set(module_names))
        ):
            raise ValueError("Compressed checkpoint metadata has invalid module_names.")
    return dict(metadata)


def _apply_metadata_compression_(backbone, metadata: Mapping[str, Any]):
    module_names = metadata.get("module_names")
    if module_names is None:
        return compress_decoupled_(backbone, metadata["rank"])

    selected = set(module_names)
    available = {name for name, _module in backbone.named_modules()}
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"Compressed checkpoint refers to unknown modules: {missing}")
    compress_decoupled_(
        backbone,
        metadata["rank"],
        module_filter=lambda name, _module: name in selected,
    )
    return backbone


def _resolve_module(root, dotted_path: str):
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Model has no module at '{dotted_path}' (missing '{part}').")
        current = getattr(current, part)
    return current


def apply_checkpoint_compression_(model, checkpoint: Mapping[str, Any] | Any):
    """Mutate ``model`` so it matches compression metadata in ``checkpoint``."""
    metadata = get_compression_metadata(checkpoint)
    if metadata is None:
        return model
    for path in metadata["backbone_paths"]:
        _apply_metadata_compression_(_resolve_module(model, path), metadata)
    return model


def apply_backbone_compression_(backbone, checkpoint: Mapping[str, Any] | Any, *, backbone_path: str = "dnn"):
    """Prepare an extracted backbone for a full or DNN-only compressed checkpoint."""
    metadata = get_compression_metadata(checkpoint)
    if metadata is None and isinstance(checkpoint, Mapping) and checkpoint.get("method") == COMPRESSION_METHOD:
        metadata = dict(checkpoint)
    if metadata is None:
        return backbone
    if backbone_path in metadata["backbone_paths"]:
        _apply_metadata_compression_(backbone, metadata)
    return backbone
