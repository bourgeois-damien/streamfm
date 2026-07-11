"""Checkpoint metadata and architecture setup for decoupled-SVD compression."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sgmse.backbones.streaming_unet import compress_decoupled_


COMPRESSION_METADATA_KEY = "streamfm_compression"
COMPRESSION_METHOD = "decoupled_svd"


def make_decoupled_svd_metadata(*, rank: int, backbone_paths: list[str]) -> dict[str, Any]:
    """Describe the architecture change stored in a compressed checkpoint."""
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")
    if not backbone_paths:
        raise ValueError("At least one backbone path must be provided.")
    return {
        "format_version": 1,
        "method": COMPRESSION_METHOD,
        "rank": rank,
        "backbone_paths": list(backbone_paths),
    }


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
    return dict(metadata)


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
        compress_decoupled_(_resolve_module(model, path), metadata["rank"])
    return model


def apply_backbone_compression_(backbone, checkpoint: Mapping[str, Any] | Any, *, backbone_path: str = "dnn"):
    """Prepare an extracted backbone for a full or DNN-only compressed checkpoint."""
    metadata = get_compression_metadata(checkpoint)
    if metadata is None and isinstance(checkpoint, Mapping) and checkpoint.get("method") == COMPRESSION_METHOD:
        metadata = dict(checkpoint)
    if metadata is None:
        return backbone
    if backbone_path in metadata["backbone_paths"]:
        compress_decoupled_(backbone, metadata["rank"])
    return backbone
