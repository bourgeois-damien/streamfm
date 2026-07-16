"""Checkpoint loading for the benchmark backbones.

Instantiates a backbone from its Hydra config and loads either DNN-only weights
or the DNN slice of a full training checkpoint, applying the SGMSE compression
metadata. Exposes per-task loaders: ``load_flow_model``, ``load_se_predictor``,
``load_se_flow`` and ``load_se_full``.
"""

from __future__ import annotations

from experiments.core.repo import ensure_repo_importable
from experiments.core.paths import BenchmarkPaths, checkpoint_path
from sgmse.util.model_compression import apply_backbone_compression_, get_compression_metadata


# task -> (hydra config name, DNN-only checkpoint, full training checkpoint fallback)
FLOW_TASK_CHECKPOINTS = {
    "stftpr": ("streamfm_stftpr", "streamfm_stftpr_dnn_only.pt", "streamfm_stftpr.ckpt"),
    "bwe": ("streamfm_bwe", "streamfm_bwe_dnn_only.pt", "streamfm_bwe.ckpt"),
    "derev": ("streamfm_derev", "streamfm_derev_dnn_only.pt", "streamfm_derev.ckpt"),
    "lyra": ("streamfm_lyra", "streamfm_lyra_dnn_only.pt", "streamfm_lyra.ckpt"),
}


def _try_checkpoint_path(filename: str, paths: BenchmarkPaths) -> str | None:
    """First existing match across the checkpoint roots, or None."""
    for root in paths.checkpoint_roots:
        path = root / filename
        if path.exists():
            return str(path)
    return None


def _normalize_checkpoint_state(state):
    """Unwrap a Lightning-style {"state_dict": ...} container to the bare state dict."""
    if isinstance(state, dict) and "state_dict" in state:
        candidate = state["state_dict"]
        if isinstance(candidate, dict):
            return candidate
    return state


def _extract_checkpoint_prefix(state_dict: dict, prefix: str) -> dict:
    """Slice out the sub-module under `prefix.` (keys renamed to drop the prefix).

    Also tries `model.{prefix}` because the Lightning module nesting differs
    between training setups; raises KeyError if neither prefix matches.
    """
    prefix = prefix.rstrip(".")
    extracted = {
        key[len(prefix) + 1 :]: value
        for key, value in state_dict.items()
        if key == prefix or key.startswith(prefix + ".")
    }
    if extracted:
        return extracted

    alt_prefix = f"model.{prefix}"
    extracted = {
        key[len(alt_prefix) + 1 :]: value
        for key, value in state_dict.items()
        if key == alt_prefix or key.startswith(alt_prefix + ".")
    }
    if extracted:
        return extracted

    raise KeyError(
        f"No state_dict keys found for prefix '{prefix}' or '{alt_prefix}' in full checkpoint."
    )


def _extract_backbone_state_from_full_checkpoint(checkpoint, prefix: str | None):
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Full checkpoint does not contain a 'state_dict' entry.")

    state_dict = _normalize_checkpoint_state(state_dict)
    if prefix is None:
        return state_dict

    return _extract_checkpoint_prefix(state_dict, prefix)


def _load_backbone_state(*, dnn_checkpoint_name, full_checkpoint_name, paths, state_prefix):
    """Load DNN weights, preferring the small DNN-only file over the full training checkpoint.

    Returns (state_dict, loaded path, compression metadata). The DNN-only
    export is a few MB of pure tensors (weights_only=True); the full .ckpt
    carries pickled training objects and needs weights_only=False.
    """
    import torch

    dnn_path = _try_checkpoint_path(dnn_checkpoint_name, paths)
    if dnn_path is not None:
        state = torch.load(dnn_path, map_location="cpu", weights_only=True)
        metadata = get_compression_metadata(state)
        if isinstance(state, dict) and "state_dict" in state:
            state = _extract_backbone_state_from_full_checkpoint(state, prefix=state_prefix)
        else:
            state = _normalize_checkpoint_state(state)
        return state, dnn_path, metadata

    if full_checkpoint_name is None:
        raise FileNotFoundError(
            f"DNN checkpoint '{dnn_checkpoint_name}' not found and no full checkpoint provided."
        )
    full_path = checkpoint_path(full_checkpoint_name, paths)
    checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
    return (
        _extract_backbone_state_from_full_checkpoint(checkpoint, prefix=state_prefix),
        full_path,
        get_compression_metadata(checkpoint),
    )


def load_backbone_from_checkpoint(
    *,
    config_name,
    checkpoint_name,
    device,
    dtype,
    paths,
    full_checkpoint_name=None,
    full_checkpoint_state_prefix=None,
):
    """Instantiate a configured backbone and load DNN-only or full-checkpoint DNN weights."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    ensure_repo_importable(paths.repo_root)
    with initialize_config_dir(config_dir=str(paths.config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name)

    backbone_cfg = cfg.model.backbone
    backbone = instantiate(backbone_cfg)
    backbone_state, loaded_path, compression_metadata = _load_backbone_state(
        dnn_checkpoint_name=checkpoint_name,
        full_checkpoint_name=full_checkpoint_name,
        paths=paths,
        state_prefix=full_checkpoint_state_prefix,
    )
    # Compression (SVD-factorized layers) rewrites module shapes, so the
    # architecture must be adapted BEFORE the strict state-dict load.
    apply_backbone_compression_(backbone, compression_metadata, backbone_path=full_checkpoint_state_prefix or "dnn")
    backbone.load_state_dict(backbone_state, strict=True)
    backbone = backbone.eval().to(device=device, dtype=dtype)
    return backbone, cfg


def load_flow_model(device, dtype, paths, task="stftpr", checkpoint_name: str | None = None):
    """Load a flow-only Stream.FM backbone for STFTPR/BWE/DEREV/LYRA."""
    task = task.lower().replace("-", "_")
    if task not in FLOW_TASK_CHECKPOINTS:
        supported = ", ".join(sorted(FLOW_TASK_CHECKPOINTS))
        raise ValueError(f"Unsupported flow task '{task}'. Supported flow tasks: {supported}.")
    config_name, dnn_checkpoint_name, full_checkpoint_name = FLOW_TASK_CHECKPOINTS[task]
    return load_backbone_from_checkpoint(
        config_name=config_name,
        checkpoint_name=checkpoint_name or dnn_checkpoint_name,
        full_checkpoint_name=None if checkpoint_name else full_checkpoint_name,
        full_checkpoint_state_prefix="dnn",
        device=device,
        dtype=dtype,
        paths=paths,
    )


def load_se_predictor(device, dtype, paths, checkpoint_name: str | None = None):
    """Load only the SE initial predictor streaming backbone."""
    return load_backbone_from_checkpoint(
        config_name="streamfm_se_predictor",
        checkpoint_name=checkpoint_name or "streamfm_se_predictor_dnn_only.pt",
        full_checkpoint_state_prefix="initial_predictor.dnn" if checkpoint_name else None,
        device=device,
        dtype=dtype,
        paths=paths,
    )


def load_se_flow(device, dtype, paths, checkpoint_name: str | None = None):
    """Load the SE flow-only backbone using the SE predgen config."""
    return load_backbone_from_checkpoint(
        config_name="streamfm_se_predgen",
        checkpoint_name=checkpoint_name or "streamfm_se_predgen_dnn_only.pt",
        full_checkpoint_name=None if checkpoint_name else "streamfm_se_predgen.ckpt",
        full_checkpoint_state_prefix="dnn",
        device=device,
        dtype=dtype,
        paths=paths,
    )


def load_se_full(device, dtype, paths, checkpoint_name: str | None = None):
    """Load the full SE chain: predictor + flow + sigma_e from one predgen checkpoint.

    The two DNNs live in the same training checkpoint under different
    prefixes: 'initial_predictor.*' for the predictor, 'dnn.*' (or
    'model.dnn.*') for the flow backbone.
    """
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    ensure_repo_importable(paths.repo_root)
    with initialize_config_dir(config_dir=str(paths.config_dir), version_base="1.3"):
        cfg = compose(config_name="streamfm_se_predgen")

    flow = instantiate(cfg.model.backbone).eval().to(device=device, dtype=dtype)
    checkpoint_file = checkpoint_name or "streamfm_se_predgen.ckpt"
    state = torch.load(checkpoint_path(checkpoint_file, paths), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict")
    if state_dict is None:
        raise KeyError("Full SE checkpoint does not contain a 'state_dict' entry.")

    if "initial_predictor" in state_dict:
        predictor_state = {
            key[len("initial_predictor.") :]: value
            for key, value in state_dict.items()
            if key.startswith("initial_predictor.")
        }
    else:
        raise KeyError("Full SE checkpoint does not contain initial predictor weights.")

    if "dnn" in state_dict:
        flow_state = {
            key[len("dnn.") :]: value
            for key, value in state_dict.items()
            if key.startswith("dnn.")
        }
    elif "model.dnn" in state_dict:
        flow_state = {
            key[len("model.dnn.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.dnn.")
        }
    else:
        raise KeyError("Full SE checkpoint does not contain flow backbone weights under 'dnn' or 'model.dnn'.")

    compression_metadata = get_compression_metadata(state)
    apply_backbone_compression_(flow, compression_metadata, backbone_path="dnn")
    predictor = load_se_predictor(
        device=device,
        dtype=dtype,
        paths=paths,
        checkpoint_name=checkpoint_file if compression_metadata else None,
    )[0]
    flow.load_state_dict(flow_state, strict=True)

    return {
        "predictor": predictor,
        "flow": flow,
        "sigma_e": float(cfg.model.sigma_e),
    }
