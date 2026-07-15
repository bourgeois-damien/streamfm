"""Shared Modal cache configuration.

``configure_shared_modal_cache`` points the HF/torch caches at the persistent
Modal volume so weights and datasets survive between runs;
``CACHE_LAYOUT_VERSION`` namespaces that layout.
"""

from __future__ import annotations

import os


CACHE_LAYOUT_VERSION = "v2_torch2_7"


def _safe_cache_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "unknown"


def configure_shared_modal_cache(*, volume_root: str, hardware: str) -> dict[str, str]:
    """Share content-addressed compiler caches between experiment combinations."""
    hardware_label = _safe_cache_label(hardware.upper())
    shared_root = f"{volume_root}/cache/shared/{CACHE_LAYOUT_VERSION}"
    cache_root = f"{shared_root}/{hardware_label}"

    os.environ["TORCH_HOME"] = f"{volume_root}/cache/shared/torch"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_root}/torchinductor"
    os.environ["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    os.environ["XDG_CACHE_HOME"] = f"{cache_root}/xdg"
    for path in (
        cache_root,
        os.environ["TORCH_HOME"],
        os.environ["TORCHINDUCTOR_CACHE_DIR"],
        os.environ["TRITON_CACHE_DIR"],
        os.environ["XDG_CACHE_HOME"],
        f"{os.environ['XDG_CACHE_HOME']}/torch/kernels",
    ):
        os.makedirs(path, exist_ok=True)

    return {
        "cache_layout": CACHE_LAYOUT_VERSION,
        "cache_root": cache_root,
        "torch_home": os.environ["TORCH_HOME"],
        "torchinductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"],
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        "xdg_cache_home": os.environ["XDG_CACHE_HOME"],
    }
