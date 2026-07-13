"""Create a compressed Stream.FM checkpoint on Modal and store it in the shared volume."""

from __future__ import annotations

import sys
from pathlib import Path

import modal


REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    raise RuntimeError("Could not find the Stream.FM repository root.")


LOCAL_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT, "STREAMFM_SKIP_DYNAMO_CONFIG": "1"})
    .pip_install("torch==2.7.0", "einops==0.8.1", "hydra-core==1.3.2", "numpy==1.26.4")
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse")
    .add_local_file(str(LOCAL_ROOT / "compress_checkpoint.py"), remote_path=f"{REMOTE_ROOT}/compress_checkpoint.py")
)

for checkpoint_name in ("streamfm_stftpr.ckpt", "streamfm_se_predgen.ckpt"):
    image = image.add_local_file(
        str(LOCAL_ROOT / "checkpoints" / checkpoint_name),
        remote_path=f"{REMOTE_ROOT}/checkpoints/{checkpoint_name}",
    )


app = modal.App("streamfm-compression", image=image)


@app.function(timeout=7200, cpu=8, volumes={VOLUME_ROOT: CACHE_VOLUME})
def compress(
    *,
    config_name: str,
    source_name: str,
    output_name: str,
    rank: int,
) -> dict:
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    from compress_checkpoint import compress_checkpoint

    result = compress_checkpoint(
        config_name=config_name,
        source_checkpoint=f"{REMOTE_ROOT}/checkpoints/{source_name}",
        output_checkpoint=f"{VOLUME_ROOT}/checkpoints/compressed/{output_name}",
        rank=rank,
        config_dir=f"{REMOTE_ROOT}/config",
    )
    CACHE_VOLUME.commit()
    return result


@app.local_entrypoint()
def main(
    config_name: str = "streamfm_stftpr",
    source_name: str = "streamfm_stftpr.ckpt",
    output_name: str = "streamfm_stftpr_k6.ckpt",
    rank: int = 6,
) -> None:
    print(
        compress.remote(
            config_name=config_name,
            source_name=source_name,
            output_name=output_name,
            rank=rank,
        )
    )
