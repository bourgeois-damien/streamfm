"""Modal entrypoint: profile CausalNCSNpp on L4 (and optionally run cuda_graph bench)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)


def _find_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current_file.parent


LOCAL_ROOT = _find_repo_root()
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .apt_install("libsndfile1")
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "numpy==1.26.4",
        "soundfile==0.12.1",
    )
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    .add_local_dir(str(LOCAL_ROOT / "experiments"), remote_path=f"{REMOTE_ROOT}/experiments")
    .add_local_dir(str(LOCAL_ROOT / "flow_autoparams"), remote_path=f"{REMOTE_ROOT}/flow_autoparams")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse")
)

for checkpoint_name in ("streamfm_stftpr_dnn_only.pt",):
    local_checkpoint = LOCAL_ROOT / "checkpoints" / checkpoint_name
    if local_checkpoint.exists():
        image = image.add_local_file(
            str(local_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{checkpoint_name}",
        )

app = modal.App("streamfm-backbone-profile", image=image)


def _remote_paths():
    from experiments.core.paths import make_benchmark_paths

    return make_benchmark_paths(
        repo_root=REMOTE_ROOT,
        config_dir=f"{REMOTE_ROOT}/config",
        checkpoint_roots=(
            f"{VOLUME_ROOT}/checkpoints",
            f"{REMOTE_ROOT}/checkpoints",
        ),
    )


@app.function(gpu="L4", timeout=1200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def profile_l4(
    task: str = "stftpr",
    dtype: str = "fp16",
    memory_format: str = "channels_last",
    iterations: int = 50,
    warmup: int = 15,
) -> dict:
    """Detailed eager backbone profile on Modal L4 (stage + aten breakdown)."""
    from experiments.benchmarks.profiling.backbone import run_backbone_profile
    from experiments.core.modal_cache import configure_shared_modal_cache

    configure_shared_modal_cache(volume_root=VOLUME_ROOT, hardware="L4")
    return run_backbone_profile(
        task=task,
        device="cuda",
        dtype_name=dtype,
        memory_format=memory_format,
        iterations=iterations,
        warmup=warmup,
        paths=_remote_paths(),
    )


@app.function(gpu="L4", timeout=1200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark_l4_best(
    iterations: int = 100,
    warmup: int = 20,
) -> list[dict]:
    """Best known L4 config: cuda_graph + fp16 + channels_last (+ eager/compiled refs)."""
    from experiments.benchmarks.runner import run_benchmark
    from experiments.core.modal_cache import configure_shared_modal_cache
    import torch

    cache_info = configure_shared_modal_cache(volume_root=VOLUME_ROOT, hardware="L4")
    device = torch.device("cuda")
    paths = _remote_paths()
    results: list[dict] = []
    for execution in ("cuda_graph", "compiled", "eager"):
        rows = run_benchmark(
            task="stftpr",
            part="model",
            pipeline="model_only",
            execution=execution,
            steps="1",
            iterations=iterations,
            warmup=warmup,
            model_dtype_name="fp16",
            device=device,
            paths=paths,
            backend="modal",
            hardware="L4",
            cache_info=cache_info,
            model_memory_format="channels_last",
            preallocate_model_buffers=False,
            save_audio=False,
            profile=True,
            profile_file="",
        )
        results.extend(rows)
    return results


@app.local_entrypoint()
def main(
    dtype: str = "fp16",
    memory_format: str = "channels_last",
    iterations: int = 50,
    warmup: int = 15,
    also_best_bench: bool = True,
):
    """Run L4 eager profile (+ optional cuda_graph/compiled/eager benches)."""
    out_dir = Path("outputs/benchmark_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Launching L4 eager backbone profile…")
    report = profile_l4.remote(
        dtype=dtype,
        memory_format=memory_format,
        iterations=iterations,
        warmup=warmup,
    )
    profile_path = out_dir / "backbone_profile_l4_fp16_cl.json"
    profile_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {profile_path}")
    print(json.dumps({"wall_ms": report["wall_ms"], "gpu": report["config"].get("gpu_name")}, indent=2))

    if also_best_bench:
        print("Launching L4 cuda_graph/compiled/eager reference benches with profiler…")
        benches = benchmark_l4_best.remote()
        bench_path = out_dir / "backbone_l4_best_configs.json"
        # Strip huge profile_summary duplication for readability in a side file
        slim = []
        for row in benches:
            slim_row = {k: v for k, v in row.items() if k != "profile_summary"}
            slim.append(slim_row)
            summary = row.get("profile_summary") or ""
            if summary:
                exe = row.get("requested_execution") or row.get("execution")
                (out_dir / f"l4_{exe}_profile.txt").write_text(summary, encoding="utf-8")
        bench_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"Wrote {bench_path}")
        for row in slim:
            exe = row.get("requested_execution") or row.get("execution")
            mean = row.get("total_mean_ms") or row.get("mean_ms") or row.get("model_mean_ms")
            print(f"  {exe:12s} mean={mean:.3f} ms")
