"""Modal entrypoint: real per-frame latency of the depth-pruned model on L4.

The quality curve (``modal_ablation.py``) says what dropping the k least
influential residual blocks costs.  This says what it buys, measured rather
than estimated: the same deployed configuration -- fp16, channels-last,
preallocated buffers, STFT + solver + ISTFT captured as a single CUDA Graph --
run once per k inside one container, so every number comes from the same GPU,
the same clocks and the same warm cache.

    modal run experiments/pruning/modal_prune_latency.py --ks 0,1,2,3

Because the graph is captured after the transform, a pruned block is absent
from the capture entirely: no kernel is recorded for it, so the replay is
genuinely shorter rather than launching cheap no-ops.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

# Reuse the benchmark image (config/, sgmse/, experiments/ and the checkpoints)
# and the shared cache volume.
from experiments.benchmarks.modal_streamfm_benchmark import (
    CACHE_VOLUME,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image,
)
from experiments.pruning.modal_ablation import DEREV_RANKING

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

app = modal.App("streamfm-prune-latency", image=image)


@app.function(gpu="L4", timeout=7200, volumes={VOLUME_ROOT: CACHE_VOLUME})
def latency(
    task: str,
    ranking: list[str],
    ks: list[int],
    pipeline: str,
    execution: str,
    steps: str,
    iterations: int,
    warmup: int,
    model_dtype_name: str,
    model_memory_format: str,
    preallocate_model_buffers: bool,
    float32_matmul_precision: str,
    tf32_mode: str,
    cudnn_benchmark: bool,
    checkpoint_name: str,
) -> list[dict]:
    from functools import partial

    import torch

    from experiments.benchmarks.modal_streamfm_benchmark import (
        _configure_persistent_cache_env,
        _remote_paths,
    )
    from experiments.benchmarks.runner import run_benchmark
    from sgmse.backbones.streaming_unet import prune_resblocks_

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this Modal container.")

    device = torch.device("cuda")
    cache_info = _configure_persistent_cache_env("L4")
    paths = _remote_paths()

    rows: list[dict] = []
    for k in ks:
        drop = list(ranking[:k])
        transform = partial(prune_resblocks_, drop=drop) if drop else None
        print(f"\n=== k={k}: dropping {drop or '[baseline]'} ===")

        results = run_benchmark(
            task=task,
            part="model",
            pipeline=pipeline,
            execution=execution,
            steps=steps,
            iterations=iterations,
            warmup=warmup,
            model_dtype_name=model_dtype_name,
            device=device,
            paths=paths,
            backend="modal",
            hardware="L4",
            cache_info=cache_info,
            float32_matmul_precision=float32_matmul_precision,
            preallocate_model_buffers=preallocate_model_buffers,
            model_memory_format=model_memory_format,
            tf32_mode=tf32_mode,
            cudnn_benchmark=cudnn_benchmark,
            checkpoint_name=checkpoint_name,
            backbone_transform=transform,
        )
        for row in results:
            row = json.loads(json.dumps(row))
            row["k"] = k
            row["dropped"] = drop
            rows.append(row)

    return rows


@app.local_entrypoint()
def main(
    task: str = "derev",
    ranking: str = "",
    ks: str = "0,1,2,3",
    # audio + cuda_graph_full = the deployed configuration: STFT + solver +
    # ISTFT in a single CUDA Graph replay, one launch per frame.
    pipeline: str = "audio",
    execution: str = "cuda_graph_full",
    steps: str = "1",
    iterations: int = 500,
    warmup: int = 100,
    model_dtype_name: str = "fp16",
    model_memory_format: str = "channels_last",
    preallocate_model_buffers: bool = True,
    float32_matmul_precision: str = "high",
    tf32_mode: str = "on",
    cudnn_benchmark: bool = True,
    checkpoint_name: str = "",
    out: str = "results/pruning/prune_latency_derev.json",
):
    ranking_list = [name.strip() for name in ranking.split(",") if name.strip()] or DEREV_RANKING
    k_list = [int(value) for value in ks.split(",") if value.strip()]

    rows = latency.remote(
        task=task,
        ranking=ranking_list,
        ks=k_list,
        pipeline=pipeline,
        execution=execution,
        steps=steps,
        iterations=iterations,
        warmup=warmup,
        model_dtype_name=model_dtype_name,
        model_memory_format=model_memory_format,
        preallocate_model_buffers=preallocate_model_buffers,
        float32_matmul_precision=float32_matmul_precision,
        tf32_mode=tf32_mode,
        cudnn_benchmark=cudnn_benchmark,
        checkpoint_name=checkpoint_name,
    )

    baseline = next((r["total_mean_ms"] for r in rows if r["k"] == 0), None)
    print("\nPer-frame latency vs blocks removed (L4, fp16, full CUDA graph):")
    print(f"{'k':>2}  {'mean ms':>9}  {'p50 ms':>9}  {'p99 ms':>9}  {'budget':>8}  {'speedup':>8}   dropped")
    for r in rows:
        speedup = f"{baseline / r['total_mean_ms']:>7.3f}x" if baseline else f"{'NA':>8}"
        last = r["dropped"][-1] if r["dropped"] else "[baseline]"
        print(
            f"{r['k']:>2}  {r['total_mean_ms']:>9.4f}  {r.get('total_p50_ms', float('nan')):>9.4f}  "
            f"{r.get('total_p99_ms', float('nan')):>9.4f}  {r['budget_ratio_mean']:>7.1%}  {speedup}   +{last}"
        )
    if rows:
        print(f"\nFrame budget: {rows[0]['frame_budget_ms']:.3f} ms")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "task": task,
                    "ranking": ranking_list,
                    "pipeline": pipeline,
                    "execution": execution,
                    "model_dtype": model_dtype_name,
                    "iterations": iterations,
                    "warmup": warmup,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote latency curve to {out}")
