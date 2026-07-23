"""Compare the exact 1- and 2-ResBlock DEREV architectures on an NVIDIA L4.

This benchmark intentionally instantiates both Hydra architectures directly.
It does not require a trained 1-ResBlock checkpoint because parameter values do
not affect the execution shapes or CUDA Graph latency.  Quality evaluation,
however, still requires weights trained with the matching architecture.

    modal run experiments/pruning/modal_true_1resblock_latency.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

from experiments.benchmarks.modal_streamfm_benchmark import (
    CACHE_VOLUME,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image,
)

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

app = modal.App("streamfm-true-1resblock-latency", image=image)


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def benchmark(iterations: int, warmup: int) -> list[dict]:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from experiments.benchmarks.modal_streamfm_benchmark import (
        _configure_persistent_cache_env,
        _remote_paths,
    )
    from experiments.benchmarks.runner import _streaming_config_from_model_cfg
    from experiments.core.tensors import apply_model_memory_format
    from experiments.streaming.pipeline import (
        make_synthetic_audio,
        run_streaming_audio_pipeline_with_full_cuda_graph,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container.")

    _configure_persistent_cache_env("L4")
    paths = _remote_paths()
    device = torch.device("cuda")
    dtype = torch.float16
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    rows = []
    for config_name in (
        "streamfm_derev",
        "study_ablation_streamfm_derev_1resblock",
    ):
        with initialize_config_dir(config_dir=str(paths.config_dir), version_base="1.3"):
            cfg = compose(config_name=config_name)
        model = instantiate(cfg.model.backbone).eval().to(device=device, dtype=dtype)
        model = apply_model_memory_format(model, "channels_last")
        streaming_config = _streaming_config_from_model_cfg(cfg)
        audio = make_synthetic_audio(
            num_samples=(warmup + iterations) * streaming_config.hop_length,
            sample_rate=streaming_config.sample_rate,
            device=device,
        )
        summary = run_streaming_audio_pipeline_with_full_cuda_graph(
            model,
            audio,
            device=device,
            steps=1,
            iterations=iterations,
            warmup=warmup,
            use_compiled=True,
            config=streaming_config,
            model_dtype=dtype,
            model_memory_format="channels_last",
        )
        summary.update(
            {
                "config_name": config_name,
                "num_res_blocks": int(cfg.model.backbone.get("num_res_blocks", 2)),
                "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "hardware": "NVIDIA L4",
            }
        )
        rows.append(summary)
    return rows


@app.local_entrypoint()
def main(
    iterations: int = 500,
    warmup: int = 100,
    out: str = "results/pruning/true_1resblock_latency_derev.json",
):
    rows = benchmark.remote(iterations=iterations, warmup=warmup)
    baseline_ms = rows[0]["total_mean_ms"]
    one_block_ms = rows[1]["total_mean_ms"]
    result = {
        "comparison": {
            "latency_reduction_pct": 100.0 * (1.0 - one_block_ms / baseline_ms),
            "speedup": baseline_ms / one_block_ms,
            "parameter_reduction_pct": 100.0
            * (1.0 - rows[1]["num_parameters"] / rows[0]["num_parameters"]),
        },
        "rows": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))

