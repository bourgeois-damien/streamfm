"""Five-candidate TorchInductor search for the best L4 NFE=1 configuration.

Every candidate runs in a fresh subprocess with a separate persistent cache.
The Modal volume therefore keeps both the ordinary Inductor/Triton cache and a
portable cache bundle returned by ``torch.compiler.save_cache_artifacts``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal

from experiments.benchmarks.modal_streamfm_benchmark import (
    CACHE_VOLUME,
    LOCAL_ROOT,
    REMOTE_ROOT,
    VOLUME_ROOT,
    image as benchmark_image,
)


CAMPAIGN_ID = "l4_nfe1_inductor5_20260722_v1"
DEFAULT_LOCAL_OUTPUT = "outputs/final_runs/l4_nfe1_inductor5_20260722.json"
INPUT_AUDIO_NAME = "benchmark_input_10s.wav"
LOCAL_INPUT_AUDIO = LOCAL_ROOT / "inputs" / "test_clips" / INPUT_AUDIO_NAME
REMOTE_INPUT_AUDIO = f"{REMOTE_ROOT}/inputs/test_clips/{INPUT_AUDIO_NAME}"


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline_max_autotune",
        "description": "Current best max-autotune configuration.",
        "inductor_options": {},
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "epilogue_1x1_search",
        "description": "Explore more fused epilogues and lower 1x1 convolutions as GEMMs.",
        "inductor_options": {
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 4,
            "conv_1x1_as_mm": True,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "combo_kernel_search",
        "description": "Autotune experimental combo kernels for independent small kernels.",
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "forced_big_gpu_combo",
        "description": (
            "Force is_big_gpu()=True to unlock Triton GEMM autotune templates on L4 "
            "(hardcoded 68-SM gate, L4 has ~58), stacked on top of the combo-kernel win."
        ),
        "force_big_gpu": True,
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "combo_mixed_sizes_max",
        "description": (
            "combo_kernel_search plus combo_kernel_allow_mixed_sizes=2 (enable for all, "
            "including foreach), to group more differently-sized independent kernels."
        ),
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "combo_kernel_allow_mixed_sizes": 2,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "combo_cpp_wrapper",
        "description": (
            "combo_kernel_search plus cpp_wrapper=True, to cut Python/dispatch launch "
            "overhead per kernel -- the workload is ~1ms total so overhead may dominate."
        ),
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "cpp_wrapper": True,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "combo_foreach_dynamic",
        "description": (
            "combo_kernel_search plus combo_kernel_foreach_dynamic_shapes=True, for the "
            "per-block streaming state passed through forward_step as lists."
        ),
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "combo_kernel_foreach_dynamic_shapes": True,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "fusion_benchmark_no_reorder",
        "description": (
            "Retry fusion_benchmark_search without loop_ordering_after_fusion, which "
            "crashed with InductorError: AttributeError 'NoneType' object has no "
            "attribute 'reorder_iter_loops'. Stacked on combo_kernels."
        ),
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "benchmark_fusion": True,
            "aggressive_fusion": True,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
    {
        "name": "combo_conv_1x1_as_mm",
        "description": (
            "combo_kernel_search plus conv_1x1_as_mm=True, rerouting 1x1 convs to cuBLAS "
            "gemm instead of cuDNN; flat on its own in epilogue_1x1_search but untested "
            "stacked with combo kernel fusion."
        ),
        "inductor_options": {
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "conv_1x1_as_mm": True,
        },
        "cudnn_benchmark": False,
        "cudnn_benchmark_limit": 10,
    },
)


if not LOCAL_INPUT_AUDIO.exists():
    raise FileNotFoundError(f"Required benchmark clip not found: {LOCAL_INPUT_AUDIO}")

search_image = benchmark_image.add_local_file(
    str(LOCAL_INPUT_AUDIO),
    remote_path=REMOTE_INPUT_AUDIO,
)
app = modal.App("streamfm-l4-inductor-search", image=search_image)


def _benchmark_args(
    candidate: dict[str, Any],
    remote_output: str,
    remote_campaign_root: str,
    task: str,
    prune_drop: list[str],
) -> list[str]:
    args = [
        "--backend", "local",
        "--hardware", "cuda",
        "--task", task,
        "--part", "model",
        "--pipeline", "audio",
        "--execution", "cuda_graph_full",
        "--steps", "1",
        "--iterations", "500",
        "--warmup", "100",
        "--dtype", "fp16",
        "--memory-format", "channels_last",
        "--matmul-precision", "high",
        "--tf32", "auto",
        "--input-audio", REMOTE_INPUT_AUDIO,
        "--cudnn-benchmark-limit", str(candidate["cudnn_benchmark_limit"]),
        "--history-json", f"{remote_campaign_root}/history.json",
    ]
    if prune_drop:
        args += ["--prune-drop", ",".join(prune_drop)]
    args.append("--cudnn-benchmark" if candidate["cudnn_benchmark"] else "--no-cudnn-benchmark")
    return args


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def run_search(
    task: str = "stftpr",
    prune_drop: list[str] | None = None,
    campaign_id: str = CAMPAIGN_ID,
    candidates_json: str = "",
) -> dict[str, Any]:
    prune_drop = prune_drop or []
    candidates: tuple[dict[str, Any], ...] = (
        tuple(json.loads(candidates_json)) if candidates_json else CANDIDATES
    )
    remote_campaign_root = f"{VOLUME_ROOT}/cache/inductor_search/{campaign_id}"
    campaign_root = Path(remote_campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)
    candidate_outputs: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        name = candidate["name"]
        candidate_root = campaign_root / name
        candidate_root.mkdir(parents=True, exist_ok=True)
        output_path = candidate_root / "result.json"
        manifest_path = candidate_root / "manifest.json"

        # A completed candidate is immutable and immediately reusable.  This
        # also makes interrupted campaigns resumable without paying for a
        # second compilation or changing the selected artifact.
        if manifest_path.exists() and output_path.exists():
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached_rows = cached_manifest.get("results", [])
            if (
                cached_manifest.get("candidate") == name
                and cached_manifest.get("inductor_options", {})
                == candidate.get("inductor_options", {})
                and cached_manifest.get("force_big_gpu", False)
                == bool(candidate.get("force_big_gpu", False))
                and cached_manifest.get("task", "stftpr") == task
                and cached_manifest.get("prune_drop", []) == prune_drop
                and cached_rows
            ):
                first = cached_rows[0]
                candidate_outputs.append(
                    {
                        "candidate": candidate,
                        "returncode": 0,
                        "cache_reused": True,
                        "remote_cache_root": str(candidate_root),
                        "remote_output_json": str(output_path),
                        "remote_manifest_json": str(manifest_path),
                        "manifest": cached_manifest,
                        "total_mean_ms": first.get("total_mean_ms", first.get("mean_ms")),
                        "total_p50_ms": first.get("total_p50_ms", first.get("p50_ms")),
                        "total_p90_ms": first.get("total_p90_ms", first.get("p90_ms")),
                        "total_p99_ms": first.get("total_p99_ms", first.get("p99_ms")),
                    }
                )
                print(f"[{index}/{len(candidates)}] Reusing completed {name}", flush=True)
                continue

        env = os.environ.copy()
        # Candidates that only patch Python-level behavior (e.g. force_big_gpu
        # monkeypatching torch._inductor.utils.is_big_gpu) don't change any
        # torch._inductor.config value, so Inductor's FX graph cache key can't
        # tell them apart from a config-identical candidate. Sharing a native
        # cache directory with such a candidate silently replays the other
        # candidate's compiled kernels instead of exercising the patch. Give
        # these an isolated cache, same as the baseline.
        needs_isolated_cache = name == "baseline_max_autotune" or bool(
            candidate.get("force_big_gpu")
        )
        native_cache_root = (
            candidate_root
            if needs_isolated_cache
            else campaign_root / "shared_native_variant_cache"
        )
        env.update(
            {
                "PYTHONPATH": REMOTE_ROOT,
                "TORCH_HOME": f"{VOLUME_ROOT}/cache/shared/torch",
                "TORCHINDUCTOR_CACHE_DIR": str(native_cache_root / "torchinductor"),
                "TRITON_CACHE_DIR": str(native_cache_root / "triton"),
                "XDG_CACHE_HOME": str(native_cache_root / "xdg"),
                "STREAMFM_INDUCTOR_CANDIDATE_ROOT": str(candidate_root),
            }
        )
        for directory in (
            native_cache_root / "torchinductor",
            native_cache_root / "triton",
            native_cache_root / "xdg",
            native_cache_root / "xdg" / "torch" / "kernels",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benchmarks.inductor_candidate_runner",
            "--candidate-json",
            json.dumps(candidate, sort_keys=True),
            "--output-json",
            str(output_path),
            "--manifest-json",
            str(manifest_path),
            "--",
            *_benchmark_args(candidate, str(output_path), remote_campaign_root, task, prune_drop),
        ]
        print(f"[{index}/{len(candidates)}] Starting {name}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=REMOTE_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=720,
            )
            completed_output = completed.stdout
            completed_returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timeout_output = exc.stdout or b""
            if isinstance(timeout_output, bytes):
                timeout_output = timeout_output.decode("utf-8", errors="replace")
            completed_output = timeout_output + "\nCandidate exceeded the 720 s build limit."
            completed_returncode = 124
        print(completed_output, flush=True)
        item: dict[str, Any] = {
            "candidate": candidate,
            "returncode": completed_returncode,
            "remote_cache_root": str(candidate_root),
            "remote_output_json": str(output_path),
            "remote_manifest_json": str(manifest_path),
        }
        if completed_returncode == 0 and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item["manifest"] = manifest
            rows = manifest.get("results", [])
            if rows:
                first = rows[0]
                item["total_mean_ms"] = first.get("total_mean_ms", first.get("mean_ms"))
                item["total_p50_ms"] = first.get("total_p50_ms", first.get("p50_ms"))
                item["total_p90_ms"] = first.get("total_p90_ms", first.get("p90_ms"))
                item["total_p99_ms"] = first.get("total_p99_ms", first.get("p99_ms"))
            CACHE_VOLUME.commit()
        else:
            item["error_output"] = completed_output[-12000:]
        candidate_outputs.append(item)

    successful = [item for item in candidate_outputs if item.get("total_mean_ms") is not None]
    successful.sort(key=lambda item: item["total_mean_ms"])
    summary = {
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": "L4",
        "protocol": {
            "task": task,
            "prune_drop": prune_drop,
            "pipeline": "audio",
            "execution": "cuda_graph_full",
            "steps": 1,
            "iterations": 500,
            "warmup": 100,
            "dtype": "fp16",
            "memory_format": "channels_last",
            "input_audio": INPUT_AUDIO_NAME,
        },
        "remote_campaign_root": remote_campaign_root,
        "candidate_outputs": candidate_outputs,
        "ranking": [
            {
                "rank": rank,
                "candidate": item["candidate"]["name"],
                "mean_ms": item["total_mean_ms"],
                "p50_ms": item.get("total_p50_ms"),
                "p90_ms": item.get("total_p90_ms"),
                "p99_ms": item.get("total_p99_ms"),
                "remote_cache_root": item["remote_cache_root"],
                "portable_cache": item["manifest"].get("portable_cache", {}),
            }
            for rank, item in enumerate(successful, start=1)
        ],
    }
    (campaign_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    CACHE_VOLUME.commit()
    return summary


@app.local_entrypoint()
def main(
    output: str = DEFAULT_LOCAL_OUTPUT,
    task: str = "stftpr",
    prune_drop: str = "",
    campaign_id: str = CAMPAIGN_ID,
    candidates_json: str = "",
) -> None:
    prune_drop_list = [name.strip() for name in prune_drop.split(",") if name.strip()]
    summary = run_search.remote(
        task=task,
        prune_drop=prune_drop_list,
        campaign_id=campaign_id,
        candidates_json=candidates_json,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = LOCAL_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved local summary to {output_path}")
    for row in summary.get("ranking", []):
        print(
            f"#{row['rank']} {row['candidate']}: mean={row['mean_ms']:.6f} ms "
            f"p99={row['p99_ms']:.6f} ms cache={row['remote_cache_root']}"
        )
