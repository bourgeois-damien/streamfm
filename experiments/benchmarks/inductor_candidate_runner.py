"""Run one benchmark under an explicitly patched TorchInductor configuration.

This module is intentionally a small subprocess entrypoint.  TorchInductor
reads several settings and cache locations during import, so each candidate is
started in a fresh interpreter with its own persistent cache directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any


def _set_dotted_attribute(root: Any, dotted_name: str, value: Any) -> None:
    target = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _jsonable_cache_info(cache_info: Any) -> Any:
    if is_dataclass(cache_info):
        return asdict(cache_info)
    if hasattr(cache_info, "_asdict"):
        return cache_info._asdict()
    if isinstance(cache_info, (dict, list, str, int, float, bool)) or cache_info is None:
        return cache_info
    return repr(cache_info)


def _directory_summary(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch TorchInductor, run one benchmark, and export its compiler cache."
    )
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    args, benchmark_args = parser.parse_known_args()
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]

    candidate = json.loads(args.candidate_json)
    candidate_name = str(candidate["name"])
    patches = dict(candidate.get("inductor_options", {}))
    force_big_gpu = bool(candidate.get("force_big_gpu", False))

    # Import and patch Inductor before importing the model, whose forward_step
    # function is decorated with torch.compile at module-import time.
    import torch
    from torch._inductor import config as inductor_config

    if force_big_gpu:
        # is_big_gpu() gates Triton GEMM template autotuning behind a
        # hardcoded 68-SM heuristic (torch/_inductor/utils.py). The L4 (~58
        # SMs) falls under that bar, so max-autotune-gemm never runs on it.
        # It's lru_cache'd and only referenced within utils.py itself, so
        # patching the module attribute before the first compile is enough.
        import torch._inductor.utils as inductor_utils

        inductor_utils.is_big_gpu = lambda *_args, **_kwargs: True

    available_options = set(torch._inductor.list_options())
    unknown = sorted(name for name in patches if name not in available_options)
    if unknown:
        raise ValueError(f"Unsupported TorchInductor option(s): {', '.join(unknown)}")
    for name, value in patches.items():
        _set_dotted_attribute(inductor_config, name, value)

    output_path = Path(args.output_json)
    manifest_path = Path(args.manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    from experiments.benchmarks.streamfm_benchmark import main as benchmark_main

    sys.argv = ["streamfm_benchmark", *benchmark_args, "--output-json", str(output_path)]
    benchmark_main()

    rows = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a result list in {output_path}, got {type(rows).__name__}.")

    cache_root = Path(os.environ["STREAMFM_INDUCTOR_CANDIDATE_ROOT"])
    portable_path = cache_root / "portable_torch_compile_cache.pt2"
    portable_info: dict[str, Any] = {
        "path": str(portable_path),
        "saved": False,
        "bytes": 0,
        "cache_info": None,
    }
    artifacts = torch.compiler.save_cache_artifacts()
    if artifacts is not None:
        artifact_bytes, cache_info = artifacts
        portable_path.write_bytes(artifact_bytes)
        portable_info.update(
            {
                "saved": True,
                "bytes": len(artifact_bytes),
                "cache_info": _jsonable_cache_info(cache_info),
            }
        )

    compiler_metadata = {
        "candidate": candidate_name,
        "description": str(candidate.get("description", "")),
        "inductor_options": patches,
        "force_big_gpu": force_big_gpu,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "candidate_cache_root": str(cache_root),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME", ""),
        "portable_cache": portable_info,
    }
    for row in rows:
        row["compiler_candidate"] = candidate_name
        row["compiler_candidate_description"] = compiler_metadata["description"]
        row["compiler_inductor_options"] = patches
        row["compiler_candidate_cache_root"] = str(cache_root)
        row["compiler_portable_cache_path"] = str(portable_path) if portable_info["saved"] else ""
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        **compiler_metadata,
        "benchmark_args": benchmark_args,
        "output_json": str(output_path),
        "result_count": len(rows),
        "results": rows,
        "cache_directory_summary": _directory_summary(cache_root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
