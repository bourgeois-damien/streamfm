from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

CACHE_MODES = ("off", "read", "write", "readwrite")

ENGINE_FILENAME = "engine.ep"
METADATA_FILENAME = "metadata.json"

DEFAULT_CACHE_DIR_ENV = "STREAMFM_TRT_ENGINE_CACHE_DIR"


def normalize_cache_mode(mode: str | None) -> str:
    if not mode:
        return "off"
    normalized = str(mode).strip().lower().replace("-", "").replace("_", "")
    if normalized not in CACHE_MODES:
        raise ValueError(
            f"Engine cache mode must be one of {CACHE_MODES}, got {mode!r}."
        )
    return normalized


def resolve_cache_dir(cache_dir: str | Path | None) -> Path | None:
    candidate = cache_dir or os.environ.get(DEFAULT_CACHE_DIR_ENV, "")
    if not candidate:
        return None
    return Path(candidate).expanduser()


def toolchain_signature() -> dict[str, str]:
    import torch

    signature = {"torch": torch.__version__}
    for module_name, label in (("torch_tensorrt", "torch_tensorrt"), ("tensorrt", "tensorrt")):
        try:
            module = __import__(module_name)
        except ImportError:
            signature[label] = "unavailable"
        else:
            signature[label] = str(getattr(module, "__version__", "unknown"))
    return signature


def device_signature() -> dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        return {"gpu_name": "cpu", "compute_capability": "none"}
    major, minor = torch.cuda.get_device_capability()
    return {
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": f"{major}.{minor}",
    }


def model_signature(model) -> str:
    import torch

    digest = hashlib.blake2b(digest_size=16)
    state_dict = model.state_dict()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        if not tensor.numel():
            continue
        flat = tensor.detach().to("cpu").contiguous().flatten()
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_cache_config(
    *,
    model_hash: str,
    precision: str,
    dtype: str,
    int8_fallback_dtype: str,
    calibration_steps: int,
    quant_scope: str,
    quant_coverage: float,
    calibration_source: str,
    calibration_files: int,
    calibration_seconds: float,
    calibration_split: str,
    calibration_solver_steps: str,
    calibration_seed: int,
    memory_format: str,
    allow_tf32: bool | None,
    optimization_level: int,
    num_avg_timing_iters: int,
    workspace_size_bytes: int,
    require_full_compilation: bool,
    input_channels: int,
    input_freqs: int,
    state_signature: str,
) -> dict[str, Any]:
    quantized = precision in {"int8", "fp8"}
    config: dict[str, Any] = {
        "model_hash": model_hash,
        "precision": precision,
        "dtype": dtype,
        "calibration_steps": calibration_steps if quantized else None,
        "quant_scope": quant_scope if quantized else None,
        "quant_coverage": quant_coverage if quantized else None,
        "calibration_source": calibration_source if quantized else None,
        "calibration_files": calibration_files if quantized else None,
        "calibration_seconds": calibration_seconds if quantized else None,
        "calibration_split": calibration_split if quantized else None,
        "calibration_solver_steps": calibration_solver_steps if quantized else None,
        "calibration_seed": calibration_seed if quantized else None,
        "int8_fallback_dtype": int8_fallback_dtype if quantized else None,
        "memory_format": memory_format,
        "allow_tf32": allow_tf32,
        "optimization_level": optimization_level,
        "num_avg_timing_iters": num_avg_timing_iters,
        "workspace_size_bytes": workspace_size_bytes,
        "require_full_compilation": require_full_compilation,
        "input_channels": input_channels,
        "input_freqs": input_freqs,
        "state_signature": state_signature,
    }
    config.update(toolchain_signature())
    config.update(device_signature())
    return config


def compute_cache_key(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


class EngineCache:
    def __init__(self, mode: str | None, cache_dir: str | Path | None = None):
        self.mode = normalize_cache_mode(mode)
        self.root = resolve_cache_dir(cache_dir)
        if self.mode != "off" and self.root is None:
            raise ValueError(
                f"engine cache mode={self.mode!r} needs a cache dir "
                f"(or ${DEFAULT_CACHE_DIR_ENV})."
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def may_read(self) -> bool:
        return self.mode in {"read", "readwrite"}

    @property
    def may_write(self) -> bool:
        return self.mode in {"write", "readwrite"}

    def entry_dir(self, precision: str, key: str) -> Path:
        assert self.root is not None
        return self.root / "engines" / precision / key

    def lookup(self, precision: str, key: str) -> Path | None:
        if not self.may_read:
            return None
        entry = self.entry_dir(precision, key)
        if (entry / ENGINE_FILENAME).exists() and (entry / METADATA_FILENAME).exists():
            return entry
        return None

    def read_metadata(self, entry: Path) -> dict[str, Any]:
        try:
            return json.loads((entry / METADATA_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def load(self, entry: Path):
        import torch_tensorrt

        engine_path = entry / ENGINE_FILENAME
        loaded = torch_tensorrt.load(str(engine_path))
        module = loaded.module() if hasattr(loaded, "module") else loaded
        return module

    def store(
        self,
        precision: str,
        key: str,
        engine,
        *,
        config: dict[str, Any],
        arg_inputs,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.may_write:
            return None
        import torch_tensorrt

        entry = self.entry_dir(precision, key)
        staging = entry.with_name(entry.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            torch_tensorrt.save(
                engine,
                str(staging / ENGINE_FILENAME),
                output_format="exported_program",
                arg_inputs=list(arg_inputs),
                retrace=False,
            )
            metadata = {"cache_key": key, "config": config}
            if extra_metadata:
                metadata.update(extra_metadata)
            (staging / METADATA_FILENAME).write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if entry.exists():
            shutil.rmtree(entry, ignore_errors=True)
        staging.rename(entry)
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if self.root is None:
            return []
        engines_root = self.root / "engines"
        if not engines_root.exists():
            return []
        listed = []
        for metadata_path in sorted(engines_root.glob("*/*/" + METADATA_FILENAME)):
            entry = metadata_path.parent
            engine_path = entry / ENGINE_FILENAME
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {"error": "unreadable metadata"}
            listed.append(
                {
                    "path": str(entry),
                    "precision": entry.parent.name,
                    "cache_key": entry.name,
                    "size_mib": (
                        engine_path.stat().st_size / (1024.0 * 1024.0)
                        if engine_path.exists()
                        else 0.0
                    ),
                    "metadata": metadata,
                }
            )
        return listed

    def remove(self, precision: str, key: str) -> Path | None:
        entry = self.entry_dir(precision, key)
        if not entry.exists():
            return None
        shutil.rmtree(entry)
        return entry


def _format_entry(entry: dict[str, Any], *, verbose: bool) -> str:
    config = entry.get("metadata", {}).get("config", {})
    header = (
        f"{entry['precision']:>5}  {entry['cache_key']}  "
        f"{entry['size_mib']:8.1f} MiB  "
        f"{config.get('gpu_name', '?')} / trt {config.get('tensorrt', '?')}"
    )
    if not verbose:
        return header
    detail = json.dumps(config, indent=2, sort_keys=True)
    return header + "\n" + "\n".join(f"    {line}" for line in detail.splitlines())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir",
        default="",
        help=f"Cache root (defaults to ${DEFAULT_CACHE_DIR_ENV}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="List cached engines.")
    list_parser.add_argument(
        "--verbose", action="store_true", help="Print the full build configuration."
    )
    remove_parser = subparsers.add_parser("remove", help="Delete one cached engine.")
    remove_parser.add_argument("precision", help="Precision subdirectory, e.g. fp16.")
    remove_parser.add_argument("cache_key", help="Exact cache key to delete.")

    args = parser.parse_args(argv)
    # 'read' rather than 'readwrite': inspection must never create anything.
    cache = EngineCache("read", args.cache_dir or None)

    if args.command == "list":
        entries = cache.entries()
        if not entries:
            print(f"No cached engines under {cache.root}.")
            return 0
        total = sum(entry["size_mib"] for entry in entries)
        for entry in entries:
            print(_format_entry(entry, verbose=args.verbose))
        print(f"\n{len(entries)} engine(s), {total:.1f} MiB total, under {cache.root}.")
        return 0

    removed = cache.remove(args.precision, args.cache_key)
    if removed is None:
        print(f"No such entry: {args.precision}/{args.cache_key}")
        return 1
    print(f"Removed {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
