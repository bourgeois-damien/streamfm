from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.streaming.stft import (
    StreamingSTFTConfig,
    compensate_streaming_delay,
    streaming_algorithmic_delay,
    streaming_num_frames,
)


@dataclass(frozen=True)
class QualityRunOptions:
    split: str = "test"
    data_path: str = ""  # overrides the split path from the checkpoint config
    data_format: str = ""
    limit: int = 0  # 0 = every file in the split
    offset: int = 0
    selection: str = "random"  # 'random' | 'first' | 'sequential'
    selection_seed: int = 42
    # per-file seed = seed + index (same as eval) so noise matches offline refs
    seed: int = 42
    crop_mode: str = "full"
    output_dir: str = ""
    run_id: str = ""
    overwrite: bool = False
    continue_on_error: bool = False


def build_split_dataset(cfg, split: str, data_path: str = "", data_format: str = ""):
    from hydra.utils import instantiate

    if not hasattr(cfg, "model") or not hasattr(cfg.model, "data_module"):
        raise ValueError("Config has no model.data_module; cannot iterate a split.")

    data_module_cfg = cfg.model.data_module
    if data_path:
        setattr(data_module_cfg, f"{split}_path", data_path)
    if data_format:
        data_module_cfg.format = data_format

    data_module = instantiate(data_module_cfg)
    if split == "test":
        data_module.setup(stage="test")
        return data_module.test_set
    if split == "valid":
        data_module.setup(stage="fit")
        return data_module.valid_set
    if split == "train":
        data_module.setup(stage="_train_only")
        return data_module.train_set
    raise ValueError(f"Unsupported split: {split}")


def _safe_stem(path: str, fallback: str) -> str:
    stem = Path(path).stem if path else ""
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-")
    return stem or fallback


def run_streaming_quality_sweep(
    *,
    pipeline_fn,
    pipeline_kwargs: dict,
    model,
    cfg,
    config: StreamingSTFTConfig,
    device,
    steps_list: tuple[int, ...],
    options: QualityRunOptions,
    extra_manifest: dict | None = None,
) -> list[dict]:
    """Streaming enhance over a split subset; one manifest per step count for score_manifest.
    pipeline_fn should already be bound to the built model so engines/graphs are reused across files.
    """
    import numpy as np
    import torch
    import torchaudio

    from experiments.evaluation.runner import select_eval_indices

    dataset = build_split_dataset(
        cfg, options.split, options.data_path, options.data_format
    )
    indices = select_eval_indices(
        num_available=len(dataset),
        limit=options.limit,
        offset=options.offset,
        selection=options.selection,
        selection_seed=options.selection_seed,
    )
    if not indices:
        raise ValueError("File selection is empty; check --limit/--offset/--split.")

    base_dir = (
        Path(options.output_dir)
        if options.output_dir
        else Path("outputs") / "benchmark_quality"
    )
    run_dir = base_dir / (options.run_id or "run")

    summaries = []
    for step_count in steps_list:
        step_dir = run_dir / f"steps{step_count}"
        enhanced_dir = step_dir / "enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        files: list[dict] = []
        errors: list[dict] = []
        started_at = time.perf_counter()

        for idx in indices:
            item_started_at = time.perf_counter()
            try:
                x, y, info = dataset.__getitem__(
                    idx, no_crop=(options.crop_mode == "full")
                )
                sr = int(info["sr"])
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                if y.shape[0] > 1:
                    y = y[0:1]
                num_samples = int(y.shape[-1])

                file_seed = options.seed + idx
                np.random.seed(file_seed)
                torch.manual_seed(file_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(file_seed)

                summary = pipeline_fn(
                    model,
                    y.to(device),
                    device=device,
                    steps=step_count,
                    iterations=streaming_num_frames(num_samples, config),
                    warmup=0,
                    config=config,
                    seed=file_seed,
                    return_audio=True,
                    **pipeline_kwargs,
                )
                enhanced = summary.pop("audio", None)
                if enhanced is None:
                    raise ValueError("Streaming pipeline returned no audio.")
                if enhanced.ndim == 1:
                    enhanced = enhanced.unsqueeze(0)
                enhanced = compensate_streaming_delay(enhanced.float(), num_samples, config)

                stem = _safe_stem(info.get("y_path", ""), f"{idx:06d}")
                enhanced_path = enhanced_dir / f"{idx:06d}_{stem}.wav"
                if enhanced_path.exists() and not options.overwrite:
                    raise FileExistsError(
                        f"Output exists. Use --overwrite to replace: {enhanced_path}"
                    )
                torchaudio.save(
                    str(enhanced_path), enhanced.cpu().clamp(-1, 1), sample_rate=sr
                )

                files.append(
                    {
                        "index": idx,
                        "clean_path": info.get("x_path", ""),
                        "noisy_path": info.get("y_path", ""),
                        "enhanced_path": str(enhanced_path),
                        "saved_clean_path": "",
                        "saved_noisy_path": "",
                        "sample_rate": sr,
                        "num_samples": num_samples,
                        "duration_s": round(num_samples / float(sr), 3),
                        "elapsed_s": time.perf_counter() - item_started_at,
                    }
                )
            except Exception as exc:
                if not options.continue_on_error:
                    raise
                errors.append({"index": idx, "error": f"{type(exc).__name__}: {exc}"})

        manifest = {
            **asdict(options),
            "steps": step_count,
            "solver": "euler",
            "part": "model",
            "pipeline": "streaming",
            "source": "benchmark",
            "selected_indices": indices,
            "num_available": len(dataset),
            "num_files": len(files),
            "num_errors": len(errors),
            "streaming": {
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "sample_rate": config.sample_rate,
                "algorithmic_delay_samples": streaming_algorithmic_delay(config),
            },
            "files": files,
            "errors": errors,
            **(extra_manifest or {}),
        }
        manifest_path = step_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        summaries.append(
            {
                "steps": step_count,
                "manifest_path": str(manifest_path),
                "output_dir": str(step_dir),
                "num_files": len(files),
                "num_errors": len(errors),
                "elapsed_s": time.perf_counter() - started_at,
            }
        )
        print(f"Wrote quality manifest for steps={step_count}: {manifest_path}")

    return summaries
