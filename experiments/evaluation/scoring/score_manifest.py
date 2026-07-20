"""Score an eval manifest against clean references.

Aligns each enhanced/clean pair and computes PESQ, ESTOI, SI-SDR, LSD and PSNR,
with a phaseless variant. Runs locally or dispatches to Modal, and can emit
per-file scores for the subset-convergence analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from torchaudio import functional as AF

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgmse.util.other import si_sdr
from experiments.evaluation.runner import select_eval_indices

MODAL_VOLUME_NAME = "streamfm-cache"


def _modal_executable() -> str:
    modal = shutil.which("modal")
    if modal:
        return modal
    repo_modal = REPO_ROOT / ".venv" / "bin" / "modal"
    if repo_modal.exists():
        return str(repo_modal)
    return "modal"


def _local_log_root(args: argparse.Namespace) -> Path:
    return Path(args.local_log_dir) if args.local_log_dir else REPO_ROOT / "outputs" / "evaluation_logs"


def _write_local_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "score"


def _score_suffix(limit: int) -> str:
    return f"limit{limit}" if limit > 0 else "all"


def _command_dict(args: argparse.Namespace) -> dict:
    return {
        "backend": args.backend,
        "source": args.source,
        "manifest": str(args.manifest) if args.manifest is not None else "",
        "run_name": args.run_name,
        "limit": args.limit,
        "offset": args.offset,
        "selection": args.selection,
        "selection_seed": args.selection_seed,
        "task": args.task,
        "split": args.split,
        "data_path": args.data_path,
        "data_format": args.data_format,
        "crop_mode": args.crop_mode,
        "with_distillmos": args.with_distillmos,
        "output_json": str(args.output_json) if args.output_json is not None else "",
        "score_target": args.score_target,
        "include_stats": args.include_stats,
        "include_per_file": args.include_per_file,
        "local_log_dir": args.local_log_dir,
        "no_local_log": args.no_local_log,
    }


def _local_metrics_path(args: argparse.Namespace) -> Path:
    suffix = _score_suffix(args.limit)
    root = _local_log_root(args)
    if args.output_json is not None and args.backend == "local":
        return args.output_json
    if args.source == "manifest" and args.run_name:
        return root / args.run_name / f"metrics_{suffix}.json"
    if args.source == "dataset":
        label = _safe_label(f"{args.task}_{args.split}_noisy_{suffix}_{args.selection}_seed{args.selection_seed}")
        return root / "dataset_scores" / f"{label}.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return root / "scores" / f"metrics_{suffix}_{timestamp}.json"


def _remote_metrics_path(args: argparse.Namespace) -> str | None:
    suffix = _score_suffix(args.limit)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        if output_path.is_absolute():
            if str(output_path).startswith("/data/"):
                return "/" + str(output_path.relative_to("/data"))
            return None
        return "/" + str(output_path)
    if args.source == "dataset":
        return f"/outputs/dataset_scores/{args.task}_{args.split}_noisy_{suffix}.json"
    if args.run_name:
        return f"/outputs/eval_runs/{args.run_name}/metrics_{suffix}.json"
    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        if manifest_path.is_absolute():
            if str(manifest_path).startswith("/data/"):
                return "/" + str(manifest_path.relative_to("/data").parent / f"metrics_{suffix}.json")
            return None
        return "/" + str(manifest_path.parent / f"metrics_{suffix}.json")
    return None


def _download_modal_file(remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _modal_executable(),
        "volume",
        "get",
        "--force",
        MODAL_VOLUME_NAME,
        remote_path,
        str(local_path),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode == 0


def _load_mono(path: str) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(path)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shaped [channels, samples], got {tuple(audio.shape)} for {path}")
    return audio[:1].float(), int(sample_rate)


def _resample(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return audio
    return AF.resample(audio, source_sr, target_sr, lowpass_filter_width=64)


def _align_pair(clean: torch.Tensor, enhanced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(clean.shape[-1], enhanced.shape[-1])
    if length <= 0:
        raise ValueError("Cannot score empty audio.")
    return clean[..., :length], enhanced[..., :length]


def _peak_normalize(audio: torch.Tensor) -> torch.Tensor:
    max_amplitude = torch.max(torch.abs(audio))
    if max_amplitude > 0:
        return audio / max_amplitude
    return audio


def _center_crop_or_pad(audio: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current_len = audio.shape[-1]
    if target_num_samples <= 0 or current_len == target_num_samples:
        return audio
    if current_len > target_num_samples:
        start = int((current_len - target_num_samples) / 2)
        return audio[..., start : start + target_num_samples]
    pad = target_num_samples - current_len
    return torch.nn.functional.pad(audio, (pad // 2, pad // 2 + (pad % 2)), mode="constant")


def _psnr(clean: torch.Tensor, enhanced: torch.Tensor, max_value: float = 1.0) -> float:
    mse = torch.mean((clean - enhanced) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def _lsd(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> float:
    """Log-spectral distance using 32 ms Hann windows and 75% overlap."""
    n_fft = int(round(0.032 * sample_rate))
    hop = int(round(0.008 * sample_rate))
    window = torch.hann_window(n_fft, device=clean.device)
    clean_stft = torch.stft(
        clean[0],
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    enhanced_stft = torch.stft(
        enhanced[0],
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    eps = 1e-8
    clean_db = 20.0 * torch.log10(clean_stft.clamp_min(eps))
    enhanced_db = 20.0 * torch.log10(enhanced_stft.clamp_min(eps))
    frame_lsd = torch.sqrt(torch.mean((clean_db - enhanced_db) ** 2, dim=0))
    return float(frame_lsd.mean().item())


def _distillmos_model(device: torch.device):
    try:
        import distillmos
    except ImportError:
        return None
    model = distillmos.ConvTransformerSQAModel()
    return model.eval().to(device)


def _score_aligned_pair(clean: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> dict:
    clean_16k = _resample(clean, sample_rate, 16000)
    estimate_16k = _resample(estimate, sample_rate, 16000)
    clean_16k, estimate_16k = _align_pair(clean_16k, estimate_16k)

    row = {
        "si_sdr": float(si_sdr(clean[0].numpy(), estimate[0].numpy())),
        "estoi": float(stoi(clean_16k[0].numpy(), estimate_16k[0].numpy(), 16000, extended=True)),
        "lsd": _lsd(clean_16k, estimate_16k, 16000),
        "psnr": _psnr(clean, estimate),
    }
    try:
        row["pesq"] = float(pesq(16000, clean_16k[0].numpy(), estimate_16k[0].numpy(), "wb"))
    except Exception as exc:
        warnings.warn(f"PESQ failed: {exc}")
        row["pesq"] = None
    return row


@torch.inference_mode()
def score_pair(
    clean_path: str,
    enhanced_path: str,
    *,
    noisy_path: str = "",
    distillmos_model=None,
    device: torch.device,
    peak_normalize_clean: bool = False,
    target_num_samples: int = 0,
    cached_noisy_metrics: dict | None = None,
) -> dict:
    clean, clean_sr = _load_mono(clean_path)
    enhanced, enhanced_sr = _load_mono(enhanced_path)
    if clean_sr != enhanced_sr:
        clean = _resample(clean, clean_sr, enhanced_sr)
        clean_sr = enhanced_sr
    if peak_normalize_clean:
        clean = _peak_normalize(clean)
    if target_num_samples > 0:
        clean = _center_crop_or_pad(clean, target_num_samples)
    clean, enhanced = _align_pair(clean, enhanced)

    row = {
        "clean_path": clean_path,
        "enhanced_path": enhanced_path,
        "sample_rate": clean_sr,
        "duration_s": float(clean.shape[-1] / clean_sr),
    }
    row.update(_score_aligned_pair(clean, enhanced, clean_sr))

    if noisy_path:
        row["noisy_path"] = noisy_path
        if cached_noisy_metrics is not None:
            row["noisy_metrics"] = dict(cached_noisy_metrics)
        else:
            noisy, noisy_sr = _load_mono(noisy_path)
            if noisy_sr != clean_sr:
                noisy = _resample(noisy, noisy_sr, clean_sr)
            if target_num_samples > 0:
                noisy = _center_crop_or_pad(noisy, target_num_samples)
            clean_for_noisy, noisy = _align_pair(clean, noisy)
            row["noisy_metrics"] = _score_aligned_pair(clean_for_noisy, noisy, clean_sr)
        row["metric_delta_vs_noisy"] = {
            key: (
                None
                if row.get(key) is None or row["noisy_metrics"].get(key) is None
                else float(row[key] - row["noisy_metrics"][key])
            )
            for key in ("pesq", "estoi", "si_sdr", "lsd", "psnr")
        }

    if distillmos_model is not None:
        enhanced_16k = _resample(enhanced, clean_sr, 16000)
        mos = distillmos_model(enhanced_16k.to(device))
        row["distillmos"] = float(mos.item())
    return row


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None and math.isfinite(value)]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _stats(values: list[float | None]) -> dict:
    filtered = np.array(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if filtered.size == 0:
        return {
            "mean": None,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(filtered)),
        "min": float(np.min(filtered)),
        "median": float(np.median(filtered)),
        "max": float(np.max(filtered)),
    }


def _manifest_items(manifest_path: Path, limit: int) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(manifest.get("task", "")).lower()
    crop_mode = str(manifest.get("crop_mode", "")).lower()
    items = []
    for item in manifest.get("files", []):
        saved_clean_path = item.get("saved_clean_path")
        clean_path = saved_clean_path or item.get("clean_path")
        noisy_path = item.get("saved_noisy_path") or item.get("noisy_path") or ""
        enhanced_path = item.get("enhanced_path")
        if clean_path and enhanced_path:
            target_num_samples = 0
            peak_normalize_clean = False
            if not saved_clean_path and task == "se":
                peak_normalize_clean = True
            if not saved_clean_path and crop_mode == "config":
                target_num_samples = int(item.get("num_samples") or 0)
            items.append(
                {
                    "clean_path": clean_path,
                    "noisy_path": noisy_path,
                    "enhanced_path": enhanced_path,
                    "peak_normalize_clean": peak_normalize_clean,
                    "target_num_samples": target_num_samples,
                    "reference_note": (
                        "reconstructed_from_original_clean"
                        if peak_normalize_clean or target_num_samples > 0
                        else "original_clean_file"
                    ),
                }
            )
        if limit > 0 and len(items) >= limit:
            break
    if not items:
        errors = manifest.get("errors") or []
        if errors:
            first_error = errors[0]
            raise ValueError(
                f"No scoreable clean/enhanced pairs found in {manifest_path}. "
                f"Manifest has num_files={manifest.get('num_files', 0)} and "
                f"num_errors={manifest.get('num_errors', len(errors))}. "
                f"First error at index {first_error.get('index')}: "
                f"{first_error.get('error_type')}: {first_error.get('error')}"
            )
        raise ValueError(f"No scoreable clean/enhanced pairs found in {manifest_path}")
    return items


NOISY_CACHE_VERSION = 1


def _default_noisy_cache_path(manifest_path: Path) -> Path:
    """Use one persistent cache file without creating a file per audio pair."""
    if manifest_path.is_absolute() and str(manifest_path).startswith("/data/"):
        return Path("/data/outputs/evaluation_cache/noisy_metrics.json")
    return REPO_ROOT / "outputs" / "evaluation_cache" / "noisy_metrics.json"


def _noisy_cache_key(item: dict) -> str:
    payload = {
        "version": NOISY_CACHE_VERSION,
        "clean_path": str(item["clean_path"]),
        "noisy_path": str(item["noisy_path"]),
        "peak_normalize_clean": bool(item.get("peak_normalize_clean", False)),
        "target_num_samples": int(item.get("target_num_samples", 0)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_noisy_metric_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != NOISY_CACHE_VERSION:
        return {}
    entries = payload.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def _write_noisy_metric_cache(path: Path, entries: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps({"version": NOISY_CACHE_VERSION, "entries": entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_data_format(task: str) -> str:
    task = task.lower().replace("-", "_")
    if task in {"se", "stftpr", "melflow"}:
        return "ears_wham"
    if task == "derev":
        return "ears_reverb"
    if task in {"bwe", "lyra"}:
        return "paired_dirs"
    raise ValueError(f"No default data format for task '{task}'. Pass --data-format.")


def _dataset_items(
    *,
    task: str,
    data_path: str,
    data_format: str,
    split: str,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    crop_mode: str,
) -> list[dict]:
    from sgmse.data_module import AudioDataset

    if not data_path:
        raise ValueError("--data-path is required with --source dataset.")
    data_format = data_format or _default_data_format(task)
    dataset = AudioDataset(
        path=data_path,
        format=data_format,
        random_crop=False,
        target_duration=32512,
        sampling_rate=16000,
        whichset=split,
        spatial_channels=1,
        crop_paired_to_shorter=data_format == "paired_dirs",
        peak_normalize_clean=False,
        random_neg_gain_noisy=None,
    )
    indices = select_eval_indices(
        num_available=len(dataset),
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
    )
    target_num_samples = 32512 if crop_mode == "config" else 0
    items = []
    for idx in indices:
        clean_path = dataset.clean_files[idx]
        noisy_path = dataset.noisy_files[idx]
        items.append(
            {
                "index": idx,
                "clean_path": clean_path,
                "noisy_path": noisy_path,
                "target_num_samples": target_num_samples,
            }
        )
    return items


def score_dataset_noisy(
    *,
    task: str,
    data_path: str,
    data_format: str,
    split: str,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    crop_mode: str,
    output_json: Path | None,
    include_per_file: bool = False,
    include_stats: bool = False,
) -> dict:
    crop_mode = crop_mode.lower().replace("-", "_")
    if crop_mode not in {"full", "config"}:
        raise ValueError("--crop-mode must be 'full' or 'config'.")
    items = _dataset_items(
        task=task,
        data_path=data_path,
        data_format=data_format,
        split=split,
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
        crop_mode=crop_mode,
    )
    per_file = []
    for item in items:
        clean, clean_sr = _load_mono(item["clean_path"])
        noisy, noisy_sr = _load_mono(item["noisy_path"])
        if noisy_sr != clean_sr:
            noisy = _resample(noisy, noisy_sr, clean_sr)
        if item["target_num_samples"] > 0:
            clean = _center_crop_or_pad(clean, item["target_num_samples"])
            noisy = _center_crop_or_pad(noisy, item["target_num_samples"])
        clean, noisy = _align_pair(clean, noisy)
        row = {
            "index": item["index"],
            "clean_path": item["clean_path"],
            "noisy_path": item["noisy_path"],
            "sample_rate": clean_sr,
            "duration_s": float(clean.shape[-1] / clean_sr),
            **_score_aligned_pair(clean, noisy, clean_sr),
        }
        per_file.append(row)

    metric_names = ("pesq", "estoi", "si_sdr", "lsd", "psnr")
    summary = {
        "source": "dataset",
        "target": "noisy",
        "task": task,
        "data_path": data_path,
        "data_format": data_format or _default_data_format(task),
        "split": split,
        "crop_mode": crop_mode,
        "selection": selection,
        "selection_seed": selection_seed,
        "offset": offset,
        "limit": limit,
        "num_files": len(per_file),
        "noisy": {name: _mean([row.get(name) for row in per_file]) for name in metric_names},
    }
    if include_stats:
        summary["noisy_stats"] = {
            name: _stats([row.get(name) for row in per_file])
            for name in metric_names
        }
    if include_per_file:
        summary["per_file"] = per_file
    print(json.dumps(summary, indent=2, sort_keys=True))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _build_stftpr_extractor():
    """Feature extractor matching config/streamfm_stftpr.yaml (n_fft=512, hop=256,
    sqrt-Hann, alpha=0.5, normalized STFT, Nyquist bin cut). Used to build the
    phase-retrieval degraded input from clean audio."""
    from sgmse.feature_extractors import CompressedAmplitudeComplexSTFT

    return CompressedAmplitudeComplexSTFT(
        window="hann",
        n_fft=512,
        hop_length=256,
        sampling_rate=16000,
        alpha=0.5,
        beta=1.0,
        compression_is_learnable=False,
        normalized_stft=True,
        cut_highest_freqs=1,
        sqrt_window=True,
    )


def _build_melflow_extractor():
    """Feature extractor matching config/streamfm_melflow.yaml (n_fft=512, hop=256,
    plain Hann, alpha=0.5, normalized STFT, Nyquist bin cut). Differs from the STFT-PR
    extractor only in sqrt_window=False (melflow matches HiFi-GAN's plain Hann)."""
    from sgmse.feature_extractors import CompressedAmplitudeComplexSTFT

    return CompressedAmplitudeComplexSTFT(
        window="hann",
        n_fft=512,
        hop_length=256,
        sampling_rate=16000,
        alpha=0.5,
        beta=1.0,
        compression_is_learnable=False,
        normalized_stft=True,
        cut_highest_freqs=1,
        sqrt_window=False,
    )


def _build_melflow_projector():
    """Mel bottleneck matching config/streamfm_melflow.yaml post_Y_fn: magnitude ->
    80-band slaney Mel -> Mel pseudoinverse. This is the ``M†`` the paper applies before
    the zero-phase reconstruction for the Mel vocoding degraded baseline."""
    from sgmse.util.diffphase import PhaselessMelAndBack

    return PhaselessMelAndBack(
        n_mels=80,
        sample_rate=16000,
        f_min=0.0,
        f_max=8000,
        n_stft=256,
        norm="slaney",
        mel_scale="slaney",
        alpha=0.5,
    )


def _phase_retrieval_degrade(
    clean: torch.Tensor,
    extractor,
    *,
    phase_mode: str,
    seed: int,
    mel_projector=None,
) -> torch.Tensor:
    """Build the phase-retrieval degraded input for a clean signal.

    Mirrors what the FlowModel sees at inference: it keeps only the STFT
    magnitude (``post_Y_fn=ComplexAbs``) and must recover the phase. Here we
    reconstruct a time-domain signal from that magnitude with either zeroed or
    random phase, so metrics quantify the starting-point degradation.

    clean: [1, T] mono tensor. Returns a [1, T] degraded tensor.
    """
    x = clean.unsqueeze(0)  # [B=1, C=1, T]
    spec = extractor.forward(x)  # compressed complex STFT [1, 1, F, T]
    if mel_projector is not None:
        # Mel bottleneck (M†): magnitude -> Mel -> Mel pseudoinverse. Matches the
        # melflow model's post_Y_fn, so the baseline is what that model starts from.
        spec = mel_projector(spec)
    magnitude = spec.abs()
    if phase_mode == "zero":
        degraded_spec = magnitude + 0j
    elif phase_mode == "random":
        generator = torch.Generator().manual_seed(seed)
        phase = (torch.rand(magnitude.shape, generator=generator) * 2.0 - 1.0) * math.pi
        degraded_spec = torch.polar(magnitude, phase)
    else:
        raise ValueError("phase_mode must be 'zero' or 'random'.")
    degraded = extractor.invert(degraded_spec, T_orig=x.shape[-1])  # [1, 1, T]
    return degraded[0]  # [1, T]


def score_dataset_phaseless(
    *,
    task: str,
    data_path: str,
    data_format: str,
    split: str,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    crop_mode: str,
    output_json: Path | None,
    phase_mode: str = "random",
    phase_seed: int = 1234,
    include_per_file: bool = False,
    include_stats: bool = False,
) -> dict:
    """Score the phase-retrieval degraded input (magnitude-only reconstruction)
    against clean, per file. No model inference: the degraded signal is derived
    from the clean STFT magnitude, matching what the stftpr model starts from."""
    crop_mode = crop_mode.lower().replace("-", "_")
    if crop_mode not in {"full", "config"}:
        raise ValueError("--crop-mode must be 'full' or 'config'.")
    phase_mode = phase_mode.lower().strip()
    if phase_mode not in {"random", "zero"}:
        raise ValueError("--phase-mode must be 'random' or 'zero'.")
    items = _dataset_items(
        task=task,
        data_path=data_path,
        data_format=data_format,
        split=split,
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
        crop_mode=crop_mode,
    )
    # melflow starts from a Mel-bottlenecked magnitude (M†), stftpr from the full
    # STFT magnitude; the extractor windowing also differs (see the builders).
    if task == "melflow":
        extractor = _build_melflow_extractor()
        mel_projector = _build_melflow_projector()
    else:
        extractor = _build_stftpr_extractor()
        mel_projector = None
    per_file = []
    for item in items:
        clean, clean_sr = _load_mono(item["clean_path"])
        if item["target_num_samples"] > 0:
            clean = _center_crop_or_pad(clean, item["target_num_samples"])
        # deterministic per-file phase so subsampling variance is only from
        # which files are selected, not from the phase realization
        degraded = _phase_retrieval_degrade(
            clean,
            extractor,
            phase_mode=phase_mode,
            seed=phase_seed + int(item["index"]),
            mel_projector=mel_projector,
        )
        clean, degraded = _align_pair(clean, degraded)
        row = {
            "index": item["index"],
            "clean_path": item["clean_path"],
            "sample_rate": clean_sr,
            "duration_s": float(clean.shape[-1] / clean_sr),
            **_score_aligned_pair(clean, degraded, clean_sr),
        }
        per_file.append(row)

    metric_names = ("pesq", "estoi", "si_sdr", "lsd", "psnr")
    summary = {
        "source": "dataset_phaseless",
        "target": "degraded",
        "phase_mode": phase_mode,
        "phase_seed": phase_seed,
        "task": task,
        "data_path": data_path,
        "data_format": data_format or _default_data_format(task),
        "split": split,
        "crop_mode": crop_mode,
        "selection": selection,
        "selection_seed": selection_seed,
        "offset": offset,
        "limit": limit,
        "num_files": len(per_file),
        "degraded": {name: _mean([row.get(name) for row in per_file]) for name in metric_names},
    }
    if include_stats:
        summary["degraded_stats"] = {
            name: _stats([row.get(name) for row in per_file])
            for name in metric_names
        }
    if include_per_file:
        summary["per_file"] = per_file
    print(json.dumps(summary, indent=2, sort_keys=True))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def score_manifest(
    manifest_path: Path,
    *,
    limit: int,
    with_distillmos: bool,
    output_json: Path | None,
    include_per_file: bool = False,
    include_stats: bool = False,
    score_target: str = "enhanced",
    noisy_cache_path: Path | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distillmos_model = _distillmos_model(device) if with_distillmos else None
    score_target = score_target.lower().replace("-", "_")
    if score_target not in {"enhanced", "noisy"}:
        raise ValueError("--score-target must be 'enhanced' or 'noisy'.")
    items = _manifest_items(manifest_path, limit)
    noisy_cache_path = noisy_cache_path or _default_noisy_cache_path(manifest_path)
    noisy_cache = _load_noisy_metric_cache(noisy_cache_path)
    noisy_cache_hits = 0
    noisy_cache_misses = 0
    per_file = []
    for item in items:
        cache_key = _noisy_cache_key(item) if item["noisy_path"] else ""
        cached_entry = noisy_cache.get(cache_key) if cache_key else None
        if isinstance(cached_entry, dict) and isinstance(cached_entry.get("metrics"), dict):
            cached_noisy_metrics = cached_entry["metrics"]
        elif isinstance(cached_entry, dict):
            # Backward-compatible with the initial cache format.
            cached_noisy_metrics = cached_entry
        else:
            cached_noisy_metrics = None
        if cached_noisy_metrics is not None:
            noisy_cache_hits += 1
        elif cache_key:
            noisy_cache_misses += 1
        row = score_pair(
            item["clean_path"],
            item["enhanced_path"],
            noisy_path=item["noisy_path"],
            distillmos_model=distillmos_model,
            device=device,
            peak_normalize_clean=item["peak_normalize_clean"],
            target_num_samples=item["target_num_samples"],
            cached_noisy_metrics=cached_noisy_metrics,
        )
        if cache_key and cached_noisy_metrics is None and row.get("noisy_metrics"):
            noisy_cache[cache_key] = {
                "clean_path": item["clean_path"],
                "noisy_path": item["noisy_path"],
                "peak_normalize_clean": item["peak_normalize_clean"],
                "target_num_samples": item["target_num_samples"],
                "metrics": row["noisy_metrics"],
            }
        row["reference_note"] = item["reference_note"]
        per_file.append(row)
    if noisy_cache_misses:
        _write_noisy_metric_cache(noisy_cache_path, noisy_cache)
    metric_names = ("pesq", "estoi", "si_sdr", "lsd", "psnr", "distillmos")
    noisy_metric_names = ("pesq", "estoi", "si_sdr", "lsd", "psnr")
    if score_target == "noisy":
        summary = {
            "manifest_path": str(manifest_path),
            "num_files": len(per_file),
            "target": "noisy",
            "noisy": {
                name: _mean([row.get("noisy_metrics", {}).get(name) for row in per_file])
                for name in noisy_metric_names
            },
            "noisy_cache": {
                "path": str(noisy_cache_path),
                "hits": noisy_cache_hits,
                "misses": noisy_cache_misses,
            },
        }
        if include_stats:
            summary["noisy_stats"] = {
                name: _stats([row.get("noisy_metrics", {}).get(name) for row in per_file])
                for name in noisy_metric_names
            }
        if include_per_file:
            summary["per_file"] = per_file
        print(json.dumps(summary, indent=2, sort_keys=True))
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    summary = {
        "manifest_path": str(manifest_path),
        "num_files": len(per_file),
        "target": "enhanced",
        "enhanced": {name: _mean([row.get(name) for row in per_file]) for name in metric_names},
        "noisy": {
            name: _mean([row.get("noisy_metrics", {}).get(name) for row in per_file])
            for name in noisy_metric_names
        },
        "delta_vs_noisy": {
            name: _mean([row.get("metric_delta_vs_noisy", {}).get(name) for row in per_file])
            for name in noisy_metric_names
        },
        "noisy_cache": {
            "path": str(noisy_cache_path),
            "hits": noisy_cache_hits,
            "misses": noisy_cache_misses,
        },
    }
    if include_stats:
        summary["enhanced_stats"] = {
            name: _stats([row.get(name) for row in per_file])
            for name in metric_names
        }
        summary["noisy_stats"] = {
            name: _stats([row.get("noisy_metrics", {}).get(name) for row in per_file])
            for name in noisy_metric_names
        }
        summary["delta_vs_noisy_stats"] = {
            name: _stats([row.get("metric_delta_vs_noisy", {}).get(name) for row in per_file])
            for name in noisy_metric_names
        }
    if include_per_file:
        summary["per_file"] = per_file
    print(json.dumps(summary, indent=2, sort_keys=True))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _resolve_local_manifest(manifest: Path | None, run_name: str) -> Path:
    if manifest is not None:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        return manifest
    if not run_name:
        raise ValueError("Pass either a manifest path or --run-name.")
    manifest_path = REPO_ROOT / "outputs" / "eval_runs" / run_name / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    eval_runs_dir = REPO_ROOT / "outputs" / "eval_runs"
    existing_runs = sorted(path.name for path in eval_runs_dir.iterdir() if path.is_dir()) if eval_runs_dir.exists() else []
    matches = get_close_matches(run_name, existing_runs, n=5, cutoff=0.45)
    hint = f" Close matches: {', '.join(matches)}." if matches else ""
    raise FileNotFoundError(f"Manifest not found for run-name '{run_name}': {manifest_path}.{hint}")


def _run_modal(args: argparse.Namespace) -> None:
    local_metrics_path = _local_metrics_path(args)
    if not args.no_local_log:
        _write_local_json(local_metrics_path.parent / "score_command.json", _command_dict(args))
    command = [
        _modal_executable(),
        "run",
        "experiments/evaluation/scoring/modal_score_manifest.py",
        "--source",
        args.source,
        "--limit",
        str(args.limit),
    ]
    command.extend(["--offset", str(args.offset)])
    command.extend(["--selection", args.selection])
    command.extend(["--selection-seed", str(args.selection_seed)])
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    if args.manifest is not None:
        command.extend(["--manifest", str(args.manifest)])
    if args.task:
        command.extend(["--task", args.task])
    if args.split:
        command.extend(["--split", args.split])
    if args.data_path:
        command.extend(["--data-path", args.data_path])
    if args.data_format:
        command.extend(["--data-format", args.data_format])
    command.extend(["--crop-mode", args.crop_mode])
    if args.with_distillmos:
        command.append("--with-distillmos")
    if args.output_json is not None:
        command.extend(["--output-json", str(args.output_json)])
    command.extend(["--score-target", args.score_target])
    if args.include_stats:
        command.append("--include-stats")
    if args.include_per_file:
        command.append("--include-per-file")
    subprocess.run(command, check=True)
    if args.no_local_log:
        return
    remote_path = _remote_metrics_path(args)
    if remote_path is None:
        _write_local_json(
            local_metrics_path.parent / "score_modal_volume_paths.json",
            {
                "warning": "Automatic download skipped because the remote metrics path is not inside /data.",
                "output_json": str(args.output_json) if args.output_json is not None else "",
            },
        )
        return
    downloaded = _download_modal_file(remote_path, local_metrics_path)
    _write_local_json(
        local_metrics_path.parent / "score_modal_volume_paths.json",
        {
            "volume": MODAL_VOLUME_NAME,
            "remote_metrics_path": remote_path,
            "local_metrics_path": str(local_metrics_path),
            "downloaded": downloaded,
        },
    )
    if downloaded:
        print(f"Local metrics saved to {local_metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Stream.FM eval manifest against clean references.")
    parser.add_argument("manifest", type=Path, nargs="?", help="Path to outputs/eval_runs/<run>/manifest.json.")
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--source", choices=("manifest", "dataset"), default="manifest")
    parser.add_argument("--run-name", default="", help="Run directory name under outputs/eval_runs.")
    parser.add_argument("--limit", type=int, default=0, help="Score N selected files. 0 means all files.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("first", "random"),
        default="random",
        help="random selects a reproducible subset; first selects files after offset in dataset order.",
    )
    parser.add_argument("--selection-seed", type=int, default=42, help="Seed used when --selection random.")
    parser.add_argument("--task", default="se")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-format", default="")
    parser.add_argument("--crop-mode", choices=("full", "config"), default="full")
    parser.add_argument("--with-distillmos", action="store_true", help="Also compute DistillMOS if the package is installed.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--score-target", choices=("enhanced", "noisy"), default="enhanced")
    parser.add_argument("--include-stats", action="store_true", help="Include mean/min/median/max metric stats.")
    parser.add_argument("--include-per-file", action="store_true", help="Include per-file metric rows in the JSON output.")
    parser.add_argument(
        "--local-log-dir",
        default=str(REPO_ROOT / "outputs" / "evaluation_logs"),
        help="Local directory where score command/metrics JSON copies are saved.",
    )
    parser.add_argument("--no-local-log", action="store_true", help="Disable local score metadata copies.")
    args = parser.parse_args()

    if args.backend == "modal":
        _run_modal(args)
        return

    if not args.no_local_log:
        _write_local_json(_local_metrics_path(args).parent / "score_command.json", _command_dict(args))

    if args.source == "dataset":
        score_dataset_noisy(
            task=args.task,
            data_path=args.data_path,
            data_format=args.data_format,
            split=args.split,
            limit=args.limit,
            offset=args.offset,
            selection=args.selection,
            selection_seed=args.selection_seed,
            crop_mode=args.crop_mode,
            output_json=args.output_json if args.output_json is not None else (
                None if args.no_local_log else _local_metrics_path(args)
            ),
            include_per_file=args.include_per_file,
            include_stats=args.include_stats,
        )
        return

    manifest_path = _resolve_local_manifest(args.manifest, args.run_name)
    score_manifest(
        manifest_path,
        limit=args.limit,
        with_distillmos=args.with_distillmos,
        output_json=args.output_json if args.output_json is not None else (
            None if args.no_local_log else _local_metrics_path(args)
        ),
        include_per_file=args.include_per_file,
        include_stats=args.include_stats,
        score_target=args.score_target,
    )


if __name__ == "__main__":
    main()
