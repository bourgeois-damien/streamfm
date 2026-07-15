"""From-scratch speech-enhancement baseline for Stream.FM.

Loads the SE model from its Hydra config and checkpoint and runs a plain
streaming forward loop with per-step timing. This is the reference point the
optimized benchmark and eval paths are measured against.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from experiments.core.devices import select_torch_device, sync_device
from experiments.core.streaming_state import forward_step
from experiments.core.timing import summarize_ms


def select_device(name: str = "auto") -> torch.device:
    """Resolve a user device name to the best available torch device."""
    return select_torch_device(name)


def load_streamfm_se_model(
    device: torch.device,
    config_name: str = "streamfm_se_predgen",
    ckpt_path: str | Path = "checkpoints/streamfm_se_predgen.ckpt",
):
    """Load the Stream.FM speech-enhancement model from config and checkpoint."""
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path

    with initialize_config_dir(config_dir=str(repo_root / "config"), version_base="1.3"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)
    return model, cfg


def load_mono_audio(path: str | Path, device: torch.device, target_sr: int | None = None):
    """Load audio, keep one channel, optionally resample, and move to device."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr, lowpass_filter_width=64)
        sr = target_sr
    return wav.to(device), sr


def run_offline_inference(
    model,
    wav: torch.Tensor,
    sr: int,
    device: torch.device,
    solver: str = "euler",
    steps: int = 1,
    seed: int = 0,
):
    """Run the model's offline enhance path and return real-time metrics."""
    torch.manual_seed(seed)
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.enhance(wav, sr, solver=solver, N=int(steps))
    sync_device(device)
    elapsed_s = time.perf_counter() - start
    duration_s = wav.shape[-1] / sr
    return {
        "mode": "offline_enhance",
        "device": device.type,
        "solver": solver,
        "steps": int(steps),
        "audio_duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "rtf": elapsed_s / duration_s,
        "output": out.detach().cpu(),
    }


def benchmark_frame_steps(
    model,
    device: torch.device,
    steps_list: Iterable[int] = (1,),
    iterations: int = 100,
    warmup: int = 10,
    use_compiled: bool = False,
) -> list[dict[str, float | int | str | bool]]:
    """Benchmark predictor plus flow latency for one-frame streaming steps."""
    predictor = model.initial_predictor.dnn.eval()
    flow = model.dnn.eval()
    results = []

    for steps in steps_list:
        steps = int(steps)
        y_frame = torch.randn(1, 2, 256, 1, device=device)
        predictor_state = predictor.init_state()
        flow_states = [flow.init_state() for _ in range(steps)]
        times_ms = []

        with torch.inference_mode():
            for frame_idx in range(warmup + iterations):
                sync_device(device)
                start = time.perf_counter()

                e_frame, predictor_state = forward_step(
                    predictor,
                    y_frame,
                    state=predictor_state,
                    use_compiled=use_compiled,
                )

                x_t = e_frame
                for step_idx in range(steps):
                    t = torch.full((1,), step_idx / max(steps, 1), device=device)
                    dnn_input = torch.cat([x_t, e_frame, y_frame], dim=1)
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input,
                        state=flow_states[step_idx],
                        time_cond=t,
                        use_compiled=use_compiled,
                    )
                    x_t = x_t + v / steps

                sync_device(device)
                if frame_idx >= warmup:
                    times_ms.append((time.perf_counter() - start) * 1000.0)

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_predictor_plus_flow",
                "device": device.type,
                "steps": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "frame_budget_ms": 16.0,
                "budget_ratio_mean": summary["mean_ms"] / 16.0,
            }
        )
        results.append(summary)

    return results


def write_results(path: str | Path, rows: list[dict]) -> None:
    """Write benchmark rows as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
