"""Local simulated-streaming runner for Stream.FM STFTPR.

Loads the STFTPR backbone and drives the streaming pipeline over an audio file
(or synthetic audio), writing the reconstructed output. The CLI entry point for
trying the streaming path locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from experiments.streaming.pipeline import StreamingSTFTConfig, run_streaming_audio_pipeline


def select_device(name: str | None = None) -> torch.device:
    """Resolve a CLI device value to a torch device."""
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_stftpr_backbone(device: torch.device):
    """Instantiate the STFTPR backbone and load its lightweight DNN weights."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base="1.3"):
        cfg = compose(config_name="streamfm_stftpr")
    model = instantiate(cfg.model.backbone)
    state = torch.load(
        REPO_ROOT / "checkpoints/streamfm_stftpr_dnn_only.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.eval().to(device)


def default_audio_output_path(args: argparse.Namespace, device: torch.device) -> Path:
    """Build a descriptive output filename when the user did not provide one."""
    compiled_label = "compiled" if not args.no_compiled else "eager"
    input_stem = Path(args.input).stem
    filename = (
        f"stftpr_audio_device-{device.type}_steps-{args.steps}_"
        f"{compiled_label}_input-{input_stem}.wav"
    )
    return Path("outputs/audio") / filename


def main() -> None:
    """Run the local streaming STFTPR pipeline and write audio plus metrics."""
    parser = argparse.ArgumentParser(
        description="Simulated streaming audio pipeline for Stream.FM STFTPR.",
    )
    parser.add_argument("--input", default="inputs/test_clips/benchmark_input_10s.wav")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compiled", action="store_true")
    args = parser.parse_args()

    # No audio ships with the repo: fail with a usable message rather than a
    # backend decoding error.
    if not Path(args.input).is_file():
        parser.error(
            f"input audio not found: {args.input}\n"
            "Pass --input with your own 16 kHz mono WAV, or drop one at that path."
        )

    device = select_device(args.device)
    model = load_stftpr_backbone(device)
    # The model is trained on 16 kHz mono: keep the first channel and resample
    # anything else before framing.
    wav, sr = torchaudio.load(args.input)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000, lowpass_filter_width=64)
        sr = 16000

    config = StreamingSTFTConfig(sample_rate=sr)
    summary = run_streaming_audio_pipeline(
        model,
        wav,
        device=device,
        steps=args.steps,
        iterations=args.iterations,
        warmup=args.warmup,
        use_compiled=not args.no_compiled,
        config=config,
        return_audio=True,
    )
    audio = summary.pop("audio")
    summary["device"] = device.type
    summary["input"] = str(Path(args.input))
    output_path = Path(args.output) if args.output else default_audio_output_path(args, device)
    summary["output"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, audio.clamp(-1, 1), sample_rate=sr)
    print(json.dumps(summary, indent=2))
    print(f"Wrote audio to {output_path}")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {json_path}")


if __name__ == "__main__":
    main()
