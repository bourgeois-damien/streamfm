from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from experiments.streaming.pipeline import StreamingSTFTConfig, run_streaming_audio_pipeline


def select_device(name: str | None = None) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_stftpr_backbone(device: torch.device):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulated streaming audio pipeline for Stream.FM STFTPR.",
    )
    parser.add_argument("--input", default="inputs/test_clips/audio_43m28_10s.wav")
    parser.add_argument("--output", default="outputs/streaming_audio_local_stftpr.wav")
    parser.add_argument("--json", default="outputs/streaming_audio_local_stftpr.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compiled", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_stftpr_backbone(device)
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
    summary["output"] = str(Path(args.output))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, audio.clamp(-1, 1), sample_rate=sr)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote audio to {output_path}")
    print(f"Wrote metrics to {json_path}")


if __name__ == "__main__":
    main()
