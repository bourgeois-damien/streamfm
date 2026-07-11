from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import modal

VOLUME_ROOT = "/data"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")

app = modal.App("streamfm-dataset-duration")


def _duration_s(path: Path) -> float:
    sample_rate = None
    block_align = None
    data_size = None
    with path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            raise ValueError(f"Not a RIFF WAV file: {path}")
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            raise ValueError(f"Not a WAVE file: {path}")
        while True:
            chunk_id = handle.read(4)
            if not chunk_id:
                break
            chunk_size_data = handle.read(4)
            if len(chunk_size_data) != 4:
                break
            chunk_size = struct.unpack("<I", chunk_size_data)[0]
            chunk_start = handle.tell()
            if chunk_id == b"fmt ":
                fmt = handle.read(min(chunk_size, 16))
                if len(fmt) < 16:
                    raise ValueError(f"Invalid fmt chunk in WAV file: {path}")
                _, _, sample_rate, _, block_align, _ = struct.unpack("<HHIIHH", fmt[:16])
            elif chunk_id == b"data":
                data_size = chunk_size
            handle.seek(chunk_start + chunk_size + (chunk_size % 2))
            if sample_rate and block_align and data_size is not None:
                return data_size / float(block_align * sample_rate)
    raise ValueError(f"Could not read WAV duration from: {path}")


def _summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "total_s": sum(values),
        "mean_s": sum(values) / len(values),
        "min_s": ordered[0],
        "p50_s": ordered[len(ordered) // 2],
        "p90_s": ordered[int(0.9 * (len(ordered) - 1))],
        "p99_s": ordered[int(0.99 * (len(ordered) - 1))],
        "max_s": ordered[-1],
    }


def _read_ears_wham_split_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for raw in reader:
            if len(raw) == len(header):
                rows.append(dict(zip(header, raw)))
            elif len(raw) == 12:
                rows.append(
                    dict(
                        zip(
                            [
                                "id",
                                "speaker",
                                "speech_file",
                                "speech_start",
                                "speech_end",
                                "noise_file",
                                "noise_start",
                                "noise_end",
                                "speech_dB",
                                "noise_dB",
                                "mixture_dB",
                                "snr_dB",
                            ],
                            raw,
                        )
                    )
                )
            else:
                raise ValueError(f"Unexpected CSV row width {len(raw)} in {csv_path}: {raw[:5]}")
    return rows


@app.function(timeout=20 * 60, volumes={VOLUME_ROOT: CACHE_VOLUME})
def dataset_duration_remote(dataset_dir: str, split: str, paired: bool) -> dict:
    dataset = Path(dataset_dir)
    root = dataset / split
    rows = _read_ears_wham_split_rows(dataset / f"{split}.csv")
    clean_files = [root / "clean" / row["speaker"] / f"{row['id']}.wav" for row in rows]
    noisy_files = (
        [
            root / "noisy" / row["speaker"] / f"{row['id']}_{row['snr_dB']}dB.wav"
            for row in rows
        ]
        if paired
        else []
    )
    clean_durations = [_duration_s(path) for path in clean_files]
    noisy_durations = [_duration_s(path) for path in noisy_files] if paired else []
    result = {
        "dataset_dir": dataset_dir,
        "split": split,
        "paired": paired,
        "clean": _summarize(clean_durations),
        "noisy": _summarize(noisy_durations) if paired else "not_read_assumed_same_duration",
        "first_clean_files": [str(path) for path in clean_files[:5]],
        "first_noisy_files": [str(path) for path in noisy_files[:5]],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    dataset_dir: str = f"{VOLUME_ROOT}/datasets/EARS-WHAM_v2_16k",
    split: str = "test",
    paired: bool = False,
):
    result = dataset_duration_remote.remote(dataset_dir=dataset_dir, split=split, paired=paired)
    print(json.dumps(result, indent=2, sort_keys=True))
