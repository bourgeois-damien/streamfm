from __future__ import annotations

import json
import os
from pathlib import Path

import modal

VOLUME_ROOT = "/data"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")

app = modal.App("streamfm-dataset-progress")


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def _file_info(path: Path) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "size": _human_bytes(stat.st_size),
    }


def _ears_zip_progress(ears_dir: Path) -> dict:
    expected = [f"p{idx:03d}.zip" for idx in range(1, 108)]
    zip_present = []
    extracted_present = []
    missing = []
    for name in expected:
        path = ears_dir / name
        stem = Path(name).stem
        extracted_candidates = (
            ears_dir / stem,
            ears_dir / "audio" / stem,
            ears_dir / "speech" / stem,
        )
        has_zip = path.exists()
        has_extracted = any(candidate.exists() for candidate in extracted_candidates)
        if has_zip:
            zip_present.append(name)
        if has_extracted:
            extracted_present.append(name)
        if has_zip or has_extracted:
            continue
        matching_extracted_files = list(ears_dir.rglob(f"*{stem}*"))
        if matching_extracted_files:
            extracted_present.append(name)
        else:
            missing.append(name)

    present = sorted(set(zip_present) | set(extracted_present))
    last_present = present[-1] if present else ""
    last_zip_present = zip_present[-1] if zip_present else ""
    next_expected = missing[0] if missing else ""
    return {
        "expected_zip_count": len(expected),
        "present_zip_count": len(present),
        "zip_files_currently_present_count": len(zip_present),
        "extracted_or_processed_count": len(extracted_present),
        "missing_zip_count": len(missing),
        "progress_percent_by_zip_count": round(100.0 * len(present) / len(expected), 2),
        "last_present_zip": last_present,
        "last_zip_file_currently_present": last_zip_present,
        "last_zip_file_currently_present_info": _file_info(ears_dir / last_zip_present) if last_zip_present else {},
        "next_expected_zip": next_expected,
        "first_missing_zips": missing[:10],
    }


def _dataset_status(root: Path) -> dict:
    paths = {
        "volume_root": root,
        "datasets": root / "datasets",
        "ears_raw": root / "datasets" / "EARS",
        "wham_raw": root / "datasets" / "WHAM48kHz",
        "ears_wham_16k": root / "datasets" / "EARS-WHAM_v2_16k",
        "ears_reverb_16k": root / "datasets" / "EARS-Reverb_v2_16k",
        "repos": root / "repos",
        "cache": root / "cache",
        "checkpoints": root / "checkpoints",
    }
    sizes = {}
    exists = {}
    for label, path in paths.items():
        exists[label] = path.exists()
        sizes[label] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": _dir_size(path),
            "size": _human_bytes(_dir_size(path)),
        }

    generated = paths["ears_wham_16k"]
    generated_files = {}
    for rel in ("train.csv", "valid.csv", "test.csv"):
        generated_files[rel] = _file_info(generated / rel)

    return {
        "exists": exists,
        "sizes": sizes,
        "ears_zip_progress": _ears_zip_progress(paths["ears_raw"]),
        "generated_ears_wham_files": generated_files,
    }


@app.function(timeout=20 * 60, volumes={VOLUME_ROOT: CACHE_VOLUME})
def dataset_progress_remote() -> dict:
    result = _dataset_status(Path(VOLUME_ROOT))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main():
    result = dataset_progress_remote.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
