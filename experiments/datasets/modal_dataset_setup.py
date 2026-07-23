"""Provision the training/eval datasets on the Modal volume.

Downloads and extracts the EARS-WHAM sources (and the official recipe assets),
installing the reverb dependencies needed to build the derev variant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from urllib.parse import urlparse
from urllib.request import urlretrieve

import modal

VOLUME_ROOT = "/data"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")

app = modal.App("streamfm-dataset-setup")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bzip2", "curl", "git", "libsndfile1", "tar", "unzip", "wget")
    .pip_install(
        "gdown==5.2.1",
        "librosa==0.10.2.post1",
        "numpy==1.26.4",
        "pyloudnorm==0.1.1",
        "scipy==1.15.2",
        "soundfile==0.12.1",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "tqdm==4.67.1",
    )
)

OFFICIAL_EARS_REPO = "https://github.com/sp-uhh/ears_benchmark.git"


DATASET_DEFAULTS = {
    "ears_wham_v2_16k": {
        "target_dir": f"{VOLUME_ROOT}/datasets/EARS-WHAM_v2_16k",
        "description": "EARS-WHAM v2 16 kHz speech-enhancement dataset.",
    },
    "ears_reverb_v2_16k": {
        "target_dir": f"{VOLUME_ROOT}/datasets/EARS-Reverb_v2_16k",
        "description": "EARS-Reverb v2 16 kHz dereverberation dataset.",
    },
    "ears_bwr_v2_16k": {
        "target_dir": f"{VOLUME_ROOT}/datasets/EARS_v2_16k_BWR",
        "description": "EARS v2 16 kHz bandwidth-restoration dataset.",
    },
    "ears_lyra_v2_16k": {
        "target_dir": f"{VOLUME_ROOT}/datasets/EARS_v2_16k_Lyra",
        "description": "EARS v2 16 kHz Lyra restoration dataset.",
    },
}


def _normalize_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "ears_wham": "ears_wham_v2_16k",
        "ears_wham_v2": "ears_wham_v2_16k",
        "ears_reverb": "ears_reverb_v2_16k",
        "ears_reverb_v2": "ears_reverb_v2_16k",
        "ears_bwr": "ears_bwr_v2_16k",
        "ears_lyra": "ears_lyra_v2_16k",
    }
    return aliases.get(normalized, normalized)


def _split_sources(sources: str) -> list[str]:
    return [source.strip() for source in sources.split(",") if source.strip()]


def _is_gdrive_folder(url: str) -> bool:
    parsed = urlparse(url)
    return "drive.google.com" in parsed.netloc and "/folders/" in parsed.path


def _is_gdrive_file(url: str) -> bool:
    parsed = urlparse(url)
    return "drive.google.com" in parsed.netloc and "/file/" in parsed.path


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "downloaded_file"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _ensure_official_repo(repo_url: str, repo_dir: Path) -> None:
    """Clone or update the official EARS benchmark generation repo."""
    if (repo_dir / ".git").is_dir():
        _run(["git", "fetch", "--depth", "1", "origin", "main"], cwd=repo_dir)
        _run(["git", "checkout", "main"], cwd=repo_dir)
        _run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])


def _install_reverb_dependencies() -> None:
    """Install dependencies only needed by generate_ears_reverb.py."""
    packages = [
        "mat73==0.65",
        # The upstream generation script imports `sofa`, which comes from
        # python-sofa. The bare `sofa` distribution on PyPI is an unrelated
        # package whose pyDNS dependency is Python 2 only and fails to build.
        # Keep this runtime-scoped so EARS-WHAM setup is not blocked by SOFA packaging changes.
        "python-sofa==0.2.0",
    ]
    _run([sys.executable, "-m", "pip", "install", *packages])


def _official_recipe(dataset: str) -> dict:
    if dataset == "ears_wham_v2_16k":
        return {
            "download_script": "download_ears_wham.sh",
            "generate_script": "generate_ears_wham.py",
            "generated_dir_name": "EARS-WHAM_v2_16k",
            "requires_reverb_deps": False,
        }
    if dataset == "ears_reverb_v2_16k":
        return {
            "download_script": "download_ears_reverb.sh",
            "generate_script": "generate_ears_reverb.py",
            "generated_dir_name": "EARS-Reverb_v2_16k",
            "requires_reverb_deps": True,
        }
    raise ValueError(
        "The official ears_benchmark recipe is only available for "
        "ears_wham_v2_16k and ears_reverb_v2_16k. Use --recipe sources for custom archives."
    )


def _official_dataset_complete(dataset_dir: Path) -> bool:
    required = ("train.csv", "valid.csv", "test.csv")
    return dataset_dir.is_dir() and all((dataset_dir / name).is_file() for name in required)


def _expected_ears_zip_names() -> list[str]:
    return [f"p{idx:03d}.zip" for idx in range(1, 108)]


def _ears_raw_complete(ears_dir: Path) -> bool:
    """Best-effort check that the official EARS raw download has been processed."""
    if not ears_dir.is_dir():
        return False
    processed = 0
    for name in _expected_ears_zip_names():
        stem = Path(name).stem
        if (ears_dir / name).exists():
            processed += 1
            continue
        if list(ears_dir.rglob(f"*{stem}*")):
            processed += 1
    return processed == len(_expected_ears_zip_names())


def _wham_raw_complete(wham_dir: Path) -> bool:
    """Check the file required by generate_ears_wham.py."""
    return (wham_dir / "high_res_wham" / "high_res_metadata.csv").is_file()


def _zip_file_complete(zip_path: Path) -> bool:
    """Return whether a zip file appears structurally complete."""
    if not zip_path.is_file():
        return False
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _remove_incomplete_official_inputs(dataset: str, data_dir: Path, overwrite: bool) -> list[str]:
    removed = []
    if dataset == "ears_wham_v2_16k":
        ears_dir = data_dir / "EARS"
        wham_dir = data_dir / "WHAM48kHz"
        wham_zip = data_dir / "WHAM48kHz.zip"
        if ears_dir.exists() and not overwrite and not _ears_raw_complete(ears_dir):
            print(f"Removing incomplete EARS raw directory before retry: {ears_dir}")
            shutil.rmtree(ears_dir)
            removed.append(str(ears_dir))
        if wham_dir.exists() and not overwrite and not _wham_raw_complete(wham_dir):
            print(f"Removing incomplete WHAM raw directory before retry: {wham_dir}")
            shutil.rmtree(wham_dir)
            removed.append(str(wham_dir))
        if wham_zip.exists() and not overwrite and not _wham_raw_complete(wham_dir) and not _zip_file_complete(wham_zip):
            print(f"Removing incomplete WHAM zip before retry: {wham_zip}")
            wham_zip.unlink()
            removed.append(str(wham_zip))
    return removed


def _run_official_ears_recipe(
    *,
    dataset: str,
    data_dir: Path,
    target_dir: str,
    repo_url: str,
    repo_dir: Path,
    overwrite: bool,
    generate_16k: bool,
) -> dict:
    recipe = _official_recipe(dataset)
    generated = data_dir / recipe["generated_dir_name"]
    if target_dir and Path(target_dir) != generated:
        raise ValueError(
            "The official generation scripts write to a fixed directory under --data-dir. "
            f"For {dataset}, use --target-dir '' or {generated}. "
            "Use --data-dir to control the parent directory."
        )

    _ensure_official_repo(repo_url, repo_dir)

    if generated.exists() and overwrite:
        print(f"Removing existing generated dataset because --overwrite=True: {generated}")
        shutil.rmtree(generated)

    if generated.exists() and not overwrite and _official_dataset_complete(generated):
        print(f"Skipping official generation because dataset is complete: {generated}")
        return {
            "official_repo": repo_url,
            "official_repo_dir": str(repo_dir),
            "data_dir": str(data_dir),
            "target_dir": str(generated),
            "download_script": recipe["download_script"],
            "generate_script": recipe["generate_script"],
            "generated": False,
            "skipped_existing": True,
        }

    if generated.exists() and not overwrite:
        print(
            "Generated dataset directory exists but is incomplete; removing it before regeneration: "
            f"{generated}"
        )
        shutil.rmtree(generated)

    if recipe["requires_reverb_deps"]:
        _install_reverb_dependencies()

    data_dir.mkdir(parents=True, exist_ok=True)
    removed_incomplete_inputs = _remove_incomplete_official_inputs(dataset, data_dir, overwrite)
    _run(["bash", recipe["download_script"], str(data_dir)], cwd=repo_dir)

    generate_cmd = [sys.executable, recipe["generate_script"], "--data_dir", str(data_dir)]
    if generate_16k:
        generate_cmd.append("--16k")
    _run(generate_cmd, cwd=repo_dir)

    return {
        "official_repo": repo_url,
        "official_repo_dir": str(repo_dir),
        "data_dir": str(data_dir),
        "target_dir": str(generated),
        "download_script": recipe["download_script"],
        "generate_script": recipe["generate_script"],
        "generated": True,
        "skipped_existing": False,
        "removed_incomplete_inputs": removed_incomplete_inputs,
    }


def _download_source(source: str, download_dir: Path, source_kind: str) -> list[Path]:
    """Download one source into download_dir and return newly present files."""
    before = {path for path in download_dir.rglob("*") if path.is_file()}
    kind = source_kind
    if kind == "auto":
        if _is_gdrive_folder(source):
            kind = "gdrive_folder"
        elif _is_gdrive_file(source):
            kind = "gdrive_file"
        else:
            kind = "url"

    if kind == "gdrive_folder":
        _run(["gdown", source, "-O", str(download_dir), "--folder", "--remaining-ok"])
    elif kind == "gdrive_file":
        _run(["gdown", source, "-O", str(download_dir)])
    elif kind == "url":
        output_path = download_dir / _filename_from_url(source)
        if output_path.exists():
            print(f"Skipping existing download: {output_path}")
        else:
            print(f"Downloading {source} -> {output_path}")
            urlretrieve(source, output_path)
    else:
        raise ValueError("Unsupported source kind. Use auto, url, gdrive_file, or gdrive_folder.")

    after = {path for path in download_dir.rglob("*") if path.is_file()}
    return sorted(after - before)


def _extract_archives(download_dir: Path, target_dir: Path, overwrite: bool) -> list[dict]:
    """Extract supported archives and return extraction records."""
    archive_suffixes = {
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
    }
    records = []
    for archive_path in sorted(path for path in download_dir.rglob("*") if path.is_file()):
        suffixes = "".join(archive_path.suffixes).lower()
        if archive_path.suffix.lower() not in archive_suffixes and not any(
            suffixes.endswith(ext) for ext in (".tar.gz", ".tar.bz2", ".tar.xz")
        ):
            continue
        extract_dir = target_dir
        marker = target_dir / ".extract_markers" / f"{archive_path.name}.done"
        if marker.exists() and not overwrite:
            print(f"Skipping already extracted archive: {archive_path}")
            records.append({"archive": str(archive_path), "skipped": True, "target_dir": str(extract_dir)})
            continue
        marker.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive_path} -> {extract_dir}")
        shutil.unpack_archive(str(archive_path), str(extract_dir))
        marker.write_text(str(time.time()), encoding="utf-8")
        records.append({"archive": str(archive_path), "skipped": False, "target_dir": str(extract_dir)})
    return records


def _count_files(root: Path) -> dict:
    counts = {
        "wav": 0,
        "flac": 0,
        "csv": 0,
        "archives": 0,
        "total_files": 0,
        "total_bytes": 0,
    }
    archive_exts = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        counts["total_files"] += 1
        try:
            counts["total_bytes"] += path.stat().st_size
        except OSError:
            pass
        lower = path.name.lower()
        if lower.endswith(".wav"):
            counts["wav"] += 1
        elif lower.endswith(".flac"):
            counts["flac"] += 1
        elif lower.endswith(".csv"):
            counts["csv"] += 1
        elif lower.endswith(archive_exts):
            counts["archives"] += 1
    return counts


def _find_likely_paths(root: Path) -> dict:
    csvs = {path.name: str(path) for path in sorted(root.rglob("*.csv"))}
    likely = {}
    for split in ("train", "valid", "test"):
        for name, path in csvs.items():
            if name.lower() == f"{split}.csv":
                likely[f"{split}_csv"] = path
                break
    for split in ("train", "valid", "test"):
        candidate = root / split
        if candidate.is_dir():
            likely[f"{split}_dir"] = str(candidate)
    return {"csvs": csvs, "likely": likely}


@app.function(image=image, timeout=24 * 60 * 60, volumes={VOLUME_ROOT: CACHE_VOLUME})
def setup_dataset_remote(
    *,
    dataset: str,
    sources: str,
    source_kind: str,
    recipe: str,
    target_dir: str,
    download_dir: str,
    data_dir: str,
    repo_url: str,
    repo_dir: str,
    extract: bool,
    overwrite: bool,
    generate_16k: bool,
) -> dict:
    """Download and optionally extract a dataset into the Modal Volume."""
    dataset = _normalize_dataset_name(dataset)
    defaults = DATASET_DEFAULTS.get(dataset, {})
    if not download_dir:
        download_dir = f"{VOLUME_ROOT}/downloads/{dataset}"
    if not data_dir:
        data_dir = f"{VOLUME_ROOT}/datasets"
    if not repo_dir:
        repo_dir = f"{VOLUME_ROOT}/repos/ears_benchmark"

    source_list = _split_sources(sources)
    selected_recipe = recipe.lower().strip()
    if selected_recipe == "auto":
        selected_recipe = "sources" if source_list else "official"

    if not target_dir:
        if selected_recipe == "official" and dataset in {"ears_wham_v2_16k", "ears_reverb_v2_16k"}:
            target_dir = str(Path(data_dir) / _official_recipe(dataset)["generated_dir_name"])
        else:
            target_dir = defaults.get("target_dir") or f"{VOLUME_ROOT}/datasets/{dataset}"

    target = Path(target_dir)
    downloads = Path(download_dir)

    started_at = time.perf_counter()
    official = None
    downloaded = []
    extractions = []

    if selected_recipe == "official":
        official = _run_official_ears_recipe(
            dataset=dataset,
            data_dir=Path(data_dir),
            target_dir=target_dir,
            repo_url=repo_url,
            repo_dir=Path(repo_dir),
            overwrite=overwrite,
            generate_16k=generate_16k,
        )
        target = Path(official["target_dir"])
    elif selected_recipe == "sources":
        target.mkdir(parents=True, exist_ok=True)
        downloads.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("Unsupported recipe. Use official, sources, or auto.")

    if selected_recipe == "sources":
        if not source_list:
            raise ValueError(
                "No source URL was provided. Pass --sources with a direct URL, Google Drive file URL, "
                "or Google Drive folder URL for the dataset."
            )

        for source in source_list:
            downloaded.extend(str(path) for path in _download_source(source, downloads, source_kind))

        extractions = _extract_archives(downloads, target, overwrite=overwrite) if extract else []

    counts = _count_files(target)
    discovered = _find_likely_paths(target)

    manifest = {
        "dataset": dataset,
        "description": defaults.get("description", ""),
        "recipe": selected_recipe,
        "official": official,
        "sources": source_list,
        "source_kind": source_kind,
        "target_dir": str(target),
        "download_dir": str(downloads),
        "data_dir": data_dir,
        "repo_url": repo_url,
        "repo_dir": repo_dir,
        "downloaded_files": downloaded,
        "extractions": extractions,
        "counts": counts,
        "paths": discovered,
        "elapsed_s": time.perf_counter() - started_at,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = target / ".streamfm_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        CACHE_VOLUME.commit()
    except Exception as exc:
        print(f"[Warning] Could not commit Modal volume explicitly: {exc}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@app.local_entrypoint()
def main(
    dataset: str = "ears_wham_v2_16k",
    sources: str = "",
    source_kind: str = "auto",
    recipe: str = "auto",
    target_dir: str = "",
    download_dir: str = "",
    data_dir: str = "",
    repo_url: str = OFFICIAL_EARS_REPO,
    repo_dir: str = "",
    extract: bool = True,
    overwrite: bool = False,
    generate_16k: bool = True,
):
    """Download/extract a dataset into the persistent Modal Volume."""
    result = setup_dataset_remote.remote(
        dataset=dataset,
        sources=sources,
        source_kind=source_kind,
        recipe=recipe,
        target_dir=target_dir,
        download_dir=download_dir,
        data_dir=data_dir,
        repo_url=repo_url,
        repo_dir=repo_dir,
        extract=extract,
        overwrite=overwrite,
        generate_16k=generate_16k,
    )
    print("Dataset setup complete.")
    print("Target:", result["target_dir"])
    print("Manifest:", str(Path(result["target_dir"]) / ".streamfm_dataset_manifest.json"))
