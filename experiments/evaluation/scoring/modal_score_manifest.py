"""Modal entrypoint that scores an eval manifest remotely.

Resolves the manifest on the volume and runs the scoring on Modal. Invoked via
``modal run``.
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import sys

import modal

REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"

if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "experiments").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current_file.parent


LOCAL_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .apt_install("build-essential", "libsndfile1")
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "audiomentations==0.41.0",
        "distillmos==0.9.1",
        "einops==0.8.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pesq==0.0.4",
        "pystoi==0.3.3",
        "pytorch-lightning==2.5.1.post0",
        "scipy==1.15.2",
    )
    .add_local_dir(str(LOCAL_ROOT / "experiments"), remote_path=f"{REMOTE_ROOT}/experiments")
    .add_local_dir(str(LOCAL_ROOT / "sgmse"), remote_path=f"{REMOTE_ROOT}/sgmse")
)

app = modal.App("streamfm-score-manifest", image=image)


def _resolve_manifest(run_name: str, manifest: str) -> Path:
    if manifest:
        path = Path(manifest)
        resolved = path if path.is_absolute() else Path(VOLUME_ROOT) / path
        if not resolved.exists():
            raise FileNotFoundError(f"Manifest not found: {resolved}")
        return resolved
    if not run_name:
        raise ValueError("Pass either --run-name or --manifest.")
    manifest_path = Path(VOLUME_ROOT) / "outputs" / "eval_runs" / run_name / "manifest.json"
    if manifest_path.exists():
        return manifest_path

    eval_runs_dir = Path(VOLUME_ROOT) / "outputs" / "eval_runs"
    existing_runs = sorted(path.name for path in eval_runs_dir.iterdir() if path.is_dir()) if eval_runs_dir.exists() else []
    matches = get_close_matches(run_name, existing_runs, n=5, cutoff=0.45)
    hint = f" Close matches: {', '.join(matches)}." if matches else ""
    raise FileNotFoundError(f"Manifest not found for run-name '{run_name}': {manifest_path}.{hint}")


@app.function(timeout=2 * 60 * 60, volumes={VOLUME_ROOT: CACHE_VOLUME})
def score_remote(
    *,
    source: str,
    run_name: str,
    manifest: str,
    task: str,
    split: str,
    data_path: str,
    data_format: str,
    crop_mode: str,
    limit: int,
    offset: int,
    selection: str,
    selection_seed: int,
    with_distillmos: bool,
    output_json: str,
    include_per_file: bool,
    include_stats: bool,
    score_target: str,
    phase_mode: str = "random",
    phase_seed: int = 1234,
) -> dict:
    from experiments.evaluation.scoring.score_manifest import (
        score_dataset_noisy,
        score_dataset_phaseless,
        score_manifest,
    )

    if output_json:
        output_path = Path(output_json)
        if not output_path.is_absolute():
            output_path = Path(VOLUME_ROOT) / output_path
    elif source == "dataset":
        suffix = f"limit{limit}" if limit > 0 else "all"
        output_path = Path(VOLUME_ROOT) / "outputs" / "dataset_scores" / f"{task}_{split}_noisy_{suffix}.json"
    elif source == "dataset_phaseless":
        suffix = f"limit{limit}" if limit > 0 else "all"
        output_path = Path(VOLUME_ROOT) / "outputs" / "dataset_scores" / f"{task}_{split}_phaseless_{phase_mode}_{suffix}.json"
    else:
        manifest_path = _resolve_manifest(run_name, manifest)
        suffix = f"limit{limit}" if limit > 0 else "all"
        output_path = manifest_path.parent / f"metrics_{suffix}.json"

    if source == "dataset":
        result = score_dataset_noisy(
            task=task,
            data_path=data_path,
            data_format=data_format,
            split=split,
            limit=limit,
            offset=offset,
            selection=selection,
            selection_seed=selection_seed,
            crop_mode=crop_mode,
            output_json=output_path,
            include_per_file=include_per_file,
            include_stats=include_stats,
        )
        CACHE_VOLUME.commit()
        return result

    if source == "dataset_phaseless":
        result = score_dataset_phaseless(
            task=task,
            data_path=data_path,
            data_format=data_format,
            split=split,
            limit=limit,
            offset=offset,
            selection=selection,
            selection_seed=selection_seed,
            crop_mode=crop_mode,
            output_json=output_path,
            phase_mode=phase_mode,
            phase_seed=phase_seed,
            include_per_file=include_per_file,
            include_stats=include_stats,
        )
        CACHE_VOLUME.commit()
        return result

    manifest_path = _resolve_manifest(run_name, manifest)
    result = score_manifest(
        manifest_path,
        limit=limit,
        with_distillmos=with_distillmos,
        output_json=output_path,
        include_per_file=include_per_file,
        include_stats=include_stats,
        score_target=score_target,
    )
    CACHE_VOLUME.commit()
    return result


@app.local_entrypoint()
def main(
    source: str = "manifest",
    run_name: str = "",
    manifest: str = "",
    task: str = "se",
    split: str = "test",
    data_path: str = "",
    data_format: str = "",
    crop_mode: str = "full",
    limit: int = 0,
    offset: int = 0,
    selection: str = "random",
    selection_seed: int = 42,
    with_distillmos: bool = False,
    output_json: str = "",
    include_per_file: bool = False,
    include_stats: bool = False,
    score_target: str = "enhanced",
    phase_mode: str = "random",
    phase_seed: int = 1234,
    local_json: str = "",
):
    import json as _json

    result = score_remote.remote(
        source=source,
        run_name=run_name,
        manifest=manifest,
        task=task,
        split=split,
        data_path=data_path,
        data_format=data_format,
        crop_mode=crop_mode,
        limit=limit,
        offset=offset,
        selection=selection,
        selection_seed=selection_seed,
        with_distillmos=with_distillmos,
        output_json=output_json,
        include_per_file=include_per_file,
        include_stats=include_stats,
        score_target=score_target,
        phase_mode=phase_mode,
        phase_seed=phase_seed,
    )
    if local_json:
        local_path = Path(local_json)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(_json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved result locally to {local_path}")
