from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.common import find_repo_root


@dataclass(frozen=True)
class BenchmarkPaths:
    repo_root: Path
    config_dir: Path
    checkpoint_roots: tuple[Path, ...]


def make_benchmark_paths(
    repo_root: Path | str | None = None,
    config_dir: Path | str | None = None,
    checkpoint_roots: tuple[Path | str, ...] | None = None,
) -> BenchmarkPaths:
    """Create normalized paths for local or remote execution."""
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    cfg_dir = Path(config_dir).resolve() if config_dir is not None else root / "config"
    if checkpoint_roots:
        ckpt_roots = tuple(Path(path).resolve() for path in checkpoint_roots)
    else:
        ckpt_roots = (root / "checkpoints",)
    return BenchmarkPaths(repo_root=root, config_dir=cfg_dir, checkpoint_roots=ckpt_roots)


def checkpoint_path(filename: str, paths: BenchmarkPaths) -> str:
    """Resolve a checkpoint by searching configured checkpoint roots."""
    requested = Path(filename).expanduser()
    if requested.is_absolute() or requested.parent != Path("."):
        if requested.exists():
            return str(requested)
        raise FileNotFoundError(f"Checkpoint not found: {requested}")
    candidates = [root / filename for root in paths.checkpoint_roots]
    for path in candidates:
        if path.exists():
            return str(path)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint '{filename}' not found. Searched: {searched}.")
