from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current.parent


def ensure_repo_importable(repo_root: Path) -> None:
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
