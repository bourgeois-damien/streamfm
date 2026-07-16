"""Locate the repository root and make it importable from standalone scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for the Stream.FM source layout.

    Walks upward until a directory contains both ``config/`` and ``sgmse/``
    (only the repo root has both), so scripts work no matter which directory
    they are launched from. Falls back to the starting directory's parent.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    return current.parent


def ensure_repo_importable(repo_root: Path) -> None:
    """Put the repository root on sys.path for direct script execution."""
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
