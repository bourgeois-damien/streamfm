"""Atomic, lock-protected JSON history writes shared by benchmarks and evaluation.

Benchmark and evaluation runs append to append-only JSON history files that
may be written by concurrent processes (e.g. parallel Modal jobs). These two
primitives keep those writes safe: an exclusive file lock for serialization and
a temp-file-then-rename for atomic replacement.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def history_file_lock(history_path: Path):
    """Serialize history updates across concurrent benchmark processes.

    The lock lives in a separate ``.lock`` file because write_json_atomic
    replaces the history file's inode on every write — locking the history
    file itself would leave later writers locking a dead inode.
    """
    lock_path = history_path.with_name(f"{history_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, data) -> None:
    """Write JSON through a temporary file, then atomically replace the target.

    Readers never observe a half-written file: they see either the old
    version or the new one. The temp file sits in the same directory so the
    final rename stays on one filesystem (a cross-device rename would copy).
    """
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
