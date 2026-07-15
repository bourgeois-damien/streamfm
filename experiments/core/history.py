from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def history_file_lock(history_path: Path):
    lock_path = history_path.with_name(f"{history_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, data) -> None:
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
