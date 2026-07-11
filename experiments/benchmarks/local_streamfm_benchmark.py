from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.streamfm_benchmark import main as run_unified_benchmark


def main() -> None:
    """Compatibility wrapper for the unified benchmark launcher."""
    if "--local" not in sys.argv and "--backend" not in sys.argv:
        sys.argv.insert(1, "--local")
    run_unified_benchmark()


if __name__ == "__main__":
    main()
