from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluation.scoring.score_manifest import (
    _load_noisy_metric_cache,
    _manifest_items,
    _noisy_cache_key,
    _write_noisy_metric_cache,
)


class EvalNoisyCacheTest(unittest.TestCase):
    def test_manifest_uses_original_noisy_path_without_saved_copy(self) -> None:
        manifest = {
            "task": "stftpr",
            "crop_mode": "full",
            "files": [
                {
                    "clean_path": "/data/clean.wav",
                    "noisy_path": "/data/noisy.wav",
                    "enhanced_path": "/data/enhanced.wav",
                    "saved_clean_path": "",
                    "saved_noisy_path": "",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            items = _manifest_items(path, limit=0)

        self.assertEqual(items[0]["noisy_path"], "/data/noisy.wav")

    def test_noisy_metrics_are_stored_in_one_reusable_cache_file(self) -> None:
        item = {
            "clean_path": "/data/clean.wav",
            "noisy_path": "/data/noisy.wav",
            "peak_normalize_clean": False,
            "target_num_samples": 0,
        }
        key = _noisy_cache_key(item)
        metrics = {"pesq": 1.2, "estoi": 0.5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noisy_metrics.json"
            _write_noisy_metric_cache(
                path,
                {
                    key: {
                        "clean_path": item["clean_path"],
                        "noisy_path": item["noisy_path"],
                        "metrics": metrics,
                    }
                },
            )
            loaded = _load_noisy_metric_cache(path)

        self.assertEqual(loaded[key]["metrics"], metrics)
        self.assertEqual(loaded[key]["noisy_path"], "/data/noisy.wav")


if __name__ == "__main__":
    unittest.main()
