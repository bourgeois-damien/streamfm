from __future__ import annotations

import os
import tempfile
import unittest

from experiments.core.modal_cache import CACHE_LAYOUT_VERSION, configure_shared_modal_cache


class ModalCacheTest(unittest.TestCase):
    def test_combinations_share_cache_on_same_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = configure_shared_modal_cache(volume_root=directory, hardware="L4")
            second = configure_shared_modal_cache(volume_root=directory, hardware="l4")

        self.assertEqual(first["cache_root"], second["cache_root"])
        self.assertIn(CACHE_LAYOUT_VERSION, first["cache_root"])
        self.assertTrue(first["cache_root"].endswith("/L4"))

    def test_hardware_tiers_remain_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            l4 = configure_shared_modal_cache(volume_root=directory, hardware="L4")
            l40s = configure_shared_modal_cache(volume_root=directory, hardware="L40S")

        self.assertNotEqual(l4["triton_cache_dir"], l40s["triton_cache_dir"])
        self.assertEqual(l4["torch_home"], l40s["torch_home"])
        self.assertEqual(os.environ["TORCH_HOME"], l40s["torch_home"])


if __name__ == "__main__":
    unittest.main()
