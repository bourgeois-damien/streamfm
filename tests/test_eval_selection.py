from __future__ import annotations

import unittest

from experiments.evaluation.runner import select_eval_indices


class EvalSelectionTest(unittest.TestCase):
    def test_first_selection_respects_offset_and_limit(self) -> None:
        self.assertEqual(
            select_eval_indices(
                num_available=10,
                limit=3,
                offset=2,
                selection="first",
                selection_seed=42,
            ),
            [2, 3, 4],
        )

    def test_random_selection_is_reproducible_and_sorted(self) -> None:
        first = select_eval_indices(
            num_available=100,
            limit=10,
            offset=0,
            selection="random",
            selection_seed=42,
        )
        second = select_eval_indices(
            num_available=100,
            limit=10,
            offset=0,
            selection="random",
            selection_seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)

    def test_random_selection_respects_offset(self) -> None:
        indices = select_eval_indices(
            num_available=100,
            limit=10,
            offset=50,
            selection="random",
            selection_seed=42,
        )
        self.assertTrue(all(idx >= 50 for idx in indices))


if __name__ == "__main__":
    unittest.main()
