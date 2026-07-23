from __future__ import annotations
import unittest
from experiments.evaluation.modal_defaults import modal_default_data_path, resolve_modal_data_path

class ModalEvalDefaultsTest(unittest.TestCase):
    
    def test_bwe_uses_paired_dir_split(self = None):
        self.assertEqual(modal_default_data_path('bwe', 'test'), '/data/datasets/EARS_v2_16k_BWR/test')

    
    def test_csv_tasks_use_split_csv(self = None):
        self.assertEqual(modal_default_data_path('stftpr', 'valid'), '/data/datasets/EARS-WHAM_v2_16k/valid.csv')
        self.assertEqual(modal_default_data_path('derev', 'test'), '/data/datasets/EARS-Reverb_v2_16k/test.csv')

    
    def test_explicit_path_wins(self = None):
        self.assertEqual(
            resolve_modal_data_path('/custom/test', task='bwe', split='test'),
            '/custom/test',
        )


if __name__ == '__main__':
    unittest.main()