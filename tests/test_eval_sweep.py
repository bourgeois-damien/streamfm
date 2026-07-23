from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.evaluation.sweeps.run_eval_sweep import build_eval_trial_command, load_eval_sweep_trials


class EvalSweepTest(unittest.TestCase):
    def test_presets_are_applied_after_grid_exclusions(self) -> None:
        payload = """
parameters:
  variant: {values: [baseline, quant]}
  dtype: {values: [fp32, fp16]}
presets:
  baseline: {ckpt: baseline.ckpt}
  quant:
    ckpt: quant.ckpt
    config_overrides: [model.quant.bits=8]
exclude:
  - variant: quant
    dtype: fp16
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.yaml"
            path.write_text(payload, encoding="utf-8")
            trials = load_eval_sweep_trials(path)

        self.assertEqual(len(trials), 3)
        quant = next(trial for trial in trials if trial["variant"] == "quant")
        self.assertEqual(quant["dtype"], "fp32")
        self.assertEqual(quant["ckpt"], "quant.ckpt")
        self.assertEqual(quant["config_overrides"], ["model.quant.bits=8"])

    def test_trial_command_enables_scoring_wandb_and_overrides(self) -> None:
        command = build_eval_trial_command(
            {
                "backend": "modal",
                "variant": "svd_50",
                "steps": 5,
                "score_after_run": True,
                "wandb": True,
                "config_overrides": ["model.svd.rank_ratio=0.5"],
            },
            run_name="quality-001-svd_50",
            default_project="streamfm-evals",
            default_group="quality",
        )

        self.assertIn("--score-after-run", command)
        self.assertIn("--wandb", command)
        self.assertIn("--config-override", command)
        self.assertIn("model.svd.rank_ratio=0.5", command)
        self.assertIn("--wandb-project", command)
        self.assertIn("streamfm-evals", command)

    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported evaluation sweep parameter"):
            build_eval_trial_command({"quantization": "int8"}, run_name="bad")


if __name__ == "__main__":
    unittest.main()
