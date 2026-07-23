from __future__ import annotations

import unittest

from experiments.core.options import normalize_cli_options, parse_steps
from experiments.core.devices import normalize_tf32_mode
from experiments.benchmarks.sweeps.run_benchmark_sweep import build_sweep_command, sweep_config_get
from experiments.benchmarks.results import build_benchmark_records
from experiments.tools.upload_history_to_wandb import group_history_rows
from experiments.evaluation.results import build_eval_wandb_record


class BenchmarkOptionsTest(unittest.TestCase):
    def test_normalize_tf32_mode(self) -> None:
        self.assertEqual(normalize_tf32_mode("AUTO"), "auto")
        self.assertEqual(normalize_tf32_mode("on"), "on")
        self.assertEqual(normalize_tf32_mode("off"), "off")
        with self.assertRaises(ValueError):
            normalize_tf32_mode("maybe")

    def test_parse_steps(self) -> None:
        self.assertEqual(parse_steps("1, 2,5"), (1, 2, 5))

    def test_sweep_command_carries_explicit_tf32_mode(self) -> None:
        command = build_sweep_command({"tf32": "off"})
        self.assertEqual(command["tf32"], "off")

    def test_sweep_command_carries_cudnn_benchmark(self) -> None:
        command = build_sweep_command(
            {"cudnn_benchmark": True, "cudnn_benchmark_limit": 20}
        )
        self.assertTrue(command["cudnn_benchmark"])
        self.assertEqual(command["cudnn_benchmark_limit"], 20)

    def test_sweep_command_carries_compressed_checkpoint(self) -> None:
        command = build_sweep_command({"ckpt": "compressed/streamfm_stftpr_k7.ckpt"})
        self.assertEqual(command["checkpoint_name"], "compressed/streamfm_stftpr_k7.ckpt")

    def test_se_audio_cuda_graph_maps_to_full_audio_graph(self) -> None:
        resolved = normalize_cli_options(
            task="se",
            part="model",
            pipeline="audio",
            execution="cuda_graph",
        )

        self.assertEqual(resolved["internal_task"], "se_full")
        self.assertEqual(resolved["internal_pipeline"], "audio_graph_model")
        self.assertTrue(resolved["use_compiled"])

    def test_se_flow_audio_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "SE audio pipeline supports only"):
            normalize_cli_options(
                task="se",
                part="flow",
                pipeline="audio",
                execution="eager",
            )

    def test_build_benchmark_records_creates_one_record_per_result(self) -> None:
        records = build_benchmark_records(
            results=[
                {
                    "task": "stftpr",
                    "pipeline": "audio",
                    "execution": "cuda_graph",
                    "steps": 1,
                    "total_mean_ms": 12.3,
                    "audio": object(),
                },
                {
                    "task": "stftpr",
                    "pipeline": "audio",
                    "execution": "cuda_graph",
                    "steps": 2,
                    "total_mean_ms": 20.1,
                },
            ],
            command={"task": "stftpr", "execution": "cuda_graph"},
            run_id="abc123",
            run_started_at="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["group"], "abc123")
        self.assertEqual(records[0]["config"]["steps"], 1)
        self.assertEqual(records[1]["config"]["steps"], 2)
        self.assertEqual(records[0]["config"]["command_task"], "stftpr")
        self.assertEqual(records[0]["metrics"]["total_mean_ms"], 12.3)
        self.assertNotIn("audio", records[0]["config"])
        self.assertNotIn("audio", records[0]["metrics"])

    def test_group_history_rows_preserves_run_groups(self) -> None:
        groups = group_history_rows(
            [
                {"run_id": "run-a", "run_started_at": "t0", "command": {"task": "stftpr"}, "steps": 1},
                {"run_id": "run-a", "run_started_at": "t0", "command": {"task": "stftpr"}, "steps": 2},
                {"run_id": "run-b", "run_started_at": "t1", "command": {"task": "se"}, "steps": 1},
            ]
        )

        self.assertEqual([group["run_id"] for group in groups], ["run-a", "run-b"])
        self.assertEqual(len(groups[0]["rows"]), 2)
        self.assertEqual(groups[0]["command"]["task"], "stftpr")
        self.assertEqual(groups[1]["command"]["task"], "se")

    def test_build_eval_wandb_record_adds_score_metrics(self) -> None:
        record = build_eval_wandb_record(
            result={
                "run_id": "eval-1",
                "task": "se",
                "split": "test",
                "solver": "euler",
                "steps": 5,
                "model_dtype": "fp32",
                "backend": "local",
                "device": "mps",
                "num_files": 10,
                "num_errors": 0,
                "elapsed_s": 12.5,
                "mean_file_s": 1.25,
                "manifest_path": "outputs/eval_runs/eval-1/manifest.json",
            },
            command={"task": "se", "score_target": "enhanced"},
            score_result={
                "target": "enhanced",
                "num_files": 10,
                "enhanced": {"pesq": 2.5, "si_sdr": 11.0},
                "delta_vs_noisy": {"pesq": 0.4},
            },
        )

        self.assertEqual(record["group"], "eval-1")
        self.assertEqual(record["config"]["task"], "se")
        self.assertEqual(record["metrics"]["num_files"], 10)
        self.assertEqual(record["metrics"]["elapsed_s"], 12.5)
        self.assertEqual(record["metrics"]["score_enhanced_pesq"], 2.5)
        self.assertEqual(record["metrics"]["score_delta_vs_noisy_pesq"], 0.4)

    def test_parse_ptq_components_via_sweep_defaults(self) -> None:
        from sgmse.util.ptq_int8 import parse_ptq_components

        command = build_sweep_command(
            {
                "backend": "local",
                "hardware": "cpu",
                "ptq_int8": "linear,causal_conv",
            }
        )
        self.assertEqual(command["ptq_int8"], "linear,causal_conv")
        self.assertEqual(parse_ptq_components(command["ptq_int8"]), ("linear", "causal_conv"))

    def test_sweep_config_get_uses_defaults(self) -> None:
        self.assertEqual(sweep_config_get({"task": "bwe"}, "task", "stftpr"), "bwe")
        self.assertEqual(sweep_config_get({"task": None}, "task", "stftpr"), "stftpr")
        self.assertEqual(sweep_config_get({}, "iterations", 100), 100)

    def test_build_sweep_command_maps_wandb_parameters(self) -> None:
        command = build_sweep_command(
            {
                "backend": "modal",
                "task": "stftpr",
                "execution": "cuda_graph",
                "steps": 4,
                "dtype": "fp16",
                "iterations": 50,
                "hardware": "L4",
            },
            hardware_override="cuda",
        )

        self.assertEqual(command["backend"], "modal")
        self.assertEqual(command["task"], "stftpr")
        self.assertEqual(command["execution"], "cuda_graph")
        self.assertEqual(command["steps"], "4")
        self.assertEqual(command["model_dtype"], "fp16")
        self.assertEqual(command["iterations"], 50)
        self.assertEqual(command["hardware"], "cuda")
        self.assertTrue(command["sweep"])

    def test_normalize_modal_hardware_defaults_to_l4(self) -> None:
        from experiments.benchmarks.sweeps.run_benchmark_sweep import normalize_modal_hardware

        self.assertEqual(normalize_modal_hardware("auto"), "L4")
        self.assertEqual(normalize_modal_hardware("L4"), "L4")

    def test_resolve_sweep_iterations_from_audio_duration(self) -> None:
        from experiments.benchmarks.sweeps.run_benchmark_sweep import resolve_sweep_iterations

        command = {
            "pipeline": "audio",
            "task": "stftpr",
            "iterations": 100,
            "warmup": 0,
            "audio_duration_s": 1.6,
        }
        self.assertEqual(resolve_sweep_iterations(command, input_audio_path=""), 100)

    def test_expand_parameter_grid_builds_cartesian_product(self) -> None:
        from experiments.benchmarks.sweeps.sweep_grid import expand_parameter_grid

        trials = expand_parameter_grid(
            {
                "backend": {"value": "modal"},
                "execution": {"values": ["eager", "compiled"]},
                "dtype": {"values": ["fp32", "fp16"]},
                "steps": {"value": 1},
            }
        )

        self.assertEqual(len(trials), 4)
        self.assertEqual(trials[0]["backend"], "modal")
        self.assertEqual({trial["execution"] for trial in trials}, {"eager", "compiled"})
        self.assertEqual({trial["dtype"] for trial in trials}, {"fp32", "fp16"})

    def test_filter_excluded_trials_drops_cuda_graph_preallocate(self) -> None:
        from experiments.benchmarks.sweeps.sweep_grid import expand_parameter_grid, filter_excluded_trials

        trials = expand_parameter_grid(
            {
                "execution": {"values": ["eager", "cuda_graph"]},
                "preallocate_model_buffers": {"values": [False, True]},
            }
        )
        filtered = filter_excluded_trials(
            trials,
            [{"execution": "cuda_graph", "preallocate_model_buffers": True}],
        )

        self.assertEqual(len(trials), 4)
        self.assertEqual(len(filtered), 3)
        self.assertFalse(
            any(
                trial["execution"] == "cuda_graph" and trial["preallocate_model_buffers"] is True
                for trial in filtered
            )
        )

    def test_shard_trials_splits_evenly(self) -> None:
        from experiments.benchmarks.sweeps.run_benchmark_sweep_batch import shard_trials

        trials = [{"i": i} for i in range(10)]
        shards = shard_trials(trials, workers=4)

        self.assertEqual(len(shards), 4)
        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])
        self.assertEqual([item["i"] for shard in shards for item in shard], list(range(10)))
        self.assertEqual(shard_trials(trials, workers=1), [trials])
        self.assertEqual(len(shard_trials(trials, workers=100)), 10)

    def test_needs_gpu_workers_confirmation(self) -> None:
        from experiments.benchmarks.sweeps.run_benchmark_sweep_batch import needs_gpu_workers_confirmation

        self.assertFalse(needs_gpu_workers_confirmation(["L4"], 4))
        self.assertTrue(needs_gpu_workers_confirmation(["L4"], 5))
        self.assertFalse(needs_gpu_workers_confirmation(["CPU"], 8))
        self.assertTrue(needs_gpu_workers_confirmation(["CPU", "L4"], 8))


if __name__ == "__main__":
    unittest.main()
