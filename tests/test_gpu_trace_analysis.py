import gzip
import json

import pytest

from experiments.benchmarks.profiling.analyze_trace import analyze_trace


def test_trace_analysis_separates_layout_and_compute(tmp_path):
    trace_path = tmp_path / "trace.json.gz"
    metadata_path = tmp_path / "metadata.json"
    payload = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 0,
                "dur": 2,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "convertTensor_kernel<float, float>",
                "ts": 10,
                "dur": 20,
                "args": {"grid": [1, 1, 1], "block": [32, 1, 1]},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "implicit_gemm_fprop_conv",
                "ts": 31,
                "dur": 80,
                "args": {"grid": [4, 1, 1], "block": [128, 1, 1]},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    metadata_path.write_text(json.dumps({"profile_frames": 1}), encoding="utf-8")

    report = analyze_trace(trace_path, metadata_path)

    families = {row["family"]: row for row in report["kernel_families"]}
    assert report["attribution_level"] == "runtime_to_kernel"
    assert families["tensor_layout_conversion"]["ms_per_frame"] == 0.02
    assert families["conv_gemm_compute"]["ms_per_frame"] == 0.08
    assert report["adjacency"]["tensor_layout_conversion"][
        "followed_immediately_by_conv_gemm"
    ] == 1


def test_trace_analysis_recognizes_winograd_as_convolution(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "_5x_cudnn_ampere_scudnn_winograd_128x128",
                        "ts": 0,
                        "dur": 100,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "generateWinogradTilesKernel<float>",
                        "ts": 100,
                        "dur": 20,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_trace(trace_path)

    families = {row["family"]: row for row in report["kernel_families"]}
    assert families["conv_gemm_compute"]["calls_per_frame"] == 2
    assert families["conv_gemm_compute"]["ms_per_frame"] == pytest.approx(0.12)


def test_cuda_graph_trace_marks_graph_only_attribution(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": "cudaGraphLaunch",
                        "ts": 0,
                        "dur": 5,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "some_kernel",
                        "ts": 10,
                        "dur": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_trace(trace_path)

    assert report["attribution_level"] == "graph_only"
    assert report["gpu"]["summed_kernel_ms_per_frame"] == 0.01


def test_trace_attributes_kernel_to_submitting_operator_and_reads_benchmark(tmp_path):
    trace_path = tmp_path / "streamfm_perfetto_trace.json.gz"
    metadata_path = tmp_path / "metadata.json"
    payload = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::cudnn_convolution",
                "ts": 0,
                "dur": 10,
                "args": {
                    "External id": 7,
                    "Input Dims": [[1, 64, 32, 3], [64, 64, 3, 3]],
                    "Input type": ["float", "float"],
                },
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 2,
                "dur": 1,
                "args": {"External id": 7, "correlation": 11},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "implicit_gemm_fprop_conv",
                "ts": 20,
                "dur": 50,
                "args": {"External id": 7, "correlation": 11},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    metadata_path.write_text(json.dumps({"profile_frames": 1}), encoding="utf-8")
    (tmp_path / "torch_profile.log").write_text(
        "library warning\n[\n  {\"mean_ms\": 4.2, \"iterations\": 100}\n]\n",
        encoding="utf-8",
    )

    report = analyze_trace(trace_path, metadata_path)

    assert report["operator_attribution"]["attributed_kernel_fraction"] == 1.0
    top = report["operator_attribution"]["top_operators"][0]
    assert top["operator"] == "aten::cudnn_convolution"
    assert top["input_dims"] == [[1, 64, 32, 3], [64, 64, 3, 3]]
    assert top["kernel_ms_per_frame"] == 0.05
    assert report["benchmark_result"]["mean_ms"] == 4.2


def test_trace_analysis_maps_tensorrt_layer_back_to_exported_fx_node(tmp_path):
    trace_path = tmp_path / "streamfm_perfetto_trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": []}, handle)
    layer_dir = tmp_path / "tensorrt_layer_profile"
    layer_dir.mkdir()
    (layer_dir / "engine_engine_exectuion_profile.trace").write_text(
        json.dumps(
            [
                {
                    "ph": "X",
                    "name": "[CONVOLUTION]-[aten_ops.convolution.default]-[/convolution_13]",
                    "dur": 64.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "tensorrt_exported_ops.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "name": "convolution_13",
                        "target": "aten.convolution.default",
                        "input_nodes": [
                            "x",
                            "p_model_down_modules__lvl1_rnb0___cconv_1_weight",
                        ],
                        "output": {"shape": ["1", "256", "32", "3"]},
                        "nn_module_stack": "model.down_modules.lvl1",
                        "stack_trace": "streaming_unet.py:700",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_trace(trace_path)

    layers = report["tensorrt_layers"]
    assert layers["slow_convolution_layer_events"] == 1
    assert layers["slow_convolution_total_ms"] == 0.064
    assert layers["slow_convolutions"][0]["fx_node"] == "convolution_13"
    assert layers["slow_convolutions"][0]["weight_parameter"] == (
        "model.down_modules.lvl1_rnb0.cconv_1.weight"
    )
    assert "lvl1" in layers["slow_convolutions"][0]["nn_module_stack"]
