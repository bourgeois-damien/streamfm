"""Turn StreamFM Perfetto traces into comparable per-frame timing reports.

The clean 100-frame benchmark remains the latency source of truth.  This
module analyzes the short instrumented trace to explain *where* GPU time goes:
kernel families, exact hotspots, CUDA runtime calls, copies, layout work and
the attribution lost when execution is replayed through one CUDA Graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


TRACE_NAME = "streamfm_perfetto_trace.json.gz"


def _tensorrt_layer_family(name: str) -> str:
    lower = name.lower()
    if "reformatting" in lower or "forced cast" in lower:
        return "reformat_and_cast"
    if lower.startswith("[convolution]"):
        return "convolution_and_fusions"
    if "[matrix_multiply]" in lower or "[fully_connected]" in lower:
        return "matrix_multiply"
    if "[shuffle]" in lower or "[slice]" in lower or "reshape" in lower:
        return "shape_slice_shuffle"
    if "__myl" in lower or "pwn(" in lower or "[elementwise]" in lower:
        return "pointwise_and_fusions"
    return "other"


def _tensorrt_fx_node_name(layer_name: str) -> str | None:
    match = re.search(r"/convolution(?:_(\d+))?", layer_name)
    if not match:
        return None
    return "convolution" if match.group(1) is None else f"convolution_{match.group(1)}"


def _tensorrt_convolution_index(layer_name: str) -> int | None:
    match = re.search(r"/convolution(?:_(\d+))?", layer_name)
    if not match:
        return None
    return int(match.group(1)) if match.group(1) is not None else 0


def _fx_weight_parameter(node: dict[str, Any]) -> str | None:
    """Return the lifted model-weight placeholder consumed by an exported op."""
    for name in node.get("input_nodes", []):
        if name.startswith("p_model_") and name.endswith("_weight"):
            return name
    return None


def _readable_parameter_path(placeholder: str | None) -> str | None:
    """Make torch.export's lifted parameter name recognizable to a human."""
    if placeholder is None:
        return None
    path = placeholder.removeprefix("p_model_").removesuffix("_weight")
    path = path.replace("___", ".").replace("__", ".")
    return f"model.{path}.weight"


def _analyze_tensorrt_layers(run_dir: Path) -> dict[str, Any]:
    profiles = list(run_dir.glob("tensorrt_layer_profile/*engine_exectuion_profile.trace"))
    if not profiles:
        return {"available": False}

    exported_by_name = {}
    exported_path = run_dir / "tensorrt_exported_ops.json"
    if exported_path.exists():
        exported = json.loads(exported_path.read_text(encoding="utf-8"))
        exported_by_name = {node["name"]: node for node in exported.get("nodes", [])}
    exported_convolutions = [
        node
        for node in exported_by_name.values()
        if node.get("target") in {"aten.conv1d.default", "aten.conv2d.default"}
    ]

    events = []
    for profile_path in profiles:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        events.extend(event for event in payload if event.get("ph") == "X")
    total_us = sum(float(event.get("dur", 0.0)) for event in events)
    families: dict[str, dict[str, float]] = defaultdict(lambda: {"layers": 0.0, "us": 0.0})
    rows = []
    for event in events:
        name = str(event.get("name", "unknown"))
        duration_us = float(event.get("dur", 0.0))
        family = _tensorrt_layer_family(name)
        families[family]["layers"] += 1
        families[family]["us"] += duration_us
        fx_name = _tensorrt_fx_node_name(name)
        fx_node = exported_by_name.get(fx_name, {}) if fx_name else {}
        convolution_index = _tensorrt_convolution_index(name)
        if (
            family == "convolution_and_fusions"
            and convolution_index is not None
            and convolution_index < len(exported_convolutions)
        ):
            # Torch-TensorRT renames exported conv1d/conv2d nodes to a single
            # monotonically numbered ``convolution_N`` sequence.  The order
            # is preserved, even though the original FX names are not.
            fx_node = exported_convolutions[convolution_index]
            fx_name = fx_node.get("name")
        rows.append(
            {
                "name": name,
                "family": family,
                "duration_us": duration_us,
                "percent": 100.0 * duration_us / total_us if total_us else 0.0,
                "fx_node": fx_name,
                "fx_target": fx_node.get("target"),
                "fx_input_nodes": fx_node.get("input_nodes"),
                "weight_parameter": _readable_parameter_path(
                    _fx_weight_parameter(fx_node)
                ),
                "fx_output": fx_node.get("output"),
                "nn_module_stack": fx_node.get("nn_module_stack"),
                "stack_trace": fx_node.get("stack_trace"),
            }
        )
    rows.sort(key=lambda row: row["duration_us"], reverse=True)
    family_rows = sorted(
        (
            {
                "family": family,
                "layer_events": int(values["layers"]),
                "total_ms": values["us"] / 1000.0,
                "percent": 100.0 * values["us"] / total_us if total_us else 0.0,
            }
            for family, values in families.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )
    convolutions = [row for row in rows if row["family"] == "convolution_and_fusions"]
    slow_convolutions = [row for row in convolutions if row["duration_us"] >= 50.0]
    return {
        "available": True,
        "profile_files": [str(path) for path in profiles],
        "layer_events": len(events),
        "summed_layer_ms": total_us / 1000.0,
        "families": family_rows,
        "top_layers": rows[:40],
        "convolution_layer_events": len(convolutions),
        "slow_convolution_threshold_us": 50.0,
        "slow_convolution_layer_events": len(slow_convolutions),
        "slow_convolution_total_ms": sum(row["duration_us"] for row in slow_convolutions)
        / 1000.0,
        "slow_convolutions": slow_convolutions,
        "warning": (
            "TensorRT's per-layer profiler is an explanatory one-run measurement and changes "
            "enqueue behavior; use the clean benchmark for deployment latency."
        ),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _kernel_family(name: str) -> str:
    lower = name.lower()
    if "padfilterweights" in lower:
        return "filter_weight_preparation"
    if "converttensor" in lower:
        return "tensor_layout_conversion"
    if any(token in lower for token in ("nchwtonhwc", "nhwctonchw", "tensortransform", "kcrsto")):
        return "tensor_layout_conversion"
    if any(token in lower for token in ("quant", "dequant", "round", "rouncycastle")) or (
        "cast" in lower and "copy_kernel" not in lower
    ):
        return "quantize_dequantize_cast"
    if any(token in lower for token in ("memcpy", "copy_kernel", "copy_", "clone")):
        return "copies"
    if any(token in lower for token in ("slice", "slic", "permute", "transpose", "reshape", "resh")):
        return "slice_reshape"
    # cuDNN/SCUDNN Winograd kernels do not consistently contain the literal
    # substring ``conv`` even though they implement a convolution tactic.
    if any(
        token in lower
        for token in ("scudnn_winograd", "generatewinograd", "convolve_common")
    ):
        return "conv_gemm_compute"
    if any(token in lower for token in ("pointwise", "elementwise", "triton_poi", "__myl", "silu", "relu")):
        return "pointwise_fused"
    if any(token in lower for token in ("conv", "fprop", "gemm", "gemv", "addmm", "xmma", "cutlass")):
        return "conv_gemm_compute"
    return "other"


def _merge_busy_time_us(events: Iterable[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
        for event in events
        if float(event.get("dur", 0.0)) > 0
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _representative_shape(events: list[dict[str, Any]], key: str) -> Any:
    values = [json.dumps(event.get("args", {}).get(key)) for event in events if key in event.get("args", {})]
    if not values:
        return None
    encoded = Counter(values).most_common(1)[0][0]
    return json.loads(encoded)


def _aggregate_named(events: list[dict[str, Any]], *, frames: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("name", "unknown"))].append(event)
    total_us = sum(float(event.get("dur", 0.0)) for event in events)
    rows = []
    for name, group in grouped.items():
        durations = [float(event.get("dur", 0.0)) for event in group]
        group_us = sum(durations)
        occupancy = [
            float(event.get("args", {}).get("est. achieved occupancy %"))
            for event in group
            if event.get("args", {}).get("est. achieved occupancy %") is not None
        ]
        rows.append(
            {
                "name": name,
                "family": _kernel_family(name),
                "calls": len(group),
                "calls_per_frame": len(group) / frames,
                "total_ms_per_frame": group_us / 1000.0 / frames,
                "percent_of_summed_kernel_time": 100.0 * group_us / total_us if total_us else 0.0,
                "mean_us_per_call": statistics.fmean(durations),
                "p50_us_per_call": _percentile(durations, 0.5),
                "p90_us_per_call": _percentile(durations, 0.9),
                "grid": _representative_shape(group, "grid"),
                "block": _representative_shape(group, "block"),
                "registers_per_thread": _representative_shape(group, "registers per thread"),
                "static_estimated_occupancy_percent_mean": (
                    statistics.fmean(occupancy) if occupancy else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["total_ms_per_frame"], reverse=True)


def _aggregate_families(exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "ms": 0.0})
    for row in exact_rows:
        grouped[row["family"]]["calls"] += float(row["calls_per_frame"])
        grouped[row["family"]]["ms"] += float(row["total_ms_per_frame"])
    total_ms = sum(values["ms"] for values in grouped.values())
    return sorted(
        (
            {
                "family": family,
                "calls_per_frame": values["calls"],
                "ms_per_frame": values["ms"],
                "percent_of_summed_kernel_time": 100.0 * values["ms"] / total_ms if total_ms else 0.0,
            }
            for family, values in grouped.items()
        ),
        key=lambda row: row["ms_per_frame"],
        reverse=True,
    )


def _external_id(event: dict[str, Any]) -> int | None:
    value = event.get("args", {}).get("External id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _aggregate_attributed_operators(
    kernel_events: list[dict[str, Any]],
    cpu_ops: list[dict[str, Any]],
    *,
    frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attribute kernels to the CPU op that submitted them via External id.

    CUPTI stores the submitting CPU op's ``External id`` on every directly
    launched kernel.  A CUDA Graph replay deliberately collapses this relation:
    every replayed node points back to the one graph-launch operation instead.
    """

    cpu_by_external_id = {
        external_id: event
        for event in cpu_ops
        if (external_id := _external_id(event)) is not None
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    attributed = 0
    for kernel in kernel_events:
        op = cpu_by_external_id.get(_external_id(kernel))
        if op is None:
            continue
        attributed += 1
        args = op.get("args", {})
        dims = json.dumps(args.get("Input Dims"), sort_keys=True)
        types = json.dumps(args.get("Input type"), sort_keys=True)
        grouped[(str(op.get("name", "unknown")), dims, types)].append(kernel)

    total_us = sum(float(event.get("dur", 0.0)) for event in kernel_events)
    rows = []
    for (name, dims_json, types_json), group in grouped.items():
        group_us = sum(float(event.get("dur", 0.0)) for event in group)
        rows.append(
            {
                "operator": name,
                "input_dims": json.loads(dims_json),
                "input_types": json.loads(types_json),
                "kernel_calls_per_frame": len(group) / frames,
                "kernel_ms_per_frame": group_us / 1000.0 / frames,
                "percent_of_summed_kernel_time": 100.0 * group_us / total_us if total_us else 0.0,
                "kernel_families": sorted({_kernel_family(str(event.get("name", ""))) for event in group}),
                "representative_kernel": Counter(
                    str(event.get("name", "unknown")) for event in group
                ).most_common(1)[0][0],
            }
        )
    rows.sort(key=lambda row: row["kernel_ms_per_frame"], reverse=True)
    return rows, {
        "attributed_kernel_fraction": attributed / len(kernel_events) if kernel_events else 0.0,
        "attributed_kernel_calls": attributed,
        "total_kernel_calls": len(kernel_events),
    }


def _section_gpu_breakdown(
    complete_events: list[dict[str, Any]],
    gpu_events: list[dict[str, Any]],
    *,
    frames: int,
) -> list[dict[str, Any]]:
    ranges = [
        event
        for event in complete_events
        if event.get("cat") == "gpu_user_annotation"
        and str(event.get("name", "")).startswith("streamfm/section/")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ranges:
        section = str(event.get("name", "")).rsplit("/", 1)[-1]
        grouped[section].append(event)

    rows = []
    for section, section_ranges in grouped.items():
        inside = []
        for gpu_event in gpu_events:
            start = float(gpu_event.get("ts", 0.0))
            if any(
                float(section_range.get("ts", 0.0))
                <= start
                < float(section_range.get("ts", 0.0)) + float(section_range.get("dur", 0.0))
                for section_range in section_ranges
            ):
                inside.append(gpu_event)
        rows.append(
            {
                "section": section,
                "ranges": len(section_ranges),
                "kernel_and_memory_calls_per_frame": len(inside) / frames,
                "summed_gpu_ms_per_frame": sum(float(event.get("dur", 0.0)) for event in inside)
                / 1000.0
                / frames,
                "busy_union_ms_per_frame": _merge_busy_time_us(inside) / 1000.0 / frames,
            }
        )
    return sorted(rows, key=lambda row: row["summed_gpu_ms_per_frame"], reverse=True)


def _adjacency(kernel_events: list[dict[str, Any]], source_family: str) -> dict[str, Any]:
    ordered = sorted(kernel_events, key=lambda event: float(event.get("ts", 0.0)))
    sources = 0
    followed_by_compute = 0
    gaps = []
    for index, event in enumerate(ordered[:-1]):
        if _kernel_family(str(event.get("name", ""))) != source_family:
            continue
        sources += 1
        next_event = ordered[index + 1]
        if _kernel_family(str(next_event.get("name", ""))) == "conv_gemm_compute":
            followed_by_compute += 1
            gaps.append(
                float(next_event.get("ts", 0.0))
                - (float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)))
            )
    return {
        "source_calls": sources,
        "followed_immediately_by_conv_gemm": followed_by_compute,
        "fraction_followed_by_conv_gemm": followed_by_compute / sources if sources else 0.0,
        "median_gap_us": statistics.median(gaps) if gaps else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_benchmark_result(log_path: Path) -> dict[str, Any]:
    """Extract the benchmark's final JSON row despite preceding library logs."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if "mean_ms" in value[0] and "iterations" in value[0]:
                return value[0]
    return {}


def analyze_trace(trace_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    payload = _load_json(trace_path)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path is not None and metadata_path.exists()
        else {}
    )
    benchmark_result = _extract_benchmark_result(trace_path.parent / "torch_profile.log")
    frames = max(1, int(metadata.get("profile_frames", 1)))
    complete = [event for event in payload.get("traceEvents", []) if event.get("ph") == "X"]
    kernels = [event for event in complete if event.get("cat") == "kernel"]
    gpu_memory = [
        event for event in complete if event.get("cat") in {"gpu_memcpy", "gpu_memset"}
    ]
    runtime = [event for event in complete if event.get("cat") == "cuda_runtime"]
    cpu_ops = [event for event in complete if event.get("cat") == "cpu_op"]

    exact = _aggregate_named(kernels, frames=frames)
    runtime_exact = _aggregate_named(runtime, frames=frames)
    graph_launches = [event for event in runtime if event.get("name") == "cudaGraphLaunch"]
    ordinary_launches = [
        event
        for event in runtime
        if str(event.get("name", "")).lower() in {"cudalaunchkernel", "culaunchkernel"}
        or "launchkernel" in str(event.get("name", "")).lower()
    ]
    gpu_events = kernels + gpu_memory
    gpu_span_us = (
        max(float(event["ts"]) + float(event.get("dur", 0.0)) for event in gpu_events)
        - min(float(event["ts"]) for event in gpu_events)
        if gpu_events
        else 0.0
    )
    summed_kernel_us = sum(float(event.get("dur", 0.0)) for event in kernels)
    attributed_operators, attribution = _aggregate_attributed_operators(
        kernels, cpu_ops, frames=frames
    )

    return {
        "trace": str(trace_path),
        "metadata": metadata,
        "benchmark_result": benchmark_result,
        "profile_frames": frames,
        "attribution_level": (
            "graph_only"
            if graph_launches
            else ("operator_to_kernel" if ordinary_launches and cpu_ops else "runtime_to_kernel")
        ),
        "warnings": [
            "Use the clean 100-frame benchmark for latency; this short trace is for attribution.",
            "Summed kernel durations can exceed GPU busy time when streams overlap.",
            "The occupancy field is a static launch estimate, not a hardware performance counter.",
            "CUDA runtime durations under CUPTI are instrumented and are not deployment submit costs.",
        ],
        "gpu": {
            "kernel_calls_per_frame": len(kernels) / frames,
            "summed_kernel_ms_per_frame": summed_kernel_us / 1000.0 / frames,
            "busy_union_ms_per_frame": _merge_busy_time_us(gpu_events) / 1000.0 / frames,
            "span_ms_per_frame": gpu_span_us / 1000.0 / frames,
            "memory_event_calls_per_frame": len(gpu_memory) / frames,
            "memory_event_ms_per_frame": sum(float(event.get("dur", 0.0)) for event in gpu_memory)
            / 1000.0
            / frames,
        },
        "cpu_cuda_runtime": {
            "calls_per_frame": len(runtime) / frames,
            "summed_ms_per_frame": sum(float(event.get("dur", 0.0)) for event in runtime)
            / 1000.0
            / frames,
            "cuda_graph_launches_per_frame": len(graph_launches) / frames,
            "ordinary_kernel_launches_per_frame": len(ordinary_launches) / frames,
            "cpu_ops_per_frame": len(cpu_ops) / frames,
            "top_calls": runtime_exact[:15],
        },
        "operator_attribution": {
            **attribution,
            "top_operators": attributed_operators[:30],
        },
        "streaming_sections": _section_gpu_breakdown(
            complete, gpu_events, frames=frames
        ),
        "tensorrt_layers": _analyze_tensorrt_layers(trace_path.parent),
        "kernel_families": _aggregate_families(exact),
        "top_kernels": exact[:30],
        "adjacency": {
            "filter_weight_preparation": _adjacency(kernels, "filter_weight_preparation"),
            "tensor_layout_conversion": _adjacency(kernels, "tensor_layout_conversion"),
        },
    }


def _label(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {})
    execution = metadata.get("execution", "unknown")
    precision = (
        f"int8({metadata.get('dtype', 'unknown')}-fallback)"
        if metadata.get("ptq_int8")
        else metadata.get("dtype", "unknown")
    )
    suffix = "+cg" if metadata.get("cuda_graph") or execution == "cuda_graph" else ""
    tf32 = metadata.get("tf32")
    tf32_suffix = f"[tf32-{tf32}]" if tf32 in {"on", "off"} else ""
    return f"{execution}:{precision}{tf32_suffix}{suffix}"


def render_comparison(reports: list[dict[str, Any]]) -> str:
    families = sorted(
        {row["family"] for report in reports for row in report.get("kernel_families", [])}
    )
    lines = [
        "# StreamFM GPU trace comparison",
        "",
        "The clean benchmark latency is authoritative; these profiled frames explain attribution.",
        "",
        "| configuration | attribution | benchmark mean ms | engine elapsed ms | kernels/frame | busy ms/frame |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        gpu = report["gpu"]
        benchmark = report.get("benchmark_result", {})
        lines.append(
            f"| {_label(report)} | {report['attribution_level']} | "
            f"{benchmark.get('mean_ms', 0.0):.3f} | "
            f"{benchmark.get('tensorrt_engine_gpu_mean_ms', 0.0):.3f} | "
            f"{gpu['kernel_calls_per_frame']:.1f} | "
            f"{gpu['busy_union_ms_per_frame']:.3f} |"
        )
    lines.extend(["", "| family | " + " | ".join(_label(report) for report in reports) + " |", "|---|" + "---:|" * len(reports)])
    for family in families:
        values = []
        for report in reports:
            by_family = {row["family"]: row for row in report.get("kernel_families", [])}
            values.append(f"{by_family.get(family, {}).get('ms_per_frame', 0.0):.3f} ms")
        lines.append(f"| {family} | " + " | ".join(values) + " |")
    lines.extend(["", "## Dominant kernels", ""])
    for report in reports:
        lines.append(f"### {_label(report)}")
        lines.append("")
        for row in report.get("top_kernels", [])[:8]:
            lines.append(
                f"- {row['total_ms_per_frame']:.3f} ms/frame, {row['calls_per_frame']:.1f} calls/frame — "
                f"`{row['family']}` — `{row['name']}`"
            )
        lines.append("")

    lines.extend(["", "## Dominant submitting operations", ""])
    for report in reports:
        lines.append(f"### {_label(report)}")
        lines.append("")
        attribution = report.get("operator_attribution", {})
        lines.append(
            f"Attributed kernels: {100.0 * attribution.get('attributed_kernel_fraction', 0.0):.1f}%"
        )
        lines.append("")
        for row in attribution.get("top_operators", [])[:8]:
            lines.append(
                f"- {row['kernel_ms_per_frame']:.3f} ms/frame, "
                f"{row['kernel_calls_per_frame']:.1f} kernels/frame — "
                f"`{row['operator']}` — inputs `{row['input_dims']}`"
            )
        lines.append("")
    return "\n".join(lines)


def _resolve_trace(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path / TRACE_NAME, path / "metadata.json"
    metadata = path.parent / "metadata.json"
    return path, metadata if metadata.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Trace files or run directories")
    parser.add_argument("--output", type=Path, default=Path("outputs/nsight_streamfm/trace_comparison.md"))
    args = parser.parse_args()

    reports = []
    for input_path in args.paths:
        trace_path, metadata_path = _resolve_trace(input_path)
        report = analyze_trace(trace_path, metadata_path)
        reports.append(report)
        analysis_path = trace_path.parent / "trace_analysis.json"
        analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_comparison(reports), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
