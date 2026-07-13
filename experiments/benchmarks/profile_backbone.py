#!/usr/bin/env python3
"""Profile CausalNCSNpp streaming forward_step: stage, op, and module costs."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.loading import load_flow_model
from experiments.benchmarks.paths import make_benchmark_paths
from experiments.common import apply_model_memory_format, prepare_streaming_state
from sgmse.backbones.streaming_unet import CausalConv2d, CausalDecoupledConv2d, CausalNCSNpp


def _inventory(dnn: torch.nn.Module) -> dict[str, Any]:
    conv3x3 = []
    conv1x1 = []
    other_conv = []
    decoupled = []
    linear = []
    norms = []

    for name, mod in dnn.named_modules():
        if isinstance(mod, CausalDecoupledConv2d):
            decoupled.append(name)
            continue
        if isinstance(mod, CausalConv2d):
            kh, kw = int(mod.kernel_size[0]), int(mod.kernel_size[1])
            entry = {
                "name": name,
                "in": mod.in_channels,
                "out": mod.out_channels,
                "k": [kh, kw],
                "params": sum(p.numel() for p in mod.parameters()),
            }
            if (kh, kw) == (3, 3):
                conv3x3.append(entry)
            elif (kh, kw) == (1, 1):
                conv1x1.append(entry)
            else:
                other_conv.append(entry)
        elif isinstance(mod, torch.nn.Linear):
            linear.append(
                {
                    "name": name,
                    "in": mod.in_features,
                    "out": mod.out_features,
                    "params": sum(p.numel() for p in mod.parameters()),
                }
            )
        elif "Norm" in type(mod).__name__ or "norm" in type(mod).__name__.lower():
            if list(mod.parameters()):
                norms.append({"name": name, "type": type(mod).__name__})

    def _sum_params(items: list[dict[str, Any]]) -> int:
        return int(sum(i.get("params", 0) for i in items))

    return {
        "conv3x3_count": len(conv3x3),
        "conv1x1_count": len(conv1x1),
        "other_conv_count": len(other_conv),
        "decoupled_count": len(decoupled),
        "linear_count": len(linear),
        "norm_modules_with_params": len(norms),
        "conv3x3_params": _sum_params(conv3x3),
        "conv1x1_params": _sum_params(conv1x1),
        "linear_params": _sum_params(linear),
        "total_params": int(sum(p.numel() for p in dnn.parameters())),
        "conv3x3": conv3x3,
        "conv1x1": conv1x1,
        "linear": linear,
    }


def _install_stage_labels(dnn: CausalNCSNpp):
    """Wrap uncompiled forward_step with named profiler regions."""
    raw = getattr(dnn.forward_step, "__wrapped__", None)
    if raw is None:
        raise RuntimeError("Expected @torch.compile-wrapped forward_step with __wrapped__.")

    def labeled_forward_step(self, x, time_cond=None, aux_condition=None, *, state):
        if not self.is_causal:
            raise NotImplementedError("forward_step requires a causal model.")

        with record_function("stage/temb_input"):
            temb = self.prepare_temb(time_cond)
            m_idx = 0
            h0, _ = self.input_layer.forward_step(x, state=state[m_idx])
            m_idx += 1
            hs_up = [h0]
            input_pyramid = x
            h = h0

        for l in range(self.num_resolutions):
            with record_function(f"stage/down_lvl{l}"):
                for k in range(self.num_res_blocks):
                    h, _ = self.down_modules[f"lvl{l}_rnb{k}"].forward_step(h, temb, state=state[m_idx])
                    m_idx += 1
                    hs_up.append(h)
                if l != self.num_resolutions - 1:
                    h, _ = self.down_modules[f"lvl{l}_rnb_down"].forward_step(h, temb, state=state[m_idx])
                    m_idx += 1
                    input_pyramid, state[m_idx] = self.pyramid_downsample.forward_step(
                        input_pyramid, state=state[m_idx]
                    )
                    m_idx += 1
                    h = self.down_modules[f"lvl{l}_combiner"](input_pyramid, h)

        with record_function("stage/bottleneck"):
            h, _ = self.bottleneck_modules["rnb1"].forward_step(h, temb, state=state[m_idx])
            m_idx += 1
            if self.attn_bottleneck:
                raise NotImplementedError("Bottleneck attention not implemented for forward_step.")
            h, _ = self.bottleneck_modules["rnb2"].forward_step(h, temb, state=state[m_idx])
            m_idx += 1

        pyramid = None
        for l in reversed(range(self.num_resolutions)):
            with record_function(f"stage/up_lvl{l}"):
                nrb = self.num_res_blocks + (1 if l == 0 else 0)
                for k in range(nrb):
                    h_input = self.up_modules[f"lvl{l}_combiner{k}"](hs_up.pop(), h)
                    h, _ = self.up_modules[f"lvl{l}_rnb{k}"].forward_step(h_input, temb, state=state[m_idx])
                    m_idx += 1
                pyramid_h, _ = self.up_modules[f"lvl{l}_pyramid_normconv"].forward_step(h, state=state[m_idx])
                m_idx += 1
                if l != self.num_resolutions - 1:
                    pyramid_up, _ = self.pyramid_upsample.forward_step(pyramid, state=state[m_idx])
                    m_idx += 1
                    pyramid = pyramid_up + pyramid_h
                else:
                    pyramid = pyramid_h
                if l != 0:
                    h, _ = self.up_modules[f"lvl{l}_rnb_up"].forward_step(h, temb, state=state[m_idx])
                    m_idx += 1
                else:
                    h = pyramid
        return h, state

    # Bind as a plain method; call dnn.forward_step(...) directly (eager path).
    dnn.forward_step = types.MethodType(labeled_forward_step, dnn)
    return raw


def _install_module_labels(dnn: torch.nn.Module):
    """Add a profiler range to every individual Conv/Linear module.

    Aggregate ``aten`` names tell us which kernel family was selected, but not
    which model layer produced it.  Module ranges bridge that gap and let us
    rank actual calls by latency *and* by their concrete tensor shapes/MACs.
    """
    originals: list[tuple[torch.nn.Module, str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    def _shape(tensor: torch.Tensor) -> list[int]:
        return [int(dim) for dim in tensor.shape]

    def _record_shapes(label: str, x: torch.Tensor, output: torch.Tensor) -> None:
        entry = metadata[label]
        # Each module has one stable shape in a fixed-shape streaming profile.
        # Keep a list nonetheless so shape changes are reported rather than
        # silently folded into one latency number.
        input_shape = _shape(x)
        output_shape = _shape(output)
        if input_shape not in entry["input_shapes"]:
            entry["input_shapes"].append(input_shape)
        if output_shape not in entry["output_shapes"]:
            entry["output_shapes"].append(output_shape)

    for name, module in dnn.named_modules():
        if isinstance(module, CausalConv2d):
            label = f"module/causal_conv/{name}"
            original = module.forward_step
            metadata[label] = {
                "name": name,
                "kind": "causal_conv2d",
                "kernel": [int(v) for v in module.kernel_size],
                "dilation": [int(v) for v in module.dilation],
                "groups": int(module.groups),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "input_shapes": [],
                "output_shapes": [],
            }

            def labeled_forward_step(self, x, *, state, _original=original, _label=label):
                with record_function(_label):
                    output, next_state = _original(x, state=state)
                _record_shapes(_label, x, output)
                return output, next_state

            module.forward_step = types.MethodType(labeled_forward_step, module)
            originals.append((module, "forward_step", original))
            continue

        if type(module) is torch.nn.Conv2d:
            label = f"module/conv2d/{name}"
            original = module.forward
            metadata[label] = {
                "name": name,
                "kind": "conv2d",
                "kernel": [int(v) for v in module.kernel_size],
                "dilation": [int(v) for v in module.dilation],
                "groups": int(module.groups),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "input_shapes": [],
                "output_shapes": [],
            }

            def labeled_conv_forward(self, x, *args, _original=original, _label=label, **kwargs):
                with record_function(_label):
                    output = _original(x, *args, **kwargs)
                _record_shapes(_label, x, output)
                return output

            module.forward = types.MethodType(labeled_conv_forward, module)
            originals.append((module, "forward", original))
            continue

        if isinstance(module, torch.nn.Linear):
            label = f"module/linear/{name}"
            original = module.forward
            metadata[label] = {
                "name": name,
                "kind": "linear",
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "input_shapes": [],
                "output_shapes": [],
            }

            def labeled_linear_forward(self, x, *args, _original=original, _label=label, **kwargs):
                with record_function(_label):
                    output = _original(x, *args, **kwargs)
                _record_shapes(_label, x, output)
                return output

            module.forward = types.MethodType(labeled_linear_forward, module)
            originals.append((module, "forward", original))

    return originals, metadata


def _restore_module_labels(originals: list[tuple[torch.nn.Module, str, Any]]) -> None:
    for module, attribute, original in originals:
        setattr(module, attribute, original)


def _conv_macs(entry: dict[str, Any]) -> int | None:
    """Return MACs for one concrete Conv2d call, if its output shape is known."""
    if len(entry["output_shapes"]) != 1:
        return None
    output_shape = entry["output_shapes"][0]
    if len(output_shape) != 4:
        return None
    batch, out_channels, out_height, out_width = output_shape
    kh, kw = entry["kernel"]
    return int(
        batch
        * out_channels
        * out_height
        * out_width
        * (entry["in_channels"] // entry["groups"])
        * kh
        * kw
    )


def _linear_macs(entry: dict[str, Any]) -> int | None:
    """Return MACs for one concrete Linear call, if its output shape is known."""
    if len(entry["output_shapes"]) != 1:
        return None
    output_shape = entry["output_shapes"][0]
    if not output_shape:
        return None
    rows = 1
    for dim in output_shape[:-1]:
        rows *= dim
    return int(rows * entry["in_features"] * entry["out_features"])


def _parse_profiler_table(prof: profile) -> list[dict[str, Any]]:
    rows = []
    for evt in prof.key_averages():
        rows.append(
            {
                "key": evt.key,
                "cpu_self_us": float(evt.self_cpu_time_total),
                "cpu_total_us": float(evt.cpu_time_total),
                "cuda_self_us": float(getattr(evt, "self_cuda_time_total", 0) or 0),
                "cuda_total_us": float(getattr(evt, "cuda_time_total", 0) or 0),
                "calls": int(evt.count),
            }
        )
    return rows


def run_backbone_profile(
    *,
    task: str = "stftpr",
    device: str = "cpu",
    dtype_name: str = "fp32",
    memory_format: str = "channels_last",
    iterations: int = 40,
    warmup: int = 10,
    freq_bins: int = 256,
    num_threads: int = 1,
    num_interop_threads: int = 1,
    paths=None,
    gpu_name: str = "",
) -> dict[str, Any]:
    """Profile CausalNCSNpp streaming forward_step; return a JSON-serializable report."""
    if device == "cpu":
        if num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(num_interop_threads)
            except RuntimeError:
                pass
        if num_threads > 0:
            torch.set_num_threads(num_threads)

    torch_device = torch.device(device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    if paths is None:
        paths = make_benchmark_paths(REPO_ROOT)
    model, _cfg = load_flow_model(torch_device, dtype, paths, task=task)
    dnn = model.dnn if hasattr(model, "dnn") else model
    dnn = apply_model_memory_format(dnn, memory_format)
    dnn.eval()

    if not isinstance(dnn, CausalNCSNpp):
        raise TypeError(f"Expected CausalNCSNpp backbone, got {type(dnn)}")

    if not gpu_name and torch_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch_device)

    inventory = _inventory(dnn)
    _install_stage_labels(dnn)
    module_label_originals, module_metadata = _install_module_labels(dnn)

    freq = freq_bins
    in_ch = int(getattr(dnn, "input_channels", 4))
    y = torch.randn(1, in_ch, freq, 1, device=torch_device, dtype=dtype)
    if memory_format == "channels_last":
        y = y.contiguous(memory_format=torch.channels_last)
    t = torch.rand(1, device=torch_device, dtype=dtype)
    state = prepare_streaming_state(dnn)

    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            dnn.forward_step(y, time_cond=t, state=state)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(iterations):
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            dnn.forward_step(y, time_cond=t, state=state)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    activities = [ProfilerActivity.CPU]
    if torch_device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    try:
        with torch.inference_mode():
            with profile(activities=activities, record_shapes=True, profile_memory=False) as prof:
                for _ in range(iterations):
                    dnn.forward_step(y, time_cond=t, state=state)
                if torch_device.type == "cuda":
                    torch.cuda.synchronize()
        events = _parse_profiler_table(prof)
    finally:
        _restore_module_labels(module_label_originals)

    def _rank(e: dict[str, Any]) -> float:
        return max(e["cuda_self_us"], e["cpu_self_us"])

    def _incl(e: dict[str, Any]) -> float:
        return max(e["cuda_total_us"], e["cpu_total_us"])

    events_sorted = sorted(events, key=_rank, reverse=True)
    stages = sorted(
        [e for e in events if e["key"].startswith("stage/")],
        key=_incl,
        reverse=True,
    )
    module_events = {
        e["key"]: e for e in events if e["key"].startswith("module/")
    }
    module_ops = []
    for label, metadata in module_metadata.items():
        event = module_events.get(label)
        if event is None:
            continue
        calls = max(1, int(event["calls"]))
        total_us = _incl(event)
        self_us = _rank(event)
        entry = {
            **metadata,
            "calls": calls,
            "calls_per_frame": calls / max(1, iterations),
            "self_ms_per_call": self_us / 1000.0 / calls,
            "inclusive_ms_per_call": total_us / 1000.0 / calls,
            "inclusive_ms_per_frame": total_us / 1000.0 / max(1, iterations),
        }
        if metadata["kind"] in {"causal_conv2d", "conv2d"}:
            entry["macs_per_call"] = _conv_macs(metadata)
        else:
            entry["macs_per_call"] = _linear_macs(metadata)
        module_ops.append(entry)
    module_ops.sort(key=lambda e: -e["inclusive_ms_per_frame"])
    aten = sorted(
        [
            e
            for e in events
            if e["key"].startswith(("aten::", "cudnn::", "nvrtc::", "cuda::", "mkldnn::", "thnn::"))
        ],
        key=_rank,
        reverse=True,
    )

    return {
        "config": {
            "task": task,
            "device": device,
            "gpu_name": gpu_name,
            "dtype": dtype_name,
            "memory_format": memory_format,
            "iterations": iterations,
            "warmup": warmup,
            "freq_bins": freq,
            "input_channels": in_ch,
            "num_threads": num_threads,
            "num_interop_threads": num_interop_threads,
            "execution": "eager_uncompiled",
        },
        "wall_ms": {
            "mean": sum(times) / len(times),
            "p50": sorted(times)[len(times) // 2],
            "min": min(times),
            "max": max(times),
        },
        "inventory": {
            k: v
            for k, v in inventory.items()
            if k not in {"conv3x3", "conv1x1", "linear"}
        },
        "inventory_detail": {
            "conv3x3_top_by_params": sorted(inventory["conv3x3"], key=lambda x: -x["params"])[:12],
            "conv1x1_top_by_params": sorted(inventory["conv1x1"], key=lambda x: -x["params"])[:8],
        },
        "stages": stages,
        "module_ops": module_ops,
        "aten_top": aten[:25],
        "profiler_top": events_sorted[:40],
    }


def _print_report(report: dict[str, Any]) -> None:
    iters = max(1, int(report["config"]["iterations"]))
    print(json.dumps({"wall_ms": report["wall_ms"], "inventory": report["inventory"], "config": report["config"]}, indent=2))
    print("\n=== Stages (inclusive ms/frame) ===")
    for e in report["stages"]:
        incl = max(e["cuda_total_us"], e["cpu_total_us"]) / 1000.0 / iters
        print(f"  {e['key']:22s}  {incl:8.3f} ms/frame  calls={e['calls']}")
    print("\n=== Top modules (inclusive ms/frame) ===")
    for e in report["module_ops"][:20]:
        macs = e.get("macs_per_call")
        macs_label = "?" if macs is None else f"{macs / 1e6:.2f}M MACs"
        print(
            f"  {e['kind']:14s} {e['name']:42.42s} "
            f"{e['inclusive_ms_per_frame']:7.3f} ms/frame  "
            f"{e['inclusive_ms_per_call']:7.3f} ms/call  {macs_label}"
        )
    print("\n=== Top aten (self ms/frame) ===")
    for e in report["aten_top"][:15]:
        self_ms = max(e["cuda_self_us"], e["cpu_self_us"]) / 1000.0 / iters
        print(f"  {e['key']:40s}  {self_ms:8.3f} ms/frame  calls={e['calls']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stftpr")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--memory-format", default="channels_last")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--freq-bins", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-interop-threads", type=int, default=1)
    parser.add_argument(
        "--out",
        default="outputs/benchmark_profiles/backbone_profile.json",
        help="JSON report path.",
    )
    args = parser.parse_args()

    report = run_backbone_profile(
        task=args.task,
        device=args.device,
        dtype_name=args.dtype,
        memory_format=args.memory_format,
        iterations=args.iterations,
        warmup=args.warmup,
        freq_bins=args.freq_bins,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
