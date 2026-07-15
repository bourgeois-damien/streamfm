"""Eager (non-graph) benchmark loops for the backbones.

Baseline timing loops for flow steps, the SE predictor, SE flow and the full SE
model: one warmup pass, then timed iterations.
"""

from __future__ import annotations

import time

from experiments.core.tensors import empty_model_tensor, format_model_tensor, pack_ri_channels
from experiments.core.streaming_state import forward_step
from experiments.core.timing import summarize_ms
from experiments.core.devices import sync_device


def benchmark_flow_steps(
    model,
    device,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    dtype,
    preallocate_model_buffers: bool,
    model_memory_format: str = "contiguous",
    freq_bins: int = 256,
    frame_budget_ms: float = 16.0,
) -> list[dict]:
    """Benchmark only a streaming flow model loop on random frames."""
    import torch

    flow = model.eval()
    results = []

    for steps in steps_list:
        y_frame = format_model_tensor(torch.randn(1, 2, freq_bins, 1, device=device, dtype=dtype), model_memory_format)
        flow_states = [flow.init_state() for _ in range(steps)]
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]
        dnn_input = empty_model_tensor((1, 4, freq_bins, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        x_t_buffer = torch.empty_like(y_frame)
        times_ms = []

        with torch.inference_mode():
            for frame_idx in range(warmup + iterations):
                sync_device(device)
                start = time.perf_counter()

                if preallocate_model_buffers:
                    x_t_buffer.copy_(y_frame)
                    for step_idx in range(steps):
                        pack_ri_channels(x_t_buffer, y_frame, out=dnn_input)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input,
                            state=flow_states[step_idx],
                            time_cond=t_tensors[step_idx],
                            use_compiled=use_compiled,
                        )
                        x_t_buffer.add_(v, alpha=1.0 / steps)
                else:
                    x_t = y_frame
                    for step_idx in range(steps):
                        t = torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype)
                        dnn_input_dynamic = pack_ri_channels(x_t, y_frame, memory_format=model_memory_format)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input_dynamic,
                            state=flow_states[step_idx],
                            time_cond=t,
                            use_compiled=use_compiled,
                        )
                        x_t = x_t + v / steps

                sync_device(device)
                if frame_idx >= warmup:
                    times_ms.append((time.perf_counter() - start) * 1000.0)

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_flow_only",
                "task": "stftpr",
                "device": device.type,
                "steps": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "preallocate_model_buffers": preallocate_model_buffers,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": frame_budget_ms,
                "budget_ratio_mean": summary["mean_ms"] / frame_budget_ms,
            }
        )
        results.append(summary)

    return results


def benchmark_se_predictor(
    predictor,
    device,
    iterations: int,
    warmup: int,
    use_compiled: bool,
    dtype,
    model_memory_format: str = "contiguous",
) -> list[dict]:
    """Benchmark only the SE initial predictor DNN on random streaming frames."""
    import torch

    predictor = predictor.eval()
    source_y_frames = torch.randn(warmup + iterations, 1, 2, 256, 1, device=device, dtype=dtype)
    predictor_state = predictor.init_state()
    times_ms = []

    with torch.inference_mode():
        for frame_idx in range(warmup + iterations):
            y_frame = format_model_tensor(source_y_frames[frame_idx], model_memory_format)
            sync_device(device)
            start = time.perf_counter()
            _, predictor_state = forward_step(
                predictor,
                y_frame,
                state=predictor_state,
                use_compiled=use_compiled,
            )
            sync_device(device)

            if frame_idx >= warmup:
                times_ms.append((time.perf_counter() - start) * 1000.0)

    summary = summarize_ms(times_ms)
    summary.update(
        {
            "mode": "frame_step_se_predictor_only",
            "task": "se_predictor",
            "device": device.type,
            "steps": 0,
            "predictor_calls_per_frame": 1,
            "flow_calls_per_frame": 0,
            "iterations": iterations,
            "warmup": warmup,
            "compiled": use_compiled,
            "model_memory_format": model_memory_format,
            "frame_budget_ms": 16.0,
            "budget_ratio_mean": summary["mean_ms"] / 16.0,
        }
    )
    return [summary]


def benchmark_se_flow(
    flow,
    device,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    dtype,
    sigma_e: float = 0.05,
    preallocate_model_buffers: bool = False,
    model_memory_format: str = "contiguous",
) -> list[dict]:
    """Benchmark only the SE generative flow DNN loop on random E/Y frames."""
    import torch

    flow = flow.eval()
    results = []
    total_frames = warmup + iterations
    source_e_frames = torch.randn(total_frames, 1, 2, 256, 1, device=device, dtype=dtype)
    source_y_frames = torch.randn_like(source_e_frames)
    source_noise_frames = torch.randn_like(source_e_frames)

    for steps in steps_list:
        flow_states = [flow.init_state() for _ in range(steps)]
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]
        dnn_input = empty_model_tensor((1, 6, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        x_t_buffer = empty_model_tensor((1, 2, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        times_ms = []

        with torch.inference_mode():
            for frame_idx in range(total_frames):
                e_frame = format_model_tensor(source_e_frames[frame_idx], model_memory_format)
                y_frame = format_model_tensor(source_y_frames[frame_idx], model_memory_format)
                noise_frame = format_model_tensor(source_noise_frames[frame_idx], model_memory_format)
                if preallocate_model_buffers:
                    x_t_buffer.copy_(e_frame)
                    x_t_buffer.add_(noise_frame, alpha=sigma_e)
                else:
                    x_t = e_frame + sigma_e * noise_frame

                sync_device(device)
                start = time.perf_counter()
                if preallocate_model_buffers:
                    for step_idx in range(steps):
                        pack_ri_channels(x_t_buffer, e_frame, y_frame, out=dnn_input)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input,
                            state=flow_states[step_idx],
                            time_cond=t_tensors[step_idx],
                            use_compiled=use_compiled,
                        )
                        x_t_buffer.add_(v, alpha=1.0 / steps)
                else:
                    for step_idx in range(steps):
                        t = torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype)
                        dnn_input_dynamic = pack_ri_channels(x_t, e_frame, y_frame, memory_format=model_memory_format)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input_dynamic,
                            state=flow_states[step_idx],
                            time_cond=t,
                            use_compiled=use_compiled,
                        )
                        x_t = x_t + v / steps
                sync_device(device)

                if frame_idx >= warmup:
                    times_ms.append((time.perf_counter() - start) * 1000.0)

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_se_flow_only",
                "task": "se_flow",
                "device": device.type,
                "steps": steps,
                "predictor_calls_per_frame": 0,
                "flow_calls_per_frame": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "preallocate_model_buffers": preallocate_model_buffers,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": 16.0,
                "budget_ratio_mean": summary["mean_ms"] / 16.0,
            }
        )
        results.append(summary)

    return results


def benchmark_se_full(
    predictor,
    flow,
    device,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    use_compiled: bool,
    dtype,
    sigma_e: float = 0.05,
    preallocate_model_buffers: bool = False,
    model_memory_format: str = "contiguous",
) -> list[dict]:
    """Benchmark the SE initial predictor followed by the SE flow solver."""
    import torch

    predictor = predictor.eval()
    flow = flow.eval()
    results = []
    total_frames = warmup + iterations
    source_y_frames = torch.randn(total_frames, 1, 2, 256, 1, device=device, dtype=dtype)
    source_noise_frames = torch.randn_like(source_y_frames)

    for steps in steps_list:
        predictor_state = predictor.init_state()
        flow_states = [flow.init_state() for _ in range(steps)]
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]
        dnn_input = empty_model_tensor((1, 6, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        x_t_buffer = empty_model_tensor((1, 2, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        times_ms = []

        with torch.inference_mode():
            for frame_idx in range(total_frames):
                y_frame = format_model_tensor(source_y_frames[frame_idx], model_memory_format)
                noise_frame = format_model_tensor(source_noise_frames[frame_idx], model_memory_format)

                sync_device(device)
                start = time.perf_counter()
                e_frame, predictor_state = forward_step(
                    predictor,
                    y_frame,
                    state=predictor_state,
                    use_compiled=use_compiled,
                )
                if preallocate_model_buffers:
                    x_t_buffer.copy_(e_frame)
                    x_t_buffer.add_(noise_frame, alpha=sigma_e)
                    for step_idx in range(steps):
                        pack_ri_channels(x_t_buffer, e_frame, y_frame, out=dnn_input)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input,
                            state=flow_states[step_idx],
                            time_cond=t_tensors[step_idx],
                            use_compiled=use_compiled,
                        )
                        x_t_buffer.add_(v, alpha=1.0 / steps)
                else:
                    x_t = e_frame + sigma_e * noise_frame
                    for step_idx in range(steps):
                        t = torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype)
                        dnn_input_dynamic = pack_ri_channels(x_t, e_frame, y_frame, memory_format=model_memory_format)
                        v, flow_states[step_idx] = forward_step(
                            flow,
                            dnn_input_dynamic,
                            state=flow_states[step_idx],
                            time_cond=t,
                            use_compiled=use_compiled,
                        )
                        x_t = x_t + v / steps
                sync_device(device)

                if frame_idx >= warmup:
                    times_ms.append((time.perf_counter() - start) * 1000.0)

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_se_predictor_plus_flow",
                "task": "se_full",
                "device": device.type,
                "steps": steps,
                "predictor_calls_per_frame": 1,
                "flow_calls_per_frame": steps,
                "total_dnn_calls_per_frame": 1 + steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "preallocate_model_buffers": preallocate_model_buffers,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": 16.0,
                "budget_ratio_mean": summary["mean_ms"] / 16.0,
            }
        )
        results.append(summary)

    return results
