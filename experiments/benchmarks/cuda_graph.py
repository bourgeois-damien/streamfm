from __future__ import annotations

import time

from experiments.common import (
    empty_model_tensor,
    forward_step,
    pack_ri_channels,
    prepare_streaming_state,
    summarize_ms,
    zero_streaming_state,
)


def benchmark_flow_steps_cuda_graph(
    model,
    device,
    steps_list,
    iterations,
    warmup,
    use_compiled,
    dtype,
    model_memory_format: str = "contiguous",
    freq_bins: int = 256,
    frame_budget_ms: float = 16.0,
) -> list[dict]:
    """Benchmark a flow model loop using CUDA Graph replay."""
    import torch

    if device.type != "cuda":
        raise ValueError("CUDA Graph execution requires a CUDA device.")

    flow = model.eval()
    results = []

    static_x_t = None
    static_dnn_input = None

    def run_solver(y_frame, flow_states, t_tensors, steps):
        static_x_t.copy_(y_frame)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, y_frame, out=static_dnn_input)
            v, flow_states[step_idx] = forward_step(
                flow,
                static_dnn_input,
                state=flow_states[step_idx],
                time_cond=t_tensors[step_idx],
                use_compiled=use_compiled,
            )
            static_x_t.add_(v, alpha=1.0 / steps)
        return static_x_t

    for steps in steps_list:
        static_y_frame = empty_model_tensor((1, 2, freq_bins, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        static_y_frame.normal_()
        static_output = torch.empty_like(static_y_frame)
        static_x_t = torch.empty_like(static_y_frame)
        static_dnn_input = empty_model_tensor((1, 4, freq_bins, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        source_frames = torch.randn(warmup + iterations, 1, 2, freq_bins, 1, device=device, dtype=dtype)
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]
        flow_states = [prepare_streaming_state(flow) for _ in range(steps)]

        with torch.inference_mode():
            for _ in range(3):
                static_output.copy_(run_solver(static_y_frame, flow_states, t_tensors, steps))
            torch.cuda.synchronize()
            for state in flow_states:
                zero_streaming_state(flow, state)

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_output.copy_(run_solver(static_y_frame, flow_states, t_tensors, steps))

            for frame_idx in range(warmup):
                static_y_frame.copy_(source_frames[frame_idx])
                graph.replay()
            torch.cuda.synchronize()

            times_ms = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            measured_start = time.perf_counter()
            for frame_idx in range(warmup, warmup + iterations):
                start_event.record()
                static_y_frame.copy_(source_frames[frame_idx])
                graph.replay()
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            measured_wall_s = time.perf_counter() - measured_start

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_flow_only_cuda_graph",
                "task": "stftpr",
                "pipeline": "graph_model",
                "device": device.type,
                "steps": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "cuda_graph": True,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": frame_budget_ms,
                "budget_ratio_mean": summary["mean_ms"] / frame_budget_ms,
                "measured_wall_s": measured_wall_s,
            }
        )
        results.append(summary)

    return results


def benchmark_se_predictor_cuda_graph(predictor, device, iterations, warmup, use_compiled, dtype, model_memory_format: str = "contiguous") -> list[dict]:
    """Benchmark the SE initial predictor DNN using CUDA Graph replay."""
    import torch

    if device.type != "cuda":
        raise ValueError("CUDA Graph execution requires a CUDA device.")

    predictor = predictor.eval()
    total_frames = warmup + iterations
    static_y_frame = empty_model_tensor((1, 2, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
    static_y_frame.normal_()
    static_output = torch.empty_like(static_y_frame)
    source_y_frames = torch.randn(total_frames, 1, 2, 256, 1, device=device, dtype=dtype)
    predictor_state = prepare_streaming_state(predictor)

    def run_predictor(y_frame):
        e_frame, _ = forward_step(predictor, y_frame, state=predictor_state, use_compiled=use_compiled)
        return e_frame

    with torch.inference_mode():
        for _ in range(3):
            static_output.copy_(run_predictor(static_y_frame))
        torch.cuda.synchronize()
        zero_streaming_state(predictor, predictor_state)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_output.copy_(run_predictor(static_y_frame))

        for frame_idx in range(warmup):
            static_y_frame.copy_(source_y_frames[frame_idx])
            graph.replay()
        torch.cuda.synchronize()

        times_ms = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        measured_start = time.perf_counter()
        for frame_idx in range(warmup, total_frames):
            start_event.record()
            static_y_frame.copy_(source_y_frames[frame_idx])
            graph.replay()
            end_event.record()
            end_event.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
        measured_wall_s = time.perf_counter() - measured_start

    summary = summarize_ms(times_ms)
    summary.update(
        {
            "mode": "frame_step_se_predictor_only_cuda_graph",
            "task": "se_predictor",
            "pipeline": "graph_model",
            "device": device.type,
            "steps": 0,
            "predictor_calls_per_frame": 1,
            "flow_calls_per_frame": 0,
            "iterations": iterations,
            "warmup": warmup,
            "compiled": use_compiled,
            "cuda_graph": True,
            "model_memory_format": model_memory_format,
            "frame_budget_ms": 16.0,
            "budget_ratio_mean": summary["mean_ms"] / 16.0,
            "measured_wall_s": measured_wall_s,
        }
    )
    return [summary]


def benchmark_se_flow_cuda_graph(flow, device, steps_list, iterations, warmup, use_compiled, dtype, sigma_e=0.05, model_memory_format: str = "contiguous") -> list[dict]:
    """Benchmark the SE generative flow DNN loop using CUDA Graph replay."""
    import torch

    if device.type != "cuda":
        raise ValueError("CUDA Graph execution requires a CUDA device.")

    flow = flow.eval()
    results = []
    total_frames = warmup + iterations
    source_e_frames = torch.randn(total_frames, 1, 2, 256, 1, device=device, dtype=dtype)
    source_y_frames = torch.randn_like(source_e_frames)
    source_noise_frames = torch.randn_like(source_e_frames)

    static_x_t = None
    static_dnn_input = None

    def run_flow(e_frame, y_frame, noise_frame, flow_states, t_tensors, steps):
        static_x_t.copy_(e_frame)
        static_x_t.add_(noise_frame, alpha=sigma_e)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, e_frame, y_frame, out=static_dnn_input)
            v, flow_states[step_idx] = forward_step(
                flow,
                static_dnn_input,
                state=flow_states[step_idx],
                time_cond=t_tensors[step_idx],
                use_compiled=use_compiled,
            )
            static_x_t.add_(v, alpha=1.0 / steps)
        return static_x_t

    for steps in steps_list:
        static_e_frame = empty_model_tensor((1, 2, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        static_e_frame.normal_()
        static_y_frame = torch.empty_like(static_e_frame)
        static_y_frame.normal_()
        static_noise_frame = torch.empty_like(static_e_frame)
        static_noise_frame.normal_()
        static_output = torch.empty_like(static_e_frame)
        static_x_t = torch.empty_like(static_e_frame)
        static_dnn_input = empty_model_tensor((1, 6, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]
        flow_states = [prepare_streaming_state(flow) for _ in range(steps)]

        with torch.inference_mode():
            for _ in range(3):
                static_output.copy_(run_flow(static_e_frame, static_y_frame, static_noise_frame, flow_states, t_tensors, steps))
            torch.cuda.synchronize()
            for state in flow_states:
                zero_streaming_state(flow, state)

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_output.copy_(run_flow(static_e_frame, static_y_frame, static_noise_frame, flow_states, t_tensors, steps))

            for frame_idx in range(warmup):
                static_e_frame.copy_(source_e_frames[frame_idx])
                static_y_frame.copy_(source_y_frames[frame_idx])
                static_noise_frame.copy_(source_noise_frames[frame_idx])
                graph.replay()
            torch.cuda.synchronize()

            times_ms = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            measured_start = time.perf_counter()
            for frame_idx in range(warmup, total_frames):
                start_event.record()
                static_e_frame.copy_(source_e_frames[frame_idx])
                static_y_frame.copy_(source_y_frames[frame_idx])
                static_noise_frame.copy_(source_noise_frames[frame_idx])
                graph.replay()
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            measured_wall_s = time.perf_counter() - measured_start

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_se_flow_only_cuda_graph",
                "task": "se_flow",
                "pipeline": "graph_model",
                "device": device.type,
                "steps": steps,
                "predictor_calls_per_frame": 0,
                "flow_calls_per_frame": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "cuda_graph": True,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": 16.0,
                "budget_ratio_mean": summary["mean_ms"] / 16.0,
                "measured_wall_s": measured_wall_s,
            }
        )
        results.append(summary)

    return results


def benchmark_se_full_cuda_graph(predictor, flow, device, steps_list, iterations, warmup, use_compiled, dtype, sigma_e=0.05, model_memory_format: str = "contiguous") -> list[dict]:
    """Benchmark the SE predictor plus flow solver using CUDA Graph replay."""
    import torch

    if device.type != "cuda":
        raise ValueError("CUDA Graph execution requires a CUDA device.")

    predictor = predictor.eval()
    flow = flow.eval()
    results = []
    total_frames = warmup + iterations
    source_y_frames = torch.randn(total_frames, 1, 2, 256, 1, device=device, dtype=dtype)
    source_noise_frames = torch.randn_like(source_y_frames)

    static_x_t = None
    static_dnn_input = None

    def run_full(y_frame, noise_frame, predictor_state, flow_states, t_tensors, steps):
        e_frame, _ = forward_step(predictor, y_frame, state=predictor_state, use_compiled=use_compiled)
        static_x_t.copy_(e_frame)
        static_x_t.add_(noise_frame, alpha=sigma_e)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, e_frame, y_frame, out=static_dnn_input)
            v, flow_states[step_idx] = forward_step(
                flow,
                static_dnn_input,
                state=flow_states[step_idx],
                time_cond=t_tensors[step_idx],
                use_compiled=use_compiled,
            )
            static_x_t.add_(v, alpha=1.0 / steps)
        return static_x_t

    for steps in steps_list:
        static_y_frame = empty_model_tensor((1, 2, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        static_y_frame.normal_()
        static_noise_frame = torch.empty_like(static_y_frame)
        static_noise_frame.normal_()
        static_output = torch.empty_like(static_y_frame)
        static_x_t = torch.empty_like(static_y_frame)
        static_dnn_input = empty_model_tensor((1, 6, 256, 1), device=device, dtype=dtype, memory_format=model_memory_format)
        predictor_state = prepare_streaming_state(predictor)
        flow_states = [prepare_streaming_state(flow) for _ in range(steps)]
        t_tensors = [torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype) for step_idx in range(steps)]

        with torch.inference_mode():
            for _ in range(3):
                static_output.copy_(run_full(static_y_frame, static_noise_frame, predictor_state, flow_states, t_tensors, steps))
            torch.cuda.synchronize()
            zero_streaming_state(predictor, predictor_state)
            for state in flow_states:
                zero_streaming_state(flow, state)

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_output.copy_(run_full(static_y_frame, static_noise_frame, predictor_state, flow_states, t_tensors, steps))

            for frame_idx in range(warmup):
                static_y_frame.copy_(source_y_frames[frame_idx])
                static_noise_frame.copy_(source_noise_frames[frame_idx])
                graph.replay()
            torch.cuda.synchronize()

            times_ms = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            measured_start = time.perf_counter()
            for frame_idx in range(warmup, total_frames):
                start_event.record()
                static_y_frame.copy_(source_y_frames[frame_idx])
                static_noise_frame.copy_(source_noise_frames[frame_idx])
                graph.replay()
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            measured_wall_s = time.perf_counter() - measured_start

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_se_predictor_plus_flow_cuda_graph",
                "task": "se_full",
                "pipeline": "graph_model",
                "device": device.type,
                "steps": steps,
                "predictor_calls_per_frame": 1,
                "flow_calls_per_frame": steps,
                "total_dnn_calls_per_frame": 1 + steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": use_compiled,
                "cuda_graph": True,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": 16.0,
                "budget_ratio_mean": summary["mean_ms"] / 16.0,
                "measured_wall_s": measured_wall_s,
            }
        )
        results.append(summary)

    return results
