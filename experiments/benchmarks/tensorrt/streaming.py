"""TensorRT adapter for StreamFM's recurrent ``forward_step`` contract.

The adapter is deliberately shaped like a regular streaming model: callers
still use ``init_state`` and ``forward_step``.  Unlike the old fixed-window
probe, its TensorRT engine receives every causal buffer and returns the next
version of every buffer on each frame.
"""

from __future__ import annotations

import time
import os
from typing import Any


def _flatten_tensor_state(value: Any) -> list:
    """Flatten the nested streaming-state structure into an ordered tensor list.

    The TensorRT engine signature is a flat list of tensors, so the nested
    state (lists of lists per causal conv) must round-trip through
    flatten/unflatten in a stable order.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in _flatten_tensor_state(child)]
    if value is None:
        return []
    raise TypeError(f"Unsupported streaming-state leaf: {type(value)!r}")


def _unflatten_tensor_state(template: Any, values) -> Any:
    """Rebuild the nested state structure from a flat tensor iterator (inverse of _flatten_tensor_state)."""
    import torch

    if isinstance(template, torch.Tensor):
        return next(values)
    if isinstance(template, list):
        return [_unflatten_tensor_state(child, values) for child in template]
    if isinstance(template, tuple):
        return tuple(_unflatten_tensor_state(child, values) for child in template)
    if template is None:
        return None
    raise TypeError(f"Unsupported streaming-state leaf: {type(template)!r}")


def _enable_functional_state_updates(model) -> None:
    """Make causal-convolution state transitions exportable without mutation."""
    from sgmse.backbones.streaming_unet import CausalConv2d, CausalDecoupledConv2d

    for module in model.modules():
        if isinstance(module, (CausalConv2d, CausalDecoupledConv2d)):
            module.functional_state_updates = True


def _make_step_module(model, state_template):
    """Wrap the model as a pure module: (x, t, *flat_state) -> (y, *flat_next_state).

    torch.export and TensorRT need a stateless callable with only tensors in
    the signature; __wrapped__ bypasses the torch.compile decorator on
    forward_step (same trick as core/streaming_state.py).
    """
    import torch

    raw_step = getattr(type(model).forward_step, "__wrapped__", None)
    if raw_step is None:
        raise RuntimeError("TensorRT requires the raw CausalNCSNpp.forward_step implementation.")

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x, time_cond, *flat_state):
            state = _unflatten_tensor_state(state_template, iter(flat_state))
            y, next_state = raw_step(self.model, x, time_cond=time_cond, state=state)
            return (y, *_flatten_tensor_state(next_state))

    return Step().eval()


def _preserve_modelopt_amax_buffers_for_export() -> None:
    """Avoid ModelOpt 0.17 turning calibrated Q/DQ scales into fake tensors."""
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    from modelopt.torch.quantization.utils import is_torch_export_mode

    original_get_amax = TensorQuantizer._get_amax

    def _get_amax_preserving_buffer(self, inputs):
        if is_torch_export_mode():
            amax = self._buffers.get("_amax")
            if amax is not None:
                return amax if amax.device == inputs.device else amax.to(inputs.device)
        return original_get_amax(self, inputs)

    TensorQuantizer._get_amax = _get_amax_preserving_buffer


def _apply_tensorrt_int8_ptq(
    model, *, input_channels: int, input_freqs: int, calibration_steps: int
):
    """Calibrate ModelOpt Q/DQ on actual one-frame streaming calls."""
    import torch
    import modelopt.torch.quantization as mtq

    raw_step = getattr(type(model).forward_step, "__wrapped__", None)
    if raw_step is None:
        raise RuntimeError("TensorRT INT8 requires the raw CausalNCSNpp.forward_step implementation.")

    def calibrate(module):
        with torch.inference_mode():
            # These are calibration samples, not latency samples.  Each starts
            # from a fresh streaming state and uses the true one-frame API.
            for _ in range(calibration_steps):
                state = module.init_state()
                x = torch.randn(1, input_channels, input_freqs, 1, device="cuda")
                t = torch.rand(1, device="cuda")
                raw_step(module, x, time_cond=t, state=state)

    return mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)


class TensorRTStreamingAdapter:
    """A stateful TensorRT FP32, FP16, or calibrated INT8 streaming engine."""

    def __init__(
        self,
        model,
        *,
        dtype,
        precision: str = "fp16",
        calibration_steps: int = 16,
        use_cuda_graph: bool = False,
        memory_format: str = "contiguous",
    ):
        import torch

        # 1) Validate the precision/dtype pairing before any expensive work.
        if not torch.cuda.is_available():
            raise ValueError("TensorRT streaming requires CUDA.")
        if precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("TensorRT precision must be 'fp32', 'fp16', or 'int8'.")
        if precision == "fp32" and dtype != torch.float32:
            raise ValueError("TensorRT FP32 requires --dtype fp32.")
        if precision == "fp16" and dtype != torch.float16:
            raise ValueError("TensorRT FP16 requires --dtype fp16.")
        if precision == "int8" and dtype != torch.float32:
            raise ValueError(
                "TensorRT INT8 PTQ uses FP32 model I/O and requires --dtype fp32; "
                "the engine quantizes supported operations internally."
            )

        try:
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT streaming requires torch-tensorrt in the benchmark environment."
            ) from exc

        # 2) Prepare the model: functional (non-mutating) state updates so
        # torch.export can trace it, then optional INT8 Q/DQ calibration.
        self.model = model.eval().requires_grad_(False)
        _enable_functional_state_updates(self.model)
        self.precision = precision
        self.use_cuda_graph = use_cuda_graph
        self.dtype = dtype
        self.memory_format = memory_format
        self.input_freqs = int(self.model.input_freqs)
        self.input_channels = int(self.model.input_layer.in_channels)
        if precision == "int8":
            self.model = _apply_tensorrt_int8_ptq(
                self.model,
                input_channels=self.input_channels,
                input_freqs=self.input_freqs,
                calibration_steps=calibration_steps,
            ).eval().requires_grad_(False)
            _preserve_modelopt_amax_buffers_for_export()
        # 3) Build the pure step module and example inputs for export.
        self._template = self.model.init_state()
        self._initial_state = tuple(_flatten_tensor_state(self._template))
        self._step = _make_step_module(self.model, self._template)
        torch_memory_format = (
            torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
        )
        x = torch.randn(
            1, self.input_channels, self.input_freqs, 1, device="cuda", dtype=dtype
        ).contiguous(memory_format=torch_memory_format)
        time_cond = torch.full((1,), 0.5, device="cuda", dtype=dtype)

        # 4) Export before TensorRT compilation: it fixes the 63-state
        # signature and makes every state transition visible to the compiler.
        if precision == "int8":
            from modelopt.torch.quantization.utils import export_torch_mode

            with export_torch_mode():
                program = torch.export.export(self._step, (x, time_cond, *self._initial_state), strict=True)
            self.engine = torch_tensorrt.dynamo.compile(
                program,
                arg_inputs=[x, time_cond, *self._initial_state],
                min_block_size=1,
                require_full_compilation=(
                    os.environ.get("STREAMFM_TRT_REQUIRE_FULL_COMPILATION", "0") == "1"
                ),
            )
        else:
            program = torch.export.export(self._step, (x, time_cond, *self._initial_state), strict=True)
            compile_kwargs = {
                "arg_inputs": [x, time_cond, *self._initial_state],
                "min_block_size": 1,
                "require_full_compilation": (
                    os.environ.get("STREAMFM_TRT_REQUIRE_FULL_COMPILATION", "0") == "1"
                ),
            }
            if precision == "fp16":
                compile_kwargs["enabled_precisions"] = {torch.float16}
            # Omit enabled_precisions for FP32: TensorRT's default is its
            # native FP32 path, without allowing FP16 tactics implicitly.
            self.engine = torch_tensorrt.dynamo.compile(program, **compile_kwargs)
        # 5) Self-diagnostics recorded at build time: partition inventory,
        # engine-vs-eager agreement and micro-profiles all land in the
        # benchmark rows so a bad build is visible in the history.
        self.compilation_profile = self._inspect_compiled_engine()
        # ``use_cuda_graph`` deliberately does *not* enable Torch-TensorRT's
        # engine-only CUDA-graph wrapper here.  The benchmark captures a
        # single native CUDA graph around the whole recurrent solver: RI
        # packing, TensorRT, all recurrent-state copies and the flow update.
        # Nesting an engine graph inside that outer graph would neither cover
        # the surrounding work nor give us a meaningful latency comparison.
        self.validation = self._validate_one_step(x, time_cond)
        self.runtime_profile = self._profile_engine(x, time_cond)
        self.stage_profile = self._profile_full_solver_stages(x, time_cond)

    def _inspect_compiled_engine(self) -> dict[str, Any]:
        """Inventory TensorRT and fallback submodules in the compiled graph."""
        module_types = {}
        named_modules = []
        for name, module in self.engine.named_modules():
            type_name = f"{type(module).__module__}.{type(module).__name__}"
            module_types[type_name] = module_types.get(type_name, 0) + 1
            if name:
                named_modules.append({"name": name, "type": type_name})
        lowered = [
            item
            for item in named_modules
            if "tensorrt" in item["type"].lower() or "run_on_acc" in item["name"].lower()
        ]
        fallbacks = [
            item
            for item in named_modules
            if "run_on_gpu" in item["name"].lower() or "fallback" in item["type"].lower()
        ]
        return {
            "require_full_compilation": (
                os.environ.get("STREAMFM_TRT_REQUIRE_FULL_COMPILATION", "0") == "1"
            ),
            "compiled_module_types": module_types,
            "tensorrt_partition_count": len(lowered),
            "pytorch_fallback_partition_count": len(fallbacks),
            "tensorrt_partitions": lowered,
            "pytorch_fallback_partitions": fallbacks,
        }

    def _validate_one_step(self, x, time_cond) -> dict[str, float]:
        """Compare one engine step against the exported PyTorch step (output and every state tensor)."""
        import torch

        with torch.inference_mode():
            eager_state = tuple(tensor.clone() for tensor in self._initial_state)
            engine_state = tuple(tensor.clone() for tensor in self._initial_state)
            expected = self._step(x, time_cond, *eager_state)
            actual = self.engine(x, time_cond, *engine_state)
        output_delta = (actual[0].float() - expected[0].float()).abs()
        state_max = max(
            float((actual_tensor.float() - expected_tensor.float()).abs().max())
            for actual_tensor, expected_tensor in zip(actual[1:], expected[1:])
        )
        return {
            "precision": self.precision,
            "cuda_graph": self.use_cuda_graph,
            "output_mean_abs_diff": float(output_delta.mean()),
            "output_max_abs_diff": float(output_delta.max()),
            "state_max_abs_diff": state_max,
            "state_tensor_count": len(self._initial_state),
        }

    def _profile_engine(self, x, time_cond, *, warmup: int = 10, iterations: int = 100) -> dict[str, float]:
        """Separate host submission time from GPU execution for the TRT call.

        This deliberately excludes the benchmark loop's RI packing and flow
        update.  It measures the engine boundary, including binding/copying its
        63 recurrent state inputs and returning its 63 next-state outputs.
        """
        import torch

        with torch.inference_mode():
            state = self.init_state()
            for _ in range(warmup):
                outputs = self.engine(x, time_cond, *state)
                state = tuple(outputs[1:])
            torch.cuda.synchronize()

            cpu_submit_us = []
            events = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cpu_start = time.perf_counter()
                outputs = self.engine(x, time_cond, *state)
                cpu_submit_us.append((time.perf_counter() - cpu_start) * 1_000_000.0)
                end.record()
                events.append((start, end))
                state = tuple(outputs[1:])
            torch.cuda.synchronize()
        gpu_ms = [float(start.elapsed_time(end)) for start, end in events]
        return {
            "engine_cpu_submit_mean_us": sum(cpu_submit_us) / len(cpu_submit_us),
            "engine_cpu_submit_p50_us": sorted(cpu_submit_us)[len(cpu_submit_us) // 2],
            "engine_gpu_mean_ms": sum(gpu_ms) / len(gpu_ms),
            "engine_gpu_p50_ms": sorted(gpu_ms)[len(gpu_ms) // 2],
        }

    def _profile_full_solver_stages(
        self, x, time_cond, *, warmup: int = 10, iterations: int = 100
    ) -> dict[str, float]:
        """Micro-profile the GPU work around one TensorRT engine call.

        This is deliberately *not* CUDA-graph captured: individual CUDA events
        let us attribute GPU work to input staging, RI packing, the engine,
        functional recurrent-state copies, and the Euler update.  The absolute
        total is not a deployment latency; each section has its own launch.
        The captured benchmark remains the deployment-latency measurement.
        """
        import torch

        from experiments.core.tensors import empty_model_tensor, pack_ri_channels

        if self.input_channels != 4:
            return {"stage_profile_available": False}

        y = empty_model_tensor(
            (1, 2, self.input_freqs, 1),
            device="cuda",
            dtype=self.dtype,
            memory_format=self.memory_format,
        )
        y.normal_()
        x_t = torch.empty_like(y)
        dnn_input = empty_model_tensor(
            (1, self.input_channels, self.input_freqs, 1),
            device="cuda",
            dtype=self.dtype,
            memory_format=self.memory_format,
        )
        state = self.init_state()
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state)

        def run_once(events=None, cpu_enqueue_us=None):
            def section(name, fn):
                start = torch.cuda.Event(enable_timing=True) if events is not None else None
                end = torch.cuda.Event(enable_timing=True) if events is not None else None
                if start is not None:
                    start.record()
                cpu_start = time.perf_counter()
                value = fn()
                if cpu_enqueue_us is not None:
                    cpu_enqueue_us[name].append((time.perf_counter() - cpu_start) * 1_000_000.0)
                if end is not None:
                    end.record()
                    events[name].append((start, end))
                return value

            section("input_stage", lambda: x_t.copy_(y))
            section("ri_packing", lambda: pack_ri_channels(x_t, y, out=dnn_input))
            outputs = section("engine", lambda: self.engine(dnn_input, time_cond, *state))

            def copy_next_state():
                for destination, source in zip(state, outputs[1:]):
                    destination.copy_(source)

            section("state_handoff", copy_next_state)
            section("euler_update", lambda: x_t.add_(outputs[0]))

        with torch.inference_mode():
            for _ in range(warmup):
                run_once()
            torch.cuda.synchronize()

            sections = ("input_stage", "ri_packing", "engine", "state_handoff", "euler_update")
            events = {name: [] for name in sections}
            cpu_enqueue_us = {name: [] for name in sections}
            frame_events = []
            for _ in range(iterations):
                frame_start = torch.cuda.Event(enable_timing=True)
                frame_end = torch.cuda.Event(enable_timing=True)
                frame_start.record()
                run_once(events=events, cpu_enqueue_us=cpu_enqueue_us)
                frame_end.record()
                frame_events.append((frame_start, frame_end))
            torch.cuda.synchronize()

        def summary(name: str) -> dict[str, float]:
            gpu = [float(start.elapsed_time(end)) for start, end in events[name]]
            cpu = cpu_enqueue_us[name]
            return {
                f"stage_{name}_gpu_mean_ms": sum(gpu) / len(gpu),
                f"stage_{name}_gpu_p50_ms": sorted(gpu)[len(gpu) // 2],
                f"stage_{name}_cpu_enqueue_mean_us": sum(cpu) / len(cpu),
            }

        report = {
            "stage_profile_available": True,
            "stage_profile_iterations": iterations,
            "stage_profile_steps": 1,
            "state_tensor_count": len(state),
            "state_total_bytes": state_bytes,
            "state_total_mib": state_bytes / (1024.0 * 1024.0),
        }
        for section_name in events:
            report.update(summary(section_name))
        frame_gpu = [float(start.elapsed_time(end)) for start, end in frame_events]
        report["stage_full_uncaptured_gpu_mean_ms"] = sum(frame_gpu) / len(frame_gpu)
        report["stage_full_uncaptured_gpu_p50_ms"] = sorted(frame_gpu)[len(frame_gpu) // 2]
        return report

    def init_state(self):
        """Return fresh engine-I/O buffers, matching normal model ownership."""
        return tuple(tensor.clone() for tensor in self._initial_state)

    def reset_state_(self, state) -> None:
        """Restore state buffers in place, preserving CUDA-graph addresses."""
        if len(state) != len(self._initial_state):
            raise ValueError("Unexpected TensorRT streaming-state length.")
        for destination, source in zip(state, self._initial_state):
            destination.copy_(source)

    def eval(self):
        """Match ``nn.Module.eval`` so the existing streaming loops are unchanged."""
        return self

    def forward_step(self, x, *, time_cond=None, state):
        if time_cond is None:
            raise ValueError("TensorRT flow adapter requires the step time tensor.")
        outputs = self.engine(x, time_cond, *state)
        return outputs[0], tuple(outputs[1:])


def build_tensorrt_streaming_adapter(
    model,
    *,
    dtype,
    precision: str = "fp16",
    calibration_steps: int = 16,
    use_cuda_graph: bool = False,
    memory_format: str = "contiguous",
):
    """Compile a StreamFM backbone for recurrent, one-frame TensorRT inference."""
    return TensorRTStreamingAdapter(
        model,
        dtype=dtype,
        precision=precision,
        calibration_steps=calibration_steps,
        use_cuda_graph=use_cuda_graph,
        memory_format=memory_format,
    )


def benchmark_tensorrt_flow_steps_cuda_graph(
    adapter: TensorRTStreamingAdapter,
    device,
    steps_list: tuple[int, ...],
    iterations: int,
    warmup: int,
    dtype,
    model_memory_format: str = "contiguous",
    freq_bins: int = 256,
    frame_budget_ms: float = 16.0,
) -> list[dict]:
    """Benchmark one *whole* recurrent TensorRT solver CUDA graph per frame.

    TensorRT's engine is a pure function with 63 causal states as inputs and
    63 next states as outputs.  The copies from those outputs into stable
    state buffers must therefore be captured alongside the engine.  This is
    intentionally separate from the engine-only Torch-TensorRT graph option.
    """
    import torch

    from experiments.core.tensors import empty_model_tensor, pack_ri_channels
    from experiments.core.timing import summarize_ms
    from experiments.benchmarks.cuda_profile_range import CudaProfileRange

    if device.type != "cuda":
        raise ValueError("TensorRT CUDA Graph execution requires CUDA.")
    if not adapter.use_cuda_graph:
        raise ValueError("The TensorRT adapter was not configured for CUDA Graph execution.")

    results = []
    total_frames = warmup + iterations
    for steps in steps_list:
        static_y_frame = empty_model_tensor(
            (1, 2, freq_bins, 1),
            device=device,
            dtype=dtype,
            memory_format=model_memory_format,
        )
        static_output = torch.empty_like(static_y_frame)
        static_x_t = torch.empty_like(static_y_frame)
        static_dnn_input = empty_model_tensor(
            (1, 4, freq_bins, 1),
            device=device,
            dtype=dtype,
            memory_format=model_memory_format,
        )
        source_frames = torch.randn(
            total_frames, 1, 2, freq_bins, 1, device=device, dtype=dtype
        )
        time_tensors = [
            torch.full((1,), step_idx / max(steps, 1), device=device, dtype=dtype)
            for step_idx in range(steps)
        ]
        flow_states = [adapter.init_state() for _ in range(steps)]

        def reset_states() -> None:
            for state in flow_states:
                adapter.reset_state_(state)

        def run_solver() -> None:
            static_x_t.copy_(static_y_frame)
            for step_idx in range(steps):
                pack_ri_channels(static_x_t, static_y_frame, out=static_dnn_input)
                outputs = adapter.engine(
                    static_dnn_input, time_tensors[step_idx], *flow_states[step_idx]
                )
                # Keep the recurrent input addresses fixed across replays.
                # The TensorRT output buffers may be transient, but these
                # copies and the subsequent use are both graph nodes.
                for state_buffer, next_state in zip(flow_states[step_idx], outputs[1:]):
                    state_buffer.copy_(next_state)
                static_x_t.add_(outputs[0], alpha=1.0 / steps)
            static_output.copy_(static_x_t)

        with torch.inference_mode():
            # Finish TensorRT lazy setup before capture.  Reusing graph-safe
            # static buffers is essential: CUDA Graph replays fixed pointers.
            static_y_frame.copy_(source_frames[0])
            for _ in range(3):
                run_solver()
            torch.cuda.synchronize()
            reset_states()

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                run_solver()
            torch.cuda.synchronize()
            reset_states()

            # A direct engine invocation and a graph replay must agree from
            # identical input/state buffers.  This catches accidental state
            # aliasing before latency numbers are reported.
            static_y_frame.copy_(source_frames[0])
            run_solver()
            expected_output = static_output.clone()
            expected_states = [tuple(tensor.clone() for tensor in state) for state in flow_states]
            reset_states()
            static_y_frame.copy_(source_frames[0])
            graph.replay()
            torch.cuda.synchronize()
            output_delta = (static_output.float() - expected_output.float()).abs()
            state_max_abs_diff = max(
                float((actual.float() - expected.float()).abs().max())
                for actual_state, expected_state in zip(flow_states, expected_states)
                for actual, expected in zip(actual_state, expected_state)
            )

            reset_states()
            for frame_idx in range(warmup):
                static_y_frame.copy_(source_frames[frame_idx])
                graph.replay()
            torch.cuda.synchronize()

            # Same three timing regimes as benchmark_flow_steps_cuda_graph in
            # benchmarks/cuda_graph.py (see comments there): per-frame latency,
            # idle-queue submit cost, batched throughput.
            times_ms = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            profiler_range = CudaProfileRange(
                torch, label=f"tensorrt_{adapter.precision}_cuda_graph"
            )
            profiler_range.start()
            measured_start = time.perf_counter()
            try:
                for measured_idx, frame_idx in enumerate(range(warmup, total_frames)):
                    with profiler_range.frame(measured_idx):
                        start_event.record()
                        static_y_frame.copy_(source_frames[frame_idx])
                        graph.replay()
                        end_event.record()
                        end_event.synchronize()
                        times_ms.append(start_event.elapsed_time(end_event))
                    profiler_range.finish_frame(measured_idx)
            finally:
                profiler_range.close()
            measured_wall_s = time.perf_counter() - measured_start

            reset_states()
            torch.cuda.synchronize()
            idle_copy_submit_us = []
            idle_graph_submit_us = []
            for measured_idx in range(iterations):
                torch.cuda.synchronize()
                cpu_start = time.perf_counter()
                static_y_frame.copy_(source_frames[warmup + measured_idx])
                idle_copy_submit_us.append((time.perf_counter() - cpu_start) * 1_000_000.0)
                cpu_start = time.perf_counter()
                graph.replay()
                idle_graph_submit_us.append((time.perf_counter() - cpu_start) * 1_000_000.0)
            torch.cuda.synchronize()

            submit_us = []
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            batch_start.record()
            batch_wall_start = time.perf_counter()
            for measured_idx in range(iterations):
                cpu_start = time.perf_counter()
                static_y_frame.copy_(source_frames[warmup + measured_idx])
                graph.replay()
                submit_us.append((time.perf_counter() - cpu_start) * 1_000_000.0)
            batch_end.record()
            batch_end.synchronize()
            batch_wall_s = time.perf_counter() - batch_wall_start
            batched_gpu_mean_ms = batch_start.elapsed_time(batch_end) / iterations

        summary = summarize_ms(times_ms)
        summary.update(
            {
                "mode": "frame_step_tensorrt_full_solver_cuda_graph",
                "task": "stftpr",
                "pipeline": "graph_model",
                "device": device.type,
                "steps": steps,
                "iterations": iterations,
                "warmup": warmup,
                "compiled": False,
                "cuda_graph": True,
                "cuda_graph_scope": "tensorrt_full_solver",
                "tensorrt_engine_only_cuda_graph": False,
                "preallocate_model_buffers": True,
                "model_memory_format": model_memory_format,
                "frame_budget_ms": frame_budget_ms,
                "budget_ratio_mean": summary["mean_ms"] / frame_budget_ms,
                "measured_wall_s": measured_wall_s,
                "cuda_graph_cpu_submit_mean_us": sum(submit_us) / len(submit_us),
                "cuda_graph_cpu_submit_p50_us": sorted(submit_us)[len(submit_us) // 2],
                "cuda_graph_idle_copy_submit_mean_us": sum(idle_copy_submit_us) / len(idle_copy_submit_us),
                "cuda_graph_idle_copy_submit_p50_us": sorted(idle_copy_submit_us)[len(idle_copy_submit_us) // 2],
                "cuda_graph_idle_replay_submit_mean_us": sum(idle_graph_submit_us) / len(idle_graph_submit_us),
                "cuda_graph_idle_replay_submit_p50_us": sorted(idle_graph_submit_us)[len(idle_graph_submit_us) // 2],
                "cuda_graph_batched_gpu_mean_ms": batched_gpu_mean_ms,
                "cuda_graph_batched_wall_mean_ms": batch_wall_s * 1000.0 / iterations,
                "cuda_graph_batched_intermediate_sync": False,
                "cuda_graph_engine_output_mean_abs_diff": float(output_delta.mean()),
                "cuda_graph_engine_output_max_abs_diff": float(output_delta.max()),
                "cuda_graph_engine_state_max_abs_diff": state_max_abs_diff,
            }
        )
        results.append(summary)

    return results
