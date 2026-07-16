"""CUDA Graph streaming pipelines.

Streaming audio loops that replay a captured graph of the model (plus a
TensorRT + CUDA Graph variant), used to measure per-frame latency without
launch overhead.
"""

from __future__ import annotations

import time

import torch

from experiments.core.tensors import empty_model_tensor, format_model_tensor, pack_ri_channels
from experiments.core.streaming_state import (
    forward_step,
    prepare_streaming_state,
    zero_streaming_state,
)
from experiments.core.timing import summarize_prefixed_ms
from experiments.core.devices import sync_device
from experiments.streaming.stft import (
    StreamingSTFTConfig,
    complex_to_ri_frame,
    compress_complex,
    compression_norm,
    decompress_complex,
    frequency_bins,
    pad_cut_highest_freqs,
    ri_frame_to_complex,
    sqrt_hann_window,
)


def run_streaming_audio_pipeline_with_cuda_graph_model(
    flow,
    audio: torch.Tensor,
    device: torch.device,
    steps: int = 1,
    iterations: int = 100,
    warmup: int = 10,
    use_compiled: bool = True,
    config: StreamingSTFTConfig = StreamingSTFTConfig(),
    seed: int = 0,
    model_dtype: torch.dtype = torch.float32,
    model_memory_format: str = "contiguous",
    return_audio: bool = False,
) -> dict:
    """Same pipeline as the eager version, but the whole flow solver is one CUDA Graph replay.

    A CUDA Graph records the solver's kernel sequence once and replays it with
    a single launch per frame, removing per-kernel CPU launch overhead — the
    dominant cost for this small model. Constraint: every tensor the graph
    touches must keep a fixed address, hence the static_* buffers below that
    per-frame data is copied into. STFT/ISTFT stay outside the graph (their
    slicing offsets change every frame).
    """
    assert device.type == "cuda", "CUDA Graph requires a CUDA device."
    assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio shaped [1, T]."
    assert steps > 0

    torch.manual_seed(seed)
    flow = flow.eval().to(dtype=model_dtype)
    audio = audio.to(device)
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)

    total_frames = warmup + iterations
    required_samples = total_frames * config.hop_length
    if audio.shape[-1] < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - audio.shape[-1]))

    # Fixed-address I/O buffers for the graph: y and noise are the graph's
    # inputs (written by copy_ each frame), x_t/dnn_input are solver
    # scratch, static_output is where the result lands after replay.
    freq_bins = frequency_bins(config)
    static_y_frame = empty_model_tensor((1, 2, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    static_noise_frame = torch.empty_like(static_y_frame)
    static_output = torch.empty_like(static_y_frame)
    static_x_t = torch.empty_like(static_y_frame)
    static_dnn_input = empty_model_tensor((1, 4, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    noise_frames = torch.randn(total_frames, *static_y_frame.shape, device=device, dtype=model_dtype)
    t_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    # prepare_streaming_state -> statically allocated recurrent state, so the
    # state tensors captured into the graph keep stable addresses too.
    flow_states = [prepare_streaming_state(flow) for _ in range(steps)]

    def run_solver_with_states(
        y_frame: torch.Tensor,
        noise_frame: torch.Tensor,
        states: list,
    ) -> torch.Tensor:
        """Advance the flow solver for one static CUDA Graph model frame."""
        static_x_t.copy_(y_frame)
        static_x_t.add_(noise_frame, alpha=config.sigma_y)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, y_frame, out=static_dnn_input)
            v, states[step_idx] = forward_step(
                flow,
                static_dnn_input,
                state=states[step_idx],
                time_cond=t_tensors[step_idx],
                use_compiled=use_compiled,
            )
            static_x_t.add_(v, alpha=1.0 / steps)
        return static_x_t

    def run_solver() -> torch.Tensor:
        """Replay the solver using the captured static graph buffers."""
        return run_solver_with_states(static_y_frame, static_noise_frame, flow_states)

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)

    stft_times: list[float] = []
    model_times: list[float] = []
    istft_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
        # 1) Warmup BEFORE capture: cuDNN autotuning, lazy allocations and
        # kernel selection must happen now — anything that allocates during
        # capture would fail or be baked into the graph.
        static_y_frame.normal_()
        static_noise_frame.normal_()
        for _ in range(3):
            static_output.copy_(run_solver())
        torch.cuda.synchronize()
        for state in flow_states:
            zero_streaming_state(flow, state)

        # 2) Capture: record the solver's kernel sequence once.
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_output.copy_(run_solver())

        # 3) Sanity check: replay vs an eager run on identical inputs and
        # freshly zeroed states. The diff stats go into the summary so a
        # silently-corrupt capture shows up in the results instead of
        # producing wrong audio unnoticed.
        check_y_frame = torch.randn_like(static_y_frame)
        check_noise_frame = torch.randn_like(static_noise_frame)
        eager_states = [prepare_streaming_state(flow) for _ in range(steps)]
        for state in flow_states:
            zero_streaming_state(flow, state)
        for state in eager_states:
            zero_streaming_state(flow, state)
        static_y_frame.copy_(check_y_frame)
        static_noise_frame.copy_(check_noise_frame)
        graph.replay()
        torch.cuda.synchronize()
        graph_check_output = static_output.float().detach().clone()
        eager_check_output = run_solver_with_states(
            check_y_frame,
            check_noise_frame,
            eager_states,
        ).float()
        torch.cuda.synchronize()
        graph_eager_abs_diff = (graph_check_output - eager_check_output).abs()
        graph_eager_max_abs_diff = float(graph_eager_abs_diff.max().item())
        graph_eager_mean_abs_diff = float(graph_eager_abs_diff.mean().item())
        graph_eager_ref_mean_abs = float(eager_check_output.abs().mean().item())
        # Zero the states once more: warmup/capture/check polluted the
        # recurrent state, and the streamed audio must start from silence.
        for state in flow_states:
            zero_streaming_state(flow, state)

        # 4) Frame loop — same structure as the eager pipeline, except the
        # model stage is now: copy inputs into static buffers + one replay.
        for frame_idx in range(total_frames):
            chunk_start = frame_idx * config.hop_length
            chunk = audio[:, chunk_start:chunk_start + config.hop_length]
            input_buffer = torch.cat([input_buffer[:, config.hop_length:], chunk], dim=-1)

            sync_device(device)
            total_start = time.perf_counter()

            stft_start = time.perf_counter()
            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, :-config.cut_highest_freqs]
            y_complex = compress_complex(spectrum, config)
            y_condition = y_complex.abs().to(y_complex.dtype)
            y_frame = complex_to_ri_frame(y_condition)
            sync_device(device)
            stft_ms = (time.perf_counter() - stft_start) * 1000.0

            model_start = time.perf_counter()
            static_y_frame.copy_(format_model_tensor(y_frame.to(model_dtype), model_memory_format))
            static_noise_frame.copy_(noise_frames[frame_idx])
            graph.replay()
            sync_device(device)
            model_ms = (time.perf_counter() - model_start) * 1000.0

            istft_start = time.perf_counter()
            x_complex = ri_frame_to_complex(static_output.float())
            x_complex = decompress_complex(x_complex, config)
            x_complex = pad_cut_highest_freqs(x_complex, config)
            frame_audio = torch.fft.irfft(x_complex, n=config.n_fft, norm=norm) * window
            out_start = frame_idx * config.hop_length
            output[:, out_start:out_start + config.n_fft] += frame_audio
            denom[:, out_start:out_start + config.n_fft] += window.square()
            sync_device(device)
            istft_ms = (time.perf_counter() - istft_start) * 1000.0

            total_ms = (time.perf_counter() - total_start) * 1000.0
            if frame_idx >= warmup:
                stft_times.append(stft_ms)
                model_times.append(model_ms)
                istft_times.append(istft_ms)
                total_times.append(total_ms)

    output = output[:, :required_samples]
    denom = denom[:, :required_samples].clamp_min(1e-8)
    output = output / denom

    summary = {
        "mode": "streaming_audio_pipeline_cuda_graph_model",
        "steps": steps,
        "iterations": iterations,
        "warmup": warmup,
        "compiled": use_compiled,
        "cuda_graph": True,
        "cuda_graph_model": True,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "model_memory_format": model_memory_format,
        "pre_generated_noise": True,
        "graph_eager_max_abs_diff": graph_eager_max_abs_diff,
        "graph_eager_mean_abs_diff": graph_eager_mean_abs_diff,
        "graph_eager_ref_mean_abs": graph_eager_ref_mean_abs,
        "frame_budget_ms": 1000.0 * config.hop_length / config.sample_rate,
        "audio_sample_rate": config.sample_rate,
    }
    summary.update(summarize_prefixed_ms(stft_times, "stft"))
    summary.update(summarize_prefixed_ms(model_times, "model"))
    summary.update(summarize_prefixed_ms(istft_times, "istft"))
    summary.update(summarize_prefixed_ms(total_times, "total"))
    summary["budget_ratio_mean"] = summary["total_mean_ms"] / summary["frame_budget_ms"]
    if return_audio:
        summary["audio"] = output.detach().cpu()
    return summary


def run_streaming_audio_pipeline_with_tensorrt_cuda_graph(
    flow,
    audio: torch.Tensor,
    device: torch.device,
    steps: int = 1,
    iterations: int = 100,
    warmup: int = 10,
    config: StreamingSTFTConfig = StreamingSTFTConfig(),
    seed: int = 0,
    model_dtype: torch.dtype = torch.float32,
    model_memory_format: str = "contiguous",
    return_audio: bool = False,
) -> dict:
    """Run audio streaming with one CUDA graph around the full TensorRT solver.

    STFT and overlap-add remain outside the graph because their inputs and
    output offsets change from frame to frame.  The graph does cover every
    flow operation for a frame: noise/input staging, RI packing, TensorRT,
    recurrent-state handoff and Euler update.
    """
    assert device.type == "cuda", "TensorRT CUDA Graph requires CUDA."
    assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio shaped [1, T]."
    assert steps > 0
    if not getattr(flow, "use_cuda_graph", False):
        raise ValueError("TensorRT audio graph requires --execution tensorrt_cuda_graph.")

    torch.manual_seed(seed)
    audio = audio.to(device)
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)
    total_frames = warmup + iterations
    required_samples = total_frames * config.hop_length
    if audio.shape[-1] < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - audio.shape[-1]))

    freq_bins = frequency_bins(config)
    static_y_frame = empty_model_tensor(
        (1, 2, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format
    )
    static_noise_frame = torch.empty_like(static_y_frame)
    static_output = torch.empty_like(static_y_frame)
    static_x_t = torch.empty_like(static_y_frame)
    static_dnn_input = empty_model_tensor(
        (1, 4, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format
    )
    noise_frames = torch.randn(total_frames, *static_y_frame.shape, device=device, dtype=model_dtype)
    time_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    flow_states = [flow.init_state() for _ in range(steps)]

    def reset_states() -> None:
        for state in flow_states:
            flow.reset_state_(state)

    def run_solver() -> None:
        static_x_t.copy_(static_y_frame)
        static_x_t.add_(static_noise_frame, alpha=config.sigma_y)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, static_y_frame, out=static_dnn_input)
            # Raw TensorRT engine call: outputs[0] is the velocity v,
            # outputs[1:] are the next recurrent-state tensors, copied back
            # into the fixed-address state buffers for the next frame.
            outputs = flow.engine(static_dnn_input, time_tensors[step_idx], *flow_states[step_idx])
            for state_buffer, next_state in zip(flow_states[step_idx], outputs[1:]):
                state_buffer.copy_(next_state)
            static_x_t.add_(outputs[0], alpha=1.0 / steps)
        static_output.copy_(static_x_t)

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)
    stft_times: list[float] = []
    model_times: list[float] = []
    istft_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
        # Warmup outside capture, then record — same discipline as the
        # PyTorch graph pipeline above (see comments there).
        static_y_frame.normal_()
        static_noise_frame.normal_()
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

        # Validate graph replay against the same TensorRT engine path.  This
        # is an execution-equivalence check, not an audio-quality evaluation.
        check_y = torch.randn_like(static_y_frame)
        check_noise = torch.randn_like(static_noise_frame)
        static_y_frame.copy_(check_y)
        static_noise_frame.copy_(check_noise)
        run_solver()
        expected = static_output.clone()
        reset_states()
        static_y_frame.copy_(check_y)
        static_noise_frame.copy_(check_noise)
        graph.replay()
        torch.cuda.synchronize()
        graph_engine_diff = (static_output.float() - expected.float()).abs()
        reset_states()

        for frame_idx in range(total_frames):
            chunk_start = frame_idx * config.hop_length
            chunk = audio[:, chunk_start:chunk_start + config.hop_length]
            input_buffer = torch.cat([input_buffer[:, config.hop_length:], chunk], dim=-1)

            sync_device(device)
            total_start = time.perf_counter()
            stft_start = time.perf_counter()
            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, :-config.cut_highest_freqs]
            y_condition = compress_complex(spectrum, config).abs().to(spectrum.dtype)
            y_frame = complex_to_ri_frame(y_condition)
            sync_device(device)
            stft_ms = (time.perf_counter() - stft_start) * 1000.0

            model_start = time.perf_counter()
            static_y_frame.copy_(format_model_tensor(y_frame.to(model_dtype), model_memory_format))
            static_noise_frame.copy_(noise_frames[frame_idx])
            graph.replay()
            sync_device(device)
            model_ms = (time.perf_counter() - model_start) * 1000.0

            istft_start = time.perf_counter()
            x_complex = ri_frame_to_complex(static_output.float())
            x_complex = decompress_complex(x_complex, config)
            x_complex = pad_cut_highest_freqs(x_complex, config)
            frame_audio = torch.fft.irfft(x_complex, n=config.n_fft, norm=norm) * window
            out_start = frame_idx * config.hop_length
            output[:, out_start:out_start + config.n_fft] += frame_audio
            denom[:, out_start:out_start + config.n_fft] += window.square()
            sync_device(device)
            istft_ms = (time.perf_counter() - istft_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0

            if frame_idx >= warmup:
                stft_times.append(stft_ms)
                model_times.append(model_ms)
                istft_times.append(istft_ms)
                total_times.append(total_ms)

    output = output[:, :required_samples]
    output = output / denom[:, :required_samples].clamp_min(1e-8)
    summary = {
        "mode": "streaming_audio_pipeline_tensorrt_full_solver_cuda_graph",
        "steps": steps,
        "iterations": iterations,
        "warmup": warmup,
        "compiled": False,
        "cuda_graph": True,
        "cuda_graph_model": True,
        "cuda_graph_scope": "tensorrt_full_solver",
        "tensorrt_engine_only_cuda_graph": False,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "model_memory_format": model_memory_format,
        "pre_generated_noise": True,
        "cuda_graph_engine_output_mean_abs_diff": float(graph_engine_diff.mean()),
        "cuda_graph_engine_output_max_abs_diff": float(graph_engine_diff.max()),
        "frame_budget_ms": 1000.0 * config.hop_length / config.sample_rate,
        "audio_sample_rate": config.sample_rate,
    }
    summary.update(summarize_prefixed_ms(stft_times, "stft"))
    summary.update(summarize_prefixed_ms(model_times, "model"))
    summary.update(summarize_prefixed_ms(istft_times, "istft"))
    summary.update(summarize_prefixed_ms(total_times, "total"))
    summary["budget_ratio_mean"] = summary["total_mean_ms"] / summary["frame_budget_ms"]
    if return_audio:
        summary["audio"] = output.detach().cpu()
    return summary


def run_streaming_se_audio_pipeline_with_cuda_graph_model(
    predictor,
    flow,
    audio: torch.Tensor,
    device: torch.device,
    steps: int = 1,
    iterations: int = 100,
    warmup: int = 10,
    use_compiled: bool = True,
    config: StreamingSTFTConfig = StreamingSTFTConfig(),
    sigma_e: float = 0.05,
    seed: int = 0,
    model_dtype: torch.dtype = torch.float32,
    model_memory_format: str = "contiguous",
    return_audio: bool = False,
) -> dict:
    """SE audio pipeline with predictor+flow captured as one CUDA Graph.

    Same capture/check/replay structure as
    run_streaming_audio_pipeline_with_cuda_graph_model, with the SE
    differences: complex conditioning, an extra predictor DNN inside the
    graph, x_0 = e + sigma_e*noise, and 6-channel flow input (x_t, e, y).
    """
    assert device.type == "cuda", "CUDA Graph requires a CUDA device."
    assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio shaped [1, T]."
    assert steps > 0

    torch.manual_seed(seed)
    predictor = predictor.eval().to(dtype=model_dtype)
    flow = flow.eval().to(dtype=model_dtype)
    audio = audio.to(device)
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)

    total_frames = warmup + iterations
    required_samples = total_frames * config.hop_length
    if audio.shape[-1] < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - audio.shape[-1]))

    freq_bins = frequency_bins(config)
    static_y_frame = empty_model_tensor((1, 2, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    static_noise_frame = torch.empty_like(static_y_frame)
    static_output = torch.empty_like(static_y_frame)
    static_x_t = torch.empty_like(static_y_frame)
    static_dnn_input = empty_model_tensor((1, 6, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    noise_frames = torch.randn(total_frames, *static_y_frame.shape, device=device, dtype=model_dtype)
    t_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    predictor_state = prepare_streaming_state(predictor)
    flow_states = [prepare_streaming_state(flow) for _ in range(steps)]

    def run_solver_with_states(
        y_frame: torch.Tensor,
        noise_frame: torch.Tensor,
        pred_state,
        states: list,
    ) -> torch.Tensor:
        """Advance SE predictor and flow for one static CUDA Graph model frame."""
        e_frame, pred_state = forward_step(
            predictor,
            y_frame,
            state=pred_state,
            use_compiled=use_compiled,
        )
        static_x_t.copy_(e_frame)
        static_x_t.add_(noise_frame, alpha=sigma_e)
        for step_idx in range(steps):
            pack_ri_channels(static_x_t, e_frame, y_frame, out=static_dnn_input)
            v, states[step_idx] = forward_step(
                flow,
                static_dnn_input,
                state=states[step_idx],
                time_cond=t_tensors[step_idx],
                use_compiled=use_compiled,
            )
            static_x_t.add_(v, alpha=1.0 / steps)
        return static_x_t

    def run_solver() -> torch.Tensor:
        """Replay predictor+flow using the captured static graph buffers."""
        return run_solver_with_states(static_y_frame, static_noise_frame, predictor_state, flow_states)

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)

    stft_times: list[float] = []
    model_times: list[float] = []
    istft_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
        # Warmup -> capture -> replay-vs-eager check -> state reset, exactly
        # as in the STFTPR graph pipeline above (see comments there).
        static_y_frame.normal_()
        static_noise_frame.normal_()
        for _ in range(3):
            static_output.copy_(run_solver())
        torch.cuda.synchronize()
        zero_streaming_state(predictor, predictor_state)
        for state in flow_states:
            zero_streaming_state(flow, state)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_output.copy_(run_solver())

        check_y_frame = torch.randn_like(static_y_frame)
        check_noise_frame = torch.randn_like(static_noise_frame)
        eager_predictor_state = prepare_streaming_state(predictor)
        eager_flow_states = [prepare_streaming_state(flow) for _ in range(steps)]
        zero_streaming_state(predictor, predictor_state)
        zero_streaming_state(predictor, eager_predictor_state)
        for state in flow_states:
            zero_streaming_state(flow, state)
        for state in eager_flow_states:
            zero_streaming_state(flow, state)
        static_y_frame.copy_(check_y_frame)
        static_noise_frame.copy_(check_noise_frame)
        graph.replay()
        torch.cuda.synchronize()
        graph_check_output = static_output.float().detach().clone()
        eager_check_output = run_solver_with_states(
            check_y_frame,
            check_noise_frame,
            eager_predictor_state,
            eager_flow_states,
        ).float()
        torch.cuda.synchronize()
        graph_eager_abs_diff = (graph_check_output - eager_check_output).abs()
        graph_eager_max_abs_diff = float(graph_eager_abs_diff.max().item())
        graph_eager_mean_abs_diff = float(graph_eager_abs_diff.mean().item())
        graph_eager_ref_mean_abs = float(eager_check_output.abs().mean().item())
        zero_streaming_state(predictor, predictor_state)
        for state in flow_states:
            zero_streaming_state(flow, state)

        for frame_idx in range(total_frames):
            chunk_start = frame_idx * config.hop_length
            chunk = audio[:, chunk_start:chunk_start + config.hop_length]
            input_buffer = torch.cat([input_buffer[:, config.hop_length:], chunk], dim=-1)

            sync_device(device)
            total_start = time.perf_counter()

            stft_start = time.perf_counter()
            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, :-config.cut_highest_freqs]
            y_complex = compress_complex(spectrum, config)
            y_frame = complex_to_ri_frame(y_complex)
            sync_device(device)
            stft_ms = (time.perf_counter() - stft_start) * 1000.0

            model_start = time.perf_counter()
            static_y_frame.copy_(format_model_tensor(y_frame.to(model_dtype), model_memory_format))
            static_noise_frame.copy_(noise_frames[frame_idx])
            graph.replay()
            sync_device(device)
            model_ms = (time.perf_counter() - model_start) * 1000.0

            istft_start = time.perf_counter()
            x_complex = ri_frame_to_complex(static_output.float())
            x_complex = decompress_complex(x_complex, config)
            x_complex = pad_cut_highest_freqs(x_complex, config)
            frame_audio = torch.fft.irfft(x_complex, n=config.n_fft, norm=norm) * window
            out_start = frame_idx * config.hop_length
            output[:, out_start:out_start + config.n_fft] += frame_audio
            denom[:, out_start:out_start + config.n_fft] += window.square()
            sync_device(device)
            istft_ms = (time.perf_counter() - istft_start) * 1000.0

            total_ms = (time.perf_counter() - total_start) * 1000.0
            if frame_idx >= warmup:
                stft_times.append(stft_ms)
                model_times.append(model_ms)
                istft_times.append(istft_ms)
                total_times.append(total_ms)

    output = output[:, :required_samples]
    denom = denom[:, :required_samples].clamp_min(1e-8)
    output = output / denom

    summary = {
        "mode": "streaming_se_audio_pipeline_cuda_graph_model",
        "steps": steps,
        "predictor_calls_per_frame": 1,
        "flow_calls_per_frame": steps,
        "total_dnn_calls_per_frame": 1 + steps,
        "iterations": iterations,
        "warmup": warmup,
        "compiled": use_compiled,
        "cuda_graph": True,
        "cuda_graph_model": True,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "model_memory_format": model_memory_format,
        "pre_generated_noise": True,
        "graph_eager_max_abs_diff": graph_eager_max_abs_diff,
        "graph_eager_mean_abs_diff": graph_eager_mean_abs_diff,
        "graph_eager_ref_mean_abs": graph_eager_ref_mean_abs,
        "frame_budget_ms": 1000.0 * config.hop_length / config.sample_rate,
        "audio_sample_rate": config.sample_rate,
    }
    summary.update(summarize_prefixed_ms(stft_times, "stft"))
    summary.update(summarize_prefixed_ms(model_times, "model"))
    summary.update(summarize_prefixed_ms(istft_times, "istft"))
    summary.update(summarize_prefixed_ms(total_times, "total"))
    summary["budget_ratio_mean"] = summary["total_mean_ms"] / summary["frame_budget_ms"]
    if return_audio:
        summary["audio"] = output.detach().cpu()
    return summary
