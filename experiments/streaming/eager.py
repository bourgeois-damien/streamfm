"""Eager streaming pipelines.

Plain frame-by-frame streaming audio loops for the STFTPR model and the SE
model, without graph capture.
"""

from __future__ import annotations

import time

import torch

from experiments.core.tensors import empty_model_tensor, format_model_tensor, pack_ri_channels
from experiments.core.streaming_state import forward_step
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


def run_streaming_audio_pipeline(
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
    preallocate_model_buffers: bool = False,
    model_memory_format: str = "contiguous",
    return_audio: bool = False,
) -> dict:
    """Simulate streaming STFTPR over real audio: STFT -> flow solver -> ISTFT per frame.

    Times each stage separately (stft/model/istft plus total) and returns one
    summary dict with prefixed metrics; with ``return_audio`` the enhanced
    waveform (CPU tensor) is included so quality can be checked, not just speed.
    """
    assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio shaped [1, T]."
    assert steps > 0

    # 1) Deterministic setup: fixed seed so noise (and thus output audio) is
    # reproducible across runs and executions.
    torch.manual_seed(seed)
    flow = flow.eval()
    audio = audio.to(device)
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)

    # 2) Pad the audio to exactly (warmup + iterations) hops.
    total_frames = warmup + iterations
    required_samples = total_frames * config.hop_length
    if audio.shape[-1] < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - audio.shape[-1]))

    # 3) Streaming buffers. input_buffer [1, n_fft] is the sliding analysis
    # window; output/denom accumulate the overlap-add reconstruction and the
    # summed squared synthesis windows used to normalize it at the end.
    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)
    # One recurrent state per solver step: each step k is its own causal pass
    # over the frame sequence, so its state must persist across frames.
    flow_states = [flow.init_state() for _ in range(steps)]
    freq_bins = frequency_bins(config)
    # Noise pre-generated for all frames [total, 1, 2, F, 1] so RNG cost never
    # lands inside the timed region.
    noise_frames = torch.randn(total_frames, 1, 2, freq_bins, 1, device=device, dtype=model_dtype)
    # Flow times t_k = k/steps, constant across frames -> build once.
    t_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    # Reused DNN input [1, 4, F, 1] = pack(x_t, y) and solver state x_t [1, 2, F, 1].
    dnn_input = empty_model_tensor((1, 4, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    x_t_buffer = empty_model_tensor((1, 2, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)

    stft_times: list[float] = []
    model_times: list[float] = []
    istft_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
        for frame_idx in range(total_frames):
            # 4a) Slide the analysis window by one hop of fresh samples.
            chunk_start = frame_idx * config.hop_length
            chunk = audio[:, chunk_start:chunk_start + config.hop_length]
            input_buffer = torch.cat([input_buffer[:, config.hop_length:], chunk], dim=-1)

            # Each stage is bracketed by sync_device so the per-stage times
            # measure finished GPU work, not async queue submission.
            sync_device(device)
            total_start = time.perf_counter()

            # 4b) STFT + compression. Phase retrieval: the model is
            # conditioned on the magnitude ONLY (abs) and must reconstruct
            # the phase itself.
            stft_start = time.perf_counter()
            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, :-config.cut_highest_freqs]
            y_complex = compress_complex(spectrum, config)
            y_condition = y_complex.abs().to(y_complex.dtype)
            y_frame = complex_to_ri_frame(y_condition)
            sync_device(device)
            stft_ms = (time.perf_counter() - stft_start) * 1000.0

            # 4c) Euler flow solver: x_0 ~ y + sigma_y*noise, then
            # x_{k+1} = x_k + v(x_k, y, t_k)/steps for k = 0..steps-1.
            model_start = time.perf_counter()
            y_frame_model = format_model_tensor(y_frame.to(model_dtype), model_memory_format)
            if preallocate_model_buffers:
                x_t_buffer.copy_(y_frame_model)
                x_t_buffer.add_(format_model_tensor(noise_frames[frame_idx], model_memory_format), alpha=config.sigma_y)
                for step_idx in range(steps):
                    pack_ri_channels(x_t_buffer, y_frame_model, out=dnn_input)
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input,
                        state=flow_states[step_idx],
                        time_cond=t_tensors[step_idx],
                        use_compiled=use_compiled,
                    )
                    x_t_buffer.add_(v, alpha=1.0 / steps)
                model_output = x_t_buffer
            else:
                x_t = y_frame_model + config.sigma_y * format_model_tensor(torch.randn_like(y_frame_model), model_memory_format)
                for step_idx in range(steps):
                    t = torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
                    dnn_input_dynamic = pack_ri_channels(x_t, y_frame_model, memory_format=model_memory_format)
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input_dynamic,
                        state=flow_states[step_idx],
                        time_cond=t,
                        use_compiled=use_compiled,
                    )
                    x_t = x_t + v / steps
                model_output = x_t
            sync_device(device)
            model_ms = (time.perf_counter() - model_start) * 1000.0

            # 4d) ISTFT + overlap-add of this frame's n_fft samples.
            istft_start = time.perf_counter()
            x_complex = ri_frame_to_complex(model_output.float())
            x_complex = decompress_complex(x_complex, config)
            x_complex = pad_cut_highest_freqs(x_complex, config)
            frame_audio = torch.fft.irfft(x_complex, n=config.n_fft, norm=norm) * window
            out_start = frame_idx * config.hop_length
            output[:, out_start:out_start + config.n_fft] += frame_audio
            denom[:, out_start:out_start + config.n_fft] += window.square()
            sync_device(device)
            istft_ms = (time.perf_counter() - istft_start) * 1000.0

            # Warmup frames run the exact same code but are not recorded
            # (first frames pay lazy init, cache misses, compilation).
            total_ms = (time.perf_counter() - total_start) * 1000.0
            if frame_idx >= warmup:
                stft_times.append(stft_ms)
                model_times.append(model_ms)
                istft_times.append(istft_ms)
                total_times.append(total_ms)

    # 5) Overlap-add normalization: divide by the accumulated squared window
    # so overlapping frames sum to unity gain (clamp avoids 0/0 at the edges).
    output = output[:, :required_samples]
    denom = denom[:, :required_samples].clamp_min(1e-8)
    output = output / denom

    summary = {
        "mode": "streaming_audio_pipeline",
        "steps": steps,
        "iterations": iterations,
        "warmup": warmup,
        "compiled": use_compiled,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "preallocate_model_buffers": preallocate_model_buffers,
        "model_memory_format": model_memory_format,
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


def run_streaming_se_audio_pipeline(
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
    preallocate_model_buffers: bool = False,
    model_memory_format: str = "contiguous",
    return_audio: bool = False,
) -> dict:
    """Simulate streaming SE over real audio: STFT -> predictor -> flow -> ISTFT per frame.

    Same structure and timing discipline as run_streaming_audio_pipeline; the
    SE differences are: conditioning keeps the full complex spectrum (phase
    included), an initial predictor DNN estimates e from y each frame, the
    solver starts at x_0 = e + sigma_e*noise, and the flow input packs three
    frames (x_t, e, y) into 6 channels instead of two into 4.
    """
    assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio shaped [1, T]."
    assert steps > 0

    torch.manual_seed(seed)
    predictor = predictor.eval()
    flow = flow.eval()
    audio = audio.to(device)
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)

    total_frames = warmup + iterations
    required_samples = total_frames * config.hop_length
    if audio.shape[-1] < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - audio.shape[-1]))

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)
    predictor_state = predictor.init_state()
    flow_states = [flow.init_state() for _ in range(steps)]
    freq_bins = frequency_bins(config)
    noise_frames = torch.randn(total_frames, 1, 2, freq_bins, 1, device=device, dtype=model_dtype)
    t_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    dnn_input = empty_model_tensor((1, 6, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)
    x_t_buffer = empty_model_tensor((1, 2, freq_bins, 1), device=device, dtype=model_dtype, memory_format=model_memory_format)

    stft_times: list[float] = []
    model_times: list[float] = []
    istft_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
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
            # Unlike STFTPR, SE keeps the complex spectrum: phase is available,
            # the task is denoising, not phase retrieval.
            y_complex = compress_complex(spectrum, config)
            y_frame = complex_to_ri_frame(y_complex)
            sync_device(device)
            stft_ms = (time.perf_counter() - stft_start) * 1000.0

            # Predictor gives the initial estimate e from noisy y, then the
            # flow refines from x_0 = e + sigma_e*noise.
            model_start = time.perf_counter()
            y_frame_model = format_model_tensor(y_frame.to(model_dtype), model_memory_format)
            e_frame, predictor_state = forward_step(
                predictor,
                y_frame_model,
                state=predictor_state,
                use_compiled=use_compiled,
            )
            e_frame = format_model_tensor(e_frame, model_memory_format)
            if preallocate_model_buffers:
                x_t_buffer.copy_(e_frame)
                x_t_buffer.add_(format_model_tensor(noise_frames[frame_idx], model_memory_format), alpha=sigma_e)
                for step_idx in range(steps):
                    pack_ri_channels(x_t_buffer, e_frame, y_frame_model, out=dnn_input)
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input,
                        state=flow_states[step_idx],
                        time_cond=t_tensors[step_idx],
                        use_compiled=use_compiled,
                    )
                    x_t_buffer.add_(v, alpha=1.0 / steps)
                model_output = x_t_buffer
            else:
                x_t = e_frame + sigma_e * format_model_tensor(noise_frames[frame_idx], model_memory_format)
                for step_idx in range(steps):
                    t = torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
                    dnn_input_dynamic = pack_ri_channels(x_t, e_frame, y_frame_model, memory_format=model_memory_format)
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input_dynamic,
                        state=flow_states[step_idx],
                        time_cond=t,
                        use_compiled=use_compiled,
                    )
                    x_t = x_t + v / steps
                model_output = x_t
            sync_device(device)
            model_ms = (time.perf_counter() - model_start) * 1000.0

            istft_start = time.perf_counter()
            x_complex = ri_frame_to_complex(model_output.float())
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
        "mode": "streaming_se_audio_pipeline",
        "steps": steps,
        "predictor_calls_per_frame": 1,
        "flow_calls_per_frame": steps,
        "total_dnn_calls_per_frame": 1 + steps,
        "iterations": iterations,
        "warmup": warmup,
        "compiled": use_compiled,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "preallocate_model_buffers": preallocate_model_buffers,
        "model_memory_format": model_memory_format,
        "pre_generated_noise": True,
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
