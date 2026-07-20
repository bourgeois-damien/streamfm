"""Whole-file streaming enhancement, for quality evaluation.

The pipelines in :mod:`experiments.streaming.eager` and
:mod:`experiments.streaming.cuda_graph` exist to measure latency: they run a
fixed number of frames on whatever audio is at hand, split into warmup and
measured, and their per-stage timers dominate the code.  Scoring needs the
opposite shape — every frame of a real file, no timers, the waveform as the
result — so this module reimplements the same math without the instrumentation
rather than bending a benchmark into an evaluator.

The signal path is deliberately identical to those pipelines: sqrt-Hann
analysis, magnitude-only conditioning (STFTPR reconstructs phase), Euler flow
solver over the causal backbone, weighted overlap-add.  The one thing this
module adds is an explicit account of algorithmic delay, which a latency
benchmark never has to care about but which decides whether a metric like
PESQ or SI-SDR means anything: frame ``f`` analyses input samples
``[(f+1)*hop - n_fft, (f+1)*hop)`` and is written at output offset ``f*hop``,
so the reconstruction lags the input by ``n_fft - hop`` samples.  See
:func:`streaming_algorithmic_delay`.
"""

from __future__ import annotations

import torch

from experiments.core.streaming_state import forward_step
from experiments.core.tensors import format_model_tensor, pack_ri_channels
from experiments.streaming.stft import (
    StreamingSTFTConfig,
    complex_to_ri_frame,
    compress_complex,
    compression_norm,
    decompress_complex,
    pad_cut_highest_freqs,
    ri_frame_to_complex,
    sqrt_hann_window,
)


def streaming_algorithmic_delay(config: StreamingSTFTConfig) -> int:
    """Samples by which the streamed reconstruction lags its input."""
    return config.n_fft - config.hop_length


def streaming_num_frames(num_samples: int, config: StreamingSTFTConfig) -> int:
    """Frames needed to emit at least ``num_samples`` aligned output samples."""
    # One extra frame beyond ceil(): the delay pushes the last useful samples
    # past the frame that would suffice if the pipeline were delay-free.
    span = num_samples + streaming_algorithmic_delay(config)
    return -(-span // config.hop_length) + 1


def streaming_enhance(
    flow,
    audio: torch.Tensor,
    *,
    device: torch.device,
    steps: int = 1,
    config: StreamingSTFTConfig = StreamingSTFTConfig(),
    seed: int = 0,
    model_dtype: torch.dtype = torch.float32,
    model_memory_format: str = "contiguous",
    use_compiled: bool = False,
    use_engine: bool = False,
    compensate_delay: bool = True,
) -> torch.Tensor:
    """Enhance one mono file frame by frame and return the waveform.

    ``use_engine`` calls ``flow.engine`` directly (the TensorRT adapter, whose
    engine takes the 63 causal states as explicit inputs and returns their
    successors); otherwise the causal PyTorch ``forward_step`` is used.  Both
    consume the same state objects, so the choice does not change the math —
    which is the point: it lets a quantized engine be scored against an eager
    baseline produced by this very function.

    With ``compensate_delay`` the returned waveform is trimmed so sample ``i``
    corresponds to input sample ``i``, and is truncated to the input length.
    Without it the raw overlap-add buffer is returned, delay included.
    """
    if audio.ndim != 2 or audio.shape[0] != 1:
        raise ValueError("Expected mono audio shaped [1, T].")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    audio = audio.to(device)
    num_samples = int(audio.shape[-1])
    window = sqrt_hann_window(config, device)
    norm = compression_norm(config)
    delay = streaming_algorithmic_delay(config)

    total_frames = streaming_num_frames(num_samples, config)
    required_samples = total_frames * config.hop_length
    if num_samples < required_samples:
        audio = torch.nn.functional.pad(audio, (0, required_samples - num_samples))

    input_buffer = torch.zeros(1, config.n_fft, device=device)
    output = torch.zeros(1, required_samples + config.n_fft, device=device)
    denom = torch.zeros_like(output)

    time_tensors = [
        torch.full((1,), step_idx / max(steps, 1), device=device, dtype=model_dtype)
        for step_idx in range(steps)
    ]
    flow_states = [flow.init_state() for _ in range(steps)]

    with torch.inference_mode():
        for frame_idx in range(total_frames):
            chunk_start = frame_idx * config.hop_length
            chunk = audio[:, chunk_start:chunk_start + config.hop_length]
            input_buffer = torch.cat([input_buffer[:, config.hop_length:], chunk], dim=-1)

            spectrum = torch.fft.rfft(input_buffer * window, n=config.n_fft, norm=norm)
            if config.cut_highest_freqs:
                spectrum = spectrum[:, :-config.cut_highest_freqs]
            y_complex = compress_complex(spectrum, config)
            y_condition = y_complex.abs().to(y_complex.dtype)
            y_frame = complex_to_ri_frame(y_condition)
            y_frame_model = format_model_tensor(y_frame.to(model_dtype), model_memory_format)

            noise = format_model_tensor(torch.randn_like(y_frame_model), model_memory_format)
            x_t = y_frame_model + config.sigma_y * noise
            for step_idx in range(steps):
                dnn_input = pack_ri_channels(
                    x_t, y_frame_model, memory_format=model_memory_format
                )
                if use_engine:
                    outputs = flow.engine(dnn_input, time_tensors[step_idx], *flow_states[step_idx])
                    v = outputs[0]
                    # The engine returns its 63 successor states as outputs;
                    # rebinding them (rather than copying into the previous
                    # tensors) is how the adapter itself threads the recurrence.
                    flow_states[step_idx] = tuple(outputs[1:])
                else:
                    v, flow_states[step_idx] = forward_step(
                        flow,
                        dnn_input,
                        state=flow_states[step_idx],
                        time_cond=time_tensors[step_idx],
                        use_compiled=use_compiled,
                    )
                x_t = x_t + v / steps

            x_complex = ri_frame_to_complex(x_t.float())
            x_complex = decompress_complex(x_complex, config)
            x_complex = pad_cut_highest_freqs(x_complex, config)
            frame_audio = torch.fft.irfft(x_complex, n=config.n_fft, norm=norm) * window
            out_start = frame_idx * config.hop_length
            output[:, out_start:out_start + config.n_fft] += frame_audio
            denom[:, out_start:out_start + config.n_fft] += window.square()

    output = output / denom.clamp_min(1e-8)
    if not compensate_delay:
        return output.detach().cpu()
    aligned = output[:, delay:delay + num_samples]
    if aligned.shape[-1] < num_samples:
        aligned = torch.nn.functional.pad(aligned, (0, num_samples - aligned.shape[-1]))
    return aligned.detach().cpu()
