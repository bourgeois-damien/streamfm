"""Public streaming pipeline API.

Implementation details live in focused modules: ``stft.py`` for audio framing
helpers, ``eager.py`` for regular execution, and ``cuda_graph.py`` for the CUDA
Graph variants. Import the pipeline entry points from here.
"""
from __future__ import annotations

from experiments.streaming.cuda_graph import (
    run_streaming_audio_pipeline_with_cuda_graph_model,
    run_streaming_audio_pipeline_with_full_cuda_graph,
    run_streaming_audio_pipeline_with_tensorrt_cuda_graph,
    run_streaming_se_audio_pipeline_with_cuda_graph_model,
)
from experiments.streaming.eager import (
    run_streaming_audio_pipeline,
    run_streaming_se_audio_pipeline,
)
from experiments.streaming.stft import StreamingSTFTConfig, make_synthetic_audio


__all__ = [
    "StreamingSTFTConfig",
    "make_synthetic_audio",
    "run_streaming_audio_pipeline",
    "run_streaming_audio_pipeline_with_cuda_graph_model",
    "run_streaming_audio_pipeline_with_full_cuda_graph",
    "run_streaming_audio_pipeline_with_tensorrt_cuda_graph",
    "run_streaming_se_audio_pipeline",
    "run_streaming_se_audio_pipeline_with_cuda_graph_model",
]
