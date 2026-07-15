"""Shared building blocks used across benchmarks, evaluation, streaming and inference.

These modules are intentionally small and dependency-light so the higher-level
packages can reuse them without importing across each other:

- ``repo``            locate the repo root and make it importable from scripts
- ``devices``         torch device selection, sync, and matmul precision
- ``tensors``         4D memory-format handling and real/imag channel packing
- ``streaming_state`` streaming-module step helpers (state alloc/reset/compile bypass)
- ``timing``          latency-sample summaries (mean / percentiles)
- ``history``         atomic, lock-protected JSON history writes
- ``paths``           checkpoint and output path resolution
- ``options``         CLI option parsing/normalization for running the model
- ``modal_cache``     shared Modal volume/cache configuration
"""
