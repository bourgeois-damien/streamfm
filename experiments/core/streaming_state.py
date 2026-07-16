"""Streaming-module step helpers: state allocation, reset, and compile bypass.

The causal streaming U-Net carries per-frame recurrent state. These helpers
call a module's single-frame ``forward_step`` (optionally bypassing torch.compile)
and manage the state object it threads between frames.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def forward_step(
    module: Any,
    x: "torch.Tensor",
    *,
    state: Any,
    time_cond: "torch.Tensor | None" = None,
    use_compiled: bool = False,
):
    """Run one frame through a streaming module and return (output, new_state).

    ``x`` is a single-frame model tensor [B, C, F, T=1]; ``time_cond`` is the
    flow time t in [0, 1] for solver steps (None for the SE predictor).
    """
    if use_compiled:
        return module.forward_step(x, time_cond=time_cond, state=state)

    # The model classes decorate forward_step with torch.compile at import
    # time. To benchmark true eager execution we call the undecorated function
    # kept in __wrapped__; going through the decorated attribute would trigger
    # compilation even when eager timings were requested.
    fn = getattr(module.forward_step, "__wrapped__", None)
    if fn is None:
        return module.forward_step(x, time_cond=time_cond, state=state)
    return fn(module, x, time_cond=time_cond, state=state)


def prepare_streaming_state(module: Any):
    """Allocate a fresh per-stream recurrent state for a streaming module.

    Prefers ``prepare_state`` (statically allocated buffers, required for CUDA
    Graph capture where every tensor address must stay fixed) and falls back
    to the plain ``init_state``.
    """
    if hasattr(module, "prepare_state"):
        return module.prepare_state()
    return module.init_state()


def zero_streaming_state(module: Any, state: Any) -> None:
    """Reset a streaming state in place (no reallocation) when supported.

    In-place zeroing keeps buffer addresses stable, which matters when the
    state is baked into a captured CUDA Graph.
    """
    if hasattr(module, "zero_state"):
        module.zero_state(state)
