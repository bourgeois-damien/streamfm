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
    """Call a streaming module step, bypassing torch.compile when requested."""
    if use_compiled:
        return module.forward_step(x, time_cond=time_cond, state=state)

    fn = getattr(module.forward_step, "__wrapped__", None)
    if fn is None:
        return module.forward_step(x, time_cond=time_cond, state=state)
    return fn(module, x, time_cond=time_cond, state=state)


def prepare_streaming_state(module: Any):
    """Allocate graph-capturable streaming state when supported."""
    if hasattr(module, "prepare_state"):
        return module.prepare_state()
    return module.init_state()


def zero_streaming_state(module: Any, state: Any) -> None:
    """Reset a streaming state in place when supported."""
    if hasattr(module, "zero_state"):
        module.zero_state(state)
