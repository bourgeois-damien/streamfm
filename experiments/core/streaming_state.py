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
    if use_compiled:
        return module.forward_step(x, time_cond=time_cond, state=state)

    # forward_step is compile-wrapped at import; patch it here for a true eager bench
    fn = getattr(module.forward_step, "__wrapped__", None)
    if fn is None:
        return module.forward_step(x, time_cond=time_cond, state=state)
    return fn(module, x, time_cond=time_cond, state=state)


def prepare_streaming_state(module: Any):
    if hasattr(module, "prepare_state"):
        return module.prepare_state()
    return module.init_state()


def zero_streaming_state(module: Any, state: Any) -> None:
    if hasattr(module, "zero_state"):
        module.zero_state(state)
