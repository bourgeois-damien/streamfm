"""Benchmark CLI option parsing and normalization.

Turns the raw task/part/execution/dtype/steps arguments into the canonical
forms the runner expects (``SUPPORTED_TASKS``, ``FLOW_TASKS``).
"""

from __future__ import annotations

FLOW_TASKS = {"stftpr", "bwe", "derev", "lyra"}
SUPPORTED_TASKS = FLOW_TASKS | {"se"}


def normalize_cli_options(
    *,
    task: str,
    part: str,
    pipeline: str,
    execution: str,
) -> dict:
    """Map explicit CLI options to internal benchmark names.

    Rejects invalid combinations early (before any model load) and returns the
    dict the runner dispatches on: ``internal_task`` picks the benchmark
    family, ``internal_pipeline`` picks model-only vs audio and eager vs
    CUDA-Graph variants.
    """
    requested_task = task.lower().replace("-", "_")
    requested_part = part.lower().replace("-", "_")
    requested_pipeline = pipeline.lower().replace("-", "_")
    requested_execution = execution.lower().replace("-", "_")

    if requested_task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task. Use one of: {supported}.")
    if requested_part not in {"predictor", "flow", "model"}:
        raise ValueError("Unsupported part. Use 'model', 'predictor', or 'flow'.")
    if requested_pipeline not in {"model_only", "audio"}:
        raise ValueError("Unsupported pipeline. Use 'model_only' or 'audio'.")
    if requested_execution not in {"eager", "compiled", "cuda_graph", "tensorrt", "tensorrt_cuda_graph"}:
        raise ValueError("Unsupported execution. Use eager, compiled, cuda_graph, tensorrt, or tensorrt_cuda_graph.")

    # Flow-only tasks (stftpr/bwe/...) have a single DNN, so part=model and
    # part=flow are the same thing; only SE splits into predictor + flow.
    if requested_task in FLOW_TASKS:
        if requested_part == "predictor":
            raise ValueError(f"{requested_task} has no predictor. Use '--part model' or '--part flow'.")
        internal_task = requested_task
    elif requested_part == "predictor":
        internal_task = "se_predictor"
    elif requested_part == "flow":
        internal_task = "se_flow"
    else:
        internal_task = "se_full"

    if requested_pipeline == "model_only":
        internal_pipeline = "graph_model" if requested_execution == "cuda_graph" else "model"
    else:
        if requested_task == "se" and requested_part != "model":
            raise ValueError("SE audio pipeline supports only '--part model'. Use '--pipeline model_only' for predictor/flow.")
        internal_pipeline = "audio_graph_model" if requested_execution == "cuda_graph" else "audio"

    return {
        "requested_task": requested_task,
        "requested_part": requested_part,
        "requested_pipeline": requested_pipeline,
        "execution": requested_execution,
        "internal_task": internal_task,
        "internal_pipeline": internal_pipeline,
        "use_compiled": requested_execution in {"compiled", "cuda_graph"},
        "use_tensorrt": requested_execution in {"tensorrt", "tensorrt_cuda_graph"},
        "tensorrt_cuda_graph": requested_execution == "tensorrt_cuda_graph",
    }


def parse_steps(steps: str) -> tuple[int, ...]:
    """Parse a comma-separated step list."""
    parsed = tuple(int(part.strip()) for part in steps.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one step count is required.")
    return parsed


def parse_model_dtype(dtype_name: str):
    """Map a compact dtype name to a torch dtype."""
    import torch

    dtype_name = dtype_name.lower()
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError("Unsupported dtype. Use 'fp32', 'fp16', or 'bf16'.")


def resolve_execution(execution: str, device) -> str:
    """Resolve auto execution mode from the selected device."""
    requested = execution.lower().replace("-", "_")
    if requested == "auto":
        # Map auto to sensible defaults per device:
        # - CUDA: prefer cuda_graph for best throughput when available
        # - CPU: compiled is supported and may offer speedups
        # - MPS: use eager because compiled/MPS support is experimental
        if device.type == "cuda":
            return "cuda_graph"
        if device.type == "cpu":
            return "compiled"
        if device.type == "mps":
            return "eager"
        # Fallback to eager for unknown devices
        return "eager"
    return requested
