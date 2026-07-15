"""Eval CLI option normalization.

Maps task names to their config and checkpoint defaults and normalizes the
solver, steps, split, pipeline and part selections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDefaults:
    config_name: str
    checkpoint_name: str


TASK_DEFAULTS = {
    "stftpr": TaskDefaults("streamfm_stftpr", "streamfm_stftpr.ckpt"),
    "se": TaskDefaults("streamfm_se_predgen", "streamfm_se_predgen.ckpt"),
    "bwe": TaskDefaults("streamfm_bwe", "streamfm_bwe.ckpt"),
    "derev": TaskDefaults("streamfm_derev", "streamfm_derev.ckpt"),
    "lyra": TaskDefaults("streamfm_lyra", "streamfm_lyra.ckpt"),
    "melflow": TaskDefaults("streamfm_melflow", "streamfm_melflow.ckpt"),
}


def normalize_task(task: str) -> str:
    """Normalize and validate an evaluation task name."""
    normalized = task.lower().replace("-", "_")
    if normalized not in TASK_DEFAULTS:
        supported = ", ".join(sorted(TASK_DEFAULTS))
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported}.")
    return normalized


def resolve_config_and_checkpoint(*, task: str, config_name: str, checkpoint_name: str) -> tuple[str, str]:
    """Resolve config/checkpoint defaults from a task, unless explicitly set."""
    defaults = TASK_DEFAULTS[normalize_task(task)]
    if not config_name:
        config_name = defaults.config_name
    if not checkpoint_name:
        checkpoint_name = defaults.checkpoint_name
    return config_name, checkpoint_name


def parse_solver_and_steps(solver: str, steps: int) -> tuple[str, int]:
    """Accept either ('euler', 5) or the official compact form '5xeuler'."""
    solver = solver.lower().strip()
    match = re.fullmatch(r"(\d+)x(.+)", solver)
    if match:
        parsed_steps = int(match.group(1))
        parsed_solver = match.group(2)
        if steps != 5 and steps != parsed_steps:
            raise ValueError(
                f"Conflicting solver/steps: solver='{solver}' implies {parsed_steps}, but --steps={steps}."
            )
        return parsed_solver, parsed_steps
    if steps <= 0:
        raise ValueError("--steps must be a positive integer.")
    return solver, steps


def normalize_split(split: str) -> str:
    """Normalize and validate dataset split names."""
    normalized = split.lower().strip()
    if normalized not in frozenset({"train", "valid", "test"}):
        raise ValueError("Unsupported split. Use 'train', 'valid', or 'test'.")
    return normalized


def normalize_pipeline(pipeline: str) -> str:
    """Normalize and validate evaluation pipeline names."""
    normalized = pipeline.lower().replace("-", "_")
    if normalized not in frozenset({"offline"}):
        raise ValueError("Only '--pipeline offline' is implemented for test-set inference.")
    return normalized


def normalize_part(part: str) -> str:
    """Normalize and validate model part selection for evaluation."""
    normalized = part.lower().replace("-", "_")
    if normalized not in frozenset({"model", "predictor"}):
        raise ValueError("Unsupported part. Use 'model' or 'predictor'.")
    return normalized
