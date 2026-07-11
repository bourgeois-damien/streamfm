from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any


def expand_parameter_grid(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a wandb-style sweep parameter block into explicit trial dicts."""
    varying_keys: list[str] = []
    varying_values: list[list[Any]] = []
    fixed: dict[str, Any] = {}

    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid sweep parameter block for '{key}'.")
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"Parameter '{key}' must provide a non-empty values list.")
            varying_keys.append(key)
            varying_values.append(values)
        elif "value" in spec:
            fixed[key] = spec["value"]
        else:
            raise ValueError(f"Parameter '{key}' must define either value or values.")

    if not varying_keys:
        return [dict(fixed)]

    trials: list[dict[str, Any]] = []
    for combo in itertools.product(*varying_values):
        trial = dict(fixed)
        for key, value in zip(varying_keys, combo):
            trial[key] = value
        trials.append(trial)
    return trials


def trial_matches_exclude_rule(trial: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Return True when every key in the exclude rule matches the trial."""
    if not rule:
        return False
    for key, expected in rule.items():
        if key not in trial:
            return False
        if trial[key] != expected:
            return False
    return True


def filter_excluded_trials(
    trials: list[dict[str, Any]],
    exclude_rules: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Drop trials that match any exclude rule (AND within a rule, OR across rules)."""
    if not exclude_rules:
        return trials
    return [
        trial
        for trial in trials
        if not any(trial_matches_exclude_rule(trial, rule) for rule in exclude_rules)
    ]


def _load_sweep_yaml(sweep_yaml: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load sweep YAML files.") from exc

    sweep_path = Path(sweep_yaml)
    data = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sweep YAML must contain a mapping: {sweep_path}")
    return data


def load_sweep_parameters(sweep_yaml: str | Path) -> dict[str, Any]:
    """Load the parameters block from a sweep YAML file."""
    data = _load_sweep_yaml(sweep_yaml)
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Sweep YAML must define a parameters mapping: {sweep_yaml}")
    return parameters


def load_sweep_exclude_rules(sweep_yaml: str | Path) -> list[dict[str, Any]]:
    """Load optional exclude rules from a sweep YAML file."""
    data = _load_sweep_yaml(sweep_yaml)
    exclude = data.get("exclude", [])
    if exclude is None:
        return []
    if not isinstance(exclude, list):
        raise ValueError(f"Sweep YAML exclude must be a list of mappings: {sweep_yaml}")
    rules: list[dict[str, Any]] = []
    for rule_idx, rule in enumerate(exclude):
        if not isinstance(rule, dict) or not rule:
            raise ValueError(f"Sweep YAML exclude[{rule_idx}] must be a non-empty mapping.")
        rules.append(rule)
    return rules


def load_sweep_trials(sweep_yaml: str | Path) -> list[dict[str, Any]]:
    """Load, expand, and filter trials from a sweep YAML file."""
    trials = expand_parameter_grid(load_sweep_parameters(sweep_yaml))
    return filter_excluded_trials(trials, load_sweep_exclude_rules(sweep_yaml))


def load_sweep_metadata(sweep_yaml: str | Path) -> dict[str, Any]:
    """Load sweep metadata such as project and entity from a sweep YAML file."""
    data = _load_sweep_yaml(sweep_yaml)
    return {
        key: data[key]
        for key in ("project", "entity", "method")
        if key in data
    }
