"""Generic JSON-driven search-space helpers for Optuna studies.

The goal here is simple: study manifests carry the numbers, and this file only knows how to sample
those numbers. There are no task-specific profiles in this file anymore.
"""

from __future__ import annotations

import json
from typing import Any


SUPPORTED_PARAM_TYPES = {"categorical", "float", "int", "constant"}


def _suggest_categorical(trial, name: str, values: list[Any]) -> Any:
    """Sample one categorical value, including list and dict choices."""
    if not values:
        raise ValueError(f"Categorical parameter {name!r} must define at least one value.")

    simple_scalars = (str, int, float, bool, type(None))
    if all(isinstance(value, simple_scalars) for value in values):
        return trial.suggest_categorical(name, values)

    serialized_values = [json.dumps(value, sort_keys=True) for value in values]
    selected = trial.suggest_categorical(name, serialized_values)
    return json.loads(selected)


def _suggest_value(trial, name: str, param_spec: dict[str, Any]) -> Any:
    """Sample one parameter value from its manifest spec."""
    param_type = str(param_spec.get("type", ""))
    if param_type not in SUPPORTED_PARAM_TYPES:
        raise ValueError(f"Unsupported search-space type for {name!r}: {param_type!r}.")

    if param_type == "constant":
        return param_spec.get("value")
    if param_type == "categorical":
        values = list(param_spec.get("values", []))
        return _suggest_categorical(trial, name, values)
    if param_type == "float":
        low = float(param_spec["low"])
        high = float(param_spec["high"])
        log = bool(param_spec.get("log", False))
        step = param_spec.get("step")
        if step is None:
            return float(trial.suggest_float(name, low, high, log=log))
        return float(trial.suggest_float(name, low, high, step=float(step), log=log))

    low = int(param_spec["low"])
    high = int(param_spec["high"])
    log = bool(param_spec.get("log", False))
    step = int(param_spec.get("step", 1))
    return int(trial.suggest_int(name, low, high, step=step, log=log))


def _assign_value(config: dict[str, Any], *, name: str, value: Any, param_spec: dict[str, Any]) -> None:
    """Write one sampled value into the config according to the manifest assignment rule."""
    merge = bool(param_spec.get("merge", False))
    targets = list(param_spec.get("targets", []))
    length_target = param_spec.get("length_target")

    if merge and targets:
        raise ValueError(f"Parameter {name!r} cannot use both 'merge' and 'targets'.")

    if merge:
        if not isinstance(value, dict):
            raise ValueError(f"Parameter {name!r} uses 'merge' but did not sample a dict value.")
        config.update(value)
    elif targets:
        if isinstance(value, dict):
            for target in targets:
                if target not in value:
                    raise ValueError(f"Parameter {name!r} sampled a dict without target key {target!r}.")
                config[str(target)] = value[target]
        elif len(targets) == 1:
            config[str(targets[0])] = value
        else:
            if not isinstance(value, (list, tuple)) or len(value) != len(targets):
                raise ValueError(f"Parameter {name!r} must sample a list/tuple matching its targets.")
            for target, target_value in zip(targets, value):
                config[str(target)] = target_value
    else:
        config[name] = value

    if length_target is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Parameter {name!r} uses 'length_target' but did not sample a list/tuple.")
        config[str(length_target)] = int(len(value))


def validate_search_spec(search_spec: dict[str, Any]) -> None:
    """Fail early on malformed search-space specs."""
    for name, param_spec in search_spec.items():
        if not isinstance(param_spec, dict):
            raise ValueError(f"Search parameter {name!r} must be a JSON object.")
        param_type = str(param_spec.get("type", ""))
        if param_type not in SUPPORTED_PARAM_TYPES:
            raise ValueError(f"Search parameter {name!r} has unsupported type {param_type!r}.")
        if param_type == "categorical" and not isinstance(param_spec.get("values", []), list):
            raise ValueError(f"Search parameter {name!r} categorical values must be a JSON list.")
        if param_type in {"float", "int"}:
            if "low" not in param_spec or "high" not in param_spec:
                raise ValueError(f"Search parameter {name!r} must define 'low' and 'high'.")
        if "targets" in param_spec and not isinstance(param_spec.get("targets"), list):
            raise ValueError(f"Search parameter {name!r} targets must be a JSON list.")
        if "merge" in param_spec and not isinstance(param_spec.get("merge"), bool):
            raise ValueError(f"Search parameter {name!r} merge flag must be true or false.")
        if "length_target" in param_spec and not isinstance(param_spec.get("length_target"), str):
            raise ValueError(f"Search parameter {name!r} length_target must be a string.")


def sample_algorithm_config(trial, *, base_config: dict[str, Any], search_spec: dict[str, Any]) -> dict[str, Any]:
    """Sample one full effective config from a base config and a JSON search spec."""
    validate_search_spec(search_spec)
    config = json.loads(json.dumps(base_config))
    for name, param_spec in search_spec.items():
        value = _suggest_value(trial, name, param_spec)
        _assign_value(config, name=str(name), value=value, param_spec=param_spec)
    return config
