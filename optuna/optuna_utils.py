"""Small shared helpers for the Optuna sweep tools.

This file keeps the repeated bookkeeping out of the scripts without hiding the actual sweep flow.
It defines the benchmark tasks, study manifest parsing, and the suite directory layout.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

from algorithms.registry import ALGORITHM_ORDER
from utils import OPTUNA_RESULTS_ROOT, load_json


LEARNED_ALGORITHM_ORDER = tuple(name for name in ALGORITHM_ORDER if name != "random")


@dataclass(frozen=True)
class TaskSpec:
    """Files and directories tied to one benchmark task."""

    task_id: str
    run_script: Path
    task_config: Path
    config_root: Path


@dataclass(frozen=True)
class SuitePaths:
    """All filesystem roots created for one Optuna suite."""

    suite_id: str
    root: Path
    trial_runs_root: Path
    logs_root: Path
    studies_root: Path
    exports_root: Path

    @classmethod
    def create(cls, suite_id: str, *, results_root: Path = OPTUNA_RESULTS_ROOT) -> "SuitePaths":
        root = Path(results_root) / suite_id
        paths = cls(
            suite_id=suite_id,
            root=root,
            trial_runs_root=root / "trial_runs",
            logs_root=root / "logs",
            studies_root=root / "studies",
            exports_root=root / "exports",
        )
        for path in (paths.trial_runs_root, paths.logs_root, paths.studies_root, paths.exports_root):
            path.mkdir(parents=True, exist_ok=True)
        return paths


TASK_SPECS: dict[str, TaskSpec] = {
    "simple_spread": TaskSpec(
        task_id="simple_spread",
        run_script=PROJECT_ROOT / "simple_spread" / "run.py",
        task_config=PROJECT_ROOT / "simple_spread" / "task.json",
        config_root=PROJECT_ROOT / "simple_spread" / "configs",
    ),
    "simple_spread_dynamic": TaskSpec(
        task_id="simple_spread_dynamic",
        run_script=PROJECT_ROOT / "simple_spread_dynamic" / "run.py",
        task_config=PROJECT_ROOT / "simple_spread_dynamic" / "task.json",
        config_root=PROJECT_ROOT / "simple_spread_dynamic" / "configs",
    ),
    "simple_spread_dynamic_hard": TaskSpec(
        task_id="simple_spread_dynamic_hard",
        run_script=PROJECT_ROOT / "simple_spread_dynamic_hard" / "run.py",
        task_config=PROJECT_ROOT / "simple_spread_dynamic_hard" / "task.json",
        config_root=PROJECT_ROOT / "simple_spread_dynamic_hard" / "configs",
    ),
    "robotic_warehouse_dynamic": TaskSpec(
        task_id="robotic_warehouse_dynamic",
        run_script=PROJECT_ROOT / "robotic_warehouse_dynamic" / "run.py",
        task_config=PROJECT_ROOT / "robotic_warehouse_dynamic" / "task.json",
        config_root=PROJECT_ROOT / "robotic_warehouse_dynamic" / "configs",
    ),
    "level_based_foraging_dynamic": TaskSpec(
        task_id="level_based_foraging_dynamic",
        run_script=PROJECT_ROOT / "level_based_foraging_dynamic" / "run.py",
        task_config=PROJECT_ROOT / "level_based_foraging_dynamic" / "task.json",
        config_root=PROJECT_ROOT / "level_based_foraging_dynamic" / "configs",
    ),
    "lbf_hard": TaskSpec(
        task_id="lbf_hard",
        run_script=PROJECT_ROOT / "lbf_hard" / "run.py",
        task_config=PROJECT_ROOT / "lbf_hard" / "task.json",
        config_root=PROJECT_ROOT / "lbf_hard" / "configs",
    ),
    "toy_env": TaskSpec(
        task_id="toy_env",
        run_script=PROJECT_ROOT / "toy_env" / "run.py",
        task_config=PROJECT_ROOT / "toy_env" / "task.json",
        config_root=PROJECT_ROOT / "toy_env" / "configs",
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    """Return the task spec for one benchmark id."""
    if task_id not in TASK_SPECS:
        raise KeyError(f"Unsupported task: {task_id}")
    return TASK_SPECS[task_id]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV file into a list of dict rows."""
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_study_manifest(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate one study manifest.

    The `base.json` template is intentionally rejected here so it cannot be run by accident.
    """
    manifest_path = Path(path).resolve()
    manifest = load_json(manifest_path)
    if manifest_path.name == "base.json" or bool(manifest.get("template", False)):
        raise ValueError("The template manifest `base.json` is a guide only and cannot be run.")
    if "task" not in manifest:
        raise ValueError("Study manifest must define 'task'.")
    if "algorithms" not in manifest:
        raise ValueError("Study manifest must define 'algorithms'.")
    if not isinstance(manifest["algorithms"], list) or not manifest["algorithms"]:
        raise ValueError("Study manifest 'algorithms' must be a non-empty list.")
    if not isinstance(manifest.get("task_overrides", {}), dict):
        raise ValueError("Study manifest 'task_overrides' must be a JSON object.")

    seen: set[str] = set()
    for algorithm_spec in manifest["algorithms"]:
        if not isinstance(algorithm_spec, dict):
            raise ValueError("Each algorithm entry must be a JSON object.")
        name = str(algorithm_spec.get("name", ""))
        if not name:
            raise ValueError("Each algorithm entry must define 'name'.")
        if name not in LEARNED_ALGORITHM_ORDER:
            raise ValueError(f"Study manifest only supports learned algorithms, got {name!r}.")
        if name in seen:
            raise ValueError(f"Duplicate algorithm entry: {name!r}.")
        seen.add(name)
        if "trials" not in algorithm_spec:
            raise ValueError(f"Algorithm {name!r} must define 'trials'.")
        if int(algorithm_spec["trials"]) < 1:
            raise ValueError(f"Algorithm {name!r} must request at least one trial.")
        if not isinstance(algorithm_spec.get("search", {}), dict):
            raise ValueError(f"Algorithm {name!r} search spec must be a JSON object.")
        inherit_blocks = algorithm_spec.get("inherit", [])
        if not isinstance(inherit_blocks, list):
            raise ValueError(f"Algorithm {name!r} inherit spec must be a JSON list.")
        for inherit_block in inherit_blocks:
            if not isinstance(inherit_block, dict):
                raise ValueError(f"Algorithm {name!r} inherit entries must be JSON objects.")
            source_name = str(inherit_block.get("from", ""))
            if not source_name:
                raise ValueError(f"Algorithm {name!r} inherit entries must define 'from'.")
            if not isinstance(inherit_block.get("keys", []), list):
                raise ValueError(f"Algorithm {name!r} inherit keys must be a JSON list.")
            if "required" in inherit_block and not isinstance(inherit_block["required"], bool):
                raise ValueError(f"Algorithm {name!r} inherit 'required' flag must be true or false.")
    return manifest


def default_base_config_path(task_id: str, algorithm: str) -> Path:
    """Return the task-local default config for one algorithm."""
    return get_task_spec(task_id).config_root / algorithm / "default.json"


def resolve_algorithm_base_config(*, task_id: str, algorithm_spec: dict[str, Any], manifest_path: Path) -> Path:
    """Resolve the base config path for one algorithm entry."""
    configured = algorithm_spec.get("base_config")
    if configured is None:
        return default_base_config_path(task_id, str(algorithm_spec["name"]))

    candidate = Path(str(configured))
    if candidate.is_absolute():
        return candidate

    manifest_relative = (manifest_path.parent / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (PROJECT_ROOT / candidate).resolve()


def discover_task_ids(results_root: Path, suite_ids: list[str]) -> list[str]:
    """List known tasks first, then any extra tasks found under suite study directories."""
    known_order = list(TASK_SPECS.keys())
    known_seen = set(known_order)
    extras: list[str] = []
    for suite_id in suite_ids:
        studies_dir = results_root / suite_id / "studies"
        if not studies_dir.is_dir():
            continue
        for child in sorted(studies_dir.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            task_id = child.name
            if task_id in known_seen or task_id in extras:
                continue
            extras.append(task_id)
    return [*known_order, *extras]


def load_best_effective_config(studies_root: Path, task_id: str, algorithm: str) -> Optional[dict[str, Any]]:
    """Read the best effective config already exported in one leaderboard.

    This is used for chained studies such as MAPPO -> PIMAC where later algorithms inherit a few
    keys from the best completed parent run in the same suite.
    """
    leaderboard_path = Path(studies_root) / task_id / f"{algorithm}_leaderboard.csv"
    if not leaderboard_path.is_file():
        return None
    rows = read_csv_rows(leaderboard_path)
    ranked_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            if not row.get("effective_config_json"):
                continue
            int(row["rank"])
            ranked_rows.append(row)
        except Exception:
            continue
    if not ranked_rows:
        return None
    ranked_rows.sort(key=lambda row: int(row["rank"]))
    return dict(json.loads(ranked_rows[0]["effective_config_json"]))


def safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion used when reading leaderboard rows."""
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None
