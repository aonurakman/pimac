"""Run one Optuna study manifest.

The manifest is the source of truth. It defines:
- which task to run,
- which algorithms to tune,
- how many trials each algorithm gets,
- the numeric search ranges for every tunable hyperparameter,
- and any inheritance rules such as MAPPO -> PIMAC chaining.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna as optuna_lib

from optuna_utils import (
    OPTUNA_RESULTS_ROOT,
    LEARNED_ALGORITHM_ORDER,
    SuitePaths,
    TaskSpec,
    get_task_spec,
    load_best_effective_config,
    load_study_manifest,
    resolve_algorithm_base_config,
)
from search_spaces import sample_algorithm_config, validate_search_spec
from utils import load_json, write_csv, write_json


@dataclass(frozen=True)
class TrialRun:
    """Everything needed to launch one child training run."""

    suite_id: str
    task_spec: TaskSpec
    algorithm: str
    trial_number: int
    seed: int
    algorithm_config: dict[str, Any]
    task_config: dict[str, Any]
    suite_paths: SuitePaths
    show_child_output: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.task_spec.task_id}_{self.algorithm}_trial_{self.trial_number:05d}"

    @property
    def run_output_dir(self) -> Path:
        return self.suite_paths.trial_runs_root / self.task_spec.task_id / self.algorithm / self.run_id

    @property
    def summary_path(self) -> Path:
        return self.run_output_dir / "summary.json"

    @property
    def train_history_path(self) -> Path:
        return self.run_output_dir / "train_history.csv"

    @property
    def algorithm_config_path(self) -> Path:
        return self.run_output_dir / "alg_config.json"

    @property
    def task_config_path(self) -> Path:
        return self.run_output_dir / "task_config.json"

    @property
    def log_path(self) -> Path:
        return self.suite_paths.logs_root / self.task_spec.task_id / self.algorithm / f"{self.run_id}.log"


@dataclass(frozen=True)
class TrialResult:
    """The few summary numbers that drive leaderboard ranking."""

    objective_score: float
    best_validation_mean: float
    best_checkpoint_test_mean: float
    final_checkpoint_test_mean: float
    convergence_episode_90pct: int
    final_train_moving_average: float
    reward_slope_last_window: float
    best_vs_final_drop: float
    train_return_std_last_100: float
    summary: dict[str, Any]
    summary_path: str
    run_output_dir: str


def _build_command(trial_run: TrialRun) -> list[str]:
    """Build the child command exactly as a user would call the task script."""
    return [
        sys.executable,
        str(trial_run.task_spec.run_script),
        "--algorithm",
        trial_run.algorithm,
        "--alg-config",
        str(trial_run.algorithm_config_path),
        "--task-config",
        str(trial_run.task_config_path),
        "--seed",
        str(trial_run.seed),
        "--results-root",
        str(trial_run.suite_paths.trial_runs_root),
        "--run-id",
        trial_run.run_id,
        "--skip-gif",
    ]


def _train_return_std_last_100(train_history_path: Path) -> float:
    if not train_history_path.is_file():
        return float("nan")
    import csv

    returns: list[float] = []
    with train_history_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                returns.append(float(row["train_return_mean"]))
            except Exception:
                continue
    if not returns:
        return float("nan")
    tail = returns[-100:]
    mean = sum(tail) / len(tail)
    variance = sum((value - mean) ** 2 for value in tail) / len(tail)
    return float(variance ** 0.5)


def extract_trial_result(*, summary_path: Path, train_history_path: Path, run_output_dir: Path) -> TrialResult:
    """Read one finished run and turn it into one leaderboard record."""
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    validation = summary.get("validation", {})
    test = summary.get("test", {})
    best_checkpoint = test.get("best_checkpoint", {})
    final_checkpoint = test.get("final_checkpoint", {})
    best_checkpoint_mean = float(best_checkpoint.get("overall_eval_mean", test.get("best_checkpoint_mean", float("nan"))))
    final_checkpoint_mean = float(final_checkpoint.get("overall_eval_mean", test.get("final_checkpoint_mean", float("nan"))))

    return TrialResult(
        objective_score=float(test["objective_score"]),
        best_validation_mean=float(validation["best_validation_mean"]),
        best_checkpoint_test_mean=best_checkpoint_mean,
        final_checkpoint_test_mean=final_checkpoint_mean,
        convergence_episode_90pct=int(validation["convergence_episode_90pct"]),
        final_train_moving_average=float(summary["train"]["final_moving_average"]),
        reward_slope_last_window=float(summary["train"]["reward_slope_last_window"]),
        best_vs_final_drop=float(test["best_vs_final_drop"]),
        train_return_std_last_100=_train_return_std_last_100(train_history_path),
        summary=summary,
        summary_path=str(summary_path),
        run_output_dir=str(run_output_dir),
    )


def run_trial(trial_run: TrialRun) -> TrialResult:
    """Launch one child training run and return its summary metrics."""
    trial_run.run_output_dir.mkdir(parents=True, exist_ok=True)
    trial_run.log_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(trial_run.algorithm_config_path, trial_run.algorithm_config)
    write_json(trial_run.task_config_path, trial_run.task_config)

    command = _build_command(trial_run)
    stdout_target = None if trial_run.show_child_output else subprocess.PIPE
    stderr_target = None if trial_run.show_child_output else subprocess.STDOUT
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=stdout_target,
        stderr=stderr_target,
        check=False,
    )
    if not trial_run.show_child_output:
        trial_run.log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Trial failed with exit code {completed.returncode}. See {trial_run.log_path}.")
    if not trial_run.summary_path.is_file():
        raise FileNotFoundError(f"Trial summary not found: {trial_run.summary_path}")

    return extract_trial_result(
        summary_path=trial_run.summary_path,
        train_history_path=trial_run.train_history_path,
        run_output_dir=trial_run.run_output_dir,
    )


def _trial_rows(study: optuna_lib.Study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_score": trial.value if trial.value is not None else "",
        }
        for key, value in sorted(trial.user_attrs.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, sort_keys=True)
            else:
                row[key] = value
        for key, value in sorted(trial.params.items()):
            row[f"param::{key}"] = value
        rows.append(row)
    return rows


def _completed_leaderboard(study: optuna_lib.Study) -> list[dict[str, Any]]:
    completed = [trial for trial in study.trials if trial.state == optuna_lib.trial.TrialState.COMPLETE and trial.value is not None]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(completed, start=1):
        rows.append(
            {
                "rank": rank,
                "trial_number": trial.number,
                "objective_score": float(trial.value),
                "best_validation_mean": trial.user_attrs.get("best_validation_mean", ""),
                "best_checkpoint_test_mean": trial.user_attrs.get("best_checkpoint_test_mean", ""),
                "final_checkpoint_test_mean": trial.user_attrs.get("final_checkpoint_test_mean", ""),
                "convergence_episode_90pct": trial.user_attrs.get("convergence_episode_90pct", ""),
                "best_vs_final_drop": trial.user_attrs.get("best_vs_final_drop", ""),
                "train_return_std_last_100": trial.user_attrs.get("train_return_std_last_100", ""),
                "summary_path": trial.user_attrs.get("summary_path", ""),
                "run_output_dir": trial.user_attrs.get("run_output_dir", ""),
                "effective_config_json": trial.user_attrs.get("effective_config_json", ""),
                "inherit_summary_json": trial.user_attrs.get("inherit_summary_json", ""),
            }
        )
    return rows


def _export_top_configs(*, task_spec: TaskSpec, algorithm: str, leaderboard: list[dict[str, Any]], suite_paths: SuitePaths) -> list[dict[str, Any]]:
    """Export the top few configs as plain JSON files for later replay."""
    export_root = suite_paths.exports_root / task_spec.task_id / algorithm
    export_root.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, Any]] = []
    for row in leaderboard[:5]:
        rank = int(row["rank"])
        config = json.loads(row["effective_config_json"])
        export_path = export_root / f"rank_{rank:02d}.json"
        write_json(export_path, config)
        exports.append(
            {
                "rank": rank,
                "trial_number": int(row["trial_number"]),
                "objective_score": float(row["objective_score"]),
                "export_path": str(export_path),
            }
        )
    write_json(export_root / "index.json", {"exports": exports})
    return exports


def _copy_selected_keys(destination: dict[str, Any], source: dict[str, Any] | None, keys: Sequence[str]) -> None:
    if not source:
        return
    for key in keys:
        if key in source:
            destination[str(key)] = source[key]


def _resolve_inheritance(
    *,
    studies_root: Path,
    task_id: str,
    algorithm: str,
    inherit_blocks: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load the parent configs that this algorithm wants to inherit from."""
    inherited_configs: dict[str, dict[str, Any]] = {}
    for block in inherit_blocks:
        source_algorithm = str(block["from"])
        source_config = load_best_effective_config(studies_root, task_id, source_algorithm)
        if source_config is None:
            if bool(block.get("required", True)):
                raise RuntimeError(
                    f"{algorithm} requires completed study results for {source_algorithm} before it can inherit keys."
                )
            continue
        inherited_configs[source_algorithm] = source_config
    return inherited_configs


def _apply_inheritance(
    *,
    effective_config: dict[str, Any],
    inherit_blocks: Sequence[dict[str, Any]],
    inherited_configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply manifest-defined inheritance after sampling local search values."""
    applied: list[dict[str, Any]] = []
    for block in inherit_blocks:
        source_algorithm = str(block["from"])
        source_config = inherited_configs.get(source_algorithm)
        if source_config is None:
            continue
        keys = [str(key) for key in block.get("keys", [])]
        _copy_selected_keys(effective_config, source_config, keys)
        applied.append({"from": source_algorithm, "keys": keys})
    return applied


def _algorithm_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(entry) for entry in manifest["algorithms"]]


def _completed_trial_count(study: optuna_lib.Study) -> int:
    return sum(
        1 for trial in study.trials if trial.state == optuna_lib.trial.TrialState.COMPLETE and trial.value is not None
    )


def _manifest_dependencies(algorithm_entries: Sequence[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    manifest_algorithms = {str(entry["name"]) for entry in algorithm_entries}
    manifest_dependencies: dict[str, list[str]] = {}
    external_sources: dict[str, list[str]] = {}

    for entry in algorithm_entries:
        algorithm = str(entry["name"])
        local_dependencies: list[str] = []
        local_external_sources: list[str] = []
        for block in entry.get("inherit", []):
            source_algorithm = str(block["from"])
            if source_algorithm in manifest_algorithms:
                local_dependencies.append(source_algorithm)
            else:
                local_external_sources.append(source_algorithm)
        manifest_dependencies[algorithm] = local_dependencies
        external_sources[algorithm] = local_external_sources
    return manifest_dependencies, external_sources


def _validate_manifest_dag(manifest_dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(algorithm: str) -> None:
        if algorithm in visited:
            return
        if algorithm in visiting:
            raise ValueError(f"Manifest inherit graph contains a cycle involving {algorithm!r}.")
        visiting.add(algorithm)
        for parent in manifest_dependencies.get(algorithm, []):
            dfs(parent)
        visiting.remove(algorithm)
        visited.add(algorithm)

    for algorithm in manifest_dependencies:
        dfs(algorithm)


def _run_algorithm_study(
    *,
    algorithm_entry: dict[str, Any],
    manifest_file: Path,
    task_id: str,
    task_spec: TaskSpec,
    task_config: dict[str, Any],
    suite_id: str,
    suite_paths: SuitePaths,
    resolved_seed: int,
    show_child_output: bool,
    progress_lock: threading.Lock,
) -> dict[str, Any]:
    algorithm = str(algorithm_entry["name"])
    base_config_path = resolve_algorithm_base_config(
        task_id=task_id,
        algorithm_spec=algorithm_entry,
        manifest_path=manifest_file,
    )
    base_config = load_json(base_config_path)
    search_spec = dict(algorithm_entry.get("search", {}))
    validate_search_spec(search_spec)
    inherit_blocks = list(algorithm_entry.get("inherit", []))
    inherited_configs = _resolve_inheritance(
        studies_root=suite_paths.studies_root,
        task_id=task_id,
        algorithm=algorithm,
        inherit_blocks=inherit_blocks,
    )
    target_trials = int(algorithm_entry["trials"])

    task_study_dir = suite_paths.studies_root / task_id
    task_study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = task_study_dir / f"{algorithm}.sqlite3"
    study = optuna_lib.create_study(
        study_name=f"{suite_id}_{task_id}_{algorithm}",
        direction="maximize",
        sampler=optuna_lib.samplers.TPESampler(seed=resolved_seed, multivariate=True),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    def objective(trial: optuna_lib.Trial) -> float:
        effective_config = sample_algorithm_config(
            trial,
            base_config=base_config,
            search_spec=search_spec,
            task_config=task_config,
        )
        applied_inheritance = _apply_inheritance(
            effective_config=effective_config,
            inherit_blocks=inherit_blocks,
            inherited_configs=inherited_configs,
        )

        trial.set_user_attr("effective_config_json", json.dumps(effective_config, sort_keys=True))
        trial.set_user_attr("inherit_summary_json", json.dumps(applied_inheritance, sort_keys=True))

        result = run_trial(
            TrialRun(
                suite_id=suite_id,
                task_spec=task_spec,
                algorithm=algorithm,
                trial_number=trial.number,
                seed=resolved_seed,
                algorithm_config=effective_config,
                task_config=task_config,
                suite_paths=suite_paths,
                show_child_output=show_child_output,
            )
        )
        trial.set_user_attr("best_validation_mean", result.best_validation_mean)
        trial.set_user_attr("best_checkpoint_test_mean", result.best_checkpoint_test_mean)
        trial.set_user_attr("final_checkpoint_test_mean", result.final_checkpoint_test_mean)
        trial.set_user_attr("convergence_episode_90pct", result.convergence_episode_90pct)
        trial.set_user_attr("final_train_moving_average", result.final_train_moving_average)
        trial.set_user_attr("reward_slope_last_window", result.reward_slope_last_window)
        trial.set_user_attr("best_vs_final_drop", result.best_vs_final_drop)
        trial.set_user_attr("train_return_std_last_100", result.train_return_std_last_100)
        trial.set_user_attr("summary_path", result.summary_path)
        trial.set_user_attr("run_output_dir", result.run_output_dir)
        return result.objective_score

    completed_before = _completed_trial_count(study)
    remaining_trials = max(0, target_trials - completed_before)

    def callback(active_study: optuna_lib.Study, finished_trial: optuna_lib.Trial) -> None:
        completed_trials = [
            trial for trial in active_study.trials if trial.state == optuna_lib.trial.TrialState.COMPLETE and trial.value is not None
        ]
        best_score = max((float(trial.value) for trial in completed_trials), default=None)
        with progress_lock:
            best_text = "n/a" if best_score is None else f"{best_score:.4f}"
            last_text = "n/a" if finished_trial.value is None else f"{float(finished_trial.value):.4f}"
            print(
                f"[Optuna] {task_id}/{algorithm}: {len(completed_trials)}/{target_trials} complete, "
                f"last={last_text}, best={best_text}, state={finished_trial.state.name}"
            )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=1,
            catch=(RuntimeError, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError),
            show_progress_bar=False,
            callbacks=[callback],
        )

    trial_rows = _trial_rows(study)
    leaderboard = _completed_leaderboard(study)
    write_csv(task_study_dir / f"{algorithm}_trials.csv", trial_rows)
    write_csv(task_study_dir / f"{algorithm}_leaderboard.csv", leaderboard)
    exports = _export_top_configs(
        task_spec=task_spec,
        algorithm=algorithm,
        leaderboard=leaderboard,
        suite_paths=suite_paths,
    )
    return {
        "task": task_id,
        "algorithm": algorithm,
        "target_trials": int(target_trials),
        "completed_trials": int(len(leaderboard)),
        "base_config_path": str(base_config_path),
        "leaderboard_path": str(task_study_dir / f"{algorithm}_leaderboard.csv"),
        "exports": exports,
        "inherit": inherit_blocks,
    }


def run_manifest(
    *,
    manifest_path: str | Path,
    suite_id: str,
    parallel_jobs: int,
    seed: int | None = None,
    show_child_output: bool = False,
    results_root: Path = OPTUNA_RESULTS_ROOT,
) -> dict[str, Any]:
    """Run one study manifest end to end."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_study_manifest(manifest_file)
    task_id = str(manifest["task"])
    task_spec = get_task_spec(task_id)
    suite_paths = SuitePaths.create(suite_id, results_root=results_root)

    task_config = dict(load_json(task_spec.task_config))
    task_config.update(dict(manifest.get("task_overrides", {})))
    resolved_seed = int(seed if seed is not None else manifest.get("seed", 42))

    task_study_dir = suite_paths.studies_root / task_id
    task_study_dir.mkdir(parents=True, exist_ok=True)
    progress_lock = threading.Lock()
    algorithm_entries = _algorithm_entries(manifest)
    for algorithm_entry in algorithm_entries:
        algorithm = str(algorithm_entry["name"])
        if algorithm not in LEARNED_ALGORITHM_ORDER:
            raise ValueError(f"Optuna study only supports learned algorithms, got {algorithm!r}.")
        validate_search_spec(dict(algorithm_entry.get("search", {})))

    manifest_dependencies, _ = _manifest_dependencies(algorithm_entries)
    _validate_manifest_dag(manifest_dependencies)

    pending_dependencies = {
        str(entry["name"]): set(manifest_dependencies[str(entry["name"])])
        for entry in algorithm_entries
    }
    children_by_parent: dict[str, list[str]] = {str(entry["name"]): [] for entry in algorithm_entries}
    for algorithm, parents in manifest_dependencies.items():
        for parent in parents:
            children_by_parent[parent].append(algorithm)

    entry_by_algorithm = {str(entry["name"]): entry for entry in algorithm_entries}
    ready_algorithms = [str(entry["name"]) for entry in algorithm_entries if not pending_dependencies[str(entry["name"])]]
    ready_set = set(ready_algorithms)
    running_futures: dict[Future[dict[str, Any]], str] = {}
    completed_algorithms: set[str] = set()
    failed_algorithms: dict[str, str] = {}
    results_by_algorithm: dict[str, dict[str, Any]] = {}

    def submit_ready_work(executor: ThreadPoolExecutor) -> None:
        while ready_algorithms and len(running_futures) < max(1, int(parallel_jobs)):
            algorithm = ready_algorithms.pop(0)
            ready_set.remove(algorithm)
            future = executor.submit(
                _run_algorithm_study,
                algorithm_entry=entry_by_algorithm[algorithm],
                manifest_file=manifest_file,
                task_id=task_id,
                task_spec=task_spec,
                task_config=task_config,
                suite_id=suite_id,
                suite_paths=suite_paths,
                resolved_seed=resolved_seed,
                show_child_output=show_child_output,
                progress_lock=progress_lock,
            )
            running_futures[future] = algorithm

    with ThreadPoolExecutor(max_workers=max(1, int(parallel_jobs))) as executor:
        submit_ready_work(executor)
        while running_futures:
            done_futures, _ = wait(tuple(running_futures.keys()), return_when=FIRST_COMPLETED)
            for future in done_futures:
                algorithm = running_futures.pop(future)
                try:
                    results_by_algorithm[algorithm] = future.result()
                    completed_algorithms.add(algorithm)
                    for child in children_by_parent.get(algorithm, []):
                        child_dependencies = pending_dependencies[child]
                        child_dependencies.discard(algorithm)
                        if not child_dependencies and child not in completed_algorithms and child not in ready_set:
                            ready_algorithms.append(child)
                            ready_set.add(child)
                except Exception as exc:
                    failed_algorithms[algorithm] = f"{type(exc).__name__}: {exc}"
            if failed_algorithms:
                continue
            submit_ready_work(executor)

    if failed_algorithms:
        failure_messages = "; ".join(f"{algorithm}: {message}" for algorithm, message in failed_algorithms.items())
        raise RuntimeError(f"Optuna manifest failed before all dependencies completed: {failure_messages}")

    study_results = [results_by_algorithm[str(entry["name"])] for entry in algorithm_entries]

    summary = {
        "suite_id": suite_id,
        "manifest_path": str(manifest_file),
        "task": task_id,
        "seed": resolved_seed,
        "parallel_jobs": int(parallel_jobs),
        "task_config": task_config,
        "results": study_results,
    }
    write_json(task_study_dir / "study_summary.json", summary)
    return summary


def build_dry_run(manifest_path: str | Path, *, suite_id: str, results_root: Path = OPTUNA_RESULTS_ROOT) -> dict[str, Any]:
    """Resolve one manifest without launching child processes."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_study_manifest(manifest_file)
    task_id = str(manifest["task"])
    task_spec = get_task_spec(task_id)
    task_config = dict(load_json(task_spec.task_config))
    task_config.update(dict(manifest.get("task_overrides", {})))
    algorithm_entries = _algorithm_entries(manifest)
    manifest_dependencies, external_sources = _manifest_dependencies(algorithm_entries)
    _validate_manifest_dag(manifest_dependencies)
    algorithm_data: dict[str, Any] = {}
    for algorithm_entry in algorithm_entries:
        algorithm = str(algorithm_entry["name"])
        validate_search_spec(dict(algorithm_entry.get("search", {})))
        algorithm_data[algorithm] = {
            "target_trials": int(algorithm_entry["trials"]),
            "base_config_path": str(
                resolve_algorithm_base_config(
                    task_id=task_id,
                    algorithm_spec=algorithm_entry,
                    manifest_path=manifest_file,
                )
            ),
            "search_keys": list(dict(algorithm_entry.get("search", {})).keys()),
            "inherit": list(algorithm_entry.get("inherit", [])),
            "manifest_dependencies": list(manifest_dependencies[algorithm]),
            "external_inherit_sources": list(external_sources[algorithm]),
        }
    return {
        "suite_id": suite_id,
        "manifest_path": str(manifest_file),
        "results_root": str(results_root),
        "task": task_id,
        "run_script": str(task_spec.run_script),
        "task_config": task_config,
        "algorithms": algorithm_data,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Optuna study manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the study JSON manifest.")
    parser.add_argument("--suite-id", type=str, default="study_01")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-child-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_dry_run(args.manifest, suite_id=args.suite_id), indent=2, sort_keys=True))
        return 0
    summary = run_manifest(
        manifest_path=args.manifest,
        suite_id=args.suite_id,
        parallel_jobs=int(args.parallel_jobs),
        seed=args.seed,
        show_child_output=bool(args.show_child_output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
