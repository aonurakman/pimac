"""Backfill rollout-level final evaluation returns and optional extra-count re-evaluations.

This is intended for final multi-seed task runs stored under `results/`.
In its original mode it reevaluates saved checkpoints using the persisted task/config snapshots
and writes `eval_rollout_returns.csv` next to the existing `eval_by_count.csv`.

It also supports a supplementary reanalysis mode for dynamic-team suites: evaluate only missing
extra counts (for example `n=9` for RWARE), merge those with the existing `eval_by_count.csv`,
and write recomputed suite-level CSV summaries under a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import get_algorithm_class
from plotting.plot_learning_curves import PRESETS
from robotic_warehouse_dynamic import run as rware_run
from robotic_warehouse_dynamic.utils import close_env_cache as rware_close_env_cache
from robotic_warehouse_dynamic.utils import evaluate_one_count as rware_evaluate_one_count
from robotic_warehouse_dynamic.utils import grouped_eval_metrics as rware_grouped_eval_metrics
from simple_spread_dynamic_hard import run as spread_run
from simple_spread_dynamic_hard.utils import close_env_cache as spread_close_env_cache
from simple_spread_dynamic_hard.utils import evaluate_one_count as spread_evaluate_one_count
from simple_spread_dynamic_hard.utils import grouped_eval_metrics as spread_grouped_eval_metrics
from lbf_hard import run as lbf_run
from lbf_hard.utils import close_env_cache as lbf_close_env_cache
from lbf_hard.utils import evaluate_one_count as lbf_evaluate_one_count
from lbf_hard.utils import grouped_eval_metrics as lbf_grouped_eval_metrics


@dataclass(frozen=True)
class TaskHelper:
    task_names: tuple[str, ...]
    build_env_spec: Callable[[dict, int], object]
    grouped_eval_metrics: Callable[[dict, Sequence[Any]], dict[str, float]]
    evaluate_checkpoint: Callable[..., list[dict[str, object]]]


def _evaluate_spread_checkpoint(
    learner,
    task_config: dict,
    *,
    seed: int,
    checkpoint_episode: int,
    phase: str,
    counts: Sequence[int],
) -> list[dict[str, object]]:
    env_cache: dict = {}
    try:
        rows: list[dict[str, object]] = []
        for n_agents in [int(value) for value in counts]:
            returns = spread_evaluate_one_count(
                learner,
                task_config,
                env_cache,
                seed=seed,
                n_agents=n_agents,
                rollout_count=int(task_config["test_rollouts"]),
                seed_offset=int(task_config["test_seed_offset"]),
                make_env_fn=spread_run.make_env,
            )
            rows.extend(
                {
                    "phase": phase,
                    "checkpoint_episode": int(checkpoint_episode),
                    "n_agents": int(n_agents),
                    "rollout_index": int(rollout_index),
                    "episode_return": float(episode_return),
                }
                for rollout_index, episode_return in enumerate(returns)
            )
        return rows
    finally:
        spread_close_env_cache(env_cache)


def _evaluate_lbf_checkpoint(
    learner,
    task_config: dict,
    *,
    seed: int,
    checkpoint_episode: int,
    phase: str,
    counts: Sequence[int],
) -> list[dict[str, object]]:
    env_cache: dict = {}
    env_spec = lbf_run.build_env_spec(task_config, seed)
    try:
        rows: list[dict[str, object]] = []
        for n_agents in [int(value) for value in counts]:
            returns = lbf_evaluate_one_count(
                learner,
                task_config,
                env_spec,
                env_cache,
                seed=seed,
                n_agents=n_agents,
                rollout_count=int(task_config["test_rollouts"]),
                seed_offset=int(task_config["test_seed_offset"]),
                make_env_fn=lbf_run.make_env,
                summarize_episode_return_fn=lbf_run.summarize_episode_return,
            )
            rows.extend(
                {
                    "phase": phase,
                    "checkpoint_episode": int(checkpoint_episode),
                    "n_agents": int(n_agents),
                    "rollout_index": int(rollout_index),
                    "episode_return": float(episode_return),
                }
                for rollout_index, episode_return in enumerate(returns)
            )
        return rows
    finally:
        lbf_close_env_cache(env_cache)


def _evaluate_rware_checkpoint(
    learner,
    task_config: dict,
    *,
    seed: int,
    checkpoint_episode: int,
    phase: str,
    counts: Sequence[int],
) -> list[dict[str, object]]:
    env_cache: dict = {}
    env_spec = rware_run.build_env_spec(task_config, seed)
    try:
        rows: list[dict[str, object]] = []
        for n_agents in [int(value) for value in counts]:
            returns = rware_evaluate_one_count(
                learner,
                task_config,
                env_spec,
                env_cache,
                seed=seed,
                n_agents=n_agents,
                rollout_count=int(task_config["test_rollouts"]),
                seed_offset=int(task_config["test_seed_offset"]),
                make_env_fn=rware_run.make_env,
            )
            rows.extend(
                {
                    "phase": phase,
                    "checkpoint_episode": int(checkpoint_episode),
                    "n_agents": int(n_agents),
                    "rollout_index": int(rollout_index),
                    "episode_return": float(episode_return),
                }
                for rollout_index, episode_return in enumerate(returns)
            )
        return rows
    finally:
        rware_close_env_cache(env_cache)


TASK_HELPERS: dict[str, TaskHelper] = {
    "simple_spread_dynamic_hard/task.json": TaskHelper(
        task_names=("simple_spread_dynamic_hard",),
        build_env_spec=spread_run.build_env_spec,
        grouped_eval_metrics=spread_grouped_eval_metrics,
        evaluate_checkpoint=_evaluate_spread_checkpoint,
    ),
    "lbf_hard/task.json": TaskHelper(
        task_names=("lbf_hard",),
        build_env_spec=lbf_run.build_env_spec,
        grouped_eval_metrics=lbf_grouped_eval_metrics,
        evaluate_checkpoint=_evaluate_lbf_checkpoint,
    ),
    "robotic_warehouse_dynamic/task.json": TaskHelper(
        task_names=("robotic_warehouse_dynamic",),
        build_env_spec=rware_run.build_env_spec,
        grouped_eval_metrics=rware_grouped_eval_metrics,
        evaluate_checkpoint=_evaluate_rware_checkpoint,
    ),
}

PHASE_TO_CHECKPOINT = {
    "final_checkpoint_test": "final_checkpoint.pt",
    "best_checkpoint_test": "best_checkpoint.pt",
}


def _task_helper_key_for_snapshot(snapshot: dict) -> str:
    task_config_path = str(snapshot.get("task_config_path", ""))
    for key in TASK_HELPERS:
        if task_config_path.endswith(key):
            return key

    task_name = str(snapshot.get("task_config", {}).get("task_name", ""))
    for key, helper in TASK_HELPERS.items():
        if task_name in helper.task_names:
            return key
    raise KeyError(f"Unsupported task snapshot for backfill: task_config_path={task_config_path!r}, task_name={task_name!r}")


def _merge_eval_counts(existing_counts: Sequence[int], extra_counts: Sequence[int]) -> list[int]:
    return sorted(dict.fromkeys(int(value) for value in [*existing_counts, *extra_counts]))


def _csv_rows_to_eval_results(task_helper_key: str, rows: Sequence[dict[str, object]]) -> list[Any]:
    if task_helper_key == "simple_spread_dynamic_hard/task.json":
        from simple_spread_dynamic_hard.utils import EvalResult
    elif task_helper_key == "lbf_hard/task.json":
        from lbf_hard.utils import EvalResult
    elif task_helper_key == "robotic_warehouse_dynamic/task.json":
        from robotic_warehouse_dynamic.utils import EvalResult
    else:  # pragma: no cover
        raise KeyError(f"Unsupported task helper key: {task_helper_key}")

    return [
        EvalResult(
            phase=str(row["phase"]),
            checkpoint_episode=int(row["checkpoint_episode"]),
            n_agents=int(row["n_agents"]),
            rollout_count=int(row["rollout_count"]),
            return_mean=float(row["return_mean"]),
            return_std=float(row["return_std"]),
            return_min=float(row["return_min"]),
            return_max=float(row["return_max"]),
        )
        for row in rows
    ]


def _recomputed_task_config(task_config: dict, *, extra_counts: Sequence[int], override_test_counts: Sequence[int]) -> dict:
    updated = dict(task_config)
    updated["eval_counts"] = _merge_eval_counts(task_config.get("eval_counts", []), extra_counts)
    if override_test_counts:
        updated["test_counts"] = [int(value) for value in override_test_counts]
    return updated


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_eval_by_count(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (run_dir / "eval_by_count.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "phase": str(row["phase"]),
                    "checkpoint_episode": int(row["checkpoint_episode"]),
                    "n_agents": int(row["n_agents"]),
                    "rollout_count": int(row["rollout_count"]),
                    "return_mean": float(row["return_mean"]),
                    "return_std": float(row["return_std"]),
                    "return_min": float(row["return_min"]),
                    "return_max": float(row["return_max"]),
                }
            )
    return rows


def _aggregate_rollout_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        key = (str(row["phase"]), int(row["checkpoint_episode"]), int(row["n_agents"]))
        grouped.setdefault(key, []).append(float(row["episode_return"]))

    aggregated_rows: list[dict[str, object]] = []
    for (phase, checkpoint_episode, n_agents), returns in sorted(grouped.items()):
        aggregated_rows.append(
            {
                "phase": phase,
                "checkpoint_episode": checkpoint_episode,
                "n_agents": n_agents,
                "rollout_count": len(returns),
                "return_mean": float(sum(returns) / len(returns)),
                "return_std": float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0,
                "return_min": float(min(returns)),
                "return_max": float(max(returns)),
            }
        )
    return aggregated_rows


def backfill_run(run_dir: Path, *, task_helper_key: str, force: bool) -> Path | None:
    output_path = run_dir / "eval_rollout_returns.csv"
    if output_path.exists() and not force:
        return None

    snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    task_config = snapshot["task_config"]
    algorithm = str(snapshot["algorithm"])
    learner_config = snapshot["algorithm_config"]
    seed = int(summary["seed"])
    checkpoint_episode = int(task_config["episodes"])

    if task_helper_key not in TASK_HELPERS:
        raise KeyError(f"Unsupported task for rollout backfill: {task_helper_key}")
    task_helper = TASK_HELPERS[task_helper_key]

    env_spec = task_helper.build_env_spec(task_config, seed)
    learner_cls = get_algorithm_class(algorithm)
    rows: list[dict[str, object]] = []

    final_checkpoint_path = run_dir / "final_checkpoint.pt"
    learner = learner_cls.load_checkpoint(
        final_checkpoint_path,
        env_spec=env_spec,
        config=learner_config,
        device="cpu",
    )
    rows.extend(
        task_helper.evaluate_checkpoint(
            learner,
            task_config,
            seed=seed,
            checkpoint_episode=checkpoint_episode,
            phase="final_checkpoint_test",
            counts=tuple(int(value) for value in task_config["eval_counts"]),
        )
    )

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    if best_checkpoint_path.is_file():
        learner = learner_cls.load_checkpoint(
            best_checkpoint_path,
            env_spec=env_spec,
            config=learner_config,
            device="cpu",
        )
        rows.extend(
            task_helper.evaluate_checkpoint(
                learner,
                task_config,
                seed=seed,
                checkpoint_episode=checkpoint_episode,
                phase="best_checkpoint_test",
                counts=tuple(int(value) for value in task_config["eval_counts"]),
            )
        )

    write_csv(output_path, rows)
    return output_path


def _discover_run_dirs_from_root(run_root: Path) -> list[Path]:
    return sorted(path.parent for path in run_root.glob("*/*/config_snapshot.json"))


def _evaluate_extra_counts_for_phase(
    run_dir: Path,
    *,
    snapshot: dict,
    summary: dict,
    task_helper_key: str,
    phase: str,
    counts: Sequence[int],
) -> list[dict[str, object]]:
    if not counts:
        return []
    checkpoint_name = PHASE_TO_CHECKPOINT.get(phase)
    if checkpoint_name is None:
        return []
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        return []

    task_helper = TASK_HELPERS[task_helper_key]
    task_config = snapshot["task_config"]
    algorithm = str(snapshot["algorithm"])
    learner_config = snapshot["algorithm_config"]
    seed = int(summary["seed"])
    checkpoint_episode = int(summary["episodes"])
    env_spec = task_helper.build_env_spec(task_config, seed)
    learner_cls = get_algorithm_class(algorithm)
    learner = learner_cls.load_checkpoint(
        checkpoint_path,
        env_spec=env_spec,
        config=learner_config,
        device="cpu",
    )
    return task_helper.evaluate_checkpoint(
        learner,
        task_config,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
        phase=phase,
        counts=counts,
    )


def recompute_suite_with_extra_counts(
    *,
    run_root: Path,
    output_dir: Path,
    extra_counts: Sequence[int],
    override_test_counts: Sequence[int],
) -> list[Path]:
    run_dirs = _discover_run_dirs_from_root(run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {run_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_rows: list[dict[str, object]] = []
    per_config_rows: list[dict[str, object]] = []
    countwise_rows: list[dict[str, object]] = []
    by_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    by_config_count: dict[tuple[str, str], dict[int, list[float]]] = {}

    total = len(run_dirs)
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{total}] {run_dir.relative_to(run_root)}", flush=True)

        snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        task_helper_key = _task_helper_key_for_snapshot(snapshot)
        task_helper = TASK_HELPERS[task_helper_key]
        existing_eval_rows = _read_eval_by_count(run_dir)

        extra_rollout_rows: list[dict[str, object]] = []
        available_phases = sorted({str(row["phase"]) for row in existing_eval_rows if str(row["phase"]) in PHASE_TO_CHECKPOINT})
        for phase in available_phases:
            existing_phase_counts = {int(row["n_agents"]) for row in existing_eval_rows if str(row["phase"]) == phase}
            missing_counts = [int(count) for count in extra_counts if int(count) not in existing_phase_counts]
            extra_rollout_rows.extend(
                _evaluate_extra_counts_for_phase(
                    run_dir,
                    snapshot=snapshot,
                    summary=summary,
                    task_helper_key=task_helper_key,
                    phase=phase,
                    counts=missing_counts,
                )
            )

        extra_eval_rows = _aggregate_rollout_rows(extra_rollout_rows)
        merged_rows = list(existing_eval_rows)
        if extra_eval_rows:
            kept_existing = {
                (str(row["phase"]), int(row["checkpoint_episode"]), int(row["n_agents"])): row
                for row in existing_eval_rows
            }
            for row in extra_eval_rows:
                kept_existing[(str(row["phase"]), int(row["checkpoint_episode"]), int(row["n_agents"]))] = row
            merged_rows = [kept_existing[key] for key in sorted(kept_existing)]

        updated_task_config = _recomputed_task_config(
            snapshot["task_config"],
            extra_counts=extra_counts,
            override_test_counts=override_test_counts,
        )
        selection_checkpoint = str(summary["test"].get("selection_checkpoint", "final_checkpoint"))
        selection_phase = f"{selection_checkpoint}_test"

        selected_rows = [
            row
            for row in merged_rows
            if str(row["phase"]) == selection_phase
        ]
        selected_metrics = task_helper.grouped_eval_metrics(
            updated_task_config,
            _csv_rows_to_eval_results(task_helper_key, selected_rows),
        )

        run_name = run_dir.name
        algorithm = str(snapshot["algorithm"])
        prefix = f"final_{updated_task_config['task_name']}_{algorithm}_"
        config_name = run_name.removeprefix(prefix).rsplit("_s", 1)[0]

        n9_values = [float(row["return_mean"]) for row in selected_rows if int(row["n_agents"]) == 9]
        n10_values = [float(row["return_mean"]) for row in selected_rows if int(row["n_agents"]) == 10]

        run_row = {
            "algorithm": algorithm,
            "config": config_name,
            "seed": int(summary["seed"]),
            "selection_checkpoint": selection_checkpoint,
            "train_mean": float(summary["train"]["final_moving_average"]),
            "val_mean": float(selected_metrics["validation_counts_mean"]),
            "test_mean": float(selected_metrics["test_counts_mean"]),
            "overall_eval_mean": float(selected_metrics["overall_eval_mean"]),
            "selection_score": float(selected_metrics["selection_score"]),
            "n9_mean": float(n9_values[0]) if n9_values else float("nan"),
            "n10_mean": float(n10_values[0]) if n10_values else float("nan"),
        }
        per_run_rows.append(run_row)
        by_config.setdefault((algorithm, config_name), []).append(run_row)
        by_config_count.setdefault((algorithm, config_name), {})
        for row in selected_rows:
            by_config_count[(algorithm, config_name)].setdefault(int(row["n_agents"]), []).append(float(row["return_mean"]))

    for (algorithm, config_name), rows in sorted(by_config.items()):
        def _mean(key: str) -> float:
            values = [float(row[key]) for row in rows]
            return float(sum(values) / len(values))

        def _sd(key: str) -> float:
            values = [float(row[key]) for row in rows]
            return float(statistics.stdev(values)) if len(values) > 1 else 0.0

        per_config_rows.append(
            {
                "algorithm": algorithm,
                "config": config_name,
                "train_mean": _mean("train_mean"),
                "train_sd": _sd("train_mean"),
                "val_mean": _mean("val_mean"),
                "val_sd": _sd("val_mean"),
                "test_mean": _mean("test_mean"),
                "test_sd": _sd("test_mean"),
                "overall_eval_mean": _mean("overall_eval_mean"),
                "overall_eval_sd": _sd("overall_eval_mean"),
                "selection_score": _mean("selection_score"),
                "selection_score_sd": _sd("selection_score"),
                "n9_mean": _mean("n9_mean"),
                "n9_sd": _sd("n9_mean"),
                "n10_mean": _mean("n10_mean"),
                "n10_sd": _sd("n10_mean"),
            }
        )

    for (algorithm, config_name), count_map in sorted(by_config_count.items()):
        for n_agents, values in sorted(count_map.items()):
            countwise_rows.append(
                {
                    "algorithm": algorithm,
                    "config": config_name,
                    "n_agents": int(n_agents),
                    "mean_return": float(sum(values) / len(values)),
                    "sd_return": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                }
            )

    outputs = [
        output_dir / "per_run_recomputed.csv",
        output_dir / "per_config_recomputed.csv",
        output_dir / "countwise_returns_recomputed.csv",
    ]
    write_csv(outputs[0], per_run_rows)
    write_csv(outputs[1], per_config_rows)
    write_csv(outputs[2], countwise_rows)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill rollout-level final evaluation returns for selected final-run presets.")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="Preset(s) whose run directories should be backfilled. Defaults to all supported presets.",
    )
    parser.add_argument("--run-root", type=Path, help="Run-root containing task-run subdirectories for supplementary reanalysis.")
    parser.add_argument(
        "--extra-count",
        dest="extra_counts",
        type=int,
        action="append",
        default=[],
        help="Extra team-size count(s) to evaluate and merge with existing eval_by_count.csv results.",
    )
    parser.add_argument(
        "--override-test-count",
        dest="override_test_counts",
        type=int,
        action="append",
        default=[],
        help="Override test-count split used when recomputing supplementary suite summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for supplementary recomputed CSVs. Required when --run-root is used.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing eval_rollout_returns.csv files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_root is not None:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required when --run-root is used.")
        outputs = recompute_suite_with_extra_counts(
            run_root=args.run_root,
            output_dir=args.output_dir,
            extra_counts=tuple(int(value) for value in args.extra_counts),
            override_test_counts=tuple(int(value) for value in args.override_test_counts),
        )
        for output_path in outputs:
            print(output_path)
        return 0

    selected_presets = args.preset or sorted(TASK_HELPERS)
    if not args.preset:
        selected_presets = [name for name, preset in PRESETS.items() if preset.task_config_path in TASK_HELPERS]

    seen_run_dirs: set[Path] = set()
    run_dirs: list[tuple[Path, str]] = []
    for preset_name in selected_presets:
        preset = PRESETS[preset_name]
        if preset.task_config_path not in TASK_HELPERS:
            continue
        for series in preset.series:
            for train_history_path in sorted(REPO_ROOT.glob(series.glob_pattern)):
                run_dir = train_history_path.parent
                if run_dir in seen_run_dirs:
                    continue
                seen_run_dirs.add(run_dir)
                run_dirs.append((run_dir, preset.task_config_path))

    total = len(run_dirs)
    for index, (run_dir, task_helper_key) in enumerate(run_dirs, start=1):
        print(f"[{index}/{total}] {run_dir.relative_to(REPO_ROOT)}", flush=True)
        written_path = backfill_run(run_dir, task_helper_key=task_helper_key, force=bool(args.force))
        if written_path is not None:
            print(written_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
