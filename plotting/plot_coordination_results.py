"""Plot compact scalar coordination heatmaps for selected final-result models."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTUNA_DIR = PROJECT_ROOT / "optuna"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPTUNA_DIR) not in sys.path:
    sys.path.insert(0, str(OPTUNA_DIR))

from algorithms.registry import get_algorithm_class
from analyze import (
    PIMAC_TRACE_ALGORITHMS,
    _build_trial_count_summary_rows,
    _cosine_similarity,
    _extract_teacher_details,
    _task_is_dynamic,
    _task_script_module,
    _transform_obs,
)
from plotting import display_algorithm_name, plot_task_count_alignment_gate_heatmap
from utils import load_json, write_csv, write_json


REPO_ROOT = PROJECT_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "plotting" / "plots" / "coordination"
DEFAULT_OUTPUT_NAME = "selected_pc3d"
DEFAULT_ROLLOUTS_PER_COUNT = 8
DEFAULT_OVERWRITE = False
DEFAULT_DPI = 300
DEFAULT_FONT_FAMILY = "Charter"
DEFAULT_SHOW_TITLE = False
DEFAULT_TASK_WORKERS = 3

# Main user-tunable selection table. These defaults intentionally mirror
# plotting/plot_learning_curves.py and tables.ipynb.
DEFAULT_TASK_MODEL_SELECTIONS: dict[str, dict[str, str]] = {
    "spread": {
        "task_id": "simple_spread_dynamic_hard",
        "task_label": "Spread",
        "results_dir": "results/spread",
        "algorithm": "pimac_v6",
        "config": "active_03",
    },
    "lbf_hard": {
        "task_id": "lbf_hard",
        "task_label": "LBF",
        "results_dir": "results/lbf_hard",
        "algorithm": "pimac_v6",
        "config": "active_01",
    },
    "rware": {
        "task_id": "robotic_warehouse_dynamic",
        "task_label": "RWARE",
        "results_dir": "results/rware",
        "algorithm": "pimac_v6",
        "config": "active_01",
    },
}
DEFAULT_TASK_ORDER: tuple[str, ...] = ("spread", "lbf_hard", "rware")
TASK_ALIASES: dict[str, str] = {
    "spread": "spread",
    "simple_spread_dynamic_hard": "spread",
    "lbf": "lbf_hard",
    "lbf_hard": "lbf_hard",
    "rware": "rware",
    "robotic_warehouse_dynamic": "rware",
}


@dataclass(frozen=True)
class CoordinationTaskSpec:
    task_key: str
    task_id: str
    task_label: str
    results_dir: Path
    algorithm: str
    config_name: str


@dataclass(frozen=True)
class ResultRun:
    algorithm: str
    config_name: str
    task_id: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    config_snapshot: dict[str, Any]


@dataclass(frozen=True)
class CoordinationTrace:
    task: CoordinationTaskSpec
    counts: tuple[int, ...]
    seeds: tuple[int, ...]
    trial_count_rows: list[dict[str, Any]]


def _checkpoint_path_for_run(run_dir: Path) -> Path:
    final_checkpoint = run_dir / "final_checkpoint.pt"
    if final_checkpoint.is_file():
        return final_checkpoint
    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.is_file():
        return best_checkpoint
    raise FileNotFoundError(f"Missing checkpoint in run directory: {run_dir}")


def _run_label(run: ResultRun) -> str:
    return f"s{int(run.seed)}"


def _task_output_name(task: CoordinationTaskSpec) -> str:
    return task.task_label.lower().replace(" ", "_")


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _default_output_dir(task_specs: Sequence[CoordinationTaskSpec]) -> Path:
    model_pairs = {(task.algorithm, task.config_name) for task in task_specs}
    if len(model_pairs) == 1:
        algorithm, config_name = next(iter(model_pairs))
        return DEFAULT_OUTPUT_ROOT / algorithm / config_name
    return DEFAULT_OUTPUT_ROOT / DEFAULT_OUTPUT_NAME


def _parse_task_model_overrides(raw_overrides: Sequence[str] | None) -> dict[str, tuple[str, str]]:
    overrides: dict[str, tuple[str, str]] = {}
    for raw_override in raw_overrides or []:
        if "=" not in raw_override or ":" not in raw_override:
            raise ValueError("Task model overrides must use TASK=ALGORITHM:CONFIG, e.g. rware=pimac_v6:active_01.")
        raw_task, raw_model = raw_override.split("=", 1)
        algorithm, config_name = raw_model.split(":", 1)
        task_key = TASK_ALIASES.get(raw_task, raw_task)
        overrides[task_key] = (algorithm, config_name)
    return overrides


def _resolve_task_specs(
    *,
    tasks: Sequence[str] | None,
    task_results_dirs: Sequence[Path] | None,
    task_model_overrides: Sequence[str] | None,
    algorithm: str | None,
    config_name: str | None,
) -> list[CoordinationTaskSpec]:
    if (algorithm is None) != (config_name is None):
        raise ValueError("--algorithm and --config must be provided together when overriding the default task selections.")

    task_keys = [TASK_ALIASES.get(str(task), str(task)) for task in (tasks or DEFAULT_TASK_ORDER)]
    dir_args = list(task_results_dirs or [])
    if dir_args and len(task_keys) != len(dir_args):
        raise ValueError("--task and --task-results-dir must be provided the same number of times.")

    model_overrides = _parse_task_model_overrides(task_model_overrides)
    specs: list[CoordinationTaskSpec] = []
    for index, task_key in enumerate(task_keys):
        default_selection = DEFAULT_TASK_MODEL_SELECTIONS.get(task_key)
        if default_selection is None:
            if not dir_args or algorithm is None or config_name is None:
                raise ValueError(
                    f"No default selection for task {task_key!r}; pass --task-results-dir plus --algorithm/--config."
                )
            task_id = task_key
            task_label = task_key
            results_dir = dir_args[index]
            selected_algorithm = algorithm
            selected_config = config_name
        else:
            task_id = str(default_selection["task_id"])
            task_label = str(default_selection["task_label"])
            results_dir = Path(dir_args[index]) if dir_args else _resolve_repo_path(default_selection["results_dir"])
            selected_algorithm = algorithm or str(default_selection["algorithm"])
            selected_config = config_name or str(default_selection["config"])

        if task_key in model_overrides:
            selected_algorithm, selected_config = model_overrides[task_key]

        specs.append(
            CoordinationTaskSpec(
                task_key=task_key,
                task_id=task_id,
                task_label=task_label,
                results_dir=_resolve_repo_path(results_dir),
                algorithm=selected_algorithm,
                config_name=selected_config,
            )
        )
    return specs


def discover_result_runs(*, task: CoordinationTaskSpec) -> list[ResultRun]:
    algorithm_dir = task.results_dir / task.algorithm
    if not algorithm_dir.is_dir():
        raise FileNotFoundError(f"Algorithm results directory does not exist: {algorithm_dir}")

    discovered: list[ResultRun] = []
    for config_snapshot_path in sorted(algorithm_dir.glob("*/config_snapshot.json")):
        run_dir = config_snapshot_path.parent
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        config_snapshot = load_json(config_snapshot_path)
        if str(config_snapshot.get("algorithm")) != task.algorithm:
            continue
        task_config = dict(config_snapshot.get("task_config", {}))
        if str(task_config.get("task_name")) != task.task_id:
            continue
        snapshot_config_name = Path(str(config_snapshot.get("algorithm_config_path", ""))).stem
        if snapshot_config_name != task.config_name:
            continue
        summary = load_json(summary_path)
        discovered.append(
            ResultRun(
                algorithm=task.algorithm,
                config_name=task.config_name,
                task_id=task.task_id,
                seed=int(summary["seed"]),
                run_dir=run_dir,
                checkpoint_path=_checkpoint_path_for_run(run_dir),
                config_snapshot=config_snapshot,
            )
        )
    return discovered


def _load_result_policy(run: ResultRun):
    task_config = dict(run.config_snapshot["task_config"])
    task_script = _task_script_module(run.task_id)
    env_spec = task_script.build_env_spec(task_config, run.seed)
    learner_cls = get_algorithm_class(run.algorithm)
    learner = learner_cls.load_checkpoint(
        run.checkpoint_path,
        env_spec=env_spec,
        config=dict(run.config_snapshot["algorithm_config"]),
        device="cpu",
    )
    learner.set_eval_mode()
    return learner, task_config, task_script, env_spec


def _analysis_counts(task_config: dict[str, Any], counts: Sequence[int] | None) -> tuple[int, ...]:
    if counts is not None:
        return tuple(int(value) for value in counts)
    if _task_is_dynamic(task_config):
        return tuple(int(value) for value in task_config["eval_counts"])
    return (int(task_config["n_agents"]),)


def collect_coordination_trace(
    *,
    task: CoordinationTaskSpec,
    rollouts_per_count: int,
    counts: Sequence[int] | None,
) -> CoordinationTrace:
    if task.algorithm not in PIMAC_TRACE_ALGORITHMS and task.algorithm != "pimac_v6_ablation":
        raise ValueError(f"Coordination tracing is only supported for PIMAC-style algorithms, got: {task.algorithm}")

    target_runs = discover_result_runs(task=task)
    if not target_runs:
        raise RuntimeError(
            f"No result runs found for task={task.task_id}, algorithm={task.algorithm}, "
            f"config={task.config_name} in {task.results_dir}."
        )

    first_task_config = dict(target_runs[0].config_snapshot["task_config"])
    dynamic_task = _task_is_dynamic(first_task_config)
    analysis_counts = _analysis_counts(first_task_config, counts)
    student_rows: list[dict[str, Any]] = []

    for run in target_runs:
        run_label = _run_label(run)
        learner, task_config, task_script, env_spec = _load_result_policy(run)
        learner.set_eval_mode()
        for n_agents in analysis_counts:
            print(
                f"[{task.task_label}] {display_algorithm_name(task.algorithm)} {task.config_name} {run_label} n={int(n_agents)}",
                flush=True,
            )
            for episode_index in range(int(rollouts_per_count)):
                rollout_seed = int(run.seed + (1000 * int(n_agents)) + episode_index)
                torch.manual_seed(rollout_seed)
                if dynamic_task:
                    env = task_script.make_env(task_config, seed=rollout_seed, n_agents=int(n_agents), render_mode=None)
                else:
                    env = task_script.make_env(task_config, seed=rollout_seed, render_mode=None)
                obs, _ = env.reset(seed=rollout_seed)
                learner.reset_episode()
                step_index = 0
                try:
                    while obs:
                        agent_ids = sorted(obs.keys(), key=str)
                        if not agent_ids:
                            break

                        step_student_rows: list[dict[str, Any]] = []
                        step_student_contexts: list[np.ndarray] = []
                        actions: dict[str, int] = {}

                        with torch.no_grad():
                            for agent_position, agent_id in enumerate(agent_ids):
                                obs_value = _transform_obs(task_script, env_spec, obs[agent_id])
                                obs_tensor = torch.as_tensor(obs_value, dtype=torch.float32, device=learner.device).view(1, 1, -1)
                                hidden_dim = learner.actor_net.rnn.hidden_size
                                initial_hidden_state = learner._get_hidden_state(agent_id, hidden_dim)
                                action_logits, updated_hidden_state, aux = learner.actor_net(
                                    obs_tensor, initial_hidden_state, return_aux=True
                                )
                                learner._set_hidden_state(agent_id, updated_hidden_state)
                                logits = action_logits.squeeze(0).squeeze(0)
                                action = int(torch.distributions.Categorical(logits=logits).sample().item())
                                actions[str(agent_id)] = action

                                student_ctx = aux["ctx_mu"].squeeze(0).squeeze(0).detach().cpu().numpy()
                                signal_key = (
                                    "ctx_reliance"
                                    if "ctx_reliance" in aux
                                    else ("ctx_log_uncertainty" if "ctx_log_uncertainty" in aux else "ctx_logvar")
                                )
                                student_gate_signal = aux[signal_key].squeeze(0).squeeze(0).detach().cpu().numpy()
                                row = {
                                    "task": task.task_id,
                                    "task_label": task.task_label,
                                    "algorithm": run.algorithm,
                                    "rank": 0,
                                    "trial_number": int(run.seed),
                                    "config_name": run.config_name,
                                    "run_label": run_label,
                                    "episode_index": int(episode_index),
                                    "step_index": int(step_index),
                                    "seed": int(rollout_seed),
                                    "n_agents": int(n_agents),
                                    "agent_id": str(agent_id),
                                    "agent_position": int(agent_position),
                                    "action": int(action),
                                    "ctx_signal_mean": float(np.mean(student_gate_signal)),
                                    "student_ctx_norm": float(np.linalg.norm(student_ctx)),
                                }
                                if signal_key == "ctx_reliance":
                                    row["ctx_reliance_mean"] = float(np.mean(student_gate_signal))
                                elif signal_key == "ctx_log_uncertainty":
                                    row["ctx_log_uncertainty_mean"] = float(np.mean(student_gate_signal))
                                else:
                                    row["ctx_logvar_mean"] = float(np.mean(student_gate_signal))
                                if "gate" in aux:
                                    row["gate"] = float(aux["gate"].squeeze(0).squeeze(0).detach().cpu().item())
                                if "delta_w_norm" in aux:
                                    row["delta_w_norm"] = float(aux["delta_w_norm"].squeeze(0).squeeze(0).detach().cpu().item())
                                if "delta_b_norm" in aux:
                                    row["delta_b_norm"] = float(aux["delta_b_norm"].squeeze(0).squeeze(0).detach().cpu().item())
                                step_student_rows.append(row)
                                step_student_contexts.append(np.asarray(student_ctx, dtype=np.float32))

                            stacked_obs = np.stack(
                                [_transform_obs(task_script, env_spec, obs[agent_id]) for agent_id in agent_ids],
                                axis=0,
                            )
                            obs_tensor = torch.as_tensor(stacked_obs, dtype=torch.float32, device=learner.device).view(
                                1, 1, len(agent_ids), -1
                            )
                            active_mask = torch.ones(1, 1, len(agent_ids), dtype=torch.float32, device=learner.device)
                            _, teacher_context = _extract_teacher_details(
                                learner.critic(obs_tensor, active_mask, return_details=True)
                            )
                            teacher_context_matrix = teacher_context.squeeze(0).squeeze(0).detach().cpu().numpy()

                        for row, student_ctx, teacher_ctx in zip(step_student_rows, step_student_contexts, teacher_context_matrix):
                            row["teacher_alignment_cosine"] = _cosine_similarity(student_ctx, teacher_ctx)
                            row["teacher_alignment_mse"] = float(np.mean((student_ctx - teacher_ctx.reshape(-1)) ** 2))
                            row["teacher_ctx_norm"] = float(np.linalg.norm(teacher_ctx))
                            student_rows.append(row)

                        obs, _, terminations, truncations, _ = env.step(actions)
                        done = {
                            agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in set(terminations) | set(truncations)
                        }
                        obs = {
                            str(agent_id): np.asarray(value, dtype=np.float32)
                            for agent_id, value in obs.items()
                            if not done.get(agent_id, False)
                        }
                        step_index += 1
                finally:
                    env.close()

    trial_count_rows = _build_trial_count_summary_rows(student_rows)
    for row in trial_count_rows:
        row["task"] = task.task_id
        row["task_label"] = task.task_label
        row["config_name"] = task.config_name

    return CoordinationTrace(
        task=task,
        counts=analysis_counts,
        seeds=tuple(int(run.seed) for run in target_runs),
        trial_count_rows=trial_count_rows,
    )


def _collect_coordination_trace_job(
    task: CoordinationTaskSpec,
    rollouts_per_count: int,
    counts: Sequence[int] | None,
) -> CoordinationTrace:
    return collect_coordination_trace(task=task, rollouts_per_count=rollouts_per_count, counts=counts)


def collect_traces_parallel(
    *,
    task_specs: Sequence[CoordinationTaskSpec],
    rollouts_per_count: int,
    counts: Sequence[int] | None,
    task_workers: int,
) -> list[CoordinationTrace]:
    workers = max(1, min(int(task_workers), len(task_specs)))
    if workers == 1:
        return [
            collect_coordination_trace(task=task, rollouts_per_count=rollouts_per_count, counts=counts)
            for task in task_specs
        ]

    traces_by_index: list[CoordinationTrace | None] = [None] * len(task_specs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_collect_coordination_trace_job, task, int(rollouts_per_count), counts): index
            for index, task in enumerate(task_specs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            traces_by_index[index] = future.result()

    return [trace for trace in traces_by_index if trace is not None]


def build_task_alignment_rows(traces: Sequence[CoordinationTrace]) -> list[dict[str, Any]]:
    heatmap_rows: list[dict[str, Any]] = []
    for trace in traces:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in trace.trial_count_rows:
            grouped[int(row["n_agents"])].append(row)
        for n_agents, rows in sorted(grouped.items()):
            alignment_values = [float(row["teacher_alignment_cosine_mean"]) for row in rows]
            gate_values = [float(row["gate_mean"]) for row in rows if row.get("gate_mean") is not None]
            heatmap_rows.append(
                {
                    "task": trace.task.task_id,
                    "task_key": trace.task.task_key,
                    "task_label": trace.task.task_label,
                    "algorithm": trace.task.algorithm,
                    "config_name": trace.task.config_name,
                    "n_agents": int(n_agents),
                    "model_count": int(len(rows)),
                    "alignment_mean": float(np.mean(alignment_values)),
                    "alignment_std": float(np.std(alignment_values, ddof=1)) if len(alignment_values) > 1 else 0.0,
                    "gate_mean": float(np.mean(gate_values)) if gate_values else None,
                    "gate_std": float(np.std(gate_values, ddof=1)) if len(gate_values) > 1 else 0.0,
                }
            )
    return heatmap_rows


def plot_result_coordination(
    *,
    algorithm: str | None = None,
    config_name: str | None = None,
    rollouts_per_count: int = DEFAULT_ROLLOUTS_PER_COUNT,
    counts: Sequence[int] | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
    tasks: Sequence[str] | None = None,
    task_results_dirs: Sequence[Path] | None = None,
    task_model_overrides: Sequence[str] | None = None,
    task_workers: int = DEFAULT_TASK_WORKERS,
    show_title: bool = DEFAULT_SHOW_TITLE,
    dpi: int = DEFAULT_DPI,
    font_family: str = DEFAULT_FONT_FAMILY,
) -> Path:
    task_specs = _resolve_task_specs(
        tasks=tasks,
        task_results_dirs=task_results_dirs,
        task_model_overrides=task_model_overrides,
        algorithm=algorithm,
        config_name=config_name,
    )

    resolved_output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(task_specs)
    if resolved_output_dir.exists() and overwrite:
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    traces = collect_traces_parallel(
        task_specs=task_specs,
        rollouts_per_count=int(rollouts_per_count),
        counts=counts,
        task_workers=int(task_workers),
    )

    all_alignment_rows = build_task_alignment_rows(traces)
    write_csv(resolved_output_dir / "alignment_heatmap_rows.csv", all_alignment_rows)

    plot_manifest: dict[str, Any] = {"tasks": {}}
    for trace in traces:
        task_dir = resolved_output_dir / _task_output_name(trace.task)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_alignment_rows = build_task_alignment_rows([trace])
        write_csv(task_dir / "summary_by_seed_count.csv", trace.trial_count_rows)
        write_csv(task_dir / "alignment_heatmap_rows.csv", task_alignment_rows)
        title = (
            f"{trace.task.task_label}: {display_algorithm_name(trace.task.algorithm)} {trace.task.config_name}"
            if show_title
            else None
        )
        plot_task_count_alignment_gate_heatmap(
            task_alignment_rows,
            output_path=task_dir / "alignment_gate_heatmap.png",
            title=title,
            dpi=int(dpi),
            font_family=font_family,
        )
        plot_manifest["tasks"][trace.task.task_key] = {
            "task": trace.task.task_id,
            "task_label": trace.task.task_label,
            "results_dir": str(trace.task.results_dir),
            "algorithm": trace.task.algorithm,
            "config_name": trace.task.config_name,
            "counts": [int(value) for value in trace.counts],
            "seeds": [int(value) for value in trace.seeds],
            "alignment_gate_heatmap": True,
        }

    write_json(
        resolved_output_dir / "analysis_manifest.json",
        {
            "rollouts_per_count": int(rollouts_per_count),
            "task_workers": int(max(1, min(int(task_workers), len(task_specs)))),
            "dpi": int(dpi),
            "font_family": font_family,
            "plots": plot_manifest,
        },
    )
    return resolved_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot scalar coordination heatmaps for selected PIMAC-style task/model configs. "
            "By default this uses the same PC3D selections as the main result plots."
        )
    )
    parser.add_argument("--algorithm", default=None, help="Optional algorithm override for all tasks, e.g. pimac_v6.")
    parser.add_argument("--config", default=None, help="Optional config override for all tasks, e.g. active_03.")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task id or alias to include. Defaults to spread, lbf_hard, and rware. Repeat to include multiple tasks.",
    )
    parser.add_argument(
        "--task-results-dir",
        type=Path,
        action="append",
        default=None,
        help="Results directory paired with --task. Repeat in the same order as --task for custom roots.",
    )
    parser.add_argument(
        "--task-model",
        action="append",
        default=None,
        help="Per-task model override as TASK=ALGORITHM:CONFIG, e.g. rware=pimac_v6:active_01.",
    )
    parser.add_argument("--rollouts-per-count", type=int, default=DEFAULT_ROLLOUTS_PER_COUNT)
    parser.add_argument("--counts", type=int, nargs="*", default=None)
    parser.add_argument("--task-workers", type=int, default=DEFAULT_TASK_WORKERS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    parser.add_argument("--show-title", action="store_true", default=DEFAULT_SHOW_TITLE)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--font-family", type=str, default=DEFAULT_FONT_FAMILY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = plot_result_coordination(
        algorithm=args.algorithm,
        config_name=args.config,
        rollouts_per_count=args.rollouts_per_count,
        counts=args.counts,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
        tasks=args.task,
        task_results_dirs=args.task_results_dir,
        task_model_overrides=args.task_model,
        task_workers=int(args.task_workers),
        show_title=bool(args.show_title),
        dpi=int(args.dpi),
        font_family=str(args.font_family),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
