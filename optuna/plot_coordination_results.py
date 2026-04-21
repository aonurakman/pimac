"""Plot PIMAC coordination traces for one final-results task/algorithm/config family.

This mirrors the suite-level Optuna coordination analysis, but reads directly from
multi-seed task result folders under `results/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_class
from analyze import (
    PIMAC_TRACE_ALGORITHMS,
    _build_count_summary_rows,
    _build_trial_count_summary_rows,
    _cosine_similarity,
    _extract_teacher_details,
    _mean_pairwise_distance,
    _task_is_dynamic,
    _task_script_module,
    _transform_obs,
)
from plotting import (
    build_grouped_pca_rows,
    plot_alignment_heatmap,
    plot_gate_agent_heatmap,
    plot_gate_alignment_3d,
    plot_pca_projection_grid,
)
from utils import load_json, write_csv, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


@dataclass(frozen=True)
class ResultRun:
    algorithm: str
    config_name: str
    task_id: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    config_snapshot: dict[str, Any]


def _checkpoint_path_for_run(run_dir: Path) -> Path:
    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.is_file():
        return best_checkpoint
    final_checkpoint = run_dir / "final_checkpoint.pt"
    if final_checkpoint.is_file():
        return final_checkpoint
    raise FileNotFoundError(f"Missing checkpoint in run directory: {run_dir}")


def discover_result_runs(
    *,
    task_results_dir: Path,
    task_id: str,
    algorithm: str,
    config_name: str,
) -> list[ResultRun]:
    algorithm_dir = Path(task_results_dir) / algorithm
    if not algorithm_dir.is_dir():
        raise FileNotFoundError(f"Algorithm results directory does not exist: {algorithm_dir}")

    discovered: list[ResultRun] = []
    for config_snapshot_path in sorted(algorithm_dir.glob("*/config_snapshot.json")):
        run_dir = config_snapshot_path.parent
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        config_snapshot = load_json(config_snapshot_path)
        if str(config_snapshot.get("algorithm")) != str(algorithm):
            continue
        task_config = dict(config_snapshot.get("task_config", {}))
        if str(task_config.get("task_name")) != str(task_id):
            continue
        snapshot_config_name = Path(str(config_snapshot.get("algorithm_config_path", ""))).stem
        if snapshot_config_name != str(config_name):
            continue
        summary = load_json(summary_path)
        discovered.append(
            ResultRun(
                algorithm=str(algorithm),
                config_name=str(config_name),
                task_id=str(task_id),
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


def _run_label(run: ResultRun) -> str:
    return f"s{int(run.seed)}"


def default_output_dir(*, task_results_dir: Path, algorithm: str, config_name: str) -> Path:
    return Path(task_results_dir) / "coordination_plots" / algorithm / config_name


def plot_result_coordination(
    *,
    task_results_dir: Path,
    task_id: str,
    algorithm: str,
    config_name: str,
    rollouts_per_count: int = 8,
    counts: Sequence[int] | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    if algorithm not in PIMAC_TRACE_ALGORITHMS and algorithm != "pimac_v6_ablation":
        raise ValueError(f"Coordination tracing is only supported for PIMAC-style algorithms, got: {algorithm}")

    target_runs = discover_result_runs(
        task_results_dir=task_results_dir,
        task_id=task_id,
        algorithm=algorithm,
        config_name=config_name,
    )
    if not target_runs:
        raise RuntimeError(
            f"No result runs found for task={task_id}, algorithm={algorithm}, config={config_name} in {task_results_dir}."
        )

    first_task_config = dict(target_runs[0].config_snapshot["task_config"])
    dynamic_task = _task_is_dynamic(first_task_config)
    analysis_counts = list(counts) if counts is not None else (
        [int(value) for value in first_task_config["eval_counts"]]
        if dynamic_task
        else [int(first_task_config["n_agents"])]
    )

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir(task_results_dir=task_results_dir, algorithm=algorithm, config_name=config_name)
    )
    if resolved_output_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    token_rows: list[dict[str, Any]] = []
    student_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for run in target_runs:
        run_label = _run_label(run)
        learner, task_config, task_script, env_spec = _load_result_policy(run)
        learner.set_eval_mode()
        for n_agents in analysis_counts:
            for episode_index in range(int(rollouts_per_count)):
                rollout_seed = int(run.seed + (1000 * int(n_agents)) + episode_index)
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

                        student_contexts: list[np.ndarray] = []
                        step_student_rows: list[dict[str, Any]] = []
                        token_vectors: np.ndarray | None = None
                        teacher_context_matrix: np.ndarray | None = None
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
                                categorical_distribution = torch.distributions.Categorical(logits=logits)
                                action = int(categorical_distribution.sample().item())
                                actions[str(agent_id)] = action

                                student_ctx = aux["ctx_mu"].squeeze(0).squeeze(0).detach().cpu().numpy()
                                signal_key = (
                                    "ctx_reliance"
                                    if "ctx_reliance" in aux
                                    else ("ctx_log_uncertainty" if "ctx_log_uncertainty" in aux else "ctx_logvar")
                                )
                                student_gate_signal = aux[signal_key].squeeze(0).squeeze(0).detach().cpu().numpy()
                                student_contexts.append(student_ctx)
                                student_row = {
                                    "algorithm": run.algorithm,
                                    "rank": 0,
                                    "trial_number": int(run.seed),
                                    "config_name": run.config_name,
                                    "run_label": run_label,
                                    "episode_index": episode_index,
                                    "step_index": step_index,
                                    "seed": rollout_seed,
                                    "n_agents": int(n_agents),
                                    "agent_id": str(agent_id),
                                    "agent_position": int(agent_position),
                                    "action": int(action),
                                    "ctx_signal_mean": float(np.mean(student_gate_signal)),
                                    "student_ctx_norm": float(np.linalg.norm(student_ctx)),
                                }
                                if signal_key == "ctx_reliance":
                                    student_row["ctx_reliance_mean"] = float(np.mean(student_gate_signal))
                                elif signal_key == "ctx_log_uncertainty":
                                    student_row["ctx_log_uncertainty_mean"] = float(np.mean(student_gate_signal))
                                else:
                                    student_row["ctx_logvar_mean"] = float(np.mean(student_gate_signal))
                                if "gate" in aux:
                                    student_row["gate"] = float(aux["gate"].squeeze(0).squeeze(0).detach().cpu().item())
                                if "delta_w_norm" in aux:
                                    student_row["delta_w_norm"] = float(
                                        aux["delta_w_norm"].squeeze(0).squeeze(0).detach().cpu().item()
                                    )
                                if "delta_b_norm" in aux:
                                    student_row["delta_b_norm"] = float(
                                        aux["delta_b_norm"].squeeze(0).squeeze(0).detach().cpu().item()
                                    )
                                for dim_index, value in enumerate(np.asarray(student_ctx, dtype=np.float32).reshape(-1)):
                                    student_row[f"dim_{dim_index:03d}"] = float(value)
                                step_student_rows.append(student_row)

                            stacked_obs = np.stack(
                                [_transform_obs(task_script, env_spec, obs[agent_id]) for agent_id in agent_ids],
                                axis=0,
                            )
                            obs_tensor = torch.as_tensor(stacked_obs, dtype=torch.float32, device=learner.device).view(
                                1, 1, len(agent_ids), -1
                            )
                            active_mask = torch.ones(1, 1, len(agent_ids), dtype=torch.float32, device=learner.device)
                            tokens, teacher_context = _extract_teacher_details(
                                learner.critic(obs_tensor, active_mask, return_details=True)
                            )
                            token_vectors = tokens.squeeze(0).squeeze(0).detach().cpu().numpy()
                            teacher_context_matrix = teacher_context.squeeze(0).squeeze(0).detach().cpu().numpy()

                        if token_vectors is not None:
                            for token_index, token_vector in enumerate(token_vectors):
                                token_rows.append(
                                    {
                                        "algorithm": run.algorithm,
                                        "rank": 0,
                                        "trial_number": int(run.seed),
                                        "config_name": run.config_name,
                                        "run_label": run_label,
                                        "episode_index": episode_index,
                                        "step_index": step_index,
                                        "seed": rollout_seed,
                                        "n_agents": int(n_agents),
                                        "token_index": int(token_index),
                                        "token_norm": float(np.linalg.norm(token_vector)),
                                    }
                                )
                                for dim_index, value in enumerate(np.asarray(token_vector, dtype=np.float32).reshape(-1)):
                                    token_rows[-1][f"dim_{dim_index:03d}"] = float(value)

                        assert teacher_context_matrix is not None
                        for row, teacher_ctx in zip(step_student_rows, teacher_context_matrix):
                            student_ctx = np.asarray(
                                [float(row[key]) for key in sorted(k for k in row if k.startswith("dim_"))],
                                dtype=np.float32,
                            )
                            row["teacher_alignment_cosine"] = _cosine_similarity(student_ctx, teacher_ctx)
                            row["teacher_alignment_mse"] = float(np.mean((student_ctx - teacher_ctx.reshape(-1)) ** 2))
                            row["teacher_ctx_norm"] = float(np.linalg.norm(teacher_ctx))
                        student_rows.extend(step_student_rows)

                        step_rows.append(
                            {
                                "algorithm": run.algorithm,
                                "rank": 0,
                                "trial_number": int(run.seed),
                                "config_name": run.config_name,
                                "run_label": run_label,
                                "episode_index": episode_index,
                                "step_index": step_index,
                                "seed": rollout_seed,
                                "n_agents": int(n_agents),
                                "token_spread": _mean_pairwise_distance(
                                    [] if token_vectors is None else [np.asarray(vector, dtype=np.float32) for vector in token_vectors]
                                ),
                                "student_ctx_spread": _mean_pairwise_distance(student_contexts),
                            }
                        )

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
    summary_rows = _build_count_summary_rows(trial_count_rows, step_rows=step_rows, token_rows=token_rows)
    token_pca_rows = build_grouped_pca_rows(token_rows, group_key="run_label", dim_prefix="dim_")
    student_pca_rows = build_grouped_pca_rows(student_rows, group_key="run_label", dim_prefix="dim_")
    gate_summary_rows = [
        row
        for row in trial_count_rows
        if row.get("gate_mean") is not None and row.get("gate_std") is not None and row.get("teacher_alignment_cosine_mean") is not None
    ]

    write_csv(resolved_output_dir / "student_rows.csv", student_rows)
    write_csv(resolved_output_dir / "token_rows.csv", token_rows)
    write_csv(resolved_output_dir / "step_metrics.csv", step_rows)
    write_csv(resolved_output_dir / "summary_by_trial_count.csv", trial_count_rows)
    write_csv(resolved_output_dir / "summary_by_count.csv", summary_rows)
    write_csv(resolved_output_dir / "token_pca_rows.csv", token_pca_rows)
    write_csv(resolved_output_dir / "student_ctx_pca_rows.csv", student_pca_rows)

    if token_pca_rows:
        plot_pca_projection_grid(
            token_pca_rows,
            output_path=resolved_output_dir / "token_pca.png",
            title=f"{task_id}: coordination-token PCA for {algorithm}/{config_name}",
        )
    if student_pca_rows:
        plot_pca_projection_grid(
            student_pca_rows,
            output_path=resolved_output_dir / "student_ctx_pca.png",
            title=f"{task_id}: student-context PCA for {algorithm}/{config_name}",
        )
    if trial_count_rows:
        plot_alignment_heatmap(
            trial_count_rows,
            output_path=resolved_output_dir / "alignment_heatmap.png",
            title=f"{task_id}: teacher-student alignment for {algorithm}/{config_name}",
        )
    if gate_summary_rows:
        plot_gate_alignment_3d(
            gate_summary_rows,
            output_path=resolved_output_dir / "gate_alignment_3d.png",
            title=f"{task_id}: gate usage vs teacher alignment for {algorithm}/{config_name}",
        )
        plot_gate_agent_heatmap(
            gate_summary_rows,
            output_path=resolved_output_dir / "gate_agentcount_heatmap.png",
            title=f"{task_id}: gate regime occupancy for {algorithm}/{config_name}",
        )

    write_json(
        resolved_output_dir / "analysis_manifest.json",
        {
            "task": task_id,
            "task_results_dir": str(task_results_dir),
            "algorithm": algorithm,
            "config_name": config_name,
            "rollouts_per_count": int(rollouts_per_count),
            "counts": [int(value) for value in analysis_counts],
            "seeds": [int(run.seed) for run in target_runs],
            "plots": {
                "token_pca": bool(token_pca_rows),
                "student_ctx_pca": bool(student_pca_rows),
                "alignment_heatmap": bool(trial_count_rows),
                "gate_alignment_3d": bool(gate_summary_rows),
                "gate_agentcount_heatmap": bool(gate_summary_rows),
            },
        },
    )
    return resolved_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot coordination traces for one results task/algorithm/config family.")
    parser.add_argument("--task-results-dir", type=Path, required=True, help="Task results directory containing algorithm subdirectories.")
    parser.add_argument("--task", required=True, help="Task id, e.g. simple_spread_dynamic_hard.")
    parser.add_argument("--algorithm", required=True, help="Algorithm id, e.g. pimac_v6.")
    parser.add_argument("--config", required=True, help="Config name stem, e.g. active_03 or best_01.")
    parser.add_argument("--rollouts-per-count", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = plot_result_coordination(
        task_results_dir=args.task_results_dir,
        task_id=args.task,
        algorithm=args.algorithm,
        config_name=args.config,
        rollouts_per_count=args.rollouts_per_count,
        counts=args.counts,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
