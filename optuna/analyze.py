"""Read-only analysis tools for Optuna suites."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import ALGORITHM_ORDER, get_algorithm_class
from optuna_utils import LEARNED_ALGORITHM_ORDER, TASK_SPECS, discover_task_ids, get_task_spec, read_csv_rows, safe_float
from utils import OPTUNA_RESULTS_ROOT, load_json, save_gif, write_csv, write_json


PIMAC_TRACE_ALGORITHMS = {"pimac_v1", "pimac_v2", "pimac_v3", "pimac_v4"}
DEFAULT_VIDEO_DYNAMIC_COUNTS = (1, 4, 8, 10)


@dataclass(frozen=True)
class BestRun:
    algorithm: str
    rank: int
    trial_number: int
    objective_score: float
    leaderboard_path: Path
    run_output_dir: Path
    checkpoint_path: Path
    summary_path: Path
    config_snapshot_path: Path
    config_snapshot: dict[str, Any]
    effective_config: dict[str, Any]


def _task_script_module(task_id: str):
    return importlib.import_module(f"{task_id}.run")


def _suite_root(suite_id: str, results_root: Path = OPTUNA_RESULTS_ROOT) -> Path:
    return Path(results_root) / suite_id


def discover_best_runs(*, suite_id: str, task_id: str, results_root: Path = OPTUNA_RESULTS_ROOT) -> list[BestRun]:
    study_dir = Path(results_root) / suite_id / "studies" / task_id
    discovered: list[BestRun] = []
    for algorithm in LEARNED_ALGORITHM_ORDER:
        leaderboard_path = study_dir / f"{algorithm}_leaderboard.csv"
        if not leaderboard_path.is_file():
            continue
        rows = read_csv_rows(leaderboard_path)
        if not rows:
            continue
        ranked_rows = sorted((int(row["rank"]), row) for row in rows if row.get("rank"))
        if not ranked_rows:
            continue
        row = ranked_rows[0][1]
        run_output_dir = Path(row["run_output_dir"])
        checkpoint_path = run_output_dir / "best_checkpoint.pt"
        summary_path = run_output_dir / "summary.json"
        config_snapshot_path = run_output_dir / "config_snapshot.json"
        if not (checkpoint_path.is_file() and summary_path.is_file() and config_snapshot_path.is_file()):
            continue
        discovered.append(
            BestRun(
                algorithm=algorithm,
                rank=int(row["rank"]),
                trial_number=int(row["trial_number"]),
                objective_score=float(row["objective_score"]),
                leaderboard_path=leaderboard_path,
                run_output_dir=run_output_dir,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                config_snapshot_path=config_snapshot_path,
                config_snapshot=load_json(config_snapshot_path),
                effective_config=json.loads(row["effective_config_json"]),
            )
        )
    return discovered


def _transform_obs(task_script, env_spec, obs: np.ndarray) -> np.ndarray:
    """Ask the task module how one raw observation should be prepared for the learner."""
    return task_script.prepare_observation(obs, env_spec)


def _task_is_dynamic(task_config: dict[str, Any]) -> bool:
    """Treat every dynamic-team variant as a dynamic task in Optuna analysis."""
    return str(task_config.get("task_type", "")).startswith("dynamic_team")


def _video_counts_for_task(task_config: dict[str, Any]) -> list[int]:
    """Resolve the default video counts for one task config."""
    return [int(value) for value in task_config.get("video_counts", DEFAULT_VIDEO_DYNAMIC_COUNTS)]


def load_best_policy(best_run: BestRun, *, task_id: str, seed: int):
    task_spec = get_task_spec(task_id)
    task_config = load_json(task_spec.task_config)
    task_script = _task_script_module(task_id)
    env_spec = task_script.build_env_spec(task_config, seed)
    learner_cls = get_algorithm_class(best_run.algorithm)
    learner = learner_cls.load_checkpoint(
        best_run.checkpoint_path,
        env_spec=env_spec,
        config=best_run.effective_config,
        device="cpu",
    )
    learner.set_eval_mode()
    return learner, task_config, task_script, env_spec


def _run_one_rollout(*, task_id: str, task_config: dict, task_script, env_spec, learner, seed: int, n_agents: int | None = None, render_mode: str | None = None, frame_budget: int | None = None):
    if _task_is_dynamic(task_config):
        env = task_script.make_env(task_config, seed=seed, n_agents=int(n_agents), render_mode=render_mode)
    else:
        env = task_script.make_env(task_config, seed=seed, render_mode=render_mode)
    obs, _ = env.reset(seed=seed)
    agent_ids = list(env.possible_agents)
    learner.reset_episode()
    total_reward = 0.0
    done = {agent_id: False for agent_id in agent_ids}
    frames: list[np.ndarray] = []
    step_index = 0
    while True:
        obs_dict = {
            agent_id: _transform_obs(task_script, env_spec, obs[agent_id])
            for agent_id in agent_ids
            if not done[agent_id]
        }
        actions = learner.act_parallel(obs_dict)
        obs, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += sum(float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids)
        if render_mode == "rgb_array" and frame_budget is not None:
            if step_index % max(1, int(task_config["frame_skip"])) == 0 and len(frames) < int(frame_budget):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        step_index += 1
        done = {
            agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
            for agent_id in agent_ids
        }
        if all(done.values()) or (frame_budget is not None and len(frames) >= int(frame_budget)):
            break
    env.close()
    return float(total_reward / max(1, len(agent_ids))), frames


def compare_best_checkpoints(
    *,
    suite_id: str,
    task_id: str,
    results_root: Path = OPTUNA_RESULTS_ROOT,
    seed: int = 42,
    fixed_rollouts: Optional[int] = None,
    dynamic_per_count_rollouts: Optional[int] = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    best_runs = discover_best_runs(suite_id=suite_id, task_id=task_id, results_root=results_root)
    if not best_runs:
        raise RuntimeError(f"No best runs discovered for {suite_id}/{task_id}.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else _suite_root(suite_id, results_root) / "best_checkpoint_comparison" / task_id
    if resolved_output_dir.exists() and overwrite:
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    task_spec = get_task_spec(task_id)
    task_config = load_json(task_spec.task_config)
    task_script = _task_script_module(task_id)
    dynamic_task = _task_is_dynamic(task_config)

    if dynamic_task:
        rollout_count = int(dynamic_per_count_rollouts or task_config["test_rollouts"])
        raw_rows: list[dict[str, Any]] = []
        for best_run in best_runs:
            learner, _, _, env_spec = load_best_policy(best_run, task_id=task_id, seed=seed)
            for n_agents in [int(value) for value in task_config["eval_counts"]]:
                for rollout_index in range(rollout_count):
                    rollout_seed = int(seed + (1000 * n_agents) + rollout_index)
                    reward, _ = _run_one_rollout(
                        task_id=task_id,
                        task_config=task_config,
                        task_script=task_script,
                        env_spec=env_spec,
                        learner=learner,
                        seed=rollout_seed,
                        n_agents=n_agents,
                    )
                    raw_rows.append(
                        {
                            "algorithm": best_run.algorithm,
                            "n_agents": n_agents,
                            "rollout_index": rollout_index,
                            "seed": rollout_seed,
                            "reward": reward,
                        }
                    )
        normalized_rows: list[dict[str, Any]] = []
        per_count_means: list[dict[str, Any]] = []
        for n_agents in [int(value) for value in task_config["eval_counts"]]:
            count_rows = [row for row in raw_rows if int(row["n_agents"]) == n_agents]
            if not count_rows:
                continue
            rewards = np.asarray([float(row["reward"]) for row in count_rows], dtype=np.float32)
            reward_min = float(np.min(rewards))
            reward_max = float(np.max(rewards))
            spread = reward_max - reward_min
            for row in count_rows:
                normalized = 0.5 if spread <= 1e-12 else (float(row["reward"]) - reward_min) / spread
                normalized_rows.append({**row, "normalized_reward": float(normalized)})
            for algorithm in [best_run.algorithm for best_run in best_runs]:
                algorithm_rows = [row for row in normalized_rows if row["algorithm"] == algorithm and int(row["n_agents"]) == n_agents]
                if not algorithm_rows:
                    continue
                per_count_means.append(
                    {
                        "algorithm": algorithm,
                        "n_agents": n_agents,
                        "mean_reward": float(np.mean([float(row["reward"]) for row in algorithm_rows])),
                        "normalized_mean_reward": float(np.mean([float(row["normalized_reward"]) for row in algorithm_rows])),
                    }
                )

        leaderboard_rows: list[dict[str, Any]] = []
        for algorithm in [best_run.algorithm for best_run in best_runs]:
            algorithm_rows = [row for row in per_count_means if row["algorithm"] == algorithm]
            algorithm_rows.sort(key=lambda row: int(row["n_agents"]))
            if not algorithm_rows:
                continue
            row = {
                "algorithm": algorithm,
                "normalized_overall_mean": float(np.mean([float(item["normalized_mean_reward"]) for item in algorithm_rows])),
                "raw_overall_mean": float(np.mean([float(item["mean_reward"]) for item in algorithm_rows])),
            }
            for item in algorithm_rows:
                row[f"n{int(item['n_agents'])}_normalized_mean"] = float(item["normalized_mean_reward"])
                row[f"n{int(item['n_agents'])}_raw_mean"] = float(item["mean_reward"])
            leaderboard_rows.append(row)
        leaderboard_rows.sort(key=lambda row: float(row["normalized_overall_mean"]), reverse=True)
        for rank, row in enumerate(leaderboard_rows, start=1):
            row["rank"] = rank

        write_csv(resolved_output_dir / "dynamic_per_count_rewards.csv", raw_rows)
        write_csv(resolved_output_dir / "dynamic_per_count_normalized_rewards.csv", normalized_rows)
        write_csv(resolved_output_dir / "dynamic_per_count_normalized_means.csv", per_count_means)
        write_csv(resolved_output_dir / "leaderboard.csv", leaderboard_rows)
        write_json(
            resolved_output_dir / "comparison_summary.json",
            {
                "suite_id": suite_id,
                "task": task_id,
                "comparison_type": "dynamic",
                "rollouts_per_count": rollout_count,
                "algorithms": [best_run.algorithm for best_run in best_runs],
            },
        )
    else:
        rollout_count = int(fixed_rollouts or task_config["test_rollouts"])
        raw_rows: list[dict[str, Any]] = []
        for best_run in best_runs:
            learner, _, _, env_spec = load_best_policy(best_run, task_id=task_id, seed=seed)
            for rollout_index in range(rollout_count):
                rollout_seed = int(seed + rollout_index)
                reward, _ = _run_one_rollout(
                    task_id=task_id,
                    task_config=task_config,
                    task_script=task_script,
                    env_spec=env_spec,
                    learner=learner,
                    seed=rollout_seed,
                )
                raw_rows.append(
                    {
                        "algorithm": best_run.algorithm,
                        "rollout_index": rollout_index,
                        "seed": rollout_seed,
                        "reward": reward,
                    }
                )
        leaderboard_rows: list[dict[str, Any]] = []
        for algorithm in [best_run.algorithm for best_run in best_runs]:
            algorithm_rows = [row for row in raw_rows if row["algorithm"] == algorithm]
            if not algorithm_rows:
                continue
            rewards = np.asarray([float(row["reward"]) for row in algorithm_rows], dtype=np.float32)
            leaderboard_rows.append(
                {
                    "algorithm": algorithm,
                    "return_mean": float(np.mean(rewards)),
                    "return_std": float(np.std(rewards)),
                    "return_min": float(np.min(rewards)),
                    "return_max": float(np.max(rewards)),
                }
            )
        leaderboard_rows.sort(key=lambda row: float(row["return_mean"]), reverse=True)
        for rank, row in enumerate(leaderboard_rows, start=1):
            row["rank"] = rank

        write_csv(resolved_output_dir / "rollout_rewards.csv", raw_rows)
        write_csv(resolved_output_dir / "leaderboard.csv", leaderboard_rows)
        write_json(
            resolved_output_dir / "comparison_summary.json",
            {
                "suite_id": suite_id,
                "task": task_id,
                "comparison_type": "fixed",
                "rollout_count": rollout_count,
                "algorithms": [best_run.algorithm for best_run in best_runs],
            },
        )

    return resolved_output_dir


def _mean_pairwise_distance(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    stacked = np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors], axis=0)
    diffs = stacked[:, None, :] - stacked[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper = distances[np.triu_indices(stacked.shape[0], k=1)]
    return 0.0 if upper.size == 0 else float(np.mean(upper))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    vector_a = np.asarray(a, dtype=np.float32).reshape(-1)
    vector_b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)


def _select_trace_runs(best_runs: Sequence[BestRun], *, top_k: int) -> list[BestRun]:
    """Keep the top-k ranked runs for each traceable PIMAC variant."""
    selected: list[BestRun] = []
    per_algorithm_limit = max(1, int(top_k))
    for algorithm in sorted(PIMAC_TRACE_ALGORITHMS):
        algorithm_runs = [run for run in best_runs if run.algorithm == algorithm]
        algorithm_runs.sort(key=lambda run: run.rank)
        selected.extend(algorithm_runs[:per_algorithm_limit])
    return selected


def analyze_pimac_coordination(
    *,
    suite_id: str,
    task_id: str,
    results_root: Path = OPTUNA_RESULTS_ROOT,
    seed: int = 42,
    top_k: int = 1,
    rollouts_per_count: int = 5,
    counts: Sequence[int] | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    if task_id not in TASK_SPECS:
        raise KeyError(f"Unsupported task for coordination analysis: {task_id}")
    task_spec = get_task_spec(task_id)
    task_config = load_json(task_spec.task_config)
    task_script = _task_script_module(task_id)
    best_runs_all = discover_best_runs(suite_id=suite_id, task_id=task_id, results_root=results_root)
    target_runs = _select_trace_runs(best_runs_all, top_k=top_k)
    if not target_runs:
        raise RuntimeError(f"No traceable PIMAC runs found for {suite_id}/{task_id}.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else _suite_root(suite_id, results_root) / "pimac_coordination_analysis" / task_id
    if resolved_output_dir.exists() and overwrite:
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    dynamic_task = _task_is_dynamic(task_config)
    analysis_counts = list(counts) if counts is not None else (
        [value for value in task_config["eval_counts"] if int(value) > 1] if dynamic_task else [int(task_config["n_agents"])]
    )

    token_rows: list[dict[str, Any]] = []
    student_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for best_run in target_runs:
        learner, _, _, env_spec = load_best_policy(best_run, task_id=task_id, seed=seed)
        learner.set_eval_mode()
        for n_agents in analysis_counts:
            for episode_index in range(int(rollouts_per_count)):
                rollout_seed = int(seed + (1000 * int(n_agents)) + episode_index)
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
                        token_vectors: np.ndarray | None = None
                        teacher_context_matrix: np.ndarray | None = None
                        actions: dict[str, int] = {}

                        with torch.no_grad():
                            for agent_position, agent_id in enumerate(agent_ids):
                                obs_value = _transform_obs(task_script, env_spec, obs[agent_id])
                                obs_tensor = torch.as_tensor(obs_value, dtype=torch.float32, device=learner.device).view(1, 1, -1)
                                hidden_dim = learner.actor_net.rnn.hidden_size
                                initial_hidden_state = learner._get_hidden_state(agent_id, hidden_dim)
                                action_logits, updated_hidden_state, aux = learner.actor_net(obs_tensor, initial_hidden_state, return_aux=True)
                                learner._set_hidden_state(agent_id, updated_hidden_state)
                                logits = action_logits.squeeze(0).squeeze(0)
                                categorical_distribution = torch.distributions.Categorical(logits=logits)
                                action = int(categorical_distribution.sample().item())
                                actions[str(agent_id)] = action

                                student_ctx = aux["ctx_mu"].squeeze(0).squeeze(0).detach().cpu().numpy()
                                student_logvar = aux["ctx_logvar"].squeeze(0).squeeze(0).detach().cpu().numpy()
                                student_contexts.append(student_ctx)
                                student_row = {
                                    "algorithm": best_run.algorithm,
                                    "rank": best_run.rank,
                                    "trial_number": best_run.trial_number,
                                    "episode_index": episode_index,
                                    "step_index": step_index,
                                    "seed": rollout_seed,
                                    "n_agents": int(n_agents),
                                    "agent_id": str(agent_id),
                                    "agent_position": int(agent_position),
                                    "action": int(action),
                                    "ctx_logvar_mean": float(np.mean(student_logvar)),
                                    "student_ctx_norm": float(np.linalg.norm(student_ctx)),
                                }
                                if "gate" in aux:
                                    student_row["gate"] = float(aux["gate"].squeeze(0).squeeze(0).detach().cpu().item())
                                if "delta_w_norm" in aux:
                                    student_row["delta_w_norm"] = float(aux["delta_w_norm"].squeeze(0).squeeze(0).detach().cpu().item())
                                if "delta_b_norm" in aux:
                                    student_row["delta_b_norm"] = float(aux["delta_b_norm"].squeeze(0).squeeze(0).detach().cpu().item())
                                for dim_index, value in enumerate(np.asarray(student_ctx, dtype=np.float32).reshape(-1)):
                                    student_row[f"dim_{dim_index:03d}"] = float(value)
                                student_rows.append(student_row)

                            stacked_obs = np.stack(
                                [_transform_obs(task_script, env_spec, obs[agent_id]) for agent_id in agent_ids],
                                axis=0,
                            )
                            obs_tensor = torch.as_tensor(stacked_obs, dtype=torch.float32, device=learner.device).view(1, 1, len(agent_ids), -1)
                            active_mask = torch.ones(1, 1, len(agent_ids), dtype=torch.float32, device=learner.device)
                            _, tokens, teacher_context = learner.critic(obs_tensor, active_mask, return_details=True)
                            token_vectors = tokens.squeeze(0).squeeze(0).detach().cpu().numpy()
                            teacher_context_matrix = teacher_context.squeeze(0).squeeze(0).detach().cpu().numpy()

                        if token_vectors is not None:
                            for token_index, token_vector in enumerate(token_vectors):
                                token_rows.append(
                                    {
                                        "algorithm": best_run.algorithm,
                                        "rank": best_run.rank,
                                        "trial_number": best_run.trial_number,
                                        "episode_index": episode_index,
                                        "step_index": step_index,
                                        "seed": rollout_seed,
                                        "n_agents": int(n_agents),
                                        "token_index": int(token_index),
                                        "token_norm": float(np.linalg.norm(token_vector)),
                                    }
                                )

                        step_student_rows = [row for row in student_rows if row["algorithm"] == best_run.algorithm and row["seed"] == rollout_seed and row["episode_index"] == episode_index and row["step_index"] == step_index]
                        for row, teacher_ctx in zip(step_student_rows, teacher_context_matrix):
                            student_ctx = np.asarray([float(row[key]) for key in sorted(k for k in row if k.startswith("dim_"))], dtype=np.float32)
                            row["teacher_alignment_cosine"] = _cosine_similarity(student_ctx, teacher_ctx)
                            row["teacher_alignment_mse"] = float(np.mean((student_ctx - teacher_ctx.reshape(-1)) ** 2))
                            row["teacher_ctx_norm"] = float(np.linalg.norm(teacher_ctx))

                        step_rows.append(
                            {
                                "algorithm": best_run.algorithm,
                                "rank": best_run.rank,
                                "trial_number": best_run.trial_number,
                                "episode_index": episode_index,
                                "step_index": step_index,
                                "seed": rollout_seed,
                                "n_agents": int(n_agents),
                                "token_spread": _mean_pairwise_distance(
                                    []
                                    if token_vectors is None
                                    else [np.asarray(vector, dtype=np.float32) for vector in token_vectors]
                                ),
                                "student_ctx_spread": _mean_pairwise_distance(student_contexts),
                            }
                        )

                        obs, _, terminations, truncations, _ = env.step(actions)
                        done = {
                            agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in set(terminations) | set(truncations)
                        }
                        obs = {str(agent_id): np.asarray(value, dtype=np.float32) for agent_id, value in obs.items() if not done.get(agent_id, False)}
                        step_index += 1
                finally:
                    env.close()

    summary_rows: list[dict[str, Any]] = []
    for algorithm in sorted({row["algorithm"] for row in student_rows}):
        for n_agents in sorted({int(row["n_agents"]) for row in student_rows if row["algorithm"] == algorithm}):
            student_group = [row for row in student_rows if row["algorithm"] == algorithm and int(row["n_agents"]) == n_agents]
            step_group = [row for row in step_rows if row["algorithm"] == algorithm and int(row["n_agents"]) == n_agents]
            token_group = [row for row in token_rows if row["algorithm"] == algorithm and int(row["n_agents"]) == n_agents]
            if not student_group:
                continue
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "n_agents": int(n_agents),
                    "samples": int(len(student_group)),
                    "teacher_alignment_cosine_mean": float(np.mean([float(row["teacher_alignment_cosine"]) for row in student_group])),
                    "teacher_alignment_mse_mean": float(np.mean([float(row["teacher_alignment_mse"]) for row in student_group])),
                    "ctx_logvar_mean": float(np.mean([float(row["ctx_logvar_mean"]) for row in student_group])),
                    "gate_mean": float(np.mean([float(row["gate"]) for row in student_group if row.get("gate") is not None])) if any(row.get("gate") is not None for row in student_group) else None,
                    "delta_w_norm_mean": float(np.mean([float(row["delta_w_norm"]) for row in student_group if row.get("delta_w_norm") is not None])) if any(row.get("delta_w_norm") is not None for row in student_group) else None,
                    "delta_b_norm_mean": float(np.mean([float(row["delta_b_norm"]) for row in student_group if row.get("delta_b_norm") is not None])) if any(row.get("delta_b_norm") is not None for row in student_group) else None,
                    "teacher_ctx_norm_mean": float(np.mean([float(row["teacher_ctx_norm"]) for row in student_group])),
                    "student_ctx_norm_mean": float(np.mean([float(row["student_ctx_norm"]) for row in student_group])),
                    "token_norm_mean": float(np.mean([float(row["token_norm"]) for row in token_group])) if token_group else None,
                    "token_spread_mean": float(np.mean([float(row["token_spread"]) for row in step_group])) if step_group else None,
                    "student_ctx_spread_mean": float(np.mean([float(row["student_ctx_spread"]) for row in step_group])) if step_group else None,
                }
            )

    write_csv(resolved_output_dir / "student_rows.csv", student_rows)
    write_csv(resolved_output_dir / "token_rows.csv", token_rows)
    write_csv(resolved_output_dir / "step_metrics.csv", step_rows)
    write_csv(resolved_output_dir / "summary_by_count.csv", summary_rows)
    write_json(
        resolved_output_dir / "analysis_manifest.json",
        {
            "suite_id": suite_id,
            "task": task_id,
            "rollouts_per_count": int(rollouts_per_count),
            "counts": [int(value) for value in analysis_counts],
            "algorithms": sorted({row["algorithm"] for row in student_rows}),
        },
    )
    return resolved_output_dir


def render_best_rollout_videos(
    *,
    suite_id: str,
    task_id: str,
    results_root: Path = OPTUNA_RESULTS_ROOT,
    seed: int = 42,
    counts: Sequence[int] | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    best_runs = discover_best_runs(suite_id=suite_id, task_id=task_id, results_root=results_root)
    if not best_runs:
        raise RuntimeError(f"No best runs discovered for {suite_id}/{task_id}.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else _suite_root(suite_id, results_root) / "videos" / task_id
    if resolved_output_dir.exists() and overwrite:
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    task_spec = get_task_spec(task_id)
    task_config = load_json(task_spec.task_config)
    task_script = _task_script_module(task_id)
    dynamic_task = _task_is_dynamic(task_config)
    target_counts = list(counts) if counts is not None else (
        _video_counts_for_task(task_config) if dynamic_task else [int(task_config["n_agents"])]
    )

    manifest_rows: list[dict[str, Any]] = []
    for best_run in best_runs:
        learner, _, _, env_spec = load_best_policy(best_run, task_id=task_id, seed=seed)
        for n_agents in target_counts:
            rollout_seed = int(seed + (1000 * int(n_agents)))
            _, frames = _run_one_rollout(
                task_id=task_id,
                task_config=task_config,
                task_script=task_script,
                env_spec=env_spec,
                learner=learner,
                seed=rollout_seed,
                n_agents=int(n_agents) if dynamic_task else None,
                render_mode="rgb_array",
                frame_budget=int(task_config["max_frames"]),
            )
            file_name = f"{best_run.algorithm}_n{int(n_agents):02d}.gif" if dynamic_task else f"{best_run.algorithm}.gif"
            save_gif(frames, resolved_output_dir / file_name)
            manifest_rows.append(
                {
                    "algorithm": best_run.algorithm,
                    "n_agents": int(n_agents),
                    "seed": rollout_seed,
                    "gif_path": str(resolved_output_dir / file_name),
                }
            )

    write_csv(resolved_output_dir / "manifest.csv", manifest_rows)
    return resolved_output_dir


def export_best_configs(
    *,
    suite_id: str,
    task_id: str | None = None,
    results_root: Path = OPTUNA_RESULTS_ROOT,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    target_task_ids = [task_id] if task_id is not None else discover_task_ids(results_root, [suite_id])
    export_rows: list[dict[str, Any]] = []

    for current_task_id in target_task_ids:
        task_spec = get_task_spec(current_task_id)
        exported_any_ranked_configs = False
        manifest_rows = [
            {
                "algorithm": "random",
                "config_file": "random/best_01.json",
                "source_suite": "manual",
                "source_rank": 1,
                "objective_score": "",
                "note": "Uniform random policy baseline.",
            }
        ]
        for algorithm in LEARNED_ALGORITHM_ORDER:
            leaderboard_path = Path(results_root) / suite_id / "studies" / current_task_id / f"{algorithm}_leaderboard.csv"
            if not leaderboard_path.is_file():
                continue
            rows = read_csv_rows(leaderboard_path)
            ranked_rows = sorted(
                (int(row["rank"]), row)
                for row in rows
                if row.get("rank") and row.get("effective_config_json")
            )[: int(top_k)]
            if not ranked_rows:
                continue
            exported_any_ranked_configs = True
            for rank, row in ranked_rows:
                config = json.loads(row["effective_config_json"])
                config_path = task_spec.config_root / algorithm / f"best_{rank:02d}.json"
                write_json(config_path, config)
                manifest_rows.append(
                    {
                        "algorithm": algorithm,
                        "config_file": f"{algorithm}/best_{rank:02d}.json",
                        "source_suite": suite_id,
                        "source_rank": rank,
                        "objective_score": row.get("objective_score", ""),
                        "note": f"Imported from {suite_id} leaderboard.",
                    }
                )
        if not exported_any_ranked_configs:
            if task_id is not None:
                raise RuntimeError(f"No ranked leaderboards found for {suite_id}/{current_task_id}.")
            continue
        write_csv(task_spec.config_root / "manifest.csv", manifest_rows)
        export_rows.append(
            {
                "task": current_task_id,
                "manifest_path": str(task_spec.config_root / "manifest.csv"),
            }
        )
    return export_rows


def merge_suites(
    *,
    suite_a: str,
    suite_b: str,
    merged_suite_id: str,
    results_root: Path = OPTUNA_RESULTS_ROOT,
    overwrite: bool = False,
) -> Path:
    target_dir = Path(results_root) / merged_suite_id
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Merged suite already exists: {target_dir}")
        shutil.rmtree(target_dir)
    (target_dir / "studies").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    manifest: dict[str, dict[str, Any]] = {}
    for task_id in discover_task_ids(results_root, [suite_a, suite_b]):
        task_study_dir = target_dir / "studies" / task_id
        task_study_dir.mkdir(parents=True, exist_ok=True)
        task_manifest: dict[str, Any] = {}
        environment_rows: list[dict[str, Any]] = []
        for algorithm in LEARNED_ALGORITHM_ORDER:
            merged_rows: list[dict[str, Any]] = []
            for suite_id in (suite_a, suite_b):
                leaderboard_path = Path(results_root) / suite_id / "studies" / task_id / f"{algorithm}_leaderboard.csv"
                if not leaderboard_path.is_file():
                    continue
                for row in read_csv_rows(leaderboard_path):
                    merged_row = dict(row)
                    merged_row["source_suite"] = suite_id
                    merged_rows.append(merged_row)
            if not merged_rows:
                continue
            merged_rows.sort(key=lambda row: (safe_float(row.get("objective_score")) or float("-inf")), reverse=True)
            for rank, row in enumerate(merged_rows, start=1):
                row["rank"] = rank
            merged_path = task_study_dir / f"{algorithm}_leaderboard.csv"
            write_csv(merged_path, merged_rows)
            top_row = merged_rows[0]
            environment_rows.append(
                {
                    "algorithm": algorithm,
                    "best_objective_score": top_row.get("objective_score", ""),
                    "source_suite": top_row.get("source_suite", ""),
                    "leaderboard_path": str(merged_path),
                    "run_output_dir": top_row.get("run_output_dir", ""),
                }
            )
            task_manifest[algorithm] = {
                "row_count": len(merged_rows),
                "leaderboard_path": str(merged_path),
                "best_source_suite": top_row.get("source_suite", ""),
            }
            summary_rows.append(
                {
                    "task": task_id,
                    "algorithm": algorithm,
                    "row_count": len(merged_rows),
                    "best_source_suite": top_row.get("source_suite", ""),
                    "best_objective_score": top_row.get("objective_score", ""),
                }
            )
        if environment_rows:
            write_csv(task_study_dir / "environment_leaderboard.csv", environment_rows)
        manifest[task_id] = task_manifest
    write_json(
        target_dir / "suite_summary.json",
        {
            "suite_id": merged_suite_id,
            "suite_type": "merged_comparison_only",
            "source_suites": [suite_a, suite_b],
            "manifest": manifest,
            "summary_rows": summary_rows,
        },
    )
    return target_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only analysis tools for Optuna suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge two suites into one comparison suite.")
    merge_parser.add_argument("--suite-a", required=True)
    merge_parser.add_argument("--suite-b", required=True)
    merge_parser.add_argument("--merged-suite-id", required=True)
    merge_parser.add_argument("--overwrite", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare best checkpoints.")
    compare_parser.add_argument("--suite-id", required=True)
    compare_parser.add_argument("--task", required=True, choices=tuple(TASK_SPECS.keys()))
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--fixed-rollouts", type=int, default=None)
    compare_parser.add_argument("--dynamic-per-count-rollouts", type=int, default=None)
    compare_parser.add_argument("--output-dir", type=str, default=None)
    compare_parser.add_argument("--overwrite", action="store_true")

    coordination_parser = subparsers.add_parser("coordination", help="Trace PIMAC coordination signals.")
    coordination_parser.add_argument("--suite-id", required=True)
    coordination_parser.add_argument("--task", required=True, choices=tuple(TASK_SPECS.keys()))
    coordination_parser.add_argument("--seed", type=int, default=42)
    coordination_parser.add_argument("--top-k", type=int, default=1)
    coordination_parser.add_argument("--rollouts-per-count", type=int, default=5)
    coordination_parser.add_argument("--counts", type=int, nargs="*", default=None)
    coordination_parser.add_argument("--output-dir", type=str, default=None)
    coordination_parser.add_argument("--overwrite", action="store_true")

    video_parser = subparsers.add_parser("videos", help="Render best-checkpoint rollout GIFs.")
    video_parser.add_argument("--suite-id", required=True)
    video_parser.add_argument("--task", required=True, choices=tuple(TASK_SPECS.keys()))
    video_parser.add_argument("--seed", type=int, default=42)
    video_parser.add_argument("--counts", type=int, nargs="*", default=None)
    video_parser.add_argument("--output-dir", type=str, default=None)
    video_parser.add_argument("--overwrite", action="store_true")

    export_parser = subparsers.add_parser("export-best", help="Export top configs into task config folders.")
    export_parser.add_argument("--suite-id", required=True)
    export_parser.add_argument("--task", type=str, default=None, choices=tuple(TASK_SPECS.keys()))
    export_parser.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "merge":
        output_dir = merge_suites(
            suite_a=args.suite_a,
            suite_b=args.suite_b,
            merged_suite_id=args.merged_suite_id,
            overwrite=bool(args.overwrite),
        )
        print(output_dir)
        return 0
    if args.command == "compare":
        output_dir = compare_best_checkpoints(
            suite_id=args.suite_id,
            task_id=args.task,
            seed=int(args.seed),
            fixed_rollouts=args.fixed_rollouts,
            dynamic_per_count_rollouts=args.dynamic_per_count_rollouts,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
        )
        print(output_dir)
        return 0
    if args.command == "coordination":
        output_dir = analyze_pimac_coordination(
            suite_id=args.suite_id,
            task_id=args.task,
            seed=int(args.seed),
            top_k=int(args.top_k),
            rollouts_per_count=int(args.rollouts_per_count),
            counts=args.counts,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
        )
        print(output_dir)
        return 0
    if args.command == "videos":
        output_dir = render_best_rollout_videos(
            suite_id=args.suite_id,
            task_id=args.task,
            seed=int(args.seed),
            counts=args.counts,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
        )
        print(output_dir)
        return 0
    if args.command == "export-best":
        print(json.dumps(export_best_configs(suite_id=args.suite_id, task_id=args.task, top_k=int(args.top_k)), indent=2))
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
