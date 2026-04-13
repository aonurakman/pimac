"""Task-local helpers for fixed-team `simple_spread_v3`.

The runner keeps the end-to-end experiment flow in `run.py`. This file only holds the small
pieces that make the runner easier to scan: evaluation summaries, rollout helpers, and plotting
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from utils import moving_average, save_gif


@dataclass(frozen=True)
class EvalResult:
    """Summary of one evaluation pass for one checkpoint."""

    phase: str
    checkpoint_episode: int
    rollout_count: int
    return_mean: float
    return_std: float
    return_min: float
    return_max: float


def run_fixed_evaluation(
    *,
    checkpoint_episode: int,
    phase: str,
    rollout_count: int,
    evaluate_rollouts_fn: Callable[[int], Sequence[float]],
) -> EvalResult:
    """Run one fixed-team evaluation batch and summarize the returns."""
    returns = np.asarray(evaluate_rollouts_fn(int(rollout_count)), dtype=np.float32)
    return EvalResult(
        phase=str(phase),
        checkpoint_episode=int(checkpoint_episode),
        rollout_count=int(rollout_count),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        return_min=float(np.min(returns)),
        return_max=float(np.max(returns)),
    )


def build_summary(
    *,
    task_config: dict,
    algorithm: str,
    seed: int,
    validation_results: Sequence[EvalResult],
    best_checkpoint_test: Optional[EvalResult],
    final_checkpoint_test: EvalResult,
    train_history_rows: Sequence[dict],
    extra_metrics: Optional[dict[str, float]] = None,
) -> dict:
    """Build the compact summary JSON written at the end of one run."""
    train_rewards = np.asarray([float(row["train_return_mean"]) for row in train_history_rows], dtype=np.float32)
    moving_average_window = min(100, max(1, train_rewards.size))
    moving_average_values = moving_average(train_rewards, moving_average_window) if train_rewards.size else np.asarray([], dtype=np.float32)

    if validation_results:
        validation_means = np.asarray([eval_result.return_mean for eval_result in validation_results], dtype=np.float32)
        validation_episodes = np.asarray([eval_result.checkpoint_episode for eval_result in validation_results], dtype=np.int64)
        best_index = int(np.argmax(validation_means))
        best_validation_mean = float(validation_means[best_index])
        best_validation_episode = int(validation_episodes[best_index])
        threshold = 0.9 * best_validation_mean
        convergence_episode = -1
        for episode, value in zip(validation_episodes, validation_means):
            if float(value) >= threshold:
                convergence_episode = int(episode)
                break
    else:
        best_validation_mean = float("nan")
        best_validation_episode = -1
        convergence_episode = -1

    final_checkpoint_mean = float(final_checkpoint_test.return_mean)
    uses_validation_selection = bool(validation_results)
    test_summary = {
        "final_checkpoint_mean": final_checkpoint_mean,
        "final_checkpoint_std": float(final_checkpoint_test.return_std),
        "objective_score": final_checkpoint_mean,
        "best_vs_final_drop": 0.0,
    }
    if uses_validation_selection:
        assert best_checkpoint_test is not None
        best_checkpoint_mean = float(best_checkpoint_test.return_mean)
        test_summary = {
            "best_checkpoint_mean": best_checkpoint_mean,
            "best_checkpoint_std": float(best_checkpoint_test.return_std),
            "final_checkpoint_mean": final_checkpoint_mean,
            "final_checkpoint_std": float(final_checkpoint_test.return_std),
            "objective_score": float(0.7 * best_checkpoint_mean + 0.3 * final_checkpoint_mean),
            "best_vs_final_drop": float(best_checkpoint_mean - final_checkpoint_mean),
        }

    return {
        "env_name": str(task_config["env_name"]),
        "algorithm": algorithm,
        "seed": int(seed),
        "episodes": int(task_config["episodes"]),
        "train": {
            "final_episode_return": float(train_rewards[-1]) if train_rewards.size else float("nan"),
            "moving_average_window": int(moving_average_window),
            "final_moving_average": float(moving_average_values[-1]) if moving_average_values.size else float("nan"),
            "reward_slope_last_window": float("nan")
            if train_rewards.size < 3
            else float(
                np.polyfit(
                    np.arange(min(100, train_rewards.size), dtype=np.float64),
                    train_rewards[-min(100, train_rewards.size):].astype(np.float64),
                    1,
                )[0]
            ),
        },
        "validation": {
            "best_validation_mean": best_validation_mean,
            "best_validation_episode": best_validation_episode,
            "convergence_episode_90pct": convergence_episode,
        },
        "test": test_summary,
        "extra_metrics": dict(sorted((extra_metrics or {}).items())),
    }


def evaluate_rollouts(
    learner,
    task_config: dict,
    *,
    seed: int,
    rollout_count: int,
    seed_offset: int,
    make_env_fn: Callable[..., object],
) -> list[float]:
    """Run plain evaluation rollouts for one fixed-team learner."""
    learner.set_eval_mode()
    returns: list[float] = []
    try:
        for rollout_index in range(int(rollout_count)):
            env_seed = int(seed + seed_offset + rollout_index)
            env = make_env_fn(task_config, seed=env_seed, render_mode=None)
            try:
                obs, _ = env.reset(seed=env_seed)
                agent_ids = list(env.possible_agents)
                learner.reset_episode()
                done = {agent_id: False for agent_id in agent_ids}
                total_reward = 0.0

                while True:
                    obs_dict = {
                        agent_id: np.asarray(obs[agent_id], dtype=np.float32)
                        for agent_id in agent_ids
                        if not done[agent_id]
                    }
                    actions = learner.act_parallel(obs_dict)
                    obs, rewards, terminations, truncations, _ = env.step(actions)
                    total_reward += sum(float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids)
                    done = {
                        agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                        for agent_id in agent_ids
                    }
                    if all(done.values()):
                        break
                returns.append(total_reward / len(agent_ids))
            finally:
                env.close()
    finally:
        learner.set_train_mode()
    return returns


def save_rollout_gif(
    learner,
    task_config: dict,
    *,
    out_dir: Path,
    seed: int,
    make_env_fn: Callable[..., object],
) -> None:
    """Render one policy rollout for quick visual inspection."""
    learner.set_eval_mode()
    try:
        env_seed = int(seed + int(task_config["gif_seed_offset"]))
        env = make_env_fn(task_config, seed=env_seed, render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=env_seed)
            agent_ids = list(env.possible_agents)
            learner.reset_episode()
            done = {agent_id: False for agent_id in agent_ids}
            frames: list[np.ndarray] = []
            step_index = 0

            while True:
                obs_dict = {
                    agent_id: np.asarray(obs[agent_id], dtype=np.float32)
                    for agent_id in agent_ids
                    if not done[agent_id]
                }
                actions = learner.act_parallel(obs_dict)
                obs, _, terminations, truncations, _ = env.step(actions)
                if step_index % max(1, int(task_config["frame_skip"])) == 0 and len(frames) < int(task_config["max_frames"]):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                step_index += 1
                done = {
                    agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    for agent_id in agent_ids
                }
                if all(done.values()) or len(frames) >= int(task_config["max_frames"]):
                    break
        finally:
            env.close()
    finally:
        learner.set_train_mode()
    save_gif(frames, out_dir / "policy_rollout.gif")
