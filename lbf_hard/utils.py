"""Task-local helpers for dynamic-team Level-Based Foraging.

The runner keeps the training loop visible in `run.py`. This file holds the environment-shaping
helpers around that loop: curriculum bookkeeping, grouped evaluation summaries, cached evaluation
environments, and the training dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym
import matplotlib
import numpy as np

from algorithms.base import ParallelEnvSpec
from utils import grouped_dashboard_metric_keys, moving_average, plot_dashboard_metric_group, save_gif

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Observation shaping and curriculum
# -----------------------------------------------------------------------------


def pad_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad one variable-width observation to the fixed width expected by the learners."""
    array = np.asarray(vec, dtype=np.float32).reshape(-1)
    if array.shape[0] >= target_dim:
        return array[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: array.shape[0]] = array
    return padded


class FixedEntityObservationWrapper:
    """Convert native LBF vectors into a fixed-slot local entity observation."""

    def __init__(self, env, *, max_agents: int, max_food: int, sight: int):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.max_agents = int(max_agents)
        self.max_food = int(max_food)
        self.sight = int(sight)
        self._coord_sentinel = float(-(self.sight + 1))
        self._entity_sentinel = np.asarray(
            [self._coord_sentinel, self._coord_sentinel, 0.0],
            dtype=np.float32,
        )
        self._observation_size = 1 + (3 * self.max_food) + (3 * max(0, self.max_agents - 1))

        raw_space = env.observation_space(self.possible_agents[0])
        raw_high = np.asarray(raw_space.high, dtype=np.float32).reshape(-1)
        max_food_level = float(np.max(raw_high[2 : (3 * self.max_food) : 3]))
        player_start = 3 * self.max_food
        max_player_level = float(np.max(raw_high[player_start + 2 :: 3]))

        low = np.full(self._observation_size, self._coord_sentinel, dtype=np.float32)
        high = np.full(self._observation_size, float(self.sight), dtype=np.float32)
        low[0] = 0.0
        high[0] = max_player_level
        for slot_index in range(self.max_food):
            base = 1 + (3 * slot_index)
            low[base : base + 3] = self._entity_sentinel
            high[base : base + 3] = (float(self.sight), float(self.sight), max_food_level)
        teammate_offset = 1 + (3 * self.max_food)
        for slot_index in range(max(0, self.max_agents - 1)):
            base = teammate_offset + (3 * slot_index)
            low[base : base + 3] = self._entity_sentinel
            high[base : base + 3] = (float(self.sight), float(self.sight), max_player_level)
        self._observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self._observation_size,),
            dtype=np.float32,
        )

    @property
    def agents(self) -> list[str]:
        return list(self.possible_agents)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def _entity_sort_key(self, entity: tuple[float, float, float]) -> tuple[float, float, float]:
        dy, dx, _ = entity
        return ((dy * dy) + (dx * dx), dy, dx)

    def _relative_triplet(self, y: float, x: float, level: float) -> tuple[float, float, float]:
        return (float(y - self.sight), float(x - self.sight), float(level))

    def _encode_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        flat = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        food_entities: list[tuple[float, float, float]] = []
        for slot_index in range(self.max_food):
            y, x, level = flat[(3 * slot_index) : (3 * slot_index) + 3]
            if y >= 0.0 and x >= 0.0:
                food_entities.append(self._relative_triplet(y, x, level))
        food_entities.sort(key=self._entity_sort_key)

        player_section = flat[3 * self.max_food :]
        player_slots = player_section.shape[0] // 3
        self_level = 0.0
        teammate_entities: list[tuple[float, float, float]] = []
        if player_slots > 0:
            _, _, self_level = player_section[:3]
            for slot_index in range(1, player_slots):
                base = 3 * slot_index
                y, x, level = player_section[base : base + 3]
                if y >= 0.0 and x >= 0.0:
                    teammate_entities.append(self._relative_triplet(y, x, level))
        teammate_entities.sort(key=self._entity_sort_key)

        encoded = np.empty(self._observation_size, dtype=np.float32)
        encoded[0] = float(self_level)
        cursor = 1
        for slot_index in range(self.max_food):
            encoded[cursor : cursor + 3] = (
                food_entities[slot_index] if slot_index < len(food_entities) else self._entity_sentinel
            )
            cursor += 3
        for slot_index in range(max(0, self.max_agents - 1)):
            encoded[cursor : cursor + 3] = (
                teammate_entities[slot_index]
                if slot_index < len(teammate_entities)
                else self._entity_sentinel
            )
            cursor += 3
        return encoded

    def observation_space(self, agent_id: str):
        del agent_id
        return self._observation_space

    def action_space(self, agent_id: str):
        return self.env.action_space(agent_id)

    def reset(self, seed: int | None = None):
        observations, info = self.env.reset(seed=seed)
        return (
            {
                agent_id: self._encode_observation(raw_obs)
                for agent_id, raw_obs in observations.items()
            },
            info,
        )

    def step(self, action_dict: dict[str, int]):
        observations, rewards, terminations, truncations, info = self.env.step(action_dict)
        return (
            {
                agent_id: self._encode_observation(raw_obs)
                for agent_id, raw_obs in observations.items()
            },
            rewards,
            terminations,
            truncations,
            info,
        )

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage as it appears in `task.json`."""

    name: str
    fraction: float
    counts: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class StageWindow:
    """One resolved stage window after fractions are converted to episode ranges."""

    stage_index: int
    name: str
    start_episode: int
    end_episode: int
    counts: tuple[int, ...]
    weights: tuple[float, ...]

    def contains(self, episode_index: int) -> bool:
        return self.start_episode <= episode_index < self.end_episode


@dataclass(frozen=True)
class EvalResult:
    """Summary of one evaluation batch for one checkpoint and one team size."""

    phase: str
    checkpoint_episode: int
    n_agents: int
    rollout_count: int
    return_mean: float
    return_std: float
    return_min: float
    return_max: float


def _configured_agent_counts(task_config: dict) -> list[int]:
    """Collect all team sizes that this task configuration expects to support."""
    counts = {
        int(task_config["train_max_agents"]),
        int(task_config["max_agents"]),
        *(int(value) for value in task_config.get("eval_counts", [])),
        *(int(value) for value in task_config.get("video_counts", [])),
    }
    for stage in task_config.get("curriculum", []):
        counts.update(int(value) for value in stage.get("counts", []))
    return sorted(counts)


def validate_agent_count_support(task_config: dict, *, make_env_fn: Callable[..., object], seed: int = 42) -> None:
    """Fail early if the environment cannot build the requested team sizes."""
    for n_agents in _configured_agent_counts(task_config):
        env = make_env_fn(task_config, seed=seed, n_agents=n_agents, render_mode=None)
        try:
            observations, _ = env.reset(seed=seed + n_agents)
            if len(observations) != n_agents:
                raise RuntimeError(f"Expected {n_agents} agents but got {len(observations)}.")
        finally:
            env.close()


def build_curriculum_windows(task_config: dict) -> list[StageWindow]:
    """Convert fractional curriculum stages into explicit episode windows."""
    total_episodes = int(task_config["episodes"])
    stages = [
        CurriculumStage(
            name=str(stage["name"]),
            fraction=float(stage["fraction"]),
            counts=tuple(int(value) for value in stage["counts"]),
            weights=tuple(float(value) for value in stage["weights"]),
        )
        for stage in task_config["curriculum"]
    ]
    raw_lengths = [stage.fraction * total_episodes for stage in stages]
    lengths = [int(np.floor(value)) for value in raw_lengths]
    remainder = total_episodes - sum(lengths)
    if remainder > 0:
        fractional_order = np.argsort([raw - np.floor(raw) for raw in raw_lengths])[::-1]
        for index in fractional_order[:remainder]:
            lengths[int(index)] += 1

    windows: list[StageWindow] = []
    cursor = 0
    for stage_index, (stage, stage_length) in enumerate(zip(stages, lengths), start=1):
        if stage_length <= 0:
            continue
        windows.append(
            StageWindow(
                stage_index=stage_index,
                name=stage.name,
                start_episode=cursor,
                end_episode=cursor + stage_length,
                counts=stage.counts,
                weights=stage.weights,
            )
        )
        cursor += stage_length

    if not windows:
        raise RuntimeError("Curriculum produced no active stages.")
    if windows[-1].end_episode != total_episodes:
        last = windows[-1]
        windows[-1] = StageWindow(
            stage_index=last.stage_index,
            name=last.name,
            start_episode=last.start_episode,
            end_episode=total_episodes,
            counts=last.counts,
            weights=last.weights,
        )
    return windows


def resolve_curriculum_stage(episode_index: int, windows: Sequence[StageWindow]) -> StageWindow:
    """Return the stage that owns one training episode."""
    for window in windows:
        if window.contains(episode_index):
            return window
    return windows[-1]


def sample_curriculum_count(
    episode_index: int,
    windows: Sequence[StageWindow],
    rng: np.random.Generator,
) -> tuple[int, StageWindow]:
    """Sample one training team size from the active curriculum stage."""
    stage = resolve_curriculum_stage(episode_index, windows)
    sampled = int(rng.choice(stage.counts, p=np.asarray(stage.weights, dtype=np.float64)))
    return sampled, stage


def validation_selection_start_episode(task_config: dict, windows: Sequence[StageWindow]) -> int:
    """Return the first episode whose checkpoints are allowed to compete for best-model selection."""
    selection_stage = int(task_config["checkpoint_selection_stage_index"])
    for window in windows:
        if window.stage_index >= selection_stage:
            return int(window.start_episode + 1)
    return int(windows[-1].start_episode + 1)


def is_checkpoint_selection_eligible(task_config: dict, checkpoint_episode: int, windows: Sequence[StageWindow]) -> bool:
    """Check whether one checkpoint belongs to the evaluation stage we use for model selection."""
    selection_stage = int(task_config["checkpoint_selection_stage_index"])
    stage = resolve_curriculum_stage(max(0, int(checkpoint_episode) - 1), windows)
    return bool(stage.stage_index >= selection_stage)


def filter_selection_eligible_results(
    task_config: dict,
    eval_results: Sequence[EvalResult],
    windows: Sequence[StageWindow],
) -> list[EvalResult]:
    """Keep only the validation results that may influence checkpoint selection."""
    return [
        eval_result
        for eval_result in eval_results
        if is_checkpoint_selection_eligible(task_config, eval_result.checkpoint_episode, windows)
    ]


# -----------------------------------------------------------------------------
# Evaluation summaries
# -----------------------------------------------------------------------------


def _mean_or_nan(values: np.ndarray) -> float:
    """Return the mean of one array, or `nan` if it is empty."""
    return float(np.mean(values)) if values.size else float("nan")


def grouped_eval_metrics(task_config: dict, eval_results: Sequence[EvalResult]) -> dict[str, float]:
    """Aggregate one checkpoint's per-count results into the summary used in reports."""
    if not eval_results:
        return {
            "overall_eval_mean": float("nan"),
            "overall_eval_std": float("nan"),
            "small_team_mean": float("nan"),
            "mid_team_mean": float("nan"),
            "large_team_mean": float("nan"),
            "best_count_return": float("nan"),
            "worst_count_return": float("nan"),
            "generalization_gap": float("nan"),
            "frontier_gap": float("nan"),
        }

    eval_counts = tuple(int(value) for value in task_config["eval_counts"])
    count_to_mean = {eval_result.n_agents: float(eval_result.return_mean) for eval_result in eval_results}
    ordered_means = np.asarray([count_to_mean[count] for count in eval_counts], dtype=np.float32)
    small = np.asarray([count_to_mean[count] for count in eval_counts if count <= 3], dtype=np.float32)
    mid = np.asarray([count_to_mean[count] for count in eval_counts if 4 <= count <= 7], dtype=np.float32)
    large = np.asarray([count_to_mean[count] for count in eval_counts if count >= 8], dtype=np.float32)
    best = float(np.max(ordered_means))
    worst = float(np.min(ordered_means))
    return {
        "overall_eval_mean": float(np.mean(ordered_means)),
        "overall_eval_std": float(np.std(ordered_means)),
        "small_team_mean": _mean_or_nan(small),
        "mid_team_mean": _mean_or_nan(mid),
        "large_team_mean": _mean_or_nan(large),
        "best_count_return": best,
        "worst_count_return": worst,
        "generalization_gap": float(best - worst),
        "frontier_gap": _mean_or_nan(large) - _mean_or_nan(small),
    }


def run_policy_evaluation(
    task_config: dict,
    *,
    checkpoint_episode: int,
    phase: str,
    rollout_count: int,
    evaluate_one_count_fn: Callable[[int, int], Sequence[float]],
) -> list[EvalResult]:
    """Evaluate one checkpoint across all configured team sizes."""
    eval_results: list[EvalResult] = []
    for n_agents in [int(value) for value in task_config["eval_counts"]]:
        returns = np.asarray(evaluate_one_count_fn(n_agents, rollout_count), dtype=np.float32)
        eval_results.append(
            EvalResult(
                phase=str(phase),
                checkpoint_episode=int(checkpoint_episode),
                n_agents=int(n_agents),
                rollout_count=int(rollout_count),
                return_mean=float(np.mean(returns)),
                return_std=float(np.std(returns)),
                return_min=float(np.min(returns)),
                return_max=float(np.max(returns)),
            )
        )
    return eval_results


def _validation_summary(task_config: dict, eval_results: Sequence[EvalResult]) -> dict[str, float]:
    """Collapse validation results by checkpoint and pick the best one."""
    if not eval_results:
        return {
            "best_validation_mean": float("nan"),
            "best_validation_episode": -1,
            "convergence_episode_90pct": -1,
        }

    grouped_by_checkpoint: dict[int, list[EvalResult]] = {}
    for eval_result in eval_results:
        grouped_by_checkpoint.setdefault(int(eval_result.checkpoint_episode), []).append(eval_result)

    checkpoints = sorted(grouped_by_checkpoint)
    means = np.asarray(
        [grouped_eval_metrics(task_config, grouped_by_checkpoint[checkpoint])["overall_eval_mean"] for checkpoint in checkpoints],
        dtype=np.float32,
    )
    best_index = int(np.argmax(means))
    best_mean = float(means[best_index])
    threshold = 0.9 * best_mean
    convergence_episode = -1
    for checkpoint, mean_value in zip(checkpoints, means):
        if float(mean_value) >= threshold:
            convergence_episode = int(checkpoint)
            break
    return {
        "best_validation_mean": best_mean,
        "best_validation_episode": int(checkpoints[best_index]),
        "convergence_episode_90pct": int(convergence_episode),
    }


def build_summary(
    *,
    task_config: dict,
    algorithm: str,
    seed: int,
    episodes: int,
    curriculum_windows: Sequence[StageWindow],
    train_history_rows: Sequence[dict],
    validation_results: Sequence[EvalResult],
    best_checkpoint_test_results: Sequence[EvalResult],
    final_checkpoint_test_results: Sequence[EvalResult],
    extra_metrics: Optional[dict[str, float]] = None,
) -> dict:
    """Build the summary JSON written at the end of one dynamic-task run."""
    train_rewards = np.asarray([float(row["train_return_mean"]) for row in train_history_rows], dtype=np.float32)
    moving_avg_window = min(100, max(1, train_rewards.size))
    moving_avg = moving_average(train_rewards, moving_avg_window) if train_rewards.size else np.asarray([], dtype=np.float32)

    logged_validation_summary = _validation_summary(task_config, validation_results)
    eligible_validation_results = filter_selection_eligible_results(task_config, validation_results, curriculum_windows)
    filtered_validation_summary = _validation_summary(task_config, eligible_validation_results)
    final_test_summary = grouped_eval_metrics(task_config, final_checkpoint_test_results)
    uses_validation_selection = bool(validation_results)
    test_summary = {
        "final_checkpoint": final_test_summary,
        "objective_score": float(final_test_summary["overall_eval_mean"]),
        "best_vs_final_drop": 0.0,
    }
    if uses_validation_selection:
        best_test_summary = grouped_eval_metrics(task_config, best_checkpoint_test_results)
        test_summary = {
            "best_checkpoint": best_test_summary,
            "final_checkpoint": final_test_summary,
            "objective_score": float(0.7 * best_test_summary["overall_eval_mean"] + 0.3 * final_test_summary["overall_eval_mean"]),
            "best_vs_final_drop": float(best_test_summary["overall_eval_mean"] - final_test_summary["overall_eval_mean"]),
        }

    return {
        "env_name": str(task_config["env_name"]),
        "algorithm": algorithm,
        "seed": int(seed),
        "episodes": int(episodes),
        "train_agent_count_max": int(task_config["train_max_agents"]),
        "eval_agent_counts": [int(value) for value in task_config["eval_counts"]],
        "curriculum": [
            {
                "stage_index": window.stage_index,
                "name": window.name,
                "start_episode": window.start_episode + 1,
                "end_episode": window.end_episode,
                "counts": list(window.counts),
                "weights": list(window.weights),
            }
            for window in curriculum_windows
        ],
        "train": {
            "final_episode_return": float(train_rewards[-1]) if train_rewards.size else float("nan"),
            "moving_average_window": int(moving_avg_window),
            "final_moving_average": float(moving_avg[-1]) if moving_avg.size else float("nan"),
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
        "validation": filtered_validation_summary,
        "test": test_summary,
        "checkpoint_selection": {
            "selection_stage_index": int(task_config["checkpoint_selection_stage_index"]),
            "selection_start_episode": validation_selection_start_episode(task_config, curriculum_windows),
            "selection_eligible_checkpoint_count": int(len({eval_result.checkpoint_episode for eval_result in eligible_validation_results})),
            "logged_checkpoint_count": int(len({eval_result.checkpoint_episode for eval_result in validation_results})),
            "logged_best_validation_mean": float(logged_validation_summary["best_validation_mean"]),
            "logged_best_validation_episode": int(logged_validation_summary["best_validation_episode"]),
        },
        "extra_metrics": dict(sorted((extra_metrics or {}).items())),
    }


# -----------------------------------------------------------------------------
# Plotting and rollout helpers
# -----------------------------------------------------------------------------


def plot_training_dashboard(
    save_path: Path,
    *,
    task_config: dict,
    algorithm: str,
    train_history_rows: Sequence[dict],
    eval_results: Sequence[EvalResult],
    update_history: Sequence[dict[str, float]],
) -> None:
    """Create the task-specific dashboard used in reports and sweeps."""
    rewards = np.asarray([float(row["train_return_mean"]) for row in train_history_rows], dtype=np.float32)
    losses = np.asarray([float(row["train_loss_mean"]) for row in train_history_rows], dtype=np.float32)
    episodes = np.arange(1, rewards.size + 1)
    eval_counts = [int(value) for value in task_config["eval_counts"]]

    eval_checkpoints = sorted({eval_result.checkpoint_episode for eval_result in eval_results})
    grouped_by_checkpoint = {
        checkpoint: grouped_eval_metrics(
            task_config,
            [eval_result for eval_result in eval_results if eval_result.checkpoint_episode == checkpoint],
        )
        for checkpoint in eval_checkpoints
    }

    metric_groups = grouped_dashboard_metric_keys(update_history)
    fig_rows = 2 + len(metric_groups) if metric_groups else 3
    fig, axes = plt.subplots(fig_rows, 1, figsize=(13, 4 * fig_rows))
    axes[0].plot(episodes, rewards, alpha=0.30, color="tab:blue", label="train return")
    reward_ma = moving_average(rewards, min(100, max(1, rewards.size))) if rewards.size else np.asarray([], dtype=np.float32)
    if reward_ma.size:
        reward_ma_x = np.arange(rewards.size - reward_ma.size + 1, rewards.size + 1)
        axes[0].plot(reward_ma_x, reward_ma, color="tab:orange", linewidth=2.0, label="moving avg")
    axes[0].set_title(f"{algorithm} on {task_config['env_name']}")
    axes[0].set_ylabel("Train team return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if grouped_by_checkpoint:
        x = np.asarray(eval_checkpoints, dtype=np.int64)
        overall = np.asarray([grouped_by_checkpoint[idx]["overall_eval_mean"] for idx in eval_checkpoints], dtype=np.float32)
        small = np.asarray([grouped_by_checkpoint[idx]["small_team_mean"] for idx in eval_checkpoints], dtype=np.float32)
        mid = np.asarray([grouped_by_checkpoint[idx]["mid_team_mean"] for idx in eval_checkpoints], dtype=np.float32)
        large = np.asarray([grouped_by_checkpoint[idx]["large_team_mean"] for idx in eval_checkpoints], dtype=np.float32)
        small_counts = [count for count in eval_counts if count <= 3]
        mid_counts = [count for count in eval_counts if 4 <= count <= 7]
        large_counts = [count for count in eval_counts if count >= 8]
        axes[1].plot(x, overall, marker="o", linewidth=2.0, label=f"overall {eval_counts[0]}..{eval_counts[-1]}")
        if small_counts:
            axes[1].plot(x, small, marker="s", linewidth=1.2, label=f"small {small_counts[0]}..{small_counts[-1]}")
        if mid_counts:
            axes[1].plot(x, mid, marker="^", linewidth=1.2, label=f"mid {mid_counts[0]}..{mid_counts[-1]}")
        if large_counts:
            axes[1].plot(x, large, marker="d", linewidth=1.2, label=f"large {large_counts[0]}..{large_counts[-1]}")
    axes[1].set_ylabel("Policy eval team return")
    axes[1].grid(True, alpha=0.3)
    if grouped_by_checkpoint:
        axes[1].legend()

    if metric_groups:
        for axis, (title, metric_keys) in zip(axes[2:], metric_groups):
            plot_dashboard_metric_group(axis, update_history, title=title, metric_keys=metric_keys)
        axes[-1].set_xlabel("Update")
    else:
        axes[2].plot(episodes, losses, alpha=0.30, color="tab:red", label="episode mean loss")
        loss_ma = moving_average(losses, min(100, max(1, losses.size))) if losses.size else np.asarray([], dtype=np.float32)
        if loss_ma.size:
            loss_ma_x = np.arange(losses.size - loss_ma.size + 1, losses.size + 1)
            axes[2].plot(loss_ma_x, loss_ma, color="tab:purple", linewidth=2.0, label="moving avg")
        axes[2].legend()
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Episode loss")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def get_or_create_env(
    env_cache: dict,
    task_config: dict,
    *,
    seed: int,
    n_agents: int,
    render_mode: str | None,
    make_env_fn: Callable[..., object],
):
    """Reuse evaluation environments across rollouts to avoid repeated construction overhead."""
    cache_key = (int(n_agents), render_mode)
    env = env_cache.get(cache_key)
    if env is None:
        env = make_env_fn(task_config, seed=seed, n_agents=n_agents, render_mode=render_mode)
        env_cache[cache_key] = env
    return env


def close_env_cache(env_cache: dict) -> None:
    """Close every cached environment instance."""
    for env in env_cache.values():
        env.close()


def evaluate_one_count(
    learner,
    task_config: dict,
    env_spec: ParallelEnvSpec,
    env_cache: dict,
    *,
    seed: int,
    n_agents: int,
    rollout_count: int,
    seed_offset: int,
    make_env_fn: Callable[..., object],
    summarize_episode_return_fn: Callable[[float, int], float],
) -> list[float]:
    """Run evaluation rollouts for one specific team size."""
    learner.set_eval_mode()
    returns: list[float] = []
    try:
        for rollout_index in range(int(rollout_count)):
            env_seed = int(seed + seed_offset + (100 * n_agents) + rollout_index)
            env = get_or_create_env(
                env_cache,
                task_config,
                seed=seed,
                n_agents=n_agents,
                render_mode=None,
                make_env_fn=make_env_fn,
            )
            observations, _ = env.reset(seed=env_seed)
            agent_ids = list(env.possible_agents)
            learner.reset_episode()
            done = {agent_id: False for agent_id in agent_ids}
            total_reward = 0.0

            while True:
                obs_dict = {
                    agent_id: pad_vector(np.asarray(observations[agent_id], dtype=np.float32), env_spec.obs_size)
                    for agent_id in agent_ids
                    if not done[agent_id]
                }
                actions = learner.act_parallel(obs_dict)
                observations, rewards, terminations, truncations, _ = env.step(actions)
                total_reward += sum(float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids)
                done = {
                    agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    for agent_id in agent_ids
                }
                if all(done.values()):
                    break
            returns.append(float(summarize_episode_return_fn(total_reward, len(agent_ids))))
    finally:
        learner.set_train_mode()
    return returns


def save_rollout_gif(
    learner,
    task_config: dict,
    env_spec: ParallelEnvSpec,
    *,
    out_dir: Path,
    seed: int,
    make_env_fn: Callable[..., object],
) -> None:
    """Render a short stitched GIF across representative team sizes."""
    env_cache: dict = {}
    frames: list[np.ndarray] = []
    step_index = 0
    learner.set_eval_mode()
    try:
        rollout_counts = tuple(int(value) for value in task_config.get("video_counts", []))
        for episode_index, n_agents in enumerate(rollout_counts):
            env_seed = int(seed + int(task_config["gif_seed_offset"]) + episode_index)
            env = get_or_create_env(
                env_cache,
                task_config,
                seed=seed,
                n_agents=int(n_agents),
                render_mode="rgb_array",
                make_env_fn=make_env_fn,
            )
            observations, _ = env.reset(seed=env_seed)
            agent_ids = list(env.possible_agents)
            learner.reset_episode()
            done = {agent_id: False for agent_id in agent_ids}

            while True:
                obs_dict = {
                    agent_id: pad_vector(np.asarray(observations[agent_id], dtype=np.float32), env_spec.obs_size)
                    for agent_id in agent_ids
                    if not done[agent_id]
                }
                actions = learner.act_parallel(obs_dict)
                observations, _, terminations, truncations, _ = env.step(actions)
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
            if len(frames) >= int(task_config["max_frames"]):
                break
    finally:
        learner.set_train_mode()
        close_env_cache(env_cache)
    save_gif(frames, out_dir / "policy_rollout.gif")
