"""Run one benchmark experiment on dynamic-team RWARE."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.base import ParallelEnvSpec, ParallelTransition
from algorithms.registry import ALGORITHM_ORDER, get_algorithm_class
from robotic_warehouse_dynamic.utils import (
    EvalResult,
    StageWindow,
    _checkpoint_selection_mode,
    _configured_periodic_eval_counts,
    build_curriculum_windows,
    build_summary,
    close_env_cache,
    evaluate_one_count,
    get_or_create_env,
    grouped_eval_metrics,
    is_checkpoint_selection_eligible,
    pad_vector,
    plot_training_dashboard,
    run_policy_evaluation,
    sample_curriculum_count,
    save_rollout_gif,
    validate_agent_count_support,
)
from utils import (
    active_agent_mask,
    flatten_update_history,
    learner_temperature,
    load_json,
    make_run_dir,
    resolve_device,
    resolve_json_path,
    save_update_history_csv,
    save_update_history_json,
    set_global_seeds,
    write_csv,
    write_json,
)

TASK_ALGORITHM_ORDER = ALGORITHM_ORDER + ("pimac_v6_ablation",)


_CELL_SIZE = 24
_CELL_GAP = 2
_MARGIN = 6
_BACKGROUND = (244, 245, 246)
_GRID = (205, 209, 214)
_HIGHWAY = (234, 236, 239)
_GOAL = (70, 73, 76)
_SHELF = (94, 89, 174)
_REQUESTED_SHELF = (27, 156, 133)
_AGENT = (217, 119, 6)
_AGENT_WITH_LOAD = (200, 67, 89)
_DIR = (22, 24, 29)
_TEXT = (250, 250, 250)
_DIRECTION_DELTAS = {
    0: (0, -0.65),
    1: (0, 0.65),
    2: (-0.65, 0),
    3: (0.65, 0),
}


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


def _cell_bounds(x: int, y: int) -> tuple[int, int, int, int]:
    left = _MARGIN + x * (_CELL_SIZE + _CELL_GAP)
    top = _MARGIN + y * (_CELL_SIZE + _CELL_GAP)
    return left, top, left + _CELL_SIZE, top + _CELL_SIZE


def _render_frame(warehouse_env) -> np.ndarray:
    """Render one top-down frame without relying on RWARE's pyglet renderer."""
    height, width = warehouse_env.grid_size
    image_width = (width * _CELL_SIZE) + ((width - 1) * _CELL_GAP) + (2 * _MARGIN)
    image_height = (height * _CELL_SIZE) + ((height - 1) * _CELL_GAP) + (2 * _MARGIN)
    image = Image.new("RGB", (image_width, image_height), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    goals = {(int(x), int(y)) for x, y in warehouse_env.goals}
    requested = {(int(shelf.x), int(shelf.y)) for shelf in warehouse_env.request_queue}
    highways = np.asarray(warehouse_env.highways, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            fill = _HIGHWAY if int(highways[y, x]) else _BACKGROUND
            if (x, y) in goals:
                fill = _GOAL
            draw.rectangle(_cell_bounds(x, y), fill=fill, outline=_GRID, width=1)

    for shelf in warehouse_env.shelfs:
        x, y = int(shelf.x), int(shelf.y)
        left, top, right, bottom = _cell_bounds(x, y)
        draw.rectangle(
            (left + 4, top + 4, right - 4, bottom - 4),
            fill=_REQUESTED_SHELF if (x, y) in requested else _SHELF,
            outline=None,
        )

    for goal_x, goal_y in goals:
        left, top, right, bottom = _cell_bounds(goal_x, goal_y)
        draw.text((left + 7, top + 3), "G", fill=_TEXT)

    for agent in warehouse_env.agents:
        x, y = int(agent.x), int(agent.y)
        left, top, right, bottom = _cell_bounds(x, y)
        fill = _AGENT_WITH_LOAD if agent.carrying_shelf is not None else _AGENT
        draw.ellipse((left + 4, top + 4, right - 4, bottom - 4), fill=fill, outline=None)

        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        dx, dy = _DIRECTION_DELTAS[int(agent.dir.value)]
        arrow_x = center_x + dx * (_CELL_SIZE * 0.32)
        arrow_y = center_y + dy * (_CELL_SIZE * 0.32)
        draw.line((center_x, center_y, arrow_x, arrow_y), fill=_DIR, width=3)
        draw.ellipse((arrow_x - 2, arrow_y - 2, arrow_x + 2, arrow_y + 2), fill=_DIR, outline=None)

    return np.asarray(image, dtype=np.uint8)


class RwareParallelAdapter:
    """Thin dict-based adapter around the tuple/list RWARE Gymnasium API."""

    def __init__(self, env):
        self.env = env
        self.possible_agents = [f"agent_{index}" for index in range(int(env.unwrapped.n_agents))]

    @property
    def agents(self) -> list[str]:
        return list(self.possible_agents)

    def _agent_index(self, agent_id: str) -> int:
        return int(str(agent_id).split("_")[-1])

    def _obs_dict(self, observations) -> dict[str, np.ndarray]:
        return {
            agent_id: np.asarray(observations[index], dtype=np.float32)
            for index, agent_id in enumerate(self.possible_agents)
        }

    def _info_dict(self, info) -> dict[str, dict]:
        if isinstance(info, dict) and set(info.keys()) == set(self.possible_agents):
            return {
                str(agent_id): dict(agent_info) if isinstance(agent_info, dict) else {}
                for agent_id, agent_info in info.items()
            }
        info_payload = dict(info) if isinstance(info, dict) else {}
        return {agent_id: dict(info_payload) for agent_id in self.possible_agents}

    def observation_space(self, agent_id: str):
        return self.env.observation_space.spaces[self._agent_index(agent_id)]

    def action_space(self, agent_id: str):
        return self.env.action_space.spaces[self._agent_index(agent_id)]

    def reset(self, seed: int | None = None):
        observations, info = self.env.reset(seed=seed)
        return self._obs_dict(observations), self._info_dict(info)

    def step(self, action_dict: dict[str, int]):
        ordered_actions = tuple(int(action_dict.get(agent_id, 0)) for agent_id in self.possible_agents)
        observations, rewards, done, truncated, info = self.env.step(ordered_actions)
        return (
            self._obs_dict(observations),
            {
                agent_id: float(rewards[index])
                for index, agent_id in enumerate(self.possible_agents)
            },
            {agent_id: bool(done) for agent_id in self.possible_agents},
            {agent_id: bool(truncated) for agent_id in self.possible_agents},
            self._info_dict(info),
        )

    def render(self):
        return _render_frame(self.env.unwrapped)

    def close(self) -> None:
        self.env.close()


def make_env(task_config: dict, seed: int, n_agents: int, render_mode: str | None = None):
    """Build one dynamic-team RWARE environment instance."""
    try:
        import gymnasium as gym
        from rware.warehouse import RewardType
    except Exception as exc:  # pragma: no cover
        raise ImportError("RWARE is required. Install `rware==2.0.0`.") from exc

    reward_type_name = str(task_config["reward_type"]).lower()
    reward_type_map = {
        "global": RewardType.GLOBAL,
        "individual": RewardType.INDIVIDUAL,
        "two_stage": RewardType.TWO_STAGE,
    }
    if reward_type_name not in reward_type_map:
        raise ValueError(f"Unsupported reward_type: {task_config['reward_type']!r}")

    env_id = str(task_config["env_id_template"]).format(n_agents=int(n_agents))
    wrapped_env = RwareParallelAdapter(
        gym.make(
            env_id,
            reward_type=reward_type_map[reward_type_name],
            max_steps=int(task_config["max_steps"]),
            render_mode=render_mode,
            disable_env_checker=True,
        )
    )
    wrapped_env.reset(seed=seed)
    return wrapped_env


def build_env_spec(task_config: dict, seed: int) -> ParallelEnvSpec:
    """Read the widest observation/action spaces used by the dynamic task."""
    env = make_env(task_config, seed=seed, n_agents=int(task_config["max_agents"]), render_mode=None)
    try:
        agent_ids = list(env.possible_agents)
        obs_size = int(env.observation_space(agent_ids[0]).shape[0])
        action_space_size = int(env.action_space(agent_ids[0]).n)
    finally:
        env.close()
    return ParallelEnvSpec(
        obs_size=obs_size,
        action_space_size=action_space_size,
        max_agents=int(task_config["max_agents"]),
    )


def prepare_observation(raw_obs: np.ndarray, env_spec: ParallelEnvSpec) -> np.ndarray:
    """Convert one raw environment observation into the learner input for this task."""
    return pad_vector(np.asarray(raw_obs, dtype=np.float32), env_spec.obs_size)


# -----------------------------------------------------------------------------
# Experiment loop
# -----------------------------------------------------------------------------


def run_task(
    *,
    algorithm: str,
    alg_config_path: str,
    task_config_path: str | None = None,
    seed: int = 42,
    results_root: str | None = None,
    run_id: str | None = None,
    skip_gif: bool = False,
    device: str = "auto",
) -> str:
    """Train one learner on the dynamic task and write the standard outputs."""
    task_path = TASK_DIR / "task.json" if task_config_path is None else resolve_json_path(task_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    alg_path = resolve_json_path(alg_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    task_config = load_json(task_path)
    learner_config = load_json(alg_path)

    set_global_seeds(seed)
    validate_agent_count_support(task_config, make_env_fn=make_env, seed=seed)
    env_spec = build_env_spec(task_config, seed)
    learner_cls = get_algorithm_class(algorithm)
    learner = learner_cls(env_spec=env_spec, config=learner_config, device=resolve_device(device))

    out_dir = Path(make_run_dir(str(task_config["task_name"]), algorithm, results_root=results_root, run_id=run_id))
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    final_ckpt_path = out_dir / "final_checkpoint.pt"
    curriculum_windows: list[StageWindow] = build_curriculum_windows(task_config)
    periodic_eval_counts = _configured_periodic_eval_counts(task_config)
    uses_validation_selection = _checkpoint_selection_mode(task_config) == "best_validation"
    curriculum_rng = np.random.default_rng(seed)
    env_cache: dict = {}

    train_history_rows: list[dict] = []
    curriculum_rows: list[dict] = []
    validation_results: list[EvalResult] = []
    learner_updates = 0
    global_step = 0
    best_eval = -float("inf")

    try:
        for episode_index in range(int(task_config["episodes"])):
            n_agents, stage = sample_curriculum_count(episode_index, curriculum_windows, curriculum_rng)
            env_seed = int(seed + episode_index)
            env = get_or_create_env(
                env_cache,
                task_config,
                seed=seed,
                n_agents=n_agents,
                render_mode=None,
                make_env_fn=make_env,
            )
            observations, _ = env.reset(seed=env_seed)
            agent_ids = list(env.possible_agents)
            learner.reset_episode()
            learner.set_train_mode()
            done = {agent_id: False for agent_id in agent_ids}
            total_reward = 0.0
            episode_reports: list[dict] = []

            while True:
                current_active = [agent_id for agent_id in agent_ids if not done[agent_id]]
                obs_dict = {
                    agent_id: pad_vector(np.asarray(observations[agent_id], dtype=np.float32), env_spec.obs_size)
                    for agent_id in current_active
                }
                actions = learner.act_parallel(obs_dict)
                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                global_step += 1

                done_dict = {
                    agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    for agent_id in agent_ids
                }
                next_active = [agent_id for agent_id in agent_ids if not done_dict[agent_id] and agent_id in next_obs]
                transition = ParallelTransition(
                    obs_dict=obs_dict,
                    action_dict=actions,
                    reward_dict={agent_id: float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids},
                    next_obs_dict={
                        agent_id: pad_vector(np.asarray(next_obs[agent_id], dtype=np.float32), env_spec.obs_size)
                        for agent_id in next_active
                    },
                    done_dict=done_dict,
                    active_agent_mask_dict=active_agent_mask(agent_ids, current_active),
                    next_active_agent_mask_dict=active_agent_mask(agent_ids, next_active),
                )
                learner.record_parallel_step(transition)
                report = learner.maybe_update(global_step=global_step, episode_index=episode_index + 1)
                if report is not None:
                    learner_updates += 1
                    episode_reports.append(report.to_flat_dict())

                total_reward += sum(float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids)
                observations = next_obs
                done = done_dict
                if all(done.values()):
                    break

            train_history_rows.append(
                {
                    "episode": episode_index + 1,
                    "train_return_mean": total_reward / len(agent_ids),
                    "train_loss_mean": float(np.mean([row["total_loss"] for row in episode_reports])) if episode_reports else 0.0,
                    "sampled_agent_count": n_agents,
                    "stage_index": stage.stage_index,
                    "stage_name": stage.name,
                    "temperature": learner_temperature(learner),
                    "learner_updates": learner_updates,
                    "global_step": global_step,
                }
            )
            curriculum_rows.append(
                {
                    "episode": episode_index + 1,
                    "stage_index": stage.stage_index,
                    "stage_name": stage.name,
                    "sampled_agent_count": n_agents,
                    "allowed_counts": "|".join(str(value) for value in stage.counts),
                    "weights": "|".join(f"{value:.2f}" for value in stage.weights),
                }
            )

            eval_every = int(task_config["eval_every_episodes"])
            if eval_every and ((episode_index + 1) % eval_every == 0):
                eval_results = run_policy_evaluation(
                    task_config,
                    checkpoint_episode=episode_index + 1,
                    phase="validation",
                    rollout_count=int(task_config["validation_rollouts"]),
                    evaluate_one_count_fn=lambda eval_n_agents, rollout_count: evaluate_one_count(
                        learner,
                        task_config,
                        env_spec,
                        env_cache,
                        seed=seed,
                        n_agents=eval_n_agents,
                        rollout_count=rollout_count,
                        seed_offset=int(task_config["validation_seed_offset"]),
                        make_env_fn=make_env,
                    ),
                    eval_counts=periodic_eval_counts,
                )
                validation_results.extend(eval_results)

                eligible_results = eval_results if is_checkpoint_selection_eligible(task_config, episode_index + 1, curriculum_windows) else []
                if uses_validation_selection and eligible_results:
                    score = float(grouped_eval_metrics(task_config, eligible_results)["selection_score"])
                    if score > (best_eval + float(task_config["min_improve"])):
                        learner.save_checkpoint(best_ckpt_path)
                        best_eval = score

        learner.save_checkpoint(final_ckpt_path)
        if uses_validation_selection and validation_results and not best_ckpt_path.is_file():
            learner.save_checkpoint(best_ckpt_path)

        best_checkpoint_test_results: list[EvalResult] = []
        best_learner = learner
        final_checkpoint_test_results = run_policy_evaluation(
            task_config,
            checkpoint_episode=int(task_config["episodes"]),
            phase="final_checkpoint_test",
            rollout_count=int(task_config["test_rollouts"]),
            evaluate_one_count_fn=lambda eval_n_agents, rollout_count: evaluate_one_count(
                learner,
                task_config,
                env_spec,
                env_cache,
                seed=seed,
                n_agents=eval_n_agents,
                rollout_count=rollout_count,
                seed_offset=int(task_config["test_seed_offset"]),
                make_env_fn=make_env,
            ),
        )
        if uses_validation_selection and validation_results:
            best_learner = type(learner).load_checkpoint(
                best_ckpt_path,
                env_spec=learner.env_spec,
                config=learner.config,
                device=str(learner.device),
            )
            best_checkpoint_test_results = run_policy_evaluation(
                task_config,
                checkpoint_episode=int(task_config["episodes"]),
                phase="best_checkpoint_test",
                rollout_count=int(task_config["test_rollouts"]),
                evaluate_one_count_fn=lambda eval_n_agents, rollout_count: evaluate_one_count(
                    best_learner,
                    task_config,
                    env_spec,
                    env_cache,
                    seed=seed,
                    n_agents=eval_n_agents,
                    rollout_count=rollout_count,
                    seed_offset=int(task_config["test_seed_offset"]),
                    make_env_fn=make_env,
                ),
            )

        update_history_rows = flatten_update_history(learner.get_update_history())
        extra_metrics = {
            str(key): float(value)
            for key, value in (update_history_rows[-1] if update_history_rows else {}).items()
            if isinstance(value, (int, float))
        }
        summary = build_summary(
            task_config=task_config,
            algorithm=algorithm,
            seed=seed,
            episodes=int(task_config["episodes"]),
            curriculum_windows=curriculum_windows,
            train_history_rows=train_history_rows,
            validation_results=validation_results,
            best_checkpoint_test_results=best_checkpoint_test_results,
            final_checkpoint_test_results=final_checkpoint_test_results,
            uses_validation_selection=uses_validation_selection,
            extra_metrics=extra_metrics,
        )

        write_json(
            out_dir / "config_snapshot.json",
            {
                "algorithm": algorithm,
                "algorithm_config_path": str(alg_path),
                "task_config_path": str(task_path),
                "algorithm_config": learner.config,
                "task_config": task_config,
            },
        )
        write_json(out_dir / "summary.json", summary)
        write_csv(out_dir / "train_history.csv", train_history_rows)
        write_csv(out_dir / "curriculum_trace.csv", curriculum_rows)
        write_csv(
            out_dir / "eval_by_count.csv",
            [
                asdict(eval_result)
                for eval_result in [*validation_results, *best_checkpoint_test_results, *final_checkpoint_test_results]
            ],
        )
        save_update_history_json(out_dir / "update_history.json", learner.get_update_history())
        save_update_history_csv(out_dir / "update_history.csv", learner.get_update_history())
        plot_training_dashboard(
            out_dir / "learning_curves.png",
            task_config=task_config,
            algorithm=algorithm,
            train_history_rows=train_history_rows,
            eval_results=validation_results,
            update_history=update_history_rows,
        )

        if not skip_gif:
            save_rollout_gif(best_learner, task_config, env_spec, out_dir=out_dir, seed=seed, make_env_fn=make_env)

        return str(out_dir)
    finally:
        close_env_cache(env_cache)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """Create the small CLI used by both direct runs and sweep tooling."""
    parser = argparse.ArgumentParser(description="Run one experiment on dynamic-team RWARE.")
    parser.add_argument("--algorithm", choices=TASK_ALGORITHM_ORDER, required=True)
    parser.add_argument("--alg-config", type=str, required=True, help="Path to the algorithm JSON config.")
    parser.add_argument("--task-config", type=str, default=None, help="Optional task JSON. Defaults to task.json in this folder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run one benchmark job."""
    args = _parser().parse_args(argv)
    run_task(
        algorithm=str(args.algorithm),
        alg_config_path=str(args.alg_config),
        task_config_path=args.task_config,
        seed=int(args.seed),
        results_root=args.results_root,
        run_id=args.run_id,
        skip_gif=bool(args.skip_gif),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
