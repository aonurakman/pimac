"""Run one benchmark experiment on fixed-team `simple_spread_v3`.

This file is intentionally direct. Open it when you want to inspect the full task flow:
load configs, build the environment and learner, train, evaluate, and write results.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.base import ParallelEnvSpec, ParallelTransition
from algorithms.registry import ALGORITHM_ORDER, get_algorithm_class
from simple_spread.utils import (
    EvalResult,
    build_summary,
    evaluate_rollouts,
    run_fixed_evaluation,
    save_rollout_gif,
)
from utils import (
    active_agent_mask,
    flatten_update_history,
    learner_temperature,
    load_json,
    make_run_dir,
    plot_basic_curves,
    resolve_device,
    resolve_json_path,
    save_update_history_csv,
    save_update_history_json,
    set_global_seeds,
    write_csv,
    write_json,
)


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


def make_env(task_config: dict, seed: int, render_mode: str | None = None):
    """Build the fixed-team PettingZoo environment for training or evaluation."""
    try:
        from pettingzoo.mpe import simple_spread_v3
    except Exception as exc:  # pragma: no cover
        raise ImportError("PettingZoo MPE is required. Install `pettingzoo[mpe]` and `pygame`.") from exc

    env = simple_spread_v3.parallel_env(
        N=int(task_config["n_agents"]),
        max_cycles=int(task_config["max_cycles"]),
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env



def build_env_spec(task_config: dict, seed: int) -> ParallelEnvSpec:
    """Read the environment spaces once and convert them into the learner spec."""
    env = make_env(task_config, seed=seed, render_mode=None)
    try:
        agent_ids = list(env.possible_agents)
        obs_size = int(env.observation_space(agent_ids[0]).shape[0])
        action_space_size = int(env.action_space(agent_ids[0]).n)
    finally:
        env.close()
    return ParallelEnvSpec(obs_size=obs_size, action_space_size=action_space_size, max_agents=len(agent_ids))


def prepare_observation(raw_obs: np.ndarray, env_spec: ParallelEnvSpec) -> np.ndarray:
    """Convert one raw environment observation into the learner input for this task."""
    del env_spec
    return np.asarray(raw_obs, dtype=np.float32)


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
    """Train one learner on fixed-team simple spread and write the standard outputs."""
    task_path = TASK_DIR / "task.json" if task_config_path is None else resolve_json_path(task_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    alg_path = resolve_json_path(alg_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    task_config = load_json(task_path)
    learner_config = load_json(alg_path)

    set_global_seeds(seed)
    env_spec = build_env_spec(task_config, seed)
    learner_cls = get_algorithm_class(algorithm)
    learner = learner_cls(env_spec=env_spec, config=learner_config, device=resolve_device(device))

    out_dir = Path(make_run_dir(str(task_config["task_name"]), algorithm, results_root=results_root, run_id=run_id))
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    final_ckpt_path = out_dir / "final_checkpoint.pt"

    train_history_rows: list[dict] = []
    validation_results: list[EvalResult] = []
    learner_updates = 0
    global_step = 0
    best_eval = -float("inf")

    for episode_index in range(int(task_config["episodes"])):
        env_seed = int(seed + episode_index)
        env = make_env(task_config, seed=env_seed, render_mode=None)
        try:
            observations, _ = env.reset(seed=env_seed)
            agent_ids = list(env.possible_agents)
            learner.reset_episode()
            learner.set_train_mode()
            done = {agent_id: False for agent_id in agent_ids}
            total_reward = 0.0
            episode_reports: list[dict] = []

            # Run one full parallel episode and feed every joint step to the learner.
            while True:
                current_active = [agent_id for agent_id in agent_ids if not done[agent_id]]
                obs_dict = {agent_id: observations[agent_id] for agent_id in current_active}
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
                    next_obs_dict={agent_id: next_obs[agent_id] for agent_id in next_active},
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
        finally:
            env.close()

        train_history_rows.append(
            {
                "episode": episode_index + 1,
                "train_return_mean": total_reward / len(agent_ids),
                "train_loss_mean": float(sum(row["total_loss"] for row in episode_reports) / len(episode_reports)) if episode_reports else 0.0,
                "temperature": learner_temperature(learner),
                "learner_updates": learner_updates,
                "global_step": global_step,
            }
        )

        eval_every = int(task_config["eval_every_episodes"])
        if eval_every and ((episode_index + 1) % eval_every == 0):
            eval_result = run_fixed_evaluation(
                checkpoint_episode=episode_index + 1,
                phase="validation",
                rollout_count=int(task_config["validation_rollouts"]),
                evaluate_rollouts_fn=lambda rollout_count: evaluate_rollouts(
                    learner,
                    task_config,
                    seed=seed,
                    rollout_count=rollout_count,
                    seed_offset=int(task_config["validation_seed_offset"]),
                    make_env_fn=make_env,
                ),
            )
            validation_results.append(eval_result)
            if eval_result.return_mean > (best_eval + float(task_config["min_improve"])):
                learner.save_checkpoint(best_ckpt_path)
                best_eval = float(eval_result.return_mean)

    learner.save_checkpoint(final_ckpt_path)
    if not best_ckpt_path.is_file():
        learner.save_checkpoint(best_ckpt_path)

    best_learner = type(learner).load_checkpoint(
        best_ckpt_path,
        env_spec=learner.env_spec,
        config=learner.config,
        device=str(learner.device),
    )
    best_checkpoint_test = run_fixed_evaluation(
        checkpoint_episode=int(task_config["episodes"]),
        phase="best_checkpoint_test",
        rollout_count=int(task_config["test_rollouts"]),
        evaluate_rollouts_fn=lambda rollout_count: evaluate_rollouts(
            best_learner,
            task_config,
            seed=seed,
            rollout_count=rollout_count,
            seed_offset=int(task_config["test_seed_offset"]),
            make_env_fn=make_env,
        ),
    )
    final_checkpoint_test = run_fixed_evaluation(
        checkpoint_episode=int(task_config["episodes"]),
        phase="final_checkpoint_test",
        rollout_count=int(task_config["test_rollouts"]),
        evaluate_rollouts_fn=lambda rollout_count: evaluate_rollouts(
            learner,
            task_config,
            seed=seed,
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
        validation_results=validation_results,
        best_checkpoint_test=best_checkpoint_test,
        final_checkpoint_test=final_checkpoint_test,
        train_history_rows=train_history_rows,
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
    write_csv(
        out_dir / "eval_history.csv",
        [asdict(eval_result) for eval_result in [*validation_results, best_checkpoint_test, final_checkpoint_test]],
    )
    save_update_history_json(out_dir / "update_history.json", learner.get_update_history())
    save_update_history_csv(out_dir / "update_history.csv", learner.get_update_history())
    plot_basic_curves(
        save_path=out_dir / "learning_curves.png",
        title=f"{algorithm} on {task_config['env_name']}",
        rewards=[row["train_return_mean"] for row in train_history_rows],
        losses=[row["train_loss_mean"] for row in train_history_rows],
        eval_x=[eval_result.checkpoint_episode for eval_result in validation_results],
        eval_rewards=[eval_result.return_mean for eval_result in validation_results],
        update_history=update_history_rows,
    )

    if not skip_gif:
        save_rollout_gif(best_learner, task_config, out_dir=out_dir, seed=seed, make_env_fn=make_env)

    return str(out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """Create the small CLI used by both direct runs and sweep tooling."""
    parser = argparse.ArgumentParser(description="Run one experiment on simple_spread_v3.")
    parser.add_argument("--algorithm", choices=ALGORITHM_ORDER, required=True)
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
