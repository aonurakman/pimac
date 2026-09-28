"""Run one fixed-roster procedural SMACv2 experiment.

The environment adapter is deliberately thin. It keeps SMACv2's native observations, shared
reward, global state, action availability, episode termination, and capability distributions. The
runner only translates them into this repository's dictionary-based parallel learner interface.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.base import ParallelEnvSpec, ParallelTransition
from algorithms.registry import ALGORITHM_ORDER, get_algorithm_class
from smacv2_task.environment import SMACv2ParallelEnv
from utils import (
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


@dataclass(frozen=True)
class EvalResult:
    phase: str
    checkpoint_step: int
    rollout_count: int
    win_rate: float
    return_mean: float
    return_std: float
    episode_length_mean: float


def _validation_objective_settings(task_config: dict[str, Any]) -> tuple[float, float]:
    """Return a bounded tie-break that cannot outweigh one validation win."""
    rollout_count = int(task_config["validation_episodes"])
    weight = float(task_config.get("validation_return_tiebreak_weight", 0.001))
    scale = float(task_config.get("validation_return_scale", 20.0))
    if rollout_count < 1:
        raise ValueError("Calibration requires at least one validation episode.")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("validation_return_tiebreak_weight must be finite and non-negative.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("validation_return_scale must be finite and positive.")
    if 2.0 * weight >= 1.0 / rollout_count:
        raise ValueError(
            "validation_return_tiebreak_weight is too large: its full range must be smaller "
            "than one validation win."
        )
    return weight, scale


def _validation_objective(
    task_config: dict[str, Any], result: EvalResult
) -> tuple[float, float]:
    weight, scale = _validation_objective_settings(task_config)
    return_tiebreak = float(weight * np.tanh(result.return_mean / scale))
    return float(result.win_rate + return_tiebreak), return_tiebreak


def make_env(task_config: dict[str, Any], seed: int) -> SMACv2ParallelEnv:
    """Build one native SMACv2 process wrapped in the shared task shape."""
    return SMACv2ParallelEnv(task_config, seed=seed)


def build_env_spec(task_config: dict[str, Any], env: SMACv2ParallelEnv) -> ParallelEnvSpec:
    """Translate SMACv2's native environment metadata without resetting the battle."""
    state_shape = env.env_info.get("state_shape")
    global_state_size = None
    if bool(task_config.get("use_native_global_state", False)):
        if state_shape is None:
            raise ValueError("SMACv2 did not report state_shape for the requested native-state path.")
        global_state_size = int(np.prod(state_shape))
    return ParallelEnvSpec(
        obs_size=int(np.prod(env.env_info["obs_shape"])),
        action_space_size=int(env.env_info["n_actions"]),
        max_agents=int(env.env_info["n_agents"]),
        global_state_size=global_state_size,
    )


def _require_action_mask_support(learner_cls: type, algorithm: str) -> None:
    if not bool(getattr(learner_cls, "supports_legal_action_masks", False)):
        raise ValueError(
            f"{algorithm} is not yet enabled for SMACv2: its action selection and update path "
            "must both consume legal-action masks. The guard runs before StarCraft II starts."
        )


def _episode_seed(base_seed: int, offset: int, episode_index: int) -> int:
    return int(base_seed) + int(offset) + int(episode_index)


@contextmanager
def _isolated_evaluation_rng():
    """Prevent stochastic validation actions from changing the training RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _evaluate(
    learner,
    task_config: dict[str, Any],
    *,
    evaluation_seed: int,
    seed_offset: int,
    rollout_count: int,
    phase: str,
    checkpoint_step: int,
) -> tuple[EvalResult, list[dict[str, Any]]]:
    with _isolated_evaluation_rng():
        set_global_seeds(evaluation_seed)
        env = make_env(task_config, evaluation_seed)
        try:
            learner.set_eval_mode()
            returns: list[float] = []
            wins: list[float] = []
            lengths: list[int] = []
            rollout_rows: list[dict[str, Any]] = []

            for rollout_index in range(int(rollout_count)):
                env_seed = _episode_seed(evaluation_seed, seed_offset, rollout_index)
                observations, _ = env.reset(seed=env_seed)
                learner.reset_episode()
                episode_return = 0.0
                final_info: dict[str, Any] = {}

                while True:
                    action_masks = env.action_masks
                    actions = learner.act_parallel(observations, action_mask_dict=action_masks)
                    observations, rewards, terminations, truncations, final_info = env.step(actions)
                    episode_return += float(next(iter(rewards.values())))
                    if all(terminations.values()) or all(truncations.values()):
                        break

                won = float(bool(final_info.get("battle_won", False)))
                episode_length = int(final_info["episode_steps"])
                returns.append(episode_return)
                wins.append(won)
                lengths.append(episode_length)
                rollout_rows.append(
                    {
                        "phase": phase,
                        "checkpoint_step": int(checkpoint_step),
                        "rollout": rollout_index,
                        "episode_seed": env_seed,
                        "return": episode_return,
                        "battle_won": int(won),
                        "episode_length": episode_length,
                        "timed_out": int(bool(final_info.get("episode_limit", False))),
                    }
                )

            result = EvalResult(
                phase=phase,
                checkpoint_step=int(checkpoint_step),
                rollout_count=int(rollout_count),
                win_rate=float(np.mean(wins)),
                return_mean=float(np.mean(returns)),
                return_std=float(np.std(returns)),
                episode_length_mean=float(np.mean(lengths)),
            )
            return result, rollout_rows
        finally:
            env.close()


def _load_checkpoint_copy(learner, checkpoint_path: Path):
    return type(learner).load_checkpoint(
        checkpoint_path,
        env_spec=learner.env_spec,
        config=learner.config,
        device=str(learner.device),
    )


def _array_payload_bytes(value: Any) -> int:
    """Count stored array/tensor payload without estimating Python-object overhead."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_array_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, deque)):
        return sum(_array_payload_bytes(item) for item in value)
    return 0


def _replay_payload_metrics(learner) -> dict[str, float]:
    memory = getattr(learner, "memory", None)
    if memory is None or not hasattr(memory, "__len__"):
        return {}
    replay_episodes = len(memory)
    capacity = getattr(memory, "maxlen", None)
    payload_bytes = _array_payload_bytes(memory)
    projected_bytes = None
    if replay_episodes > 0 and capacity is not None:
        projected_bytes = payload_bytes * int(capacity) / replay_episodes
    metrics = {
        "replay_episodes": float(replay_episodes),
        "replay_payload_mb": float(payload_bytes / (1024.0 * 1024.0)),
    }
    if projected_bytes is not None:
        metrics["projected_full_replay_payload_mb"] = float(
            projected_bytes / (1024.0 * 1024.0)
        )
    return metrics


def _build_summary(
    *,
    task_config: dict[str, Any],
    algorithm: str,
    seed: int,
    global_step: int,
    train_history: list[dict[str, Any]],
    validation_results: list[EvalResult],
    final_evaluation: EvalResult,
    final_evaluation_split: str,
    best_test: EvalResult | None,
    extra_metrics: dict[str, float],
) -> dict[str, Any]:
    train_returns = np.asarray([row["train_return_mean"] for row in train_history], dtype=np.float64)
    tail = train_returns[-min(100, len(train_returns)) :] if train_returns.size else train_returns
    validation_wins = [result.win_rate for result in validation_results]
    best_validation_index = int(np.argmax(validation_wins)) if validation_wins else None

    final_checkpoint = {
        "overall_eval_mean": final_evaluation.win_rate,
        "win_rate": final_evaluation.win_rate,
        "return_mean": final_evaluation.return_mean,
        "return_std": final_evaluation.return_std,
        "episode_length_mean": final_evaluation.episode_length_mean,
    }
    selected_checkpoint = "final_checkpoint"
    objective_score = final_evaluation.win_rate
    objective_details: dict[str, Any] = {}
    if final_evaluation_split == "validation":
        objective_score, return_tiebreak = _validation_objective(task_config, final_evaluation)
        objective_details = {
            "objective_primary_win_rate": final_evaluation.win_rate,
            "objective_return_tiebreak": return_tiebreak,
            "objective_definition": "win_rate + weight * tanh(return_mean / scale)",
            "objective_return_tiebreak_weight": float(
                task_config.get("validation_return_tiebreak_weight", 0.001)
            ),
            "objective_return_scale": float(task_config.get("validation_return_scale", 20.0)),
        }
    selection: dict[str, Any] = {
        "split": final_evaluation_split,
        "selection_checkpoint": selected_checkpoint,
        "objective_score": objective_score,
        "overall_eval_mean": final_evaluation.win_rate,
        "final_checkpoint": final_checkpoint,
        "best_vs_final_drop": 0.0,
        **objective_details,
    }
    if best_test is not None:
        best_checkpoint = {
            "overall_eval_mean": best_test.win_rate,
            "win_rate": best_test.win_rate,
            "return_mean": best_test.return_mean,
            "return_std": best_test.return_std,
            "episode_length_mean": best_test.episode_length_mean,
        }
        selected_checkpoint = "best_checkpoint"
        objective_score = float(0.7 * best_test.win_rate + 0.3 * final_evaluation.win_rate)
        selection.update(
            {
                "selection_checkpoint": selected_checkpoint,
                "objective_score": objective_score,
                "overall_eval_mean": best_test.win_rate,
                "best_checkpoint": best_checkpoint,
                "best_vs_final_drop": float(best_test.win_rate - final_evaluation.win_rate),
            }
        )

    summary = {
        "env_name": str(task_config["env_name"]),
        "algorithm": algorithm,
        "seed": int(seed),
        "training_steps": int(global_step),
        "train": {
            "episodes": len(train_history),
            "final_episode_return": float(train_returns[-1]) if train_returns.size else None,
            "final_moving_average": float(np.mean(tail)) if tail.size else None,
            "win_rate_last_100": float(np.mean([row["battle_won"] for row in train_history[-100:]]))
            if train_history
            else None,
        },
        "validation": {
            "best_validation_mean": validation_wins[best_validation_index]
            if best_validation_index is not None
            else None,
            "best_validation_step": validation_results[best_validation_index].checkpoint_step
            if best_validation_index is not None
            else None,
        },
        "selection": selection,
        "test": selection if final_evaluation_split == "test" else {"status": "not_run"},
        "extra_metrics": dict(sorted(extra_metrics.items())),
    }
    if final_evaluation_split == "validation":
        summary["validation"]["final_checkpoint"] = final_checkpoint
        summary["validation"]["objective_score"] = objective_score
    return summary


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
    """Train one learner and evaluate checkpoints in fresh, fixed-seed SC2 processes."""
    task_path = (
        TASK_DIR / "task.json"
        if task_config_path is None
        else resolve_json_path(task_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    )
    alg_path = resolve_json_path(alg_config_path, base_dir=TASK_DIR, project_root=PROJECT_ROOT)
    task_config = load_json(task_path)
    learner_config = load_json(alg_path)
    final_evaluation_split = str(task_config.get("final_evaluation_split", "test"))
    if final_evaluation_split not in {"validation", "test"}:
        raise ValueError("final_evaluation_split must be either 'validation' or 'test'.")
    if final_evaluation_split == "validation" and int(task_config.get("eval_every_steps", 0)) > 0:
        raise ValueError("Calibration validation requires eval_every_steps=0 to avoid reusing its selection pool.")
    if final_evaluation_split == "validation":
        _validation_objective_settings(task_config)

    learner_cls = get_algorithm_class(algorithm)
    _require_action_mask_support(learner_cls, algorithm)
    set_global_seeds(seed)

    env = make_env(task_config, seed)
    try:
        env_spec = build_env_spec(task_config, env)
        if algorithm == "mipi":
            learner_config = dict(learner_config)
            if learner_config.get("entity_schema") is None:
                learner_config["entity_schema"] = env.mipi_entity_schema()
            central_source = str(
                learner_config.get("central_entity_source", "local_observations")
            )
            if central_source == "native_state":
                if not bool(task_config.get("use_native_global_state", False)):
                    raise ValueError(
                        "MIPI central_entity_source='native_state' requires "
                        "use_native_global_state=true in the SMACv2 task config."
                    )
                if learner_config.get("central_entity_schema") is None:
                    learner_config["central_entity_schema"] = env.mipi_central_entity_schema()
        learner = learner_cls(env_spec=env_spec, config=learner_config, device=resolve_device(device))
        out_dir = Path(
            make_run_dir(
                str(task_config["task_name"]),
                algorithm,
                results_root=results_root,
                run_id=run_id,
            )
        )
        best_ckpt_path = out_dir / "best_checkpoint.pt"
        final_ckpt_path = out_dir / "final_checkpoint.pt"
        validation_ckpt_path = out_dir / ".validation_checkpoint.pt"

        training_steps = int(task_config["training_steps"])
        eval_every_steps = int(task_config.get("eval_every_steps", 0))
        progress_every_steps = int(task_config.get("progress_every_steps", 50_000))
        next_eval_step = eval_every_steps
        next_progress_step = progress_every_steps
        global_step = 0
        episode_index = 0
        learner_updates = 0
        best_win_rate = -float("inf")
        best_checkpoint_step = 0
        train_history: list[dict[str, Any]] = []
        validation_results: list[EvalResult] = []
        rollout_rows: list[dict[str, Any]] = []
        evaluation_seed = int(task_config["evaluation_seed"])

        while global_step < training_steps:
            env_seed = _episode_seed(seed, int(task_config["train_seed_offset"]), episode_index)
            observations, reset_info = env.reset(seed=env_seed)
            learner.reset_episode()
            learner.set_train_mode()
            episode_return = 0.0
            episode_reports: list[dict[str, Any]] = []
            final_info: dict[str, Any] = {}

            while True:
                action_masks = env.action_masks
                roster_masks = env.roster_masks
                decision_masks = env.decision_masks
                global_state = env.global_state
                actions = learner.act_parallel(observations, action_mask_dict=action_masks)
                next_observations, rewards, terminations, truncations, final_info = env.step(actions)
                global_step += 1

                if progress_every_steps > 0 and global_step >= next_progress_step:
                    print(
                        f"step={global_step}/{training_steps} episode={episode_index + 1} "
                        f"episode_step={int(final_info['episode_steps'])}",
                        flush=True,
                    )
                    while next_progress_step <= global_step:
                        next_progress_step += progress_every_steps

                done_dict = {
                    agent_id: bool(terminations[agent_id] or truncations[agent_id])
                    for agent_id in env.possible_agents
                }
                transition = ParallelTransition(
                    obs_dict=observations,
                    action_dict=actions,
                    reward_dict=rewards,
                    next_obs_dict=next_observations,
                    done_dict=done_dict,
                    truncated_dict=truncations,
                    active_agent_mask_dict=roster_masks,
                    next_active_agent_mask_dict=env.roster_masks,
                    decision_agent_mask_dict=decision_masks,
                    next_decision_agent_mask_dict=env.decision_masks,
                    action_mask_dict=action_masks,
                    next_action_mask_dict=env.action_masks,
                    global_state=global_state,
                    next_global_state=env.global_state,
                )
                learner.record_parallel_step(transition)
                report = learner.maybe_update(global_step=global_step, episode_index=episode_index + 1)
                if report is not None:
                    learner_updates += 1
                    episode_reports.append(report.to_flat_dict())

                episode_return += float(next(iter(rewards.values())))
                observations = next_observations
                if all(done_dict.values()):
                    break

            episode_index += 1
            train_history.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "episode_seed": env_seed,
                    "generation_attempt": int(reset_info["generation_attempt"]),
                    "train_return_mean": episode_return,
                    "battle_won": int(bool(final_info.get("battle_won", False))),
                    "timed_out": int(bool(final_info.get("episode_limit", False))),
                    "episode_length": int(final_info["episode_steps"]),
                    "dead_allies": int(final_info.get("dead_allies", 0)),
                    "dead_enemies": int(final_info.get("dead_enemies", 0)),
                    "train_loss_mean": float(np.mean([row["total_loss"] for row in episode_reports]))
                    if episode_reports
                    else 0.0,
                    "temperature": learner_temperature(learner),
                    "learner_updates": learner_updates,
                }
            )

            if eval_every_steps > 0 and global_step >= next_eval_step:
                learner.save_checkpoint(validation_ckpt_path)
                evaluator = _load_checkpoint_copy(learner, validation_ckpt_path)
                validation_result, validation_rollouts = _evaluate(
                    evaluator,
                    task_config,
                    evaluation_seed=evaluation_seed,
                    seed_offset=int(task_config["validation_seed_offset"]),
                    rollout_count=int(task_config["validation_episodes"]),
                    phase="validation",
                    checkpoint_step=global_step,
                )
                validation_results.append(validation_result)
                rollout_rows.extend(validation_rollouts)
                if validation_result.win_rate > best_win_rate + float(task_config.get("min_improve", 0.0)):
                    learner.save_checkpoint(best_ckpt_path)
                    best_win_rate = validation_result.win_rate
                    best_checkpoint_step = global_step
                validation_ckpt_path.unlink(missing_ok=True)
                while next_eval_step <= global_step:
                    next_eval_step += eval_every_steps

        learner.save_checkpoint(final_ckpt_path)
        final_evaluator = _load_checkpoint_copy(learner, final_ckpt_path)
        final_seed_offset = int(task_config[f"{final_evaluation_split}_seed_offset"])
        final_rollout_count = int(task_config[f"{final_evaluation_split}_episodes"])
        final_evaluation, final_rollouts = _evaluate(
            final_evaluator,
            task_config,
            evaluation_seed=evaluation_seed,
            seed_offset=final_seed_offset,
            rollout_count=final_rollout_count,
            phase=f"final_checkpoint_{final_evaluation_split}",
            checkpoint_step=global_step,
        )
        rollout_rows.extend(final_rollouts)

        best_test = None
        if validation_results:
            if not best_ckpt_path.is_file():
                learner.save_checkpoint(best_ckpt_path)
            best_evaluator = _load_checkpoint_copy(learner, best_ckpt_path)
            best_test, best_rollouts = _evaluate(
                best_evaluator,
                task_config,
                evaluation_seed=evaluation_seed,
                seed_offset=int(task_config["test_seed_offset"]),
                rollout_count=int(task_config["test_episodes"]),
                phase="best_checkpoint_test",
                checkpoint_step=best_checkpoint_step,
            )
            rollout_rows.extend(best_rollouts)

        update_history_rows = flatten_update_history(learner.get_update_history())
        extra_metrics = {
            str(key): float(value)
            for key, value in (update_history_rows[-1] if update_history_rows else {}).items()
            if isinstance(value, (int, float)) and value is not None
        }
        extra_metrics.update(_replay_payload_metrics(learner))
        summary = _build_summary(
            task_config=task_config,
            algorithm=algorithm,
            seed=seed,
            global_step=global_step,
            train_history=train_history,
            validation_results=validation_results,
            final_evaluation=final_evaluation,
            final_evaluation_split=final_evaluation_split,
            best_test=best_test,
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
                "env_info": env.env_info,
                "runtime": env.runtime_metadata(),
            },
        )
        write_json(out_dir / "summary.json", summary)
        write_csv(out_dir / "train_history.csv", train_history)
        all_eval_results = [*validation_results, final_evaluation] + ([best_test] if best_test else [])
        write_csv(out_dir / "eval_summary.csv", [asdict(result) for result in all_eval_results])
        write_csv(out_dir / "eval_rollout_returns.csv", rollout_rows)
        save_update_history_json(out_dir / "update_history.json", learner.get_update_history())
        save_update_history_csv(out_dir / "update_history.csv", learner.get_update_history())

        if not skip_gif and bool(task_config.get("save_replay", False)):
            env.save_replay()
        return str(out_dir)
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one procedural SMACv2 experiment.")
    parser.add_argument("--algorithm", choices=ALGORITHM_ORDER, required=True)
    parser.add_argument("--alg-config", type=str, required=True)
    parser.add_argument("--task-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
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
