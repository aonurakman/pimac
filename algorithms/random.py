"""Uniform random benchmark policy.

This is treated like the other algorithms so that every task script can run it through the same
public API. It does not learn, but it still exposes checkpoints and update history for symmetry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms.base import ParallelEnvSpec, ParallelLearner, ParallelTransition, UpdateReport, normalize_config

__all__ = ["RandomPolicy", "RANDOM_DEFAULT_CONFIG"]


RANDOM_DEFAULT_CONFIG = {
    "seed": 0,
}


class RandomPolicy(ParallelLearner):
    """A no-learning policy that samples uniformly from the action space."""

    supports_legal_action_masks = True

    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, RANDOM_DEFAULT_CONFIG)

    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        self.seed = int(self.config["seed"])
        self._rng = np.random.default_rng(self.seed)

    def reset_episode(self) -> None:
        return None

    def set_train_mode(self) -> None:
        self._eval_mode = False

    def set_eval_mode(self) -> None:
        self._eval_mode = True

    def act_parallel(
        self,
        obs_dict: dict[object, np.ndarray],
        action_mask_dict: Optional[dict[object, np.ndarray]] = None,
    ) -> dict[object, int]:
        if action_mask_dict is not None:
            actions: dict[object, int] = {}
            for agent_id in sorted(obs_dict.keys(), key=str):
                legal_actions = np.flatnonzero(np.asarray(action_mask_dict[agent_id], dtype=np.int8))
                if legal_actions.size == 0:
                    raise ValueError(f"Agent {agent_id!r} has no legal actions.")
                actions[agent_id] = int(self._rng.choice(legal_actions))
            return actions
        return {
            agent_id: int(self._rng.integers(0, self.action_space_size))
            for agent_id in sorted(obs_dict.keys(), key=str)
        }

    def record_parallel_step(self, transition: ParallelTransition) -> None:
        del transition

    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        del global_step, episode_index
        return None

    def _checkpoint_state(self) -> dict:
        return {
            "config": dict(self.config),
            "rng_state": self._rng.bit_generator.state,
        }

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = dict(checkpoint_state["rng_state"])
