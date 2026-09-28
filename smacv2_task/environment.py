"""Thin deterministic adapter over the native SMACv2 environment API."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from typing import Any

import numpy as np


def _build_native_environment(task_config: dict[str, Any], seed: int):
    try:
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    except Exception as exc:  # pragma: no cover - exercised only with the optional dependency
        raise ImportError(
            "SMACv2 is required for this task. Install the pinned optional dependency from "
            "the root `requirements.txt`, then install StarCraft II 4.10 plus the SMACv2 maps."
        ) from exc

    env_args = copy.deepcopy(task_config["env_args"])
    env_args["seed"] = int(seed)
    return StarCraftCapabilityEnvWrapper(**env_args)


def _distribution_tree(root: object) -> Iterator[object]:
    """Yield one SMACv2 distribution and any nested distributions once."""
    stack = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current

        children = [
            value
            for _, value in sorted(vars(current).items())
            if hasattr(value, "generate") and callable(value.generate)
        ]
        stack.extend(reversed(children))


class SMACv2ParallelEnv:
    """Expose SMACv2 through the dictionary-based parallel task shape used here.

    SMACv2 keeps a fixed episode roster. Dead units remain in the dictionaries, receive a
    no-op-only legal-action mask, and have a zero decision mask. The wrapper samples the official
    capability distributions with an explicit per-episode seed without perturbing the process-wide
    Python RNG used by learners.
    """

    def __init__(
        self,
        task_config: dict[str, Any],
        seed: int,
        *,
        native_env: object | None = None,
    ):
        self.task_config = copy.deepcopy(task_config)
        self.capture_global_state = bool(task_config.get("use_native_global_state", False))
        self.max_reset_attempts = int(task_config.get("max_reset_attempts", 5))
        if self.max_reset_attempts < 1:
            raise ValueError("max_reset_attempts must be at least one.")
        self._native = native_env if native_env is not None else _build_native_environment(task_config, seed)
        self._raw = getattr(self._native, "env", self._native)
        self.env_info = dict(self._native.get_env_info())

        self.n_agents = int(self.env_info["n_agents"])
        self.n_actions = int(self.env_info["n_actions"])
        self.episode_limit = int(self.env_info["episode_limit"])
        self.possible_agents = tuple(f"agent_{index}" for index in range(self.n_agents))

        self._episode_steps = 0
        self._episode_done = True
        self._observations: dict[str, np.ndarray] = {}
        self._action_masks: dict[str, np.ndarray] = {}
        self._decision_masks: dict[str, float] = {}
        self._state: np.ndarray | None = None

    def mipi_entity_schema(self) -> dict[str, object]:
        """Describe the native flat observation without exposing extra state."""
        move_width = int(self._raw.get_obs_move_feats_size())
        enemy_count, enemy_width = (int(value) for value in self._raw.get_obs_enemy_feats_size())
        ally_count, ally_width = (int(value) for value in self._raw.get_obs_ally_feats_size())
        own_width = int(self._raw.get_obs_own_feats_size())

        move_start = 0
        enemy_start = move_start + move_width
        ally_start = enemy_start + enemy_count * enemy_width
        own_start = ally_start + ally_count * ally_width
        feature_end = own_start + own_width
        obs_size = int(np.prod(self.env_info["obs_shape"]))
        timestep_width = obs_size - feature_end
        if own_width <= 0 or timestep_width not in (0, 1):
            raise ValueError(
                "Cannot map the native SMACv2 observation into MIPI entities: "
                f"obs_size={obs_size}, feature_end={feature_end}, own_width={own_width}."
            )

        groups: list[dict[str, object]] = [
            {
                "name": "self",
                "type": "self",
                "slice": [own_start, feature_end],
                "active_rule": "not_sentinel",
                "inactive_values": [0.0] * own_width,
            }
        ]
        if move_width:
            groups.append(
                {
                    "name": "movement",
                    "type": "movement",
                    "slice": [move_start, enemy_start],
                    "active_rule": "always",
                }
            )
        if enemy_count:
            groups.append(
                {
                    "name": "enemies",
                    "type": "enemy",
                    "slice": [enemy_start, ally_start],
                    "count": enemy_count,
                    "width": enemy_width,
                    "active_rule": "not_sentinel",
                    "inactive_values": [0.0] * enemy_width,
                }
            )
        if ally_count:
            groups.append(
                {
                    "name": "allies",
                    "type": "ally",
                    "slice": [ally_start, own_start],
                    "count": ally_count,
                    "width": ally_width,
                    "active_rule": "not_sentinel",
                    "inactive_values": [0.0] * ally_width,
                }
            )
        if timestep_width:
            groups.append(
                {
                    "name": "timestep",
                    "type": "timestep",
                    "slice": [feature_end, obs_size],
                    "active_rule": "always",
                }
            )
        return {"groups": groups}

    def mipi_central_entity_schema(self) -> dict[str, object]:
        """Describe SMACv2's native ally/enemy state for MIPI's training-only mixer."""
        local_schema = self.mipi_entity_schema()
        num_enemies = int(self._raw.get_obs_enemy_feats_size()[0])
        ally_state_dim = int(self._raw.get_ally_num_attributes())
        enemy_state_dim = int(self._raw.get_enemy_num_attributes())
        state_last_action = bool(self._raw.state_last_action)
        state_timestep = bool(self._raw.state_timestep_number)

        expected_state_size = self.n_agents * ally_state_dim + num_enemies * enemy_state_dim
        if state_last_action:
            expected_state_size += self.n_agents * self.n_actions
        if state_timestep:
            expected_state_size += 1
        reported_state_size = int(np.prod(self.env_info["state_shape"]))
        if expected_state_size != reported_state_size:
            raise ValueError(
                "Native SMACv2 state layout does not match state_shape: "
                f"expected {expected_state_size}, got {reported_state_size}."
            )

        local_to_central: list[list[int]] = []
        for agent_index in range(self.n_agents):
            teammate_indices = [index for index in range(self.n_agents) if index != agent_index]
            mapping: list[int] = []
            for group in local_schema["groups"]:
                group_count = len(group.get("slices", [])) or int(group.get("count", 1))
                group_name = str(group["name"])
                if group_name in {"self", "movement", "timestep"}:
                    mapping.extend([agent_index] * group_count)
                elif group_name == "enemies":
                    if group_count != num_enemies:
                        raise ValueError("MIPI enemy observation slots do not match native state.")
                    mapping.extend(self.n_agents + index for index in range(num_enemies))
                elif group_name == "allies":
                    if group_count != len(teammate_indices):
                        raise ValueError("MIPI ally observation slots do not match native state.")
                    mapping.extend(teammate_indices)
                else:
                    raise ValueError(f"Cannot map MIPI observation group {group_name!r} to native state.")
            local_to_central.append(mapping)

        return {
            "num_allies": self.n_agents,
            "num_enemies": num_enemies,
            "ally_state_dim": ally_state_dim,
            "enemy_state_dim": enemy_state_dim,
            "state_last_action": state_last_action,
            "state_timestep": state_timestep,
            "local_to_central_entity_indices": local_to_central,
        }

    def _rng_owners(self) -> list[object]:
        roots = list(getattr(self._native, "env_key_to_distribution_map", {}).values())
        owners: list[object] = []
        seen: set[int] = set()
        for root in roots:
            for distribution in _distribution_tree(root):
                if id(distribution) in seen:
                    continue
                seen.add(id(distribution))
                if hasattr(distribution, "rng"):
                    owners.append(distribution)
        return owners

    def _sample_episode_config(self, seed: int) -> dict[str, Any]:
        distributions = list(getattr(self._native, "env_key_to_distribution_map", {}).values())
        if not distributions:
            raise RuntimeError("SMACv2 capability wrapper exposes no procedural distributions.")

        seed_sequence = np.random.SeedSequence(int(seed))
        rng_owners = self._rng_owners()
        for owner, child_seed in zip(rng_owners, seed_sequence.spawn(len(rng_owners))):
            owner.rng = np.random.default_rng(child_seed)

        python_rng_state = random.getstate()
        random.seed(int(seed))
        try:
            reset_config: dict[str, Any] = {}
            for distribution in distributions:
                reset_config.update(distribution.generate())
            return reset_config
        finally:
            random.setstate(python_rng_state)

    @staticmethod
    def _retryable_reset_error(exc: Exception) -> bool:
        return type(exc).__name__ == "CannotResetException"

    def _cache_timestep(self, observations: Any, state: Any | None = None) -> None:
        observation_list = list(observations)
        if len(observation_list) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} observations, got {len(observation_list)}.")
        self._observations = {
            agent_id: np.asarray(observation_list[index], dtype=np.float32)
            for index, agent_id in enumerate(self.possible_agents)
        }

        action_masks = list(self._native.get_avail_actions())
        if len(action_masks) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} action masks, got {len(action_masks)}.")
        self._action_masks = {
            agent_id: np.asarray(action_masks[index], dtype=np.float32)
            for index, agent_id in enumerate(self.possible_agents)
        }
        for agent_id, action_mask in self._action_masks.items():
            if action_mask.shape != (self.n_actions,):
                raise ValueError(
                    f"Expected action mask shape {(self.n_actions,)}, got {action_mask.shape} for {agent_id}."
                )
            if not np.any(action_mask > 0):
                raise ValueError(f"Agent {agent_id} has no legal actions.")

        self._decision_masks = {
            agent_id: float(np.any(action_mask[1:] > 0.0))
            for agent_id, action_mask in self._action_masks.items()
        }
        if self.capture_global_state:
            state_value = self._native.get_state() if state is None else state
            self._state = np.asarray(state_value, dtype=np.float32)
        else:
            self._state = None

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset with one reproducible procedural capability sample."""
        episode_seed = int(seed)
        for attempt in range(self.max_reset_attempts):
            sample_seed = int(np.random.SeedSequence([episode_seed, attempt]).generate_state(1)[0])
            reset_config = self._sample_episode_config(sample_seed)
            try:
                observations, state = self._raw.reset(reset_config)
                break
            except Exception as exc:
                if not self._retryable_reset_error(exc) or attempt + 1 >= self.max_reset_attempts:
                    raise
        else:  # pragma: no cover - loop either succeeds or raises
            raise RuntimeError("SMACv2 reset exhausted all attempts.")

        self._episode_steps = 0
        self._episode_done = False
        self._cache_timestep(observations, state)
        return self.observations, {
            "episode_seed": episode_seed,
            "generation_attempt": attempt,
        }

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute one legal joint action and return parallel-style step data."""
        if self._episode_done:
            raise RuntimeError("Call reset() before stepping SMACv2.")
        if set(action_dict) != set(self.possible_agents):
            missing = sorted(set(self.possible_agents) - set(action_dict))
            extra = sorted(set(action_dict) - set(self.possible_agents))
            raise ValueError(f"Joint action must cover the fixed roster; missing={missing}, extra={extra}.")

        native_actions: list[int] = []
        for agent_id in self.possible_agents:
            action = int(action_dict[agent_id])
            if action < 0 or action >= self.n_actions or self._action_masks[agent_id][action] <= 0:
                raise ValueError(f"Illegal SMACv2 action {action} for {agent_id}.")
            native_actions.append(action)

        timeouts_before = getattr(self._raw, "timeouts", None)
        team_reward, native_done, raw_info = self._native.step(native_actions)
        timeouts_after = getattr(self._raw, "timeouts", None)
        self._episode_steps += 1
        info = dict(raw_info)

        native_timeout = None
        if isinstance(timeouts_before, (int, np.integer)) and isinstance(timeouts_after, (int, np.integer)):
            native_timeout = int(timeouts_after) > int(timeouts_before)
        timeout_fallback = self._episode_steps >= self.episode_limit and not bool(info.get("battle_won", False))
        timeout = bool(
            native_done
            and (
                bool(info.get("episode_limit", False))
                or (native_timeout if native_timeout is not None else timeout_fallback)
            )
        )
        terminal = bool(native_done and not timeout)

        next_observations = self._native.get_obs()
        next_state = self._native.get_state() if self.capture_global_state else None
        self._cache_timestep(next_observations, next_state)
        self._episode_done = bool(native_done)

        rewards = {agent_id: float(team_reward) for agent_id in self.possible_agents}
        terminations = {agent_id: terminal for agent_id in self.possible_agents}
        truncations = {agent_id: timeout for agent_id in self.possible_agents}
        info.update(
            {
                "episode_steps": self._episode_steps,
                "episode_limit": timeout,
            }
        )
        return self.observations, rewards, terminations, truncations, info

    @property
    def observations(self) -> dict[str, np.ndarray]:
        return dict(self._observations)

    @property
    def action_masks(self) -> dict[str, np.ndarray]:
        return dict(self._action_masks)

    @property
    def roster_masks(self) -> dict[str, float]:
        return {agent_id: 1.0 for agent_id in self.possible_agents}

    @property
    def decision_masks(self) -> dict[str, float]:
        return dict(self._decision_masks)

    @property
    def global_state(self) -> np.ndarray | None:
        return self._state

    def runtime_metadata(self) -> dict[str, object]:
        """Return the SC2 build selected by PySC2 after the first reset."""
        version = getattr(self._raw, "version", None)
        if version is None:
            return {}
        return {
            "game_version": str(version.game_version),
            "base_build": int(version.build_version),
            "data_version": str(version.data_version),
        }

    def save_replay(self) -> None:
        self._native.save_replay()

    def close(self) -> None:
        self._native.close()
