"""Shared parallel-learning interfaces for the benchmark algorithms.

The goal of this module is simple:
- one environment spec shape,
- one transition shape,
- one update-report shape,
- one small abstract base class.

It is intentionally lightweight. The algorithm files should stay readable on their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class ParallelEnvSpec:
    """Environment facts that every learner needs at construction time.

    `max_agents` remains part of the shared benchmark interface because some
    learners and task/tooling paths still need a configured team-size bound.
    Current PIMAC variants do not use it as a fixed architectural width in
    their actor/critic/update path; they batch with dynamic per-minibatch
    padding instead.
    """

    obs_size: int
    action_space_size: int
    max_agents: int

    @property
    def centralized_state_size(self) -> int:
        return int(self.obs_size) * int(self.max_agents)


@dataclass(frozen=True)
class ParallelTransition:
    """One full parallel-environment step.

    `active_agent_mask_dict` marks which known agents are present in the current step.
    `next_active_agent_mask_dict` does the same for the next-step roster.
    These are team-membership masks for padded joint tensors, not legal-action masks.
    """

    obs_dict: dict[object, Any]
    action_dict: dict[object, int]
    reward_dict: dict[object, float]
    next_obs_dict: dict[object, Any]
    done_dict: dict[object, bool]
    active_agent_mask_dict: Optional[dict[object, Any]] = None
    next_active_agent_mask_dict: Optional[dict[object, Any]] = None
    global_state: Optional[Any] = None
    next_global_state: Optional[Any] = None


@dataclass(frozen=True)
class UpdateReport:
    """One optimizer update summary with shared core metrics and algorithm extras."""

    update_index: int
    episode_index: int
    global_step: int
    total_loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    buffer_items: Optional[int] = None
    batch_items: Optional[int] = None
    samples_seen: Optional[int] = None
    exploration_temperature: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float | int | None]:
        flat_record = asdict(self)
        extras = flat_record.pop("extras", {})
        flat_record.update(extras)
        return flat_record


def normalize_config(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge a user config into defaults while rejecting unknown keys."""

    unknown_keys = sorted(set(config.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    merged = dict(defaults)
    merged.update(config)
    return merged


def resolve_parallel_done(done_dict: dict[object, Any] | None) -> bool:
    """Collapse parallel-env done data to one episode-terminal boolean."""

    if not done_dict:
        return False
    if "__all__" in done_dict:
        return bool(done_dict["__all__"])
    return all(bool(flag) for flag in done_dict.values())


def resolve_agent_done(done_dict: dict[object, Any] | None, agent_id: object) -> bool:
    """Return one agent's terminal flag while honoring the parallel `__all__` key."""

    if not done_dict:
        return False
    if "__all__" in done_dict and bool(done_dict["__all__"]):
        return True
    return bool(done_dict.get(agent_id, False))


class ParallelLearner(ABC):
    """Small shared base for all benchmark algorithms."""

    def __init__(self, env_spec: ParallelEnvSpec, config: dict[str, Any], device: str = "cpu"):
        self.env_spec = env_spec
        self.config = dict(config)
        self.device = torch.device(device)
        self.obs_size = int(env_spec.obs_size)
        self.action_space_size = int(env_spec.action_space_size)
        self.max_agents = int(env_spec.max_agents)
        self._eval_mode = False
        self._update_reports: list[UpdateReport] = []

    @abstractmethod
    def reset_episode(self) -> None:
        pass

    @abstractmethod
    def set_train_mode(self) -> None:
        pass

    @abstractmethod
    def set_eval_mode(self) -> None:
        pass

    @abstractmethod
    def act_parallel(self, obs_dict: dict[object, Any]) -> dict[object, int]:
        pass

    @abstractmethod
    def record_parallel_step(self, transition: ParallelTransition) -> None:
        pass

    @abstractmethod
    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        pass

    @abstractmethod
    def _checkpoint_state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _load_checkpoint_state(self, checkpoint_state: dict[str, Any]) -> None:
        pass

    def get_update_history(self) -> list[UpdateReport]:
        return list(self._update_reports)

    def _append_update_report(self, report: UpdateReport) -> UpdateReport:
        self._update_reports.append(report)
        return report

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), target)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        env_spec: ParallelEnvSpec,
        config: dict[str, Any],
        device: str = "cpu",
    ):
        learner = cls(env_spec=env_spec, config=config, device=device)
        checkpoint_state = torch.load(Path(path), map_location="cpu")
        learner._load_checkpoint_state(checkpoint_state)
        return learner
