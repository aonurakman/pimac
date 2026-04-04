"""IPPO with the shared parallel benchmark API."""

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base import (
    ParallelEnvSpec,
    ParallelLearner,
    ParallelTransition,
    UpdateReport,
    normalize_config,
    resolve_agent_done,
    resolve_parallel_done,
)

__all__ = ["ActorCriticRNN", "IPPO", "IPPO_DEFAULT_CONFIG"]


IPPO_DEFAULT_CONFIG = {
    "batch_size": 16,
    "lr": 3e-4,
    "num_epochs": 4,
    "num_hidden": 2,
    "widths": (64, 64, 64),
    "rnn_hidden_dim": 64,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_advantage": True,
    "entropy_coef": 0.1,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "buffer_size": 2048,
    "update_every_episodes": 1,
}


def _sorted_agent_ids(keys) -> list[object]:
    return sorted(list(keys), key=lambda agent_identifier: str(agent_identifier))


def _parameter_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


class ActorCriticRNN(nn.Module):
    """Shared recurrent actor-critic used by all agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "IPPO widths and number of layers mismatch."
        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.policy_head = nn.Linear(int(rnn_hidden_dim), int(action_dim))
        self.value_head = nn.Linear(int(rnn_hidden_dim), 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        batch_size, num_timesteps, obs_dim = obs_seq.shape
        encoded = self._encode(obs_seq.reshape(batch_size * num_timesteps, obs_dim)).reshape(batch_size, num_timesteps, -1)
        recurrent_features, next_hidden_state = self.rnn(encoded, h0)
        logits = self.policy_head(recurrent_features)
        values = self.value_head(recurrent_features).squeeze(-1)
        return logits, values, next_hidden_state


class IPPO(ParallelLearner):
    """Independent PPO with shared parameters and a parallel-only public API."""

    # -------------------------------------------------------------------------
    # Config normalization
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, IPPO_DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # Constructor and state
    # -------------------------------------------------------------------------
    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        config = self.config

        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.clip_eps = float(config["clip_eps"])
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.normalize_advantage = bool(config["normalize_advantage"])
        self.entropy_coef = float(config["entropy_coef"])
        self.value_coef = float(config["value_coef"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self.update_every_episodes = int(config["update_every_episodes"])

        self.actor_critic_net = ActorCriticRNN(
            obs_dim=self.obs_size,
            action_dim=self.action_space_size,
            num_hidden=int(config["num_hidden"]),
            widths=tuple(config["widths"]),
            rnn_hidden_dim=int(config["rnn_hidden_dim"]),
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(), lr=float(config["lr"]))

        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._agent_steps: dict[object, list[dict]] = defaultdict(list)
        self._episode_finished = False

    # -------------------------------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._agent_steps = defaultdict(list)
        self._episode_finished = False

    def set_eval_mode(self) -> None:
        """Switch network modules to evaluation mode without changing action sampling."""
        self._eval_mode = True
        self.actor_critic_net.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.actor_critic_net.train()

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def _get_hidden_state(self, agent_id: object) -> torch.Tensor:
        hidden_state = self._inference_hidden.get(agent_id)
        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, self.actor_critic_net.rnn.hidden_size, device=self.device)
        return hidden_state

    def _set_hidden_state(self, agent_id: object, hidden_state: torch.Tensor) -> None:
        self._inference_hidden[agent_id] = hidden_state.detach()

    def _act_one(self, agent_id: object, obs: np.ndarray) -> int:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).view(1, 1, -1)

        logits, values, next_hidden_state = self.actor_critic_net(
            obs_tensor,
            self._get_hidden_state(agent_id),
        )
        self._set_hidden_state(agent_id, next_hidden_state)

        logits_t = logits.squeeze(0).squeeze(0)
        value_t = values.squeeze(0).squeeze(0)

        distribution = torch.distributions.Categorical(logits=logits_t)
        action = int(distribution.sample().item())
        log_prob = float(distribution.log_prob(torch.tensor(action, device=logits_t.device)).item())

        self._agent_steps[agent_id].append(
            {
                "obs": obs_array,
                "action": action,
                "log_prob": log_prob,
                "value": float(value_t.item()),
            }
        )
        return action

    def act(self, state: np.ndarray, agent_index: object = 0) -> int:
        return self._act_one(agent_index, state)

    def act_parallel(self, obs_dict: dict[object, np.ndarray]) -> dict[object, int]:
        actions: dict[object, int] = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            actions[agent_id] = self._act_one(agent_id, obs_dict[agent_id])
        return actions

    # -------------------------------------------------------------------------
    # Transition recording
    # -------------------------------------------------------------------------
    def record_parallel_step(self, transition: ParallelTransition) -> None:
        reward_dict = transition.reward_dict
        done_dict = transition.done_dict
        self._episode_finished = resolve_parallel_done(done_dict)

        for agent_id, steps in list(self._agent_steps.items()):
            if not steps:
                continue
            step = steps[-1]
            step["reward"] = float(reward_dict.get(agent_id, 0.0))
            step["done"] = resolve_agent_done(done_dict, agent_id)
            if step["done"]:
                self.memory.append(self._finalize_episode(steps))
                self._agent_steps[agent_id] = []

    def _finalize_episode(self, steps: list[dict]) -> dict:
        obs = np.stack([step["obs"] for step in steps], axis=0).astype(np.float32, copy=False)
        actions = np.asarray([step["action"] for step in steps], dtype=np.int64)
        rewards = np.asarray([step["reward"] for step in steps], dtype=np.float32)
        dones = np.asarray([step["done"] for step in steps], dtype=np.float32)
        old_log_probs = np.asarray([step["log_prob"] for step in steps], dtype=np.float32)
        values = np.asarray([step["value"] for step in steps], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        next_value = 0.0
        for timestep_index in range(rewards.shape[0] - 1, -1, -1):
            non_terminal = 1.0 - dones[timestep_index]
            delta = rewards[timestep_index] + self.gamma * non_terminal * next_value - values[timestep_index]
            last_advantage = delta + self.gamma * self.gae_lambda * non_terminal * last_advantage
            advantages[timestep_index] = last_advantage
            next_value = values[timestep_index]
        returns = advantages + values
        return {
            "obs": obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "T": int(obs.shape[0]),
        }

    # -------------------------------------------------------------------------
    # Update scheduling and learning
    # -------------------------------------------------------------------------
    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        if not self._episode_finished:
            return None
        if (episode_index % self.update_every_episodes) != 0:
            return None
        if len(self.memory) < self.batch_size:
            return None
        report = self._run_update(global_step=global_step, episode_index=episode_index)
        self._episode_finished = False
        return report

    def _run_update(self, global_step: int, episode_index: int) -> UpdateReport:
        memory_items = len(self.memory)
        mean_policy_losses: list[float] = []
        mean_value_losses: list[float] = []
        mean_entropies: list[float] = []
        mean_total_losses: list[float] = []
        mean_approx_kls: list[float] = []
        mean_clip_fracs: list[float] = []
        mean_explained_variances: list[float] = []
        mean_grad_norms: list[float] = []
        total_samples_seen = 0

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            max_num_timesteps = max(int(episode["T"]) for episode in batch)

            def pad_time(array: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
                if array.shape[0] == max_num_timesteps:
                    return array
                pad_shape = (max_num_timesteps - array.shape[0],) + array.shape[1:]
                padding = np.full(pad_shape, pad_value, dtype=array.dtype)
                return np.concatenate([array, padding], axis=0)

            obs = torch.as_tensor(np.stack([pad_time(episode["obs"]) for episode in batch]), device=self.device)
            actions = torch.as_tensor(np.stack([pad_time(episode["actions"]) for episode in batch]), device=self.device)
            old_log_probs = torch.as_tensor(
                np.stack([pad_time(episode["old_log_probs"]) for episode in batch]),
                device=self.device,
            )
            advantages = torch.as_tensor(
                np.stack([pad_time(episode["advantages"]) for episode in batch]),
                device=self.device,
            )
            returns = torch.as_tensor(
                np.stack([pad_time(episode["returns"]) for episode in batch]),
                device=self.device,
            )

            lengths = torch.tensor([int(episode["T"]) for episode in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_num_timesteps, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)
            total_samples_seen += int(lengths.sum().item())

            if self.normalize_advantage:
                valid_advantages = advantages[time_mask.bool()]
                advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)

            logits, values, _ = self.actor_critic_net(obs, None)
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions.long())
            entropy = distribution.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            decision_denom = time_mask.sum().clamp(min=1.0)

            policy_loss = -(torch.min(unclipped, clipped) * time_mask).sum() / decision_denom
            value_loss = (((returns - values) ** 2) * time_mask).sum() / decision_denom
            entropy_bonus = (entropy * time_mask).sum() / decision_denom
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            with torch.no_grad():
                approx_kl = (((old_log_probs - new_log_probs) * time_mask).sum() / decision_denom).item()
                clip_frac = ((((ratio - 1.0).abs() > self.clip_eps).to(dtype=torch.float32) * time_mask).sum() / decision_denom).item()
                valid_returns = returns[time_mask.bool()]
                valid_values = values.detach()[time_mask.bool()]
                if valid_returns.numel() > 1:
                    returns_variance = torch.var(valid_returns, unbiased=False)
                    if float(returns_variance.item()) > 1e-8:
                        explained_variance = (
                            1.0
                            - torch.var(valid_returns - valid_values, unbiased=False) / (returns_variance + 1e-8)
                        ).item()
                    else:
                        explained_variance = 0.0
                else:
                    explained_variance = 0.0

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = _parameter_grad_norm(self.actor_critic_net.parameters())
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            mean_policy_losses.append(float(policy_loss.detach().item()))
            mean_value_losses.append(float(value_loss.detach().item()))
            mean_entropies.append(float(entropy_bonus.detach().item()))
            mean_total_losses.append(float(total_loss.detach().item()))
            mean_approx_kls.append(float(approx_kl))
            mean_clip_fracs.append(float(clip_frac))
            mean_explained_variances.append(float(explained_variance))
            mean_grad_norms.append(float(grad_norm))

        self.memory.clear()

        report = UpdateReport(
            update_index=len(self._update_reports) + 1,
            episode_index=int(episode_index),
            global_step=int(global_step),
            total_loss=float(np.mean(mean_total_losses)),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm=float(np.mean(mean_grad_norms)) if mean_grad_norms else None,
            buffer_items=int(memory_items),
            batch_items=int(self.batch_size),
            samples_seen=int(total_samples_seen),
            exploration_temperature=None,
            extras={
                "policy_loss": float(np.mean(mean_policy_losses)),
                "value_loss": float(np.mean(mean_value_losses)),
                "entropy": float(np.mean(mean_entropies)),
                "approx_kl": float(np.mean(mean_approx_kls)),
                "clip_frac": float(np.mean(mean_clip_fracs)),
                "explained_variance": float(np.mean(mean_explained_variances)),
            },
        )
        return self._append_update_report(report)

    # -------------------------------------------------------------------------
    # Diagnostics and checkpoint IO
    # -------------------------------------------------------------------------
    def _checkpoint_state(self) -> dict:
        return {
            "actor_critic_state_dict": self.actor_critic_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self.actor_critic_net.load_state_dict(checkpoint_state["actor_critic_state_dict"])
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
