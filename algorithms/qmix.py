"""QMIX with the shared parallel benchmark API."""

from __future__ import annotations

from collections import deque
import copy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

__all__ = ["AgentRNN", "MixingNetwork", "QMIX", "QMIX_DEFAULT_CONFIG"]


QMIX_DEFAULT_CONFIG = {
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_steps": 50000,
    "buffer_size": 5000,
    "batch_size": 32,
    "lr": 5e-4,
    "rnn_hidden_dim": 64,
    "mixing_embed_dim": 32,
    "hypernet_embed": 64,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "target_update_every": 200,
    "double_q": True,
    "rmsprop_alpha": 0.99,
    "rmsprop_eps": 1e-5,
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


class AgentRNN(nn.Module):
    """Per-agent recurrent Q-network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
    ):
        super().__init__()
        self.input_layer = nn.Linear(int(obs_dim), int(rnn_hidden_dim))
        self.rnn = nn.GRU(input_size=int(rnn_hidden_dim), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.input_layer(x))

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        batch_size, num_timesteps, obs_dim = obs_seq.shape
        encoded = self._encode(obs_seq.reshape(batch_size * num_timesteps, obs_dim)).reshape(batch_size, num_timesteps, -1)
        recurrent_features, next_hidden_state = self.rnn(encoded, h0)
        q_values = self.out(recurrent_features)
        return q_values, next_hidden_state


class MixingNetwork(nn.Module):
    """State-conditioned monotonic mixer used by QMIX."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int,
        hypernet_embed: int,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, int(hypernet_embed)),
            nn.ReLU(),
            nn.Linear(int(hypernet_embed), self.num_agents * self.mixing_embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, int(hypernet_embed)),
            nn.ReLU(),
            nn.Linear(int(hypernet_embed), self.mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.state_value = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.state_value(states).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size)


class QMIX(ParallelLearner):
    """QMIX with the shared benchmark API."""

    # -------------------------------------------------------------------------
    # Config normalization
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, QMIX_DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # Constructor and state
    # -------------------------------------------------------------------------
    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        config = self.config

        self.epsilon_start = float(config["epsilon_start"])
        self.epsilon_finish = float(config["epsilon_finish"])
        self.epsilon_anneal_steps = max(1, int(config["epsilon_anneal_steps"]))
        if not 0.0 <= self.epsilon_finish <= self.epsilon_start <= 1.0:
            raise ValueError("QMIX epsilon must satisfy 0 <= epsilon_finish <= epsilon_start <= 1.")
        self._action_steps = 0
        self.batch_size = int(config["batch_size"])
        self.gamma = float(config["gamma"])
        self.target_update_every = max(1, int(config["target_update_every"]))
        self.double_q = bool(config["double_q"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self._learn_steps = 0
        self._completed_episodes = 0
        self._last_consumed_episode = 0
        self._last_target_update_episode = 0
        self.agent_input_size = self.obs_size + self.action_space_size

        self.agent_net = AgentRNN(
            self.agent_input_size,
            self.action_space_size,
            int(config["rnn_hidden_dim"]),
        ).to(self.device)
        self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
        self.target_agent_net.eval()

        self.mixing_net = MixingNetwork(
            num_agents=self.max_agents,
            state_dim=self.env_spec.centralized_state_size,
            mixing_embed_dim=int(config["mixing_embed_dim"]),
            hypernet_embed=int(config["hypernet_embed"]),
        ).to(self.device)
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)
        self.target_mixing_net.eval()

        self.optimizer = optim.RMSprop(
            list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
            lr=float(config["lr"]),
            alpha=float(config["rmsprop_alpha"]),
            eps=float(config["rmsprop_eps"]),
        )
        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._episode_steps: list[dict] = []
        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._last_actions: dict[object, int] = {}

    # -------------------------------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._last_actions = {}

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self.agent_net.eval()
        self.target_agent_net.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.agent_net.train()
        self.mixing_net.train()

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def _get_hidden_state(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        hidden_state = self._inference_hidden.get(agent_key)
        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return hidden_state

    def _set_hidden_state(self, agent_key: object, hidden_state: torch.Tensor) -> None:
        self._inference_hidden[agent_key] = hidden_state.detach()

    def _epsilon(self) -> float:
        progress = min(1.0, max(0.0, float(self._action_steps) / float(self.epsilon_anneal_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_finish - self.epsilon_start))

    def _epsilon_greedy_action(self, q_values: torch.Tensor) -> int:
        if self._eval_mode:
            return int(torch.argmax(q_values).item())
        if random.random() < self._epsilon():
            return random.randrange(self.action_space_size)
        return int(torch.argmax(q_values).item())

    def _actor_input(self, obs: np.ndarray, agent_key: object) -> np.ndarray:
        previous_action = np.zeros(self.action_space_size, dtype=np.float32)
        if agent_key in self._last_actions:
            previous_action[self._last_actions[agent_key]] = 1.0
        return np.concatenate((np.asarray(obs, dtype=np.float32), previous_action))

    def _act_one(self, obs: np.ndarray, agent_key: object) -> int:
        obs_tensor = torch.as_tensor(self._actor_input(obs, agent_key), device=self.device).view(1, 1, -1)
        hidden_dim = self.agent_net.rnn.hidden_size
        q_seq, hidden_state = self.agent_net(obs_tensor, self._get_hidden_state(agent_key, hidden_dim))
        self._set_hidden_state(agent_key, hidden_state)
        q_values = q_seq.squeeze(0).squeeze(0)

        return self._epsilon_greedy_action(q_values)

    def act(self, state: np.ndarray, agent_index: Optional[object] = None) -> int:
        if agent_index is None:
            agent_index = 0
        action = self._act_one(state, agent_index)
        self._last_actions[agent_index] = action
        return action

    def act_parallel(self, obs_dict: dict[object, np.ndarray]) -> dict[object, int]:
        actions_by_agent_id: dict[object, int] = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            action = self._act_one(obs_dict[agent_id], agent_id)
            actions_by_agent_id[agent_id] = action
            self._last_actions[agent_id] = action
        if not self._eval_mode:
            self._action_steps += 1
        return actions_by_agent_id

    # -------------------------------------------------------------------------
    # Transition recording
    # -------------------------------------------------------------------------
    def _derive_state(self, obs_batch: np.ndarray) -> np.ndarray:
        return obs_batch.reshape(-1).astype(np.float32, copy=False)

    def record_parallel_step(self, transition: ParallelTransition) -> None:
        done_agent_ids = {agent_id for agent_id in transition.done_dict.keys() if agent_id != "__all__"}
        truncated_agent_ids = {
            agent_id
            for agent_id in (transition.truncated_dict or {}).keys()
            if agent_id != "__all__"
        }
        agent_ids = _sorted_agent_ids(
            set(transition.obs_dict.keys())
            | set(transition.action_dict.keys())
            | set(transition.reward_dict.keys())
            | set(transition.next_obs_dict.keys())
            | done_agent_ids
            | truncated_agent_ids
        )
        obs_batch = np.zeros((self.max_agents, self.obs_size), dtype=np.float32)
        next_obs_batch = np.zeros((self.max_agents, self.obs_size), dtype=np.float32)
        actions_batch = np.zeros(self.max_agents, dtype=np.int64)
        rewards_batch = np.zeros(self.max_agents, dtype=np.float32)
        active_mask = np.zeros(self.max_agents, dtype=np.float32)
        next_active_mask = np.zeros(self.max_agents, dtype=np.float32)

        for agent_index, agent_id in enumerate(agent_ids):
            if agent_index >= self.max_agents:
                break
            if agent_id in transition.obs_dict:
                obs_batch[agent_index] = np.asarray(transition.obs_dict[agent_id], dtype=np.float32)
            if agent_id in transition.next_obs_dict:
                next_obs_batch[agent_index] = np.asarray(transition.next_obs_dict[agent_id], dtype=np.float32)
            actions_batch[agent_index] = int(transition.action_dict.get(agent_id, 0))
            rewards_batch[agent_index] = float(transition.reward_dict.get(agent_id, 0.0))
            active_mask[agent_index] = (
                float(transition.active_agent_mask_dict.get(agent_id, 0.0))
                if transition.active_agent_mask_dict is not None
                else (1.0 if agent_id in transition.obs_dict else 0.0)
            )
            next_active_mask[agent_index] = (
                float(transition.next_active_agent_mask_dict.get(agent_id, 0.0))
                if transition.next_active_agent_mask_dict is not None
                else (
                    1.0
                    if agent_id in transition.next_obs_dict
                    and not (
                        resolve_agent_done(transition.done_dict, agent_id)
                        and not resolve_agent_done(transition.truncated_dict, agent_id)
                    )
                    else 0.0
                )
            )

        state = self._derive_state(obs_batch) if transition.global_state is None else np.asarray(transition.global_state, dtype=np.float32)
        next_state = self._derive_state(next_obs_batch) if transition.next_global_state is None else np.asarray(transition.next_global_state, dtype=np.float32)
        episode_finished = resolve_parallel_done(transition.done_dict)
        bootstrap_terminal = episode_finished and not resolve_parallel_done(transition.truncated_dict)

        self._episode_steps.append(
            {
                "obs": obs_batch,
                "actions": actions_batch,
                "rewards": rewards_batch,
                "active_mask": active_mask,
                "state": state,
                "next_obs": next_obs_batch,
                "next_active_mask": next_active_mask,
                "next_state": next_state,
                "done": bootstrap_terminal,
            }
        )
        if episode_finished:
            self.memory.append(self._finalize_episode(self._episode_steps))
            self._episode_steps = []
            self._completed_episodes += 1

    def _finalize_episode(self, steps: list[dict]) -> dict:
        episode = {
            "obs": np.stack([step["obs"] for step in steps], axis=0),
            "actions": np.stack([step["actions"] for step in steps], axis=0),
            "rewards": np.stack([step["rewards"] for step in steps], axis=0),
            "active_mask": np.stack([step["active_mask"] for step in steps], axis=0),
            "state": np.stack([step["state"] for step in steps], axis=0),
            "next_obs": np.stack([step["next_obs"] for step in steps], axis=0),
            "next_active_mask": np.stack([step["next_active_mask"] for step in steps], axis=0),
            "next_state": np.stack([step["next_state"] for step in steps], axis=0),
            "done": np.asarray([step["done"] for step in steps], dtype=np.float32),
            "T": int(len(steps)),
        }
        return episode

    # -------------------------------------------------------------------------
    # Update scheduling and learning
    # -------------------------------------------------------------------------
    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        if self._completed_episodes <= self._last_consumed_episode:
            return None
        self._last_consumed_episode = self._completed_episodes
        if len(self.memory) < self.batch_size:
            return None
        return self._run_update(global_step=global_step, episode_index=episode_index)

    def _update_targets(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def _agent_q_values(self, obs: torch.Tensor, network: AgentRNN) -> torch.Tensor:
        batch_size, num_timesteps, num_agents, obs_dim = obs.shape
        obs_batch = obs.permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_timesteps, obs_dim)
        q_values, _ = network(obs_batch, None)
        return q_values.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)

    def _next_agent_q_values(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        network: AgentRNN,
    ) -> torch.Tensor:
        """Unroll next-state values with the recurrent history that precedes them."""
        aligned_obs = torch.cat((obs[:, :1], next_obs), dim=1)
        return self._agent_q_values(aligned_obs, network)[:, 1:]

    def _training_agent_inputs(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append each agent's locally available previous action to recurrent inputs."""
        safe_actions = actions.clone()
        safe_actions[active_mask == 0] = 0
        action_one_hot = F.one_hot(safe_actions.long(), num_classes=self.action_space_size).to(obs.dtype)
        action_one_hot = action_one_hot * active_mask.unsqueeze(-1)
        previous_actions = torch.zeros_like(action_one_hot)
        previous_actions[:, 1:] = action_one_hot[:, :-1]
        return (
            torch.cat((obs, previous_actions), dim=-1),
            torch.cat((next_obs, action_one_hot), dim=-1),
        )

    def _mix_q_tot(
        self,
        mixer: MixingNetwork,
        chosen_q: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_timesteps, num_agents = chosen_q.shape
        q_tot = mixer(
            chosen_q.reshape(batch_size * num_timesteps, num_agents),
            states.reshape(batch_size * num_timesteps, -1),
        )
        return q_tot.reshape(batch_size, num_timesteps)

    def _run_update(self, global_step: int, episode_index: int) -> UpdateReport:
        memory_items = len(self.memory)
        batch = random.sample(self.memory, self.batch_size)
        max_t = max(int(episode["T"]) for episode in batch)

        def pad_time(array, pad_value=0.0):
            num_timesteps = array.shape[0]
            if num_timesteps == max_t:
                return array
            pad_shape = (max_t - num_timesteps,) + array.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=array.dtype)
            return np.concatenate([array, padding], axis=0)

        obs = torch.as_tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
        actions = torch.as_tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
        rewards = torch.as_tensor(np.stack([pad_time(ep["rewards"]) for ep in batch]), device=self.device)
        active_mask = torch.as_tensor(np.stack([pad_time(ep["active_mask"]) for ep in batch]), device=self.device)
        states = torch.as_tensor(np.stack([pad_time(ep["state"]) for ep in batch]), device=self.device)
        next_obs = torch.as_tensor(np.stack([pad_time(ep["next_obs"]) for ep in batch]), device=self.device)
        next_active_mask = torch.as_tensor(np.stack([pad_time(ep["next_active_mask"]) for ep in batch]), device=self.device)
        next_states = torch.as_tensor(np.stack([pad_time(ep["next_state"]) for ep in batch]), device=self.device)
        dones = torch.as_tensor(
            np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        ).squeeze(-1)

        lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
        time_mask = (
            torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(dtype=torch.float32)
        mask_count = time_mask.sum().clamp(min=1.0)

        safe_actions = actions.clone()
        safe_actions[active_mask == 0] = 0
        agent_inputs, next_agent_inputs = self._training_agent_inputs(
            obs,
            next_obs,
            safe_actions,
            active_mask,
        )

        q_all = self._agent_q_values(agent_inputs, self.agent_net)
        next_q_online = self._next_agent_q_values(
            agent_inputs,
            next_agent_inputs,
            self.agent_net,
        )
        with torch.no_grad():
            next_q_target = self._next_agent_q_values(
                agent_inputs,
                next_agent_inputs,
                self.target_agent_net,
            )

        chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1) * active_mask
        q_tot = self._mix_q_tot(self.mixing_net, chosen_q, states)

        active_counts = active_mask.sum(dim=2).clamp(min=1.0)
        team_rewards = (rewards * active_mask).sum(dim=2) / active_counts

        with torch.no_grad():
            if self.double_q:
                next_actions = torch.argmax(next_q_online, dim=-1)
            else:
                next_actions = torch.argmax(next_q_target, dim=-1)
            safe_next_actions = next_actions.clone()
            safe_next_actions[next_active_mask == 0] = 0
            next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
            next_chosen_q = next_chosen_q * next_active_mask
            q_tot_next = self._mix_q_tot(self.target_mixing_net, next_chosen_q, next_states)
            targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next

        td_error = q_tot - targets
        td_loss = F.mse_loss(q_tot, targets, reduction="none")
        loss = (td_loss * time_mask).sum() / mask_count

        parameters = list(self.agent_net.parameters()) + list(self.mixing_net.parameters())
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = _parameter_grad_norm(parameters)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, max_norm=self.max_grad_norm)
        self.optimizer.step()
        self._learn_steps += 1

        if (self._completed_episodes - self._last_target_update_episode) >= self.target_update_every:
            self._update_targets()
            self._last_target_update_episode = self._completed_episodes

        report = UpdateReport(
            update_index=len(self._update_reports) + 1,
            episode_index=int(episode_index),
            global_step=int(global_step),
            total_loss=float(loss.detach().item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm=float(grad_norm),
            buffer_items=int(memory_items),
            batch_items=int(self.batch_size),
            samples_seen=int(time_mask.sum().item() * self.max_agents),
            extras={
                "td_loss": float(loss.detach().item()),
                "q_mean": float((q_tot * time_mask).sum().item() / mask_count.item()),
                "target_mean": float((targets * time_mask).sum().item() / mask_count.item()),
                "td_error_abs": float((td_error.abs() * time_mask).sum().item() / mask_count.item()),
                "epsilon": float(self._epsilon()),
            },
        )
        return self._append_update_report(report)

    # -------------------------------------------------------------------------
    # Diagnostics and checkpoint IO
    # -------------------------------------------------------------------------
    def _checkpoint_state(self) -> dict:
        checkpoint_state = {
            "epsilon_action_steps": int(self._action_steps),
            "learn_steps": int(self._learn_steps),
            "completed_episodes": int(self._completed_episodes),
            "last_consumed_episode": int(self._last_consumed_episode),
            "last_target_update_episode": int(self._last_target_update_episode),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "mixing_state_dict": self.mixing_net.state_dict(),
            "target_mixing_state_dict": self.target_mixing_net.state_dict(),
        }
        checkpoint_state["agent_state_dict"] = self.agent_net.state_dict()
        checkpoint_state["target_agent_state_dict"] = self.target_agent_net.state_dict()
        return checkpoint_state

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self.agent_net.load_state_dict(checkpoint_state["agent_state_dict"])
        self.target_agent_net.load_state_dict(checkpoint_state["target_agent_state_dict"])
        self.mixing_net.load_state_dict(checkpoint_state["mixing_state_dict"])
        self.target_mixing_net.load_state_dict(checkpoint_state["target_mixing_state_dict"])
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self._action_steps = int(checkpoint_state.get("epsilon_action_steps", self._action_steps))
        self._learn_steps = int(checkpoint_state.get("learn_steps", self._learn_steps))
        self._completed_episodes = int(checkpoint_state.get("completed_episodes", self._completed_episodes))
        self._last_consumed_episode = int(
            checkpoint_state.get("last_consumed_episode", self._last_consumed_episode)
        )
        self._last_target_update_episode = int(
            checkpoint_state.get("last_target_update_episode", self._last_target_update_episode)
        )
