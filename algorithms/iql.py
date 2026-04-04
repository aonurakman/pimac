"""IQL with the shared parallel benchmark API."""

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Optional, Sequence

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
)

__all__ = ["RecurrentNetwork", "IQL", "IQL_DEFAULT_CONFIG"]


IQL_DEFAULT_CONFIG = {
    "temp_init": 1.0,
    "temp_decay": 0.999,
    "temp_min": 0.05,
    "buffer_size": 256,
    "batch_size": 16,
    "lr": 0.003,
    "num_epochs": 1,
    "num_hidden": 2,
    "widths": (32, 64, 32),
    "rnn_hidden_dim": 32,
    "seq_len": 8,
    "gamma": 0.99,
    "target_update_every": 100,
    "double_dqn": True,
    "tau": 1.0,
    "max_grad_norm": 10.0,
    "learning_starts": 0,
    "learn_every_steps": 1,
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


class RecurrentNetwork(nn.Module):
    """DRQN-style Q-network: MLP encoder + GRU + linear head."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "IQL widths and number of layers mismatch."
        self.input_layer = nn.Linear(int(in_size), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.out_layer = nn.Linear(int(rnn_hidden_dim), int(out_size))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: Optional[torch.Tensor] = None):
        batch_size, num_timesteps, obs_dim = obs_seq.shape
        encoded = self._encode(obs_seq.reshape(batch_size * num_timesteps, obs_dim)).reshape(batch_size, num_timesteps, -1)
        recurrent_features, next_hidden_state = self.rnn(encoded, h0)
        q_values = self.out_layer(recurrent_features)
        return q_values, next_hidden_state


class IQL(ParallelLearner):
    """Independent recurrent Q-learning with shared parameters."""

    # -------------------------------------------------------------------------
    # Config normalization
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, IQL_DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # Constructor and state
    # -------------------------------------------------------------------------
    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        config = self.config

        self.temperature = float(config["temp_init"])
        self.temp_decay = float(config["temp_decay"])
        self.temp_min = float(config["temp_min"])
        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.seq_len = max(1, int(config["seq_len"]))
        self.gamma = float(config["gamma"])
        self.target_update_every = max(1, int(config["target_update_every"]))
        self.double_dqn = bool(config["double_dqn"])
        self.tau = float(config["tau"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self.learning_starts = int(config["learning_starts"])
        self.learn_every_steps = int(config["learn_every_steps"])
        self._learn_steps = 0

        rnn_hidden_dim = (
            int(config["rnn_hidden_dim"]) if int(config["rnn_hidden_dim"]) > 0 else int(config["widths"][-1])
        )
        self.q_network = RecurrentNetwork(
            self.obs_size,
            self.action_space_size,
            int(config["num_hidden"]),
            tuple(config["widths"]),
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)
        self.target_q_network = RecurrentNetwork(
            self.obs_size,
            self.action_space_size,
            int(config["num_hidden"]),
            tuple(config["widths"]),
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(config["lr"]))
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._pending_step: dict[object, tuple[np.ndarray, int]] = {}
        self._agent_episode_steps: dict[object, list[tuple[np.ndarray, int, float, np.ndarray, bool]]] = defaultdict(list)

    # -------------------------------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._pending_step = {}
        self._agent_episode_steps = defaultdict(list)

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self.q_network.eval()
        self.target_q_network.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.q_network.train()
        self.target_q_network.eval()

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def _coerce_obs(self, obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_array.shape[0] == self.obs_size:
            return obs_array
        padded = np.zeros(self.obs_size, dtype=np.float32)
        copy_width = min(self.obs_size, obs_array.shape[0])
        if copy_width > 0:
            padded[:copy_width] = obs_array[:copy_width]
        return padded

    def _get_hidden_state(self, agent_id: object) -> Optional[torch.Tensor]:
        return self._inference_hidden.get(agent_id)

    def _set_hidden_state(self, agent_id: object, hidden_state: torch.Tensor) -> None:
        self._inference_hidden[agent_id] = hidden_state.detach()

    def _boltzmann_action(self, q_values: torch.Tensor) -> int:
        if self._eval_mode:
            return int(torch.argmax(q_values).item())
        temp = float(self.temperature)
        if temp <= 0.0:
            return int(torch.argmax(q_values).item())
        logits = q_values / temp
        logits = logits - torch.max(logits)
        distribution = torch.distributions.Categorical(logits=logits)
        return int(distribution.sample().item())

    def _act_one(self, agent_id: object, obs: np.ndarray) -> int:
        obs_array = self._coerce_obs(obs)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).view(1, 1, -1)
            q_values_seq, next_hidden_state = self.q_network(obs_tensor, self._get_hidden_state(agent_id))
            self._set_hidden_state(agent_id, next_hidden_state)
            q_values = q_values_seq[:, -1, :].squeeze(0).squeeze(0)
        action = self._boltzmann_action(q_values)
        self._pending_step[agent_id] = (obs_array, int(action))
        return int(action)

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
        for agent_id, pending_step in list(self._pending_step.items()):
            state, action = pending_step
            reward = float(transition.reward_dict.get(agent_id, 0.0))
            next_obs_raw = transition.next_obs_dict.get(agent_id)
            next_obs = (
                np.zeros(self.obs_size, dtype=np.float32)
                if next_obs_raw is None
                else self._coerce_obs(next_obs_raw)
            )
            done = resolve_agent_done(transition.done_dict, agent_id)
            self._agent_episode_steps[agent_id].append((state, action, reward, next_obs, done))
            if done:
                self.memory.append(list(self._agent_episode_steps[agent_id]))
                self._agent_episode_steps[agent_id] = []
                self._inference_hidden.pop(agent_id, None)
            self._pending_step.pop(agent_id, None)

    # -------------------------------------------------------------------------
    # Update scheduling and learning
    # -------------------------------------------------------------------------
    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        del episode_index
        if global_step < self.learning_starts:
            return None
        if (global_step % self.learn_every_steps) != 0:
            return None
        if len(self.memory) < self.batch_size:
            return None
        return self._run_update(global_step=global_step)

    def _update_target_network(self) -> None:
        if self.tau >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            return
        with torch.no_grad():
            for target_param, q_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * q_param.data)

    def _run_update(self, global_step: int) -> UpdateReport:
        memory_items = len(self.memory)
        td_losses: list[float] = []
        q_means: list[float] = []
        target_means: list[float] = []
        td_error_abs_means: list[float] = []
        grad_norms: list[float] = []
        total_samples_seen = 0

        for _ in range(self.num_epochs):
            episodes = random.sample(self.memory, self.batch_size)

            chunks = []
            max_num_timesteps = 1
            for episode in episodes:
                if not episode:
                    continue
                start_index = 0 if len(episode) <= 1 else random.randint(0, len(episode) - 1)
                end_index = min(len(episode), start_index + self.seq_len)
                chunk = episode[start_index:end_index]
                chunks.append(chunk)
                max_num_timesteps = max(max_num_timesteps, len(chunk))

            if not chunks:
                continue

            batch_size = len(chunks)
            obs = np.zeros((batch_size, max_num_timesteps, self.obs_size), dtype=np.float32)
            next_obs = np.zeros((batch_size, max_num_timesteps, self.obs_size), dtype=np.float32)
            actions = np.zeros((batch_size, max_num_timesteps), dtype=np.int64)
            rewards = np.zeros((batch_size, max_num_timesteps), dtype=np.float32)
            dones = np.ones((batch_size, max_num_timesteps), dtype=np.float32)
            time_mask = np.zeros((batch_size, max_num_timesteps), dtype=np.float32)

            for batch_index, chunk in enumerate(chunks):
                for timestep_index, (state, action, reward, next_state, done) in enumerate(chunk):
                    obs[batch_index, timestep_index] = state
                    next_obs[batch_index, timestep_index] = next_state
                    actions[batch_index, timestep_index] = int(action)
                    rewards[batch_index, timestep_index] = float(reward)
                    dones[batch_index, timestep_index] = 1.0 if bool(done) else 0.0
                    time_mask[batch_index, timestep_index] = 1.0

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(time_mask, dtype=torch.float32, device=self.device)
            total_samples_seen += int(mask_tensor.sum().item())

            q_values_seq, _ = self.q_network(obs_tensor, None)
            chosen_q = torch.gather(q_values_seq, dim=2, index=actions_tensor.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_q_online, _ = self.q_network(next_obs_tensor, None)
                next_q_target, _ = self.target_q_network(next_obs_tensor, None)
                if self.double_dqn:
                    next_actions = torch.argmax(next_q_online, dim=2, keepdim=True)
                    next_q = torch.gather(next_q_target, dim=2, index=next_actions).squeeze(-1)
                else:
                    next_q = torch.max(next_q_target, dim=2).values
                targets = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q

            td_error = chosen_q - targets
            td_loss = self.loss_fn(chosen_q, targets)
            loss = (td_loss * mask_tensor).sum() / mask_tensor.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = _parameter_grad_norm(self.q_network.parameters())
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_target_network()

            td_losses.append(float(loss.detach().item()))
            q_means.append(float((chosen_q * mask_tensor).sum().item() / mask_tensor.sum().clamp(min=1.0).item()))
            target_means.append(float((targets * mask_tensor).sum().item() / mask_tensor.sum().clamp(min=1.0).item()))
            td_error_abs_means.append(
                float((td_error.abs() * mask_tensor).sum().item() / mask_tensor.sum().clamp(min=1.0).item())
            )
            grad_norms.append(float(grad_norm))

        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

        report = UpdateReport(
            update_index=len(self._update_reports) + 1,
            episode_index=-1,
            global_step=int(global_step),
            total_loss=float(np.mean(td_losses)) if td_losses else 0.0,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
            buffer_items=int(memory_items),
            batch_items=int(self.batch_size),
            samples_seen=int(total_samples_seen),
            exploration_temperature=float(self.temperature),
            extras={
                "td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
                "q_mean": float(np.mean(q_means)) if q_means else 0.0,
                "target_mean": float(np.mean(target_means)) if target_means else 0.0,
                "td_error_abs": float(np.mean(td_error_abs_means)) if td_error_abs_means else 0.0,
            },
        )
        return self._append_update_report(report)

    # -------------------------------------------------------------------------
    # Diagnostics and checkpoint IO
    # -------------------------------------------------------------------------
    def _checkpoint_state(self) -> dict:
        return {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "temperature": float(self.temperature),
            "learn_steps": int(self._learn_steps),
        }

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self.q_network.load_state_dict(checkpoint_state["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint_state["target_q_network_state_dict"])
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.temperature = float(checkpoint_state.get("temperature", self.temperature))
        self._learn_steps = int(checkpoint_state.get("learn_steps", self._learn_steps))
