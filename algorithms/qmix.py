"""QMIX with the shared parallel benchmark API."""

from __future__ import annotations

from collections import deque
import copy
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
    resolve_parallel_done,
)

__all__ = ["AgentRNN", "MixingNetwork", "QMIX", "QMIX_DEFAULT_CONFIG"]


QMIX_DEFAULT_CONFIG = {
    "temp_init": 1.0,
    "temp_decay": 0.999,
    "temp_min": 0.05,
    "buffer_size": 2048,
    "batch_size": 32,
    "lr": 3e-4,
    "num_epochs": 1,
    "num_hidden": 2,
    "widths": (128, 128, 128),
    "rnn_hidden_dim": 64,
    "mixing_embed_dim": 32,
    "hypernet_embed": 64,
    "mixing_num_hidden": None,
    "mixing_widths": None,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "target_update_every": 200,
    "double_q": True,
    "tau": 1.0,
    "share_parameters": True,
    "mixing_weight_clip": None,
    "q_tot_clip": None,
    "use_huber_loss": True,
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


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(in_dim)
    for hidden_dim in hidden_sizes:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(out_dim)))
    return nn.Sequential(*layers)


class AgentRNN(nn.Module):
    """Per-agent recurrent Q-network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        num_hidden: int,
        widths: Sequence[int],
    ):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "QMIX widths and number of layers mismatch."
        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[idx]), int(widths[idx + 1])) for idx in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)
        self.out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

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
        hypernet_hidden_sizes: Sequence[int],
        weight_clip: Optional[float] = None,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)
        self.weight_clip = float(weight_clip) if weight_clip is not None else None

        self.hyper_w1 = _build_mlp(self.state_dim, hypernet_hidden_sizes, self.num_agents * self.mixing_embed_dim)
        self.hyper_b1 = _build_mlp(self.state_dim, hypernet_hidden_sizes, self.mixing_embed_dim)
        self.hyper_w2 = _build_mlp(self.state_dim, hypernet_hidden_sizes, self.mixing_embed_dim)
        self.hyper_b2 = _build_mlp(self.state_dim, hypernet_hidden_sizes, 1)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        w1 = F.softplus(self.hyper_w1(states))
        if self.weight_clip is not None:
            w1 = torch.clamp(w1, max=self.weight_clip)
        b1 = self.hyper_b1(states)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)

        w2 = F.softplus(self.hyper_w2(states))
        if self.weight_clip is not None:
            w2 = torch.clamp(w2, max=self.weight_clip)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

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

        self.temperature = float(config["temp_init"])
        self.temp_decay = float(config["temp_decay"])
        self.temp_min = float(config["temp_min"])
        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.gamma = float(config["gamma"])
        self.target_update_every = max(1, int(config["target_update_every"]))
        self.double_q = bool(config["double_q"])
        self.tau = float(config["tau"])
        self.share_parameters = bool(config["share_parameters"])
        self.q_tot_clip = float(config["q_tot_clip"]) if config["q_tot_clip"] is not None else None
        self.use_huber_loss = bool(config["use_huber_loss"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self.learning_starts = int(config["learning_starts"])
        self.learn_every_steps = int(config["learn_every_steps"])
        self._learn_steps = 0

        if config["mixing_widths"] is not None:
            mixing_hidden_sizes = tuple(int(value) for value in config["mixing_widths"])
            if config["mixing_num_hidden"] is not None and len(mixing_hidden_sizes) != int(config["mixing_num_hidden"]):
                raise ValueError("QMIX mixing_widths and mixing_num_hidden mismatch.")
        else:
            mixing_hidden_sizes = (int(config["hypernet_embed"]),)

        if self.share_parameters:
            self.agent_net = AgentRNN(
                self.obs_size,
                self.action_space_size,
                int(config["rnn_hidden_dim"]),
                int(config["num_hidden"]),
                tuple(config["widths"]),
            ).to(self.device)
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.agent_nets = None
            self.target_agent_nets = None
            self.target_agent_net.eval()
        else:
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList(
                [
                    AgentRNN(
                        self.obs_size,
                        self.action_space_size,
                        int(config["rnn_hidden_dim"]),
                        int(config["num_hidden"]),
                        tuple(config["widths"]),
                    ).to(self.device)
                    for _ in range(self.max_agents)
                ]
            )
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)
            self.target_agent_nets.eval()

        self.mixing_net = MixingNetwork(
            num_agents=self.max_agents,
            state_dim=self.env_spec.centralized_state_size,
            mixing_embed_dim=int(config["mixing_embed_dim"]),
            hypernet_hidden_sizes=mixing_hidden_sizes,
            weight_clip=config["mixing_weight_clip"],
        ).to(self.device)
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)
        self.target_mixing_net.eval()

        self.optimizer = optim.Adam(
            list(self._agent_parameters()) + list(self.mixing_net.parameters()),
            lr=float(config["lr"]),
        )
        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._episode_steps: list[dict] = []
        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._agent_slot_map: dict[object, int] = {}

    # -------------------------------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._agent_slot_map = {}

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.mixing_net.train()

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def _agent_parameters(self):
        if self.share_parameters:
            return self.agent_net.parameters()
        return self.agent_nets.parameters()

    def _get_hidden_state(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        hidden_state = self._inference_hidden.get(agent_key)
        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return hidden_state

    def _set_hidden_state(self, agent_key: object, hidden_state: torch.Tensor) -> None:
        self._inference_hidden[agent_key] = hidden_state.detach()

    def _actor_key(self, agent_id: object) -> object:
        if self.share_parameters:
            return agent_id
        slot = self._agent_slot_map.get(agent_id)
        if slot is None:
            slot = len(self._agent_slot_map)
            if slot >= self.max_agents:
                raise ValueError(f"Received more than {self.max_agents} agent ids in one episode.")
            self._agent_slot_map[agent_id] = slot
        return slot

    def _boltzmann_action(self, q_values: torch.Tensor) -> int:
        if self._eval_mode:
            return int(torch.argmax(q_values).item())
        if self.temperature <= 0.0:
            return int(torch.argmax(q_values).item())
        logits = q_values / float(self.temperature)
        logits = logits - torch.max(logits)
        distribution = torch.distributions.Categorical(logits=logits)
        return int(distribution.sample().item())

    def _act_one(self, obs: np.ndarray, agent_key: object) -> int:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).view(1, 1, -1)
        if self.share_parameters:
            hidden_dim = self.agent_net.rnn.hidden_size
            q_seq, hidden_state = self.agent_net(obs_tensor, self._get_hidden_state(agent_key, hidden_dim))
        else:
            network = self.agent_nets[int(agent_key)]
            hidden_dim = network.rnn.hidden_size
            q_seq, hidden_state = network(obs_tensor, self._get_hidden_state(agent_key, hidden_dim))
        self._set_hidden_state(agent_key, hidden_state)
        q_values = q_seq.squeeze(0).squeeze(0)

        return self._boltzmann_action(q_values)

    def act(self, state: np.ndarray, agent_index: Optional[object] = None) -> int:
        if agent_index is None:
            agent_index = 0
        return self._act_one(state, agent_index)

    def act_parallel(self, obs_dict: dict[object, np.ndarray]) -> dict[object, int]:
        actions_by_agent_id: dict[object, int] = {}
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            actions_by_agent_id[agent_id] = self._act_one(
                obs_dict[agent_id],
                self._actor_key(agent_id),
            )
        return actions_by_agent_id

    # -------------------------------------------------------------------------
    # Transition recording
    # -------------------------------------------------------------------------
    def _derive_state(self, obs_batch: np.ndarray) -> np.ndarray:
        return obs_batch.reshape(-1).astype(np.float32, copy=False)

    def record_parallel_step(self, transition: ParallelTransition) -> None:
        done_agent_ids = {agent_id for agent_id in transition.done_dict.keys() if agent_id != "__all__"}
        agent_ids = _sorted_agent_ids(
            set(transition.obs_dict.keys())
            | set(transition.action_dict.keys())
            | set(transition.reward_dict.keys())
            | set(transition.next_obs_dict.keys())
            | done_agent_ids
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
                    if agent_id in transition.next_obs_dict and not bool(transition.done_dict.get(agent_id, False))
                    else 0.0
                )
            )

        state = self._derive_state(obs_batch) if transition.global_state is None else np.asarray(transition.global_state, dtype=np.float32)
        next_state = self._derive_state(next_obs_batch) if transition.next_global_state is None else np.asarray(transition.next_global_state, dtype=np.float32)

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
                "done": resolve_parallel_done(transition.done_dict),
            }
        )
        if self._episode_steps[-1]["done"]:
            self.memory.append(self._finalize_episode(self._episode_steps))
            self._episode_steps = []

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
        del episode_index
        if global_step < self.learning_starts:
            return None
        if (global_step % self.learn_every_steps) != 0:
            return None
        if len(self.memory) < self.batch_size:
            return None
        return self._run_update(global_step=global_step)

    def _update_targets(self) -> None:
        if self.tau >= 1.0:
            if self.share_parameters:
                self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            else:
                self.target_agent_nets.load_state_dict(self.agent_nets.state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
            return
        with torch.no_grad():
            if self.share_parameters:
                for target_param, online_param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)
            else:
                for target_param, online_param in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)
            for target_param, online_param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def _agent_q_values(self, obs: torch.Tensor, networks, share: bool) -> torch.Tensor:
        batch_size, num_timesteps, num_agents, obs_dim = obs.shape
        if share:
            obs_batch = obs.permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_timesteps, obs_dim)
            q_values, _ = networks(obs_batch, None)
            return q_values.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)
        outputs = []
        for agent_index in range(num_agents):
            q_values, _ = networks[agent_index](obs[:, :, agent_index, :], None)
            outputs.append(q_values.unsqueeze(2))
        return torch.cat(outputs, dim=2)

    def _mix_q_tot(self, chosen_q: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size, num_timesteps, num_agents = chosen_q.shape
        q_tot = self.mixing_net(chosen_q.reshape(batch_size * num_timesteps, num_agents), states.reshape(batch_size * num_timesteps, -1))
        return q_tot.reshape(batch_size, num_timesteps)

    def _run_update(self, global_step: int) -> UpdateReport:
        memory_items = len(self.memory)
        td_losses: list[float] = []
        q_means: list[float] = []
        target_means: list[float] = []
        td_error_abs_means: list[float] = []
        grad_norms: list[float] = []
        total_samples_seen = 0

        for _ in range(self.num_epochs):
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
            total_samples_seen += int(time_mask.sum().item() * self.max_agents)

            if self.share_parameters:
                q_all = self._agent_q_values(obs, self.agent_net, share=True)
                next_q_online = self._agent_q_values(next_obs, self.agent_net, share=True)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_net, share=True)
            else:
                q_all = self._agent_q_values(obs, self.agent_nets, share=False)
                next_q_online = self._agent_q_values(next_obs, self.agent_nets, share=False)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_nets, share=False)

            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1) * active_mask
            q_tot = self._mix_q_tot(chosen_q, states)
            if self.q_tot_clip is not None:
                q_tot = torch.clamp(q_tot, -self.q_tot_clip, self.q_tot_clip)

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
                q_tot_next = self._mix_q_tot(next_chosen_q, next_states)
                if self.q_tot_clip is not None:
                    q_tot_next = torch.clamp(q_tot_next, -self.q_tot_clip, self.q_tot_clip)
                targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next
                if self.q_tot_clip is not None:
                    targets = torch.clamp(targets, -self.q_tot_clip, self.q_tot_clip)

            td_error = q_tot - targets
            if self.use_huber_loss:
                td_loss = F.smooth_l1_loss(q_tot, targets, reduction="none")
            else:
                td_loss = F.mse_loss(q_tot, targets, reduction="none")
            loss = (td_loss * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = _parameter_grad_norm(list(self._agent_parameters()) + list(self.mixing_net.parameters()))
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self._agent_parameters()) + list(self.mixing_net.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.optimizer.step()

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

            td_losses.append(float(loss.detach().item()))
            q_means.append(float((q_tot * time_mask).sum().item() / time_mask.sum().clamp(min=1.0).item()))
            target_means.append(float((targets * time_mask).sum().item() / time_mask.sum().clamp(min=1.0).item()))
            td_error_abs_means.append(float((td_error.abs() * time_mask).sum().item() / time_mask.sum().clamp(min=1.0).item()))
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
        checkpoint_state = {
            "share_parameters": bool(self.share_parameters),
            "temperature": float(self.temperature),
            "learn_steps": int(self._learn_steps),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "mixing_state_dict": self.mixing_net.state_dict(),
            "target_mixing_state_dict": self.target_mixing_net.state_dict(),
        }
        if self.share_parameters:
            checkpoint_state["agent_state_dict"] = self.agent_net.state_dict()
            checkpoint_state["target_agent_state_dict"] = self.target_agent_net.state_dict()
        else:
            checkpoint_state["agent_state_dicts"] = [network.state_dict() for network in self.agent_nets]
            checkpoint_state["target_agent_state_dicts"] = [network.state_dict() for network in self.target_agent_nets]
        return checkpoint_state

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        if self.share_parameters:
            self.agent_net.load_state_dict(checkpoint_state["agent_state_dict"])
            self.target_agent_net.load_state_dict(checkpoint_state["target_agent_state_dict"])
        else:
            for network, state_dict in zip(self.agent_nets, checkpoint_state["agent_state_dicts"]):
                network.load_state_dict(state_dict)
            for network, state_dict in zip(self.target_agent_nets, checkpoint_state["target_agent_state_dicts"]):
                network.load_state_dict(state_dict)
        self.mixing_net.load_state_dict(checkpoint_state["mixing_state_dict"])
        self.target_mixing_net.load_state_dict(checkpoint_state["target_mixing_state_dict"])
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.temperature = float(checkpoint_state.get("temperature", self.temperature))
        self._learn_steps = int(checkpoint_state.get("learn_steps", self._learn_steps))
