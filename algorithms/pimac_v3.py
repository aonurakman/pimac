"""
PIMACV3 implementation for parallel multi-agent environments.

PIMACV3 extends `pimac_v2` with a context-conditioned policy head:
- A token-based set teacher produces per-agent cooperation context through
  masked cross-attention.
- The centralized critic consumes teacher tokens for team-value estimation.
- The decentralized actor predicts student context (`mu`, `logvar`) from local
  recurrent features and uses it to generate a gated low-rank residual over the
  policy head parameters.
- Student context is trained by heteroscedastic distillation against teacher
  context, while PPO remains on-policy.

This keeps canonical MAPPO optimization semantics (PPO clip + GAE + entropy),
while adding centralized context supervision that is not available at inference.
"""

from __future__ import annotations

from collections import deque
import copy
import math
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
    resolve_parallel_done,
)

__all__ = [
    "PIMACV3ActorRNN",
    "SetTokenTeacher",
    "SetValueCritic",
    "PIMACV3",
    "PIMACV3_DEFAULT_CONFIG",
]


PIMACV3_DEFAULT_CONFIG = {
    "buffer_size": 2048,
    "batch_size": 32,
    "lr": 3e-4,
    "num_epochs": 4,
    "num_hidden": 2,
    "widths": (64, 64, 64),
    "rnn_hidden_dim": 64,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_advantage": True,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "critic_hidden_sizes": (128, 128),
    "set_embed_dim": 128,
    "set_encoder_hidden_sizes": (128, 128),
    "include_team_size_feature": True,
    "num_tokens": 4,
    "distill_weight": 0.1,
    "teacher_ema_tau": 0.01,
    "hypernet_rank": 4,
    "hypernet_hidden_sizes": (128, 128),
    "hypernet_delta_init_scale": 0.05,
    "hypernet_l2_coef": 1e-4,
    "ctx_logvar_min": -6.0,
    "ctx_logvar_max": 4.0,
    "update_every_episodes": 1,
}


def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build a feed-forward ReLU MLP."""
    layers: list[nn.Module] = []
    current_dim = int(in_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(out_dim)))
    return nn.Sequential(*layers)


def _sorted_agent_ids(keys) -> list:
    """Return deterministic ordering for mixed-type agent-id dictionaries."""
    return sorted(list(keys), key=lambda agent_identifier: str(agent_identifier))


def _parameter_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


class PIMACV3ActorRNN(nn.Module):
    """
    Shared recurrent actor with student context heads.

    The actor predicts:
    - action logits,
    - context mean (`ctx_mu`) and log-variance (`ctx_logvar`) for distillation.

    Policy conditioning follows uncertainty-gated head adaptation:
    - A scalar gate from student uncertainty scales the adaptation strength.
    - A low-rank hypernetwork predicts residual policy-head deltas from context.
    - The recurrent policy features are passed directly into the adapted head.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_hidden: int,
        widths: Sequence[int],
        rnn_hidden_dim: int,
        ctx_dim: int,
        hypernet_rank: int = 4,
        hypernet_hidden_sizes: Sequence[int] = (128, 128),
        hypernet_delta_init_scale: float = 0.05,
        ctx_logvar_min: float = -6.0,
        ctx_logvar_max: float = 4.0,
    ):
        super().__init__()
        assert len(widths) == (int(num_hidden) + 1), "PIMACV3 actor widths and layer count mismatch."
        if int(hypernet_rank) <= 0:
            raise ValueError("PIMACV3 hypernet_rank must be a positive integer.")

        self.input_layer = nn.Linear(int(obs_dim), int(widths[0]))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(int(widths[layer_index]), int(widths[layer_index + 1]))
            for layer_index in range(int(num_hidden))
        )
        self.rnn = nn.GRU(input_size=int(widths[-1]), hidden_size=int(rnn_hidden_dim), batch_first=True)

        self.hidden_dim = int(rnn_hidden_dim)
        self.action_dim = int(action_dim)
        self.hypernet_rank = int(hypernet_rank)
        self.hypernet_delta_init_scale = float(hypernet_delta_init_scale)

        self.ctx_mu_head = nn.Linear(self.hidden_dim, int(ctx_dim))
        self.ctx_logvar_head = nn.Linear(self.hidden_dim, int(ctx_dim))
        self.gate_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Learned base policy head; hypernetwork generates a context-conditioned residual.
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)
        low_rank_weight_width = self.hidden_dim * self.hypernet_rank
        low_rank_action_width = self.hypernet_rank * self.action_dim
        self._hypernet_weight_out = low_rank_weight_width
        self._hypernet_action_out = low_rank_action_width
        hypernet_output_dim = low_rank_weight_width + low_rank_action_width + self.action_dim
        self.hypernet_delta_head = _build_mlp(
            in_dim=int(ctx_dim),
            hidden_sizes=tuple(hypernet_hidden_sizes),
            out_dim=hypernet_output_dim,
        )

        self.ctx_logvar_min = float(ctx_logvar_min)
        self.ctx_logvar_max = float(ctx_logvar_max)

    def _encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode flattened observations with the actor MLP stack."""
        encoded_features = torch.relu(self.input_layer(input_features))
        for hidden_layer in self.hidden_layers:
            encoded_features = torch.relu(hidden_layer(encoded_features))
        return encoded_features

    def forward(
        self,
        obs_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Args:
            obs_seq: Observation sequence with shape [B, T, obs_dim].
            h0: Optional GRU initial state [1, B, H].
            return_aux: When true, also return student context tensors.

        Returns:
            logits: [B, T, action_dim]
            hn: [1, B, H]
            aux (optional):
                - ctx_mu: [B, T, ctx_dim]
                - ctx_logvar: [B, T, ctx_dim]
                - gate: [B, T, 1]
                - delta_w_l2: [B, T]
                - delta_b_l2: [B, T]
                - delta_w_norm: [B, T]
                - delta_b_norm: [B, T]
        """
        batch_size, num_timesteps, observation_dim = obs_seq.shape
        encoded_sequence = self._encode(obs_seq.reshape(batch_size * num_timesteps, observation_dim)).reshape(
            batch_size,
            num_timesteps,
            -1,
        )
        recurrent_features, next_hidden_state = self.rnn(encoded_sequence, h0)

        context_mean = self.ctx_mu_head(recurrent_features)
        context_log_variance = torch.clamp(
            self.ctx_logvar_head(recurrent_features),
            min=self.ctx_logvar_min,
            max=self.ctx_logvar_max,
        )

        # Higher uncertainty (larger log-variance) reduces modulation strength.
        mean_log_variance = context_log_variance.mean(dim=-1, keepdim=True)
        modulation_gate = torch.sigmoid(self.gate_weight * (-mean_log_variance) + self.gate_bias)

        # Hypernetwork predicts low-rank policy-head residual factors from context.
        hypernet_deltas = self.hypernet_delta_head(context_mean) * self.hypernet_delta_init_scale
        weight_factor_flat = hypernet_deltas[..., : self._hypernet_weight_out]
        action_factor_flat = hypernet_deltas[..., self._hypernet_weight_out : self._hypernet_weight_out + self._hypernet_action_out]
        delta_bias = hypernet_deltas[..., self._hypernet_weight_out + self._hypernet_action_out :]

        weight_factor = weight_factor_flat.reshape(batch_size, num_timesteps, self.hidden_dim, self.hypernet_rank)
        action_factor = action_factor_flat.reshape(batch_size, num_timesteps, self.hypernet_rank, self.action_dim)
        delta_weight = torch.einsum("bthr,btra->btha", weight_factor, action_factor)

        gate_for_params = modulation_gate.unsqueeze(-1)
        base_weight = self.policy_head.weight.t().contiguous().view(1, 1, self.hidden_dim, self.action_dim)
        base_bias = self.policy_head.bias.view(1, 1, self.action_dim)
        effective_weight = base_weight + gate_for_params * delta_weight
        effective_bias = base_bias + modulation_gate * delta_bias

        action_logits = torch.einsum("bth,btha->bta", recurrent_features, effective_weight) + effective_bias

        delta_w_l2 = delta_weight.pow(2).sum(dim=(-1, -2))
        delta_b_l2 = delta_bias.pow(2).sum(dim=-1)
        delta_w_norm = torch.sqrt(delta_w_l2 + 1e-12)
        delta_b_norm = torch.sqrt(delta_b_l2 + 1e-12)

        if not return_aux:
            return action_logits, next_hidden_state
        return action_logits, next_hidden_state, {
            "ctx_mu": context_mean,
            "ctx_logvar": context_log_variance,
            "gate": modulation_gate,
            "delta_w_l2": delta_w_l2,
            "delta_b_l2": delta_b_l2,
            "delta_w_norm": delta_w_norm,
            "delta_b_norm": delta_b_norm,
        }


class SetTokenTeacher(nn.Module):
    """
    Token-based set teacher with masked cross-attention.

    Inputs:
        obs: [B, T, N, obs_dim]
        active_mask: [B, T, N] in {0,1}

    Outputs:
        tokens: [B, T, K, D]
        ctx_teacher: [B, T, N, D]
    """

    def __init__(
        self,
        obs_dim: int,
        set_embed_dim: int,
        num_tokens: int,
        set_encoder_hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.set_embed_dim = int(set_embed_dim)
        self.num_tokens = int(num_tokens)

        # Shared set encoder phi(o_i).
        self.agent_mlp = _build_mlp(
            in_dim=self.obs_dim,
            hidden_sizes=tuple(set_encoder_hidden_sizes),
            out_dim=self.set_embed_dim,
        )
        # Learnable token queries used to summarize the active set.
        self.token_queries = nn.Parameter(torch.randn(self.num_tokens, self.set_embed_dim) * 0.02)

    @staticmethod
    def _masked_cross_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Mask-safe scaled dot-product attention.

        query: [B, T, Q, D]
        key: [B, T, M, D]
        value: [B, T, M, D]
        key_mask: optional [B, T, M]
        """
        scale = math.sqrt(float(query.shape[-1]))
        attention_scores = torch.einsum("btqd,btmd->btqm", query, key) / max(scale, 1e-8)

        if key_mask is None:
            attention_weights = torch.softmax(attention_scores, dim=-1)
            return torch.einsum("btqm,btmd->btqd", attention_weights, value)

        key_mask_boolean = key_mask.to(dtype=torch.bool, device=attention_scores.device)
        attention_scores = attention_scores.masked_fill(~key_mask_boolean.unsqueeze(2), -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * key_mask_boolean.unsqueeze(2).to(dtype=attention_weights.dtype)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.einsum("btqm,btmd->btqd", attention_weights, value)

    def forward(self, obs: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract set tokens and per-agent teacher context.

        Args:
            obs: [B, T, N, obs_dim]
            active_mask: [B, T, N]

        Returns:
            tokens: [B, T, K, D]
            ctx_teacher: [B, T, N, D]
        """
        batch_size, num_timesteps, num_agents, observation_dim = obs.shape

        agent_embeddings = self.agent_mlp(
            obs.reshape(batch_size * num_timesteps * num_agents, observation_dim)
        ).reshape(batch_size, num_timesteps, num_agents, self.set_embed_dim)

        expanded_token_queries = self.token_queries.view(1, 1, self.num_tokens, self.set_embed_dim).expand(
            batch_size,
            num_timesteps,
            -1,
            -1,
        )

        tokens = self._masked_cross_attention(
            expanded_token_queries,
            agent_embeddings,
            agent_embeddings,
            key_mask=active_mask,
        )
        ctx_teacher = self._masked_cross_attention(
            agent_embeddings,
            tokens,
            tokens,
            key_mask=None,
        )

        # Keep padded/inactive slots strictly zero for clean masked reductions.
        ctx_teacher = ctx_teacher * active_mask.unsqueeze(-1)
        return tokens, ctx_teacher


class SetValueCritic(nn.Module):
    """
    Centralized critic with tokenized team context.

    The teacher produces coordination tokens and per-agent context. Team value is
    predicted from pooled tokens via `rho`.
    """

    def __init__(
        self,
        obs_dim: int,
        set_embed_dim: int,
        num_tokens: int,
        set_encoder_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        include_team_size_feature: bool = True,
    ):
        super().__init__()
        self.set_embed_dim = int(set_embed_dim)
        self.include_team_size_feature = bool(include_team_size_feature)

        self.teacher = SetTokenTeacher(
            obs_dim=int(obs_dim),
            set_embed_dim=self.set_embed_dim,
            num_tokens=int(num_tokens),
            set_encoder_hidden_sizes=tuple(set_encoder_hidden_sizes),
        )

        pooled_dim = self.set_embed_dim + (1 if self.include_team_size_feature else 0)
        self.value_mlp = _build_mlp(
            in_dim=pooled_dim,
            hidden_sizes=tuple(critic_hidden_sizes),
            out_dim=1,
        )

    def forward(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        return_details: bool = False,
    ):
        """
        Args:
            obs: [B, T, N, obs_dim]
            active_mask: [B, T, N]
            return_details: include tokens/context when true.

        Returns:
            team_values: [B, T]
            (optional) tokens: [B, T, K, D], ctx_teacher: [B, T, N, D]
        """
        tokens, ctx_teacher = self.teacher(obs, active_mask)
        team_context = tokens.mean(dim=2)

        if self.include_team_size_feature:
            active_count = active_mask.sum(dim=2, keepdim=True)
            value_input = torch.cat([team_context, active_count], dim=-1)
        else:
            value_input = team_context

        batch_size, num_timesteps, context_dim = value_input.shape
        team_values = self.value_mlp(value_input.reshape(batch_size * num_timesteps, context_dim)).reshape(
            batch_size,
            num_timesteps,
        )

        if not return_details:
            return team_values
        return team_values, tokens, ctx_teacher


class PIMACV3(ParallelLearner):
    """
    Shared PIMACV3 implementation core backing the public parallel learner.

    Core responsibilities:
    - shared actor/critic construction,
    - robust replay coercion and episode finalization,
    - team-level GAE and PPO losses,
    - teacher-student distillation with EMA teacher targets,
    - diagnostics tracking for scripts/plots.

    The actor uses head-only context adaptation. FiLM feature modulation first
    appears in `pimac_v4`.
    """

    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, PIMACV3_DEFAULT_CONFIG)

    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        config = self.config

        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.clip_eps = float(config["clip_eps"])
        self.normalize_advantage = bool(config["normalize_advantage"])
        self.entropy_coef = float(config["entropy_coef"])
        self.value_coef = float(config["value_coef"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self.update_every_episodes = int(config["update_every_episodes"])

        self.set_embed_dim = int(config["set_embed_dim"])
        self.num_tokens = int(config["num_tokens"])
        self.distill_weight = float(config["distill_weight"])
        self.teacher_ema_tau = float(max(0.0, min(1.0, config["teacher_ema_tau"])))
        self.hypernet_rank = int(config["hypernet_rank"])
        self.hypernet_hidden_sizes = tuple(config["hypernet_hidden_sizes"])
        self.hypernet_delta_init_scale = float(config["hypernet_delta_init_scale"])
        self.hypernet_l2_coef = float(config["hypernet_l2_coef"])

        self.actor_net = PIMACV3ActorRNN(
            obs_dim=self.obs_size,
            action_dim=self.action_space_size,
            num_hidden=int(config["num_hidden"]),
            widths=tuple(config["widths"]),
            rnn_hidden_dim=int(config["rnn_hidden_dim"]),
            ctx_dim=self.set_embed_dim,
            hypernet_rank=self.hypernet_rank,
            hypernet_hidden_sizes=self.hypernet_hidden_sizes,
            hypernet_delta_init_scale=self.hypernet_delta_init_scale,
            ctx_logvar_min=float(config["ctx_logvar_min"]),
            ctx_logvar_max=float(config["ctx_logvar_max"]),
        ).to(self.device)

        self.critic = SetValueCritic(
            obs_dim=self.obs_size,
            set_embed_dim=self.set_embed_dim,
            num_tokens=self.num_tokens,
            set_encoder_hidden_sizes=tuple(config["set_encoder_hidden_sizes"]),
            critic_hidden_sizes=tuple(config["critic_hidden_sizes"]),
            include_team_size_feature=bool(config["include_team_size_feature"]),
        ).to(self.device)

        # EMA target teacher for stable distillation targets.
        self.target_teacher = copy.deepcopy(self.critic.teacher).to(self.device)
        self.target_teacher.eval()
        for parameter in self.target_teacher.parameters():
            parameter.requires_grad_(False)

        self.optimizer = optim.Adam(
            list(self._actor_parameters()) + list(self.critic.parameters()),
            lr=float(config["lr"]),
        )

        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._episode_steps: list[dict] = []

        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._episode_finished = False

    def _actor_parameters(self):
        """Return actor parameters for optimizer and gradient clipping."""
        return self.actor_net.parameters()

    def reset_episode(self) -> None:
        """Clear inference recurrent state at the start of a new rollout episode."""
        self._inference_hidden = {}

    def _get_hidden_state(self, agent_key: object, hidden_dim: int) -> torch.Tensor:
        """Fetch or initialize the recurrent hidden state for one actor stream."""
        previous_hidden_state = self._inference_hidden.get(agent_key)
        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros(1, 1, int(hidden_dim), device=self.device)
        return previous_hidden_state

    def _set_hidden_state(self, agent_key: object, hidden_state: torch.Tensor) -> None:
        """Persist detached recurrent state after one forward pass."""
        self._inference_hidden[agent_key] = hidden_state.detach()

    def _act_single(self, state: np.ndarray, actor_key: object) -> int:
        """Shared single-agent action path reused by the public parallel helpers."""
        observation_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        hidden_dim = self.actor_net.rnn.hidden_size
        initial_hidden_state = self._get_hidden_state(actor_key, hidden_dim)
        action_logits_sequence, updated_hidden_state = self.actor_net(observation_tensor, initial_hidden_state)
        self._set_hidden_state(actor_key, updated_hidden_state)

        action_logits = action_logits_sequence.squeeze(0).squeeze(0)
        categorical_distribution = torch.distributions.Categorical(logits=action_logits)
        return int(categorical_distribution.sample().item())

    @staticmethod
    def _ensure_state(state: Optional[np.ndarray]) -> np.ndarray:
        """Normalize optional state inputs to dense float32 arrays."""
        if state is None:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(state, dtype=np.float32)

    def _coerce_step(
        self,
        observations,
        actions,
        rewards,
        active_mask,
        global_state,
        next_observations,
        next_active_mask,
        next_global_state,
        done,
        agent_ids: Optional[Sequence[object]],
    ) -> dict:
        """
        Normalize one transition into the internal replay format.

        The coercion layer keeps dict-based and array-based callers compatible
        with one shared episode-finalization path.
        """
        observations_are_mapping = isinstance(observations, dict)
        if observations_are_mapping:
            if agent_ids is None:
                agent_ids = _sorted_agent_ids(observations.keys())
            observation_array = np.stack(
                [np.asarray(observations[agent_id], dtype=np.float32) for agent_id in agent_ids],
                axis=0,
            )
        else:
            observation_array = np.asarray(observations, dtype=np.float32)
            if observation_array.ndim == 1:
                observation_array = observation_array.reshape(1, -1)
            if agent_ids is None:
                agent_ids = list(range(observation_array.shape[0]))

        if isinstance(actions, dict):
            action_array = np.asarray([actions.get(agent_id, 0) for agent_id in agent_ids], dtype=np.int64)
        else:
            action_array = np.asarray(actions, dtype=np.int64)
            if action_array.ndim == 0:
                action_array = action_array.reshape(1)

        if isinstance(rewards, dict):
            reward_array = np.asarray([rewards.get(agent_id, 0.0) for agent_id in agent_ids], dtype=np.float32)
        else:
            reward_array = np.asarray(rewards, dtype=np.float32)
            if reward_array.ndim == 0:
                reward_array = reward_array.reshape(1)

        if active_mask is None:
            active_mask_array = np.ones(len(agent_ids), dtype=np.float32)
        elif isinstance(active_mask, dict):
            active_mask_array = np.asarray([active_mask.get(agent_id, 0.0) for agent_id in agent_ids], dtype=np.float32)
        else:
            active_mask_array = np.asarray(active_mask, dtype=np.float32)
            if active_mask_array.ndim == 0:
                active_mask_array = active_mask_array.reshape(1)

        next_observations_are_mapping = isinstance(next_observations, dict)
        if next_observations_are_mapping:
            next_agent_ids = _sorted_agent_ids(next_observations.keys())
            if next_agent_ids:
                next_observation_array = np.stack(
                    [np.asarray(next_observations[agent_id], dtype=np.float32) for agent_id in next_agent_ids],
                    axis=0,
                )
            else:
                next_observation_array = np.zeros((0, observation_array.shape[-1]), dtype=np.float32)
        else:
            next_observation_array = np.asarray(next_observations, dtype=np.float32)
            if next_observation_array.ndim == 1:
                next_observation_array = next_observation_array.reshape(1, -1)
            next_agent_ids = list(range(next_observation_array.shape[0]))

        if next_active_mask is None:
            next_active_mask_array = np.ones(len(next_agent_ids), dtype=np.float32)
        elif isinstance(next_active_mask, dict):
            next_active_mask_array = np.asarray(
                [next_active_mask.get(agent_id, 0.0) for agent_id in next_agent_ids],
                dtype=np.float32,
            )
        else:
            next_active_mask_array = np.asarray(next_active_mask, dtype=np.float32)
            if next_active_mask_array.ndim == 0:
                next_active_mask_array = next_active_mask_array.reshape(1)

        return {
            "agent_ids": list(agent_ids),
            "obs": observation_array,
            "actions": action_array,
            "rewards": reward_array,
            "active_mask": active_mask_array,
            "state": self._ensure_state(global_state),
            "next_obs": next_observation_array,
            "next_active_mask": next_active_mask_array,
            "next_state": self._ensure_state(next_global_state),
            "done": bool(done),
            "next_agent_ids": list(next_agent_ids),
        }

    def store_transition(
        self,
        observations,
        actions,
        rewards,
        active_mask,
        global_state: Optional[np.ndarray],
        next_observations,
        next_active_mask,
        next_global_state: Optional[np.ndarray],
        done: bool,
        agent_ids: Optional[Sequence[object]] = None,
    ) -> None:
        """Append one transition and finalize the episode when `done=True`."""
        self._episode_steps.append(
            self._coerce_step(
                observations,
                actions,
                rewards,
                active_mask,
                global_state,
                next_observations,
                next_active_mask,
                next_global_state,
                done,
                agent_ids,
            )
        )

        if done:
            finalized_episode = self._finalize_episode(self._episode_steps)
            self.memory.append(finalized_episode)
            self._episode_steps = []
            self._episode_finished = True

    def _actor_forward(self, obs: torch.Tensor, return_aux: bool = False):
        """
        Run shared actor over a padded minibatch tensor.

        Args:
            obs: [B, T, N, obs_dim]
            return_aux: return student context heads when true.

        Returns:
            logits: [B, T, N, action_dim]
            aux (optional):
                - ctx_mu: [B, T, N, D]
                - ctx_logvar: [B, T, N, D]
                - gate: [B, T, N, 1]
                - delta_w_l2: [B, T, N]
                - delta_b_l2: [B, T, N]
                - delta_w_norm: [B, T, N]
                - delta_b_norm: [B, T, N]
        """
        batch_size, num_timesteps, num_agents, observation_dim = obs.shape
        flattened_agent_sequences = obs.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents,
            num_timesteps,
            observation_dim,
        )

        if return_aux:
            flattened_logits, _, flattened_aux = self.actor_net(flattened_agent_sequences, None, return_aux=True)
            action_logits = flattened_logits.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)
            aux = {
                "ctx_mu": flattened_aux["ctx_mu"].reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3),
                "ctx_logvar": flattened_aux["ctx_logvar"].reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3),
                "gate": flattened_aux["gate"].reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3),
                "delta_w_l2": flattened_aux["delta_w_l2"].reshape(batch_size, num_agents, num_timesteps).permute(0, 2, 1),
                "delta_b_l2": flattened_aux["delta_b_l2"].reshape(batch_size, num_agents, num_timesteps).permute(0, 2, 1),
                "delta_w_norm": flattened_aux["delta_w_norm"].reshape(batch_size, num_agents, num_timesteps).permute(0, 2, 1),
                "delta_b_norm": flattened_aux["delta_b_norm"].reshape(batch_size, num_agents, num_timesteps).permute(0, 2, 1),
            }
            return action_logits, aux

        flattened_logits, _ = self.actor_net(flattened_agent_sequences, None)
        action_logits = flattened_logits.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)
        return action_logits

    @staticmethod
    def _team_rewards(rewards: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Compute mean active-agent reward per timestep."""
        active_reward_sum = (rewards * active_mask).sum(axis=1)
        active_agent_count = active_mask.sum(axis=1)
        timestep_team_rewards = np.zeros(rewards.shape[0], dtype=np.float32)
        valid_timesteps = active_agent_count > 0
        timestep_team_rewards[valid_timesteps] = (
            active_reward_sum[valid_timesteps] / active_agent_count[valid_timesteps]
        )
        return timestep_team_rewards

    def _compute_old_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute behavior-policy log-probs stored with finalized episodes."""
        observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(0)
        active_mask_tensor = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits = self._actor_forward(observation_tensor)
            categorical_distribution = torch.distributions.Categorical(logits=action_logits)
            old_log_probabilities = categorical_distribution.log_prob(action_tensor) * active_mask_tensor

        return old_log_probabilities.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _compute_values(self, obs: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Evaluate centralized values for one finalized episode."""
        observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        active_mask_tensor = torch.as_tensor(active_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            predicted_values = self.critic(observation_tensor, active_mask_tensor).squeeze(0)
        return predicted_values.cpu().numpy().astype(np.float32, copy=False)

    def _compute_gae(
        self,
        team_rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute team-level GAE(lambda) advantages and returns."""
        team_advantages = np.zeros_like(team_rewards, dtype=np.float32)
        running_advantage = 0.0

        for timestep_index in range(team_rewards.shape[0] - 1, -1, -1):
            non_terminal_factor = 1.0 - float(dones[timestep_index])
            temporal_difference_residual = (
                team_rewards[timestep_index]
                + self.gamma * non_terminal_factor * next_values[timestep_index]
                - values[timestep_index]
            )
            running_advantage = (
                temporal_difference_residual
                + self.gamma * self.gae_lambda * non_terminal_factor * running_advantage
            )
            team_advantages[timestep_index] = running_advantage

        team_returns = team_advantages + values
        return (
            team_advantages.astype(np.float32, copy=False),
            team_returns.astype(np.float32, copy=False),
        )

    def _finalize_episode(self, steps: list[dict]) -> dict:
        """
        Convert variable-roster step data into one roster-aligned episode record.

        The final episode contains dense [T, N, ...] arrays plus PPO fields
        (`old_log_probs`, `advantages`, `returns`).
        """
        episode_roster: list[object] = []
        roster_slot_by_agent_id: dict[object, int] = {}

        def register_agents(agent_ids_for_step: Sequence[object]) -> None:
            for agent_id in agent_ids_for_step:
                if agent_id not in roster_slot_by_agent_id:
                    roster_slot_by_agent_id[agent_id] = len(episode_roster)
                    episode_roster.append(agent_id)

        for step_data in steps:
            register_agents(step_data["agent_ids"])
            register_agents(step_data["next_agent_ids"])

        num_timesteps = len(steps)
        num_roster_agents = len(episode_roster)
        observation_dim = steps[0]["obs"].shape[-1]

        observations = np.zeros((num_timesteps, num_roster_agents, observation_dim), dtype=np.float32)
        actions = np.zeros((num_timesteps, num_roster_agents), dtype=np.int64)
        rewards = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        active_mask = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        next_observations = np.zeros((num_timesteps, num_roster_agents, observation_dim), dtype=np.float32)
        next_active_mask = np.zeros((num_timesteps, num_roster_agents), dtype=np.float32)
        global_states = np.stack([step_data["state"] for step_data in steps], axis=0)
        next_global_states = np.stack([step_data["next_state"] for step_data in steps], axis=0)
        done_flags = np.asarray([step_data["done"] for step_data in steps], dtype=np.float32)

        for timestep_index, step_data in enumerate(steps):
            for local_agent_index, agent_id in enumerate(step_data["agent_ids"]):
                roster_slot_index = roster_slot_by_agent_id[agent_id]
                observations[timestep_index, roster_slot_index] = step_data["obs"][local_agent_index]
                actions[timestep_index, roster_slot_index] = step_data["actions"][local_agent_index]
                rewards[timestep_index, roster_slot_index] = step_data["rewards"][local_agent_index]
                active_mask[timestep_index, roster_slot_index] = step_data["active_mask"][local_agent_index]

            for local_next_agent_index, agent_id in enumerate(step_data["next_agent_ids"]):
                roster_slot_index = roster_slot_by_agent_id[agent_id]
                next_observations[timestep_index, roster_slot_index] = step_data["next_obs"][local_next_agent_index]
                next_active_mask[timestep_index, roster_slot_index] = step_data["next_active_mask"][
                    local_next_agent_index
                ]

        old_log_probabilities = self._compute_old_log_probs(observations, actions, active_mask)
        team_values = self._compute_values(observations, active_mask)
        next_team_values = self._compute_values(next_observations, next_active_mask)
        team_rewards = self._team_rewards(rewards, active_mask)
        team_advantages, team_returns = self._compute_gae(team_rewards, team_values, next_team_values, done_flags)

        return {
            "obs": observations,
            "actions": actions,
            "rewards": rewards,
            "active_mask": active_mask,
            "state": global_states,
            "next_obs": next_observations,
            "next_active_mask": next_active_mask,
            "next_state": next_global_states,
            "done": done_flags,
            "old_log_probs": old_log_probabilities,
            "advantages": team_advantages,
            "returns": team_returns,
            "values": team_values,
            "team_rewards": team_rewards,
            "T": int(observations.shape[0]),
            "N": int(observations.shape[1]),
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean with finite fallback on empty supports."""
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _pad_time(time_series_array: np.ndarray, max_num_timesteps: int, pad_value: float = 0.0) -> np.ndarray:
        """Right-pad the time axis of an array to a shared batch horizon."""
        current_num_timesteps = time_series_array.shape[0]
        if current_num_timesteps == max_num_timesteps:
            return time_series_array
        padding_shape = (max_num_timesteps - current_num_timesteps,) + time_series_array.shape[1:]
        right_padding = np.full(padding_shape, pad_value, dtype=time_series_array.dtype)
        return np.concatenate([time_series_array, right_padding], axis=0)

    @staticmethod
    def _pad_time_agents(
        time_agent_array: np.ndarray,
        max_num_timesteps: int,
        max_num_agents: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """Pad [T, N, ...] tensors to one dense [max_T, max_N, ...] grid."""
        current_num_timesteps, current_num_agents = time_agent_array.shape[0], time_agent_array.shape[1]
        if current_num_timesteps == max_num_timesteps and current_num_agents == max_num_agents:
            return time_agent_array
        pad_width = [
            (0, max_num_timesteps - current_num_timesteps),
            (0, max_num_agents - current_num_agents),
        ] + [(0, 0)] * (time_agent_array.ndim - 2)
        return np.pad(time_agent_array, pad_width, mode="constant", constant_values=pad_value)

    def _build_minibatch_tensors(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Convert sampled episodes to padded training tensors.

        Padding is dynamic and local to the sampled minibatch:
        we pad to the largest team size present in this batch only.
        The PIMAC update path does not expand tensors to `env_spec.max_agents`.
        """
        max_num_timesteps = max(int(episode["T"]) for episode in batch)
        max_num_agents = max(int(episode["N"]) for episode in batch)

        observations = torch.as_tensor(
            np.stack([self._pad_time_agents(episode["obs"], max_num_timesteps, max_num_agents) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["actions"], max_num_timesteps, max_num_agents, pad_value=0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.int64,
        )
        active_mask = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["active_mask"], max_num_timesteps, max_num_agents, pad_value=0.0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.float32,
        )
        old_log_probs = torch.as_tensor(
            np.stack(
                [self._pad_time_agents(episode["old_log_probs"], max_num_timesteps, max_num_agents, pad_value=0.0) for episode in batch]
            ),
            device=self.device,
            dtype=torch.float32,
        )
        advantages = torch.as_tensor(
            np.stack([self._pad_time(episode["advantages"], max_num_timesteps, pad_value=0.0) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )
        returns = torch.as_tensor(
            np.stack([self._pad_time(episode["returns"], max_num_timesteps, pad_value=0.0) for episode in batch]),
            device=self.device,
            dtype=torch.float32,
        )

        episode_lengths = torch.as_tensor([int(episode["T"]) for episode in batch], device=self.device, dtype=torch.int64)
        time_mask = (torch.arange(max_num_timesteps, device=self.device).unsqueeze(0) < episode_lengths.unsqueeze(1)).to(
            dtype=torch.float32
        )
        combined_mask = active_mask * time_mask.unsqueeze(-1)
        valid_time = time_mask * (active_mask.sum(dim=2) > 0.0).to(dtype=torch.float32)

        return {
            "obs": observations,
            "actions": actions,
            "active_mask": active_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "time_mask": time_mask,
            "combined_mask": combined_mask,
            "valid_time": valid_time,
        }

    def _normalize_advantages(self, advantages: torch.Tensor, valid_time: torch.Tensor) -> torch.Tensor:
        """Normalize team-level advantages over valid timesteps only."""
        if not self.normalize_advantage:
            return advantages

        valid_advantages = advantages[valid_time.bool()]
        if valid_advantages.numel() == 0:
            return advantages

        advantage_mean = valid_advantages.mean()
        advantage_std = valid_advantages.std(unbiased=False)
        return (advantages - advantage_mean) / (advantage_std + 1e-8)

    @staticmethod
    def _teacher_ema_update(target_teacher: SetTokenTeacher, online_teacher: SetTokenTeacher, tau: float) -> None:
        """EMA update for teacher parameters and floating buffers."""
        for target_parameter, online_parameter in zip(target_teacher.parameters(), online_teacher.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(online_parameter.data, alpha=tau)

        for target_buffer, online_buffer in zip(target_teacher.buffers(), online_teacher.buffers()):
            if torch.is_floating_point(target_buffer):
                target_buffer.data.mul_(1.0 - tau).add_(online_buffer.data, alpha=tau)
            else:
                target_buffer.data.copy_(online_buffer.data)

    def _teacher_for_targets(self) -> SetTokenTeacher:
        """Select teacher used as distillation target source."""
        if self.teacher_ema_tau > 0.0:
            return self.target_teacher
        return self.critic.teacher

    def _update_target_teacher(self) -> None:
        """Apply EMA update to target teacher if enabled."""
        if self.teacher_ema_tau <= 0.0:
            return
        self._teacher_ema_update(self.target_teacher, self.critic.teacher, self.teacher_ema_tau)

    def _compute_update_diagnostics(
        self,
        active_mask: torch.Tensor,
        valid_time: torch.Tensor,
        combined_mask: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        ratio: torch.Tensor,
        decision_denom: torch.Tensor,
        ctx_logvar: torch.Tensor,
        gate: torch.Tensor,
        delta_w_norm: torch.Tensor,
        delta_b_norm: torch.Tensor,
        tokens: torch.Tensor,
        teacher_context: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """Compute training diagnostics for one optimizer update."""
        with torch.no_grad():
            log_probability_ratio = new_log_probs - old_log_probs
            approx_kl = ((-log_probability_ratio) * combined_mask).sum() / decision_denom
            clip_fraction = (((ratio - 1.0).abs() > self.clip_eps).to(dtype=torch.float32) * combined_mask).sum() / decision_denom

            valid_time_boolean_mask = valid_time.bool()
            if valid_time_boolean_mask.any():
                valid_returns = returns[valid_time_boolean_mask]
                valid_predicted_values = values.detach()[valid_time_boolean_mask]
                returns_variance = torch.var(valid_returns, unbiased=False)
                if float(returns_variance.item()) > 1e-8:
                    explained_variance = 1.0 - (
                        torch.var(valid_returns - valid_predicted_values, unbiased=False) / (returns_variance + 1e-8)
                    )
                else:
                    explained_variance = torch.tensor(0.0, device=self.device)
            else:
                explained_variance = torch.tensor(0.0, device=self.device)

            active_decision_count = float(combined_mask.sum().item())
            active_agents_per_timestep = active_mask.sum(dim=-1)
            active_agents_mean = self._masked_mean(active_agents_per_timestep, valid_time)

            token_norm = tokens.norm(dim=-1)
            token_norm_mean = self._masked_mean(token_norm.mean(dim=2), valid_time)

            teacher_context_norm = teacher_context.norm(dim=-1)
            teacher_ctx_norm_mean = self._masked_mean(teacher_context_norm, combined_mask)

            mean_log_variance_per_agent = ctx_logvar.mean(dim=-1)
            selected_log_variances = mean_log_variance_per_agent[combined_mask.bool()]
            if selected_log_variances.numel() > 0:
                ctx_logvar_mean = selected_log_variances.mean()
                ctx_logvar_std = selected_log_variances.std(unbiased=False)
            else:
                ctx_logvar_mean = torch.tensor(0.0, device=self.device)
                ctx_logvar_std = torch.tensor(0.0, device=self.device)

            selected_gate_values = gate[combined_mask.bool()]
            if selected_gate_values.numel() > 0:
                gate_mean = selected_gate_values.mean()
                gate_std = selected_gate_values.std(unbiased=False)
            else:
                gate_mean = torch.tensor(0.0, device=self.device)
                gate_std = torch.tensor(0.0, device=self.device)

            selected_delta_w_norm = delta_w_norm[combined_mask.bool()]
            if selected_delta_w_norm.numel() > 0:
                delta_w_norm_mean = selected_delta_w_norm.mean()
            else:
                delta_w_norm_mean = torch.tensor(0.0, device=self.device)

            selected_delta_b_norm = delta_b_norm[combined_mask.bool()]
            if selected_delta_b_norm.numel() > 0:
                delta_b_norm_mean = selected_delta_b_norm.mean()
            else:
                delta_b_norm_mean = torch.tensor(0.0, device=self.device)

        return {
            "approx_kl": approx_kl,
            "clip_frac": clip_fraction,
            "explained_variance": explained_variance,
            "active_decisions": active_decision_count,
            "active_agents_mean": active_agents_mean,
            "token_norm_mean": token_norm_mean,
            "teacher_ctx_norm_mean": teacher_ctx_norm_mean,
            "ctx_logvar_mean": ctx_logvar_mean,
            "ctx_logvar_std": ctx_logvar_std,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "delta_w_norm_mean": delta_w_norm_mean,
            "delta_b_norm_mean": delta_b_norm_mean,
        }

    def _run_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        """
        Run one PIMACV3 update phase.

        Each epoch samples on-policy episodes, computes PPO + value + distillation
        losses, applies one optimizer step, logs diagnostics, and updates EMA teacher.
        """
        memory_items = len(self.memory)
        if memory_items < self.batch_size:
            return None

        mean_epoch_losses = []
        mean_policy_losses: list[float] = []
        mean_value_losses: list[float] = []
        mean_entropies: list[float] = []
        mean_approx_kls: list[float] = []
        mean_clip_fracs: list[float] = []
        mean_explained_variances: list[float] = []
        mean_distill_losses: list[float] = []
        mean_distill_mses: list[float] = []
        mean_hyper_l2s: list[float] = []
        mean_ctx_logvar_means: list[float] = []
        mean_ctx_logvar_stds: list[float] = []
        mean_gate_means: list[float] = []
        mean_gate_stds: list[float] = []
        mean_delta_w_norms: list[float] = []
        mean_delta_b_norms: list[float] = []
        mean_token_norms: list[float] = []
        mean_teacher_norms: list[float] = []
        mean_active_decisions: list[float] = []
        mean_active_agents: list[float] = []
        mean_grad_norms: list[float] = []
        total_samples_seen = 0
        for _ in range(self.num_epochs):
            sampled_episodes = random.sample(self.memory, self.batch_size)
            minibatch_tensors = self._build_minibatch_tensors(sampled_episodes)

            observations = minibatch_tensors["obs"]
            actions = minibatch_tensors["actions"]
            active_mask = minibatch_tensors["active_mask"]
            old_log_probabilities = minibatch_tensors["old_log_probs"]
            normalized_advantages = self._normalize_advantages(
                minibatch_tensors["advantages"],
                minibatch_tensors["valid_time"],
            )
            returns = minibatch_tensors["returns"]
            combined_mask = minibatch_tensors["combined_mask"]
            valid_time = minibatch_tensors["valid_time"]
            total_samples_seen += int(valid_time.sum().item())

            action_logits, actor_aux = self._actor_forward(observations, return_aux=True)
            categorical_distribution = torch.distributions.Categorical(logits=action_logits)
            new_log_probabilities = categorical_distribution.log_prob(actions)
            action_entropy = categorical_distribution.entropy()

            broadcast_advantages = normalized_advantages.unsqueeze(-1).expand_as(new_log_probabilities)
            importance_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            unclipped_surrogate = importance_ratio * broadcast_advantages
            clipped_surrogate = torch.clamp(importance_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * broadcast_advantages

            decision_denom = combined_mask.sum().clamp(min=1.0)
            policy_loss = -(torch.min(unclipped_surrogate, clipped_surrogate) * combined_mask).sum() / decision_denom
            entropy_bonus = (action_entropy * combined_mask).sum() / decision_denom

            predicted_values, tokens, teacher_context = self.critic(observations, active_mask, return_details=True)
            value_loss = (((returns - predicted_values) ** 2) * valid_time).sum() / valid_time.sum().clamp(min=1.0)

            teacher_for_targets = self._teacher_for_targets()
            with torch.no_grad():
                _, target_teacher_context = teacher_for_targets(observations, active_mask)

            student_context_mean = actor_aux["ctx_mu"]
            student_context_log_variance = actor_aux["ctx_logvar"]
            uncertainty_gate = actor_aux["gate"].squeeze(-1)
            delta_weight_l2 = actor_aux["delta_w_l2"]
            delta_bias_l2 = actor_aux["delta_b_l2"]
            delta_weight_norm = actor_aux["delta_w_norm"]
            delta_bias_norm = actor_aux["delta_b_norm"]

            distillation_nll = (
                0.5 * torch.exp(-student_context_log_variance) * ((student_context_mean - target_teacher_context) ** 2)
                + 0.5 * student_context_log_variance
            )
            distillation_loss = (distillation_nll.sum(dim=-1) * combined_mask).sum() / decision_denom

            distillation_mse = (((student_context_mean - target_teacher_context) ** 2).mean(dim=-1) * combined_mask).sum() / decision_denom
            hypernet_l2 = ((delta_weight_l2 + delta_bias_l2) * combined_mask).sum() / decision_denom

            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_bonus
                + self.distill_weight * distillation_loss
                + self.hypernet_l2_coef * hypernet_l2
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = _parameter_grad_norm(list(self._actor_parameters()) + list(self.critic.parameters()))
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self._actor_parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.optimizer.step()
            self._update_target_teacher()

            diagnostics = self._compute_update_diagnostics(
                active_mask=active_mask,
                valid_time=valid_time,
                combined_mask=combined_mask,
                returns=returns,
                values=predicted_values,
                old_log_probs=old_log_probabilities,
                new_log_probs=new_log_probabilities,
                ratio=importance_ratio,
                decision_denom=decision_denom,
                ctx_logvar=student_context_log_variance,
                gate=uncertainty_gate,
                delta_w_norm=delta_weight_norm,
                delta_b_norm=delta_bias_norm,
                tokens=tokens,
                teacher_context=teacher_context,
            )

            total_loss_scalar = float(total_loss.detach().item())
            mean_epoch_losses.append(total_loss_scalar)
            mean_policy_losses.append(float(policy_loss.detach().item()))
            mean_value_losses.append(float(value_loss.detach().item()))
            mean_entropies.append(float(entropy_bonus.detach().item()))
            mean_approx_kls.append(float(diagnostics["approx_kl"].detach().item()))
            mean_clip_fracs.append(float(diagnostics["clip_frac"].detach().item()))
            mean_explained_variances.append(float(diagnostics["explained_variance"].detach().item()))
            mean_distill_losses.append(float(distillation_loss.detach().item()))
            mean_distill_mses.append(float(distillation_mse.detach().item()))
            mean_hyper_l2s.append(float(hypernet_l2.detach().item()))
            mean_ctx_logvar_means.append(float(diagnostics["ctx_logvar_mean"].detach().item()))
            mean_ctx_logvar_stds.append(float(diagnostics["ctx_logvar_std"].detach().item()))
            mean_gate_means.append(float(diagnostics["gate_mean"].detach().item()))
            mean_gate_stds.append(float(diagnostics["gate_std"].detach().item()))
            mean_delta_w_norms.append(float(diagnostics["delta_w_norm_mean"].detach().item()))
            mean_delta_b_norms.append(float(diagnostics["delta_b_norm_mean"].detach().item()))
            mean_token_norms.append(float(diagnostics["token_norm_mean"].detach().item()))
            mean_teacher_norms.append(float(diagnostics["teacher_ctx_norm_mean"].detach().item()))
            mean_active_decisions.append(float(diagnostics["active_decisions"]))
            mean_active_agents.append(float(diagnostics["active_agents_mean"].detach().item()))
            mean_grad_norms.append(float(grad_norm))

        self.memory.clear()
        if not mean_epoch_losses:
            return None
        report = UpdateReport(
            update_index=len(self._update_reports) + 1,
            episode_index=int(episode_index),
            global_step=int(global_step),
            total_loss=float(np.mean(mean_epoch_losses)),
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
                "distill_loss": float(np.mean(mean_distill_losses)),
                "distill_mse": float(np.mean(mean_distill_mses)),
                "hyper_l2": float(np.mean(mean_hyper_l2s)),
                "ctx_logvar_mean": float(np.mean(mean_ctx_logvar_means)),
                "ctx_logvar_std": float(np.mean(mean_ctx_logvar_stds)),
                "gate_mean": float(np.mean(mean_gate_means)),
                "gate_std": float(np.mean(mean_gate_stds)),
                "delta_w_norm_mean": float(np.mean(mean_delta_w_norms)),
                "delta_b_norm_mean": float(np.mean(mean_delta_b_norms)),
                "token_norm_mean": float(np.mean(mean_token_norms)),
                "teacher_ctx_norm_mean": float(np.mean(mean_teacher_norms)),
                "active_decisions": float(np.mean(mean_active_decisions)),
                "active_agents_mean": float(np.mean(mean_active_agents)),
            },
        )
        return self._append_update_report(report)

    def maybe_update(self, global_step: int, episode_index: int) -> Optional[UpdateReport]:
        if not self._episode_finished:
            return None
        if (int(episode_index) % self.update_every_episodes) != 0:
            return None
        report = self._run_update(global_step=global_step, episode_index=episode_index)
        self._episode_finished = False
        return report

    def set_eval_mode(self) -> None:
        """Switch actor/critic modules to evaluation mode without changing action sampling."""
        self._eval_mode = True
        self.actor_net.eval()
        self.critic.eval()
        self.target_teacher.eval()

    def set_train_mode(self) -> None:
        """Switch actor/critic modules to training mode; target teacher remains eval/frozen."""
        self._eval_mode = False
        self.actor_net.train()
        self.critic.train()
        self.target_teacher.eval()


    def act(
        self,
        state: np.ndarray,
        agent_index: Optional[object] = None,
    ) -> int:
        if agent_index is None:
            raise ValueError("PIMACV3.act requires agent_index. Prefer act_parallel for parallel environments.")
        return self._act_single(state=state, actor_key=agent_index)

    def act_parallel(self, obs_dict: dict[object, np.ndarray]) -> dict[object, int]:
        return {
            agent_id: self._act_single(state=obs_dict[agent_id], actor_key=agent_id)
            for agent_id in _sorted_agent_ids(obs_dict.keys())
        }

    def record_parallel_step(self, transition: ParallelTransition) -> None:
        self.store_transition(
            observations=transition.obs_dict,
            actions=transition.action_dict,
            rewards=transition.reward_dict,
            active_mask=transition.active_agent_mask_dict,
            global_state=transition.global_state,
            next_observations=transition.next_obs_dict,
            next_active_mask=transition.next_active_agent_mask_dict,
            next_global_state=transition.next_global_state,
            done=resolve_parallel_done(transition.done_dict),
        )

    def _checkpoint_state(self) -> dict:
        return {
            "actor_state_dict": self.actor_net.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_teacher_state_dict": self.target_teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_history": [report.to_flat_dict() for report in self._update_reports],
        }

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self.actor_net.load_state_dict(checkpoint_state["actor_state_dict"])
        self.critic.load_state_dict(checkpoint_state["critic_state_dict"])
        target_teacher_state = checkpoint_state.get("target_teacher_state_dict")
        if target_teacher_state is not None:
            self.target_teacher.load_state_dict(target_teacher_state)
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)#
