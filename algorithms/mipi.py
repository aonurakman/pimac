"""Entity-based MIPI adapted to the shared parallel benchmark API.

The runners still provide the same flat local observations used by all
methods. MIPI makes the task schema explicit inside this learner, parses those
flat vectors into local entities, and uses the entity-attention / marginal
policy / imaginary-composition machinery from the original implementation.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
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

__all__ = [
    "EntitySchema",
    "EntityAttentionLayer",
    "MIPIAgentRNN",
    "AveragePolicyRNN",
    "FlexQMixer",
    "MIPI",
    "MIPI_DEFAULT_CONFIG",
]


MIPI_DEFAULT_CONFIG = {
    "buffer_size": 2048,
    "batch_size": 32,
    "lr": 5e-4,
    "optim_alpha": 0.99,
    "optim_eps": 1e-5,
    "weight_decay": 0.0,
    "num_epochs": 1,
    "rnn_hidden_dim": 64,
    "attn_embed_dim": 128,
    "attn_n_heads": 4,
    "mixing_embed_dim": 32,
    "hypernet_embed": 128,
    "mixer_chunk_size": 1024,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "target_update_every": 200,
    "tau": 1.0,
    "share_parameters": True,
    "learning_starts": 0,
    "learn_every_steps": 1,
    "entity_schema": None,
    "mi_alpha_start": 0.1,
    "mi_alpha_end": 0.1,
    "mi_alpha_anneal_steps": 200000,
    "ba_iters": 4,
    "lmbda": 0.5,
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_steps": 500000,
    "test_greedy": True,
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


@dataclass(frozen=True)
class EntityGroup:
    name: str
    entity_type: str
    slices: tuple[tuple[int, int], ...]
    width: int
    team_related: bool
    active_rule: str
    active_count_offset: int
    active_feature_index: Optional[int]
    active_threshold: float
    inactive_values: Optional[tuple[float, ...]]


class EntitySchema:
    """Parse flat task observations into padded typed local entities."""

    def __init__(self, raw_schema: dict, obs_size: int):
        if not isinstance(raw_schema, dict):
            raise ValueError("MIPI requires an entity_schema dictionary.")
        raw_groups = raw_schema.get("groups")
        if not raw_groups:
            raise ValueError("MIPI entity_schema must contain at least one group.")

        groups: list[EntityGroup] = []
        entity_types: list[str] = []
        for raw_group in raw_groups:
            group = self._parse_group(raw_group, obs_size)
            groups.append(group)
            if group.entity_type not in entity_types:
                entity_types.append(group.entity_type)

        self.groups = tuple(groups)
        self.entity_types = tuple(entity_types)
        self.type_to_index = {entity_type: index for index, entity_type in enumerate(self.entity_types)}
        self.raw_feature_dim = max(group.width for group in self.groups)
        self.entity_dim = self.raw_feature_dim + len(self.entity_types) + 1
        self.entities_per_observation = sum(len(group.slices) for group in self.groups)

    @staticmethod
    def _parse_group(raw_group: dict, obs_size: int) -> EntityGroup:
        name = str(raw_group["name"])
        entity_type = str(raw_group.get("type", name))
        team_related = bool(raw_group.get("team_related", False))
        active_rule = str(raw_group.get("active_rule", "always"))
        active_count_offset = int(raw_group.get("active_count_offset", 0))
        active_feature_index = raw_group.get("active_feature_index")
        active_threshold = float(raw_group.get("active_threshold", 0.5))
        inactive_values = raw_group.get("inactive_values")
        if inactive_values is not None:
            inactive_values = tuple(float(value) for value in inactive_values)

        if "slices" in raw_group:
            slices = tuple((int(start), int(end)) for start, end in raw_group["slices"])
            widths = {end - start for start, end in slices}
            if len(widths) != 1:
                raise ValueError(f"All slices in entity group {name!r} must have the same width.")
            width = widths.pop()
        else:
            start, end = (int(value) for value in raw_group["slice"])
            count = int(raw_group.get("count", 1))
            if count <= 0:
                raise ValueError(f"Entity group {name!r} must have positive count.")
            width = int(raw_group.get("width", (end - start) // max(1, count)))
            if start + width * count != end:
                raise ValueError(
                    f"Entity group {name!r} slice [{start}, {end}) is not covered by "
                    f"count={count} and width={width}."
                )
            slices = tuple((start + width * index, start + width * (index + 1)) for index in range(count))

        if width <= 0:
            raise ValueError(f"Entity group {name!r} has non-positive width.")
        for start, end in slices:
            if start < 0 or end < start or end > int(obs_size):
                raise ValueError(f"Invalid slice [{start}, {end}) for observation size {obs_size}.")
            if (end - start) != width:
                raise ValueError(f"Slice [{start}, {end}) does not match width {width}.")
        if active_feature_index is not None:
            active_feature_index = int(active_feature_index)
            if active_feature_index < 0 or active_feature_index >= width:
                raise ValueError(f"Invalid active_feature_index for entity group {name!r}.")
        if inactive_values is not None and len(inactive_values) != width:
            raise ValueError(f"inactive_values for entity group {name!r} must match width {width}.")

        return EntityGroup(
            name=name,
            entity_type=entity_type,
            slices=slices,
            width=width,
            team_related=team_related,
            active_rule=active_rule,
            active_count_offset=active_count_offset,
            active_feature_index=active_feature_index,
            active_threshold=active_threshold,
            inactive_values=inactive_values,
        )

    def parse(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
        active_counts_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return entities, inactive masks, and team-related masks."""

        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape [B,T,A,D], got {tuple(obs.shape)}.")
        active_mask_bool = active_mask.to(dtype=torch.bool)
        if active_counts_override is None:
            active_counts = active_mask_bool.sum(dim=2)
        else:
            active_counts = active_counts_override.to(device=obs.device, dtype=torch.long)

        entity_tensors: list[torch.Tensor] = []
        inactive_masks: list[torch.Tensor] = []
        team_related_masks: list[torch.Tensor] = []
        type_offset = self.raw_feature_dim
        slot_offset = self.raw_feature_dim + len(self.entity_types)

        for group in self.groups:
            group_count = len(group.slices)
            type_index = self.type_to_index[group.entity_type]
            for slot_index, (start, end) in enumerate(group.slices):
                raw = obs[..., start:end]
                entity = torch.zeros(*raw.shape[:-1], self.entity_dim, dtype=obs.dtype, device=obs.device)
                entity[..., : group.width] = raw
                entity[..., type_offset + type_index] = 1.0
                if group_count > 1:
                    entity[..., slot_offset] = float(slot_index) / float(group_count - 1)
                entity_tensors.append(entity)

                present = self._entity_present(group, raw, active_mask_bool, active_counts, slot_index)
                inactive_masks.append(~present)
                team_related_masks.append(
                    torch.full_like(present, fill_value=bool(group.team_related), dtype=torch.bool)
                )

        entities = torch.stack(entity_tensors, dim=3)
        inactive = torch.stack(inactive_masks, dim=3)
        team_related = torch.stack(team_related_masks, dim=3)
        return entities, inactive, team_related

    def _entity_present(
        self,
        group: EntityGroup,
        raw: torch.Tensor,
        active_mask_bool: torch.Tensor,
        active_counts: torch.Tensor,
        slot_index: int,
    ) -> torch.Tensor:
        if group.active_rule == "always":
            present = torch.ones_like(active_mask_bool, dtype=torch.bool)
        elif group.active_rule == "first_n_agents_clamped":
            limit = torch.clamp(active_counts + int(group.active_count_offset), min=0, max=len(group.slices))
            present = slot_index < limit.unsqueeze(-1)
        elif group.active_rule == "feature_positive":
            if group.active_feature_index is None:
                raise ValueError(f"Group {group.name!r} needs active_feature_index.")
            present = raw[..., group.active_feature_index] > float(group.active_threshold)
        elif group.active_rule == "not_sentinel":
            if group.inactive_values is None:
                raise ValueError(f"Group {group.name!r} needs inactive_values.")
            sentinel = torch.as_tensor(group.inactive_values, dtype=raw.dtype, device=raw.device)
            present = ~torch.isclose(raw, sentinel, atol=1e-6, rtol=0.0).all(dim=-1)
        else:
            raise ValueError(f"Unsupported active_rule: {group.active_rule!r}.")
        return present & active_mask_bool


class EntityAttentionLayer(nn.Module):
    """Multi-head entity attention with explicit pre/post masks."""

    def __init__(self, in_dim: int, embed_dim: int, out_dim: int, n_heads: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.out_dim = int(out_dim)
        self.n_heads = int(n_heads)
        if self.embed_dim % self.n_heads != 0:
            raise AssertionError("Attention embedding dim must be divisible by attn_n_heads.")
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer("scale_factor", torch.scalar_tensor(float(self.head_dim)).sqrt())
        self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 3, bias=False)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self,
        entities: torch.Tensor,
        *,
        query_count: int,
        pre_mask: torch.Tensor,
        post_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_entities, _ = entities.shape
        query_count = int(query_count)
        query, key, value = self.in_trans(entities).chunk(3, dim=-1)
        query = query[:, :query_count]

        query = query.reshape(batch_size, query_count, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, num_entities, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        value = value.reshape(batch_size, num_entities, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        query = query.reshape(batch_size * self.n_heads, query_count, self.head_dim)
        key = key.reshape(batch_size * self.n_heads, self.head_dim, num_entities)
        value = value.reshape(batch_size * self.n_heads, num_entities, self.head_dim)

        logits = torch.bmm(query, key) / self.scale_factor
        mask = pre_mask[:, :query_count, :num_entities].repeat_interleave(self.n_heads, dim=0)
        logits = logits.masked_fill(mask.bool(), -float("inf"))
        weights = F.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        outputs = torch.bmm(weights, value)
        outputs = outputs.reshape(batch_size, self.n_heads, query_count, self.head_dim)
        outputs = outputs.permute(0, 2, 1, 3).reshape(batch_size, query_count, self.embed_dim)
        outputs = self.out_trans(outputs)
        if post_mask is not None:
            outputs = outputs.masked_fill(post_mask[:, :query_count].bool().unsqueeze(-1), 0.0)
        return outputs


class _EntityRNNBase(nn.Module):
    def __init__(self, entity_dim: int, rnn_hidden_dim: int, attn_embed_dim: int, attn_n_heads: int):
        super().__init__()
        self.entity_encoder = nn.Linear(int(entity_dim), int(attn_embed_dim))
        self.attn = EntityAttentionLayer(int(attn_embed_dim), int(attn_embed_dim), int(attn_embed_dim), int(attn_n_heads))
        self.fc = nn.Linear(int(attn_embed_dim), int(rnn_hidden_dim))
        self.rnn = nn.GRU(input_size=int(rnn_hidden_dim), hidden_size=int(rnn_hidden_dim), batch_first=True)

    def _features(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        extra_pre_mask: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_timesteps, num_agents, num_entities, entity_dim = entities.shape
        flat_entities = entities.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_agents * num_timesteps,
            num_entities,
            entity_dim,
        )
        flat_inactive = inactive_mask.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents * num_timesteps,
            num_entities,
        )
        pre_mask = flat_inactive.unsqueeze(1)
        if extra_pre_mask is not None:
            flat_extra = extra_pre_mask.permute(0, 2, 1, 3).reshape(
                batch_size * num_agents * num_timesteps,
                num_entities,
            )
            pre_mask = pre_mask | flat_extra.unsqueeze(1)
        post_mask = flat_inactive[:, :1]

        encoded = F.relu(self.entity_encoder(flat_entities))
        attended = self.attn(encoded, query_count=1, pre_mask=pre_mask, post_mask=post_mask).squeeze(1)
        recurrent_input = F.relu(self.fc(attended)).reshape(batch_size * num_agents, num_timesteps, -1)
        recurrent_features, next_hidden = self.rnn(recurrent_input, h0)
        recurrent_features = recurrent_features.reshape(batch_size, num_agents, num_timesteps, -1).permute(0, 2, 1, 3)
        return recurrent_features, next_hidden


class MIPIAgentRNN(_EntityRNNBase):
    """Entity-attention recurrent Q and stochastic-policy network."""

    def __init__(
        self,
        entity_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        attn_embed_dim: int,
        attn_n_heads: int,
    ):
        super().__init__(entity_dim, rnn_hidden_dim, attn_embed_dim, attn_n_heads)
        self.q_out = nn.Linear(int(rnn_hidden_dim), int(action_dim))
        self.policy_out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def forward(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        pre_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, next_hidden = self._features(entities, inactive_mask, pre_mask, h0)
        return self.q_out(features), self.policy_out(features), next_hidden


class AveragePolicyRNN(_EntityRNNBase):
    """Marginal policy over team-unrelated local entities."""

    def __init__(
        self,
        entity_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        attn_embed_dim: int,
        attn_n_heads: int,
    ):
        super().__init__(entity_dim, rnn_hidden_dim, attn_embed_dim, attn_n_heads)
        self.out = nn.Linear(int(rnn_hidden_dim), int(action_dim))

    def forward(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        pre_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extra_mask = team_related_mask if pre_mask is None else (team_related_mask | pre_mask)
        features, next_hidden = self._features(entities, inactive_mask, extra_mask, h0)
        return self.out(features), next_hidden


class AttentionHyperNet(nn.Module):
    """Entity-attention hypernetwork used by the FlexQMIX mixer."""

    def __init__(self, entity_dim: int, hypernet_embed: int, mixing_embed_dim: int, n_heads: int, mode: str):
        super().__init__()
        self.mode = str(mode)
        self.mixing_embed_dim = int(mixing_embed_dim)
        self.fc1 = nn.Linear(int(entity_dim), int(hypernet_embed))
        self.attn = EntityAttentionLayer(int(hypernet_embed), int(hypernet_embed), int(hypernet_embed), int(n_heads))
        self.fc2 = nn.Linear(int(hypernet_embed), int(mixing_embed_dim))

    def forward(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        num_agents: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :num_agents]
        if attn_mask is None:
            pre_mask = entity_mask.unsqueeze(1).expand(-1, int(num_agents), -1)
        else:
            pre_mask = attn_mask[:, :num_agents, :]
        attended = self.attn(encoded, query_count=int(num_agents), pre_mask=pre_mask, post_mask=agent_mask)
        outputs = self.fc2(attended)
        outputs = outputs.masked_fill(agent_mask.bool().unsqueeze(-1), 0.0)
        if self.mode == "vector":
            return outputs.mean(dim=1)
        if self.mode == "scalar":
            return outputs.mean(dim=(1, 2))
        return outputs


class FlexQMixer(nn.Module):
    """Entity-aware mixer matching the structure used by MIPI."""

    def __init__(
        self,
        *,
        num_agents: int,
        entity_dim: int,
        mixing_embed_dim: int,
        hypernet_embed: int,
        attn_n_heads: int,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.embed_dim = int(mixing_embed_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.hyper_w_1 = AttentionHyperNet(entity_dim, hypernet_embed, mixing_embed_dim, attn_n_heads, "matrix")
        self.hyper_w_final = AttentionHyperNet(entity_dim, hypernet_embed, mixing_embed_dim, attn_n_heads, "vector")
        self.hyper_b_1 = AttentionHyperNet(entity_dim, hypernet_embed, mixing_embed_dim, attn_n_heads, "vector")
        self.V = AttentionHyperNet(entity_dim, hypernet_embed, mixing_embed_dim, attn_n_heads, "scalar")

    def _chunk_ranges(self, total_rows: int):
        for start in range(0, int(total_rows), self.chunk_size):
            yield start, min(start + self.chunk_size, int(total_rows))

    def _forward_flat_chunk(
        self,
        flat_qs: torch.Tensor,
        flat_entities: torch.Tensor,
        flat_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = flat_entities.shape[0]
        w1 = self.hyper_w_1(flat_entities, flat_mask, num_agents=self.num_agents)
        b1 = self.hyper_b_1(flat_entities, flat_mask, num_agents=self.num_agents)
        w_final = self.hyper_w_final(flat_entities, flat_mask, num_agents=self.num_agents)
        v = self.V(flat_entities, flat_mask, num_agents=self.num_agents)

        w1 = F.softmax(w1.reshape(num_rows, self.num_agents, self.embed_dim), dim=-1)
        b1 = b1.reshape(num_rows, 1, self.embed_dim)
        w_final = F.softmax(w_final.reshape(num_rows, self.embed_dim, 1), dim=-2)
        v = v.reshape(num_rows, 1, 1)

        hidden = torch.bmm(flat_qs, w1) + b1
        return (torch.bmm(hidden, w_final) + v).reshape(num_rows)

    def forward(self, agent_qs: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_timesteps, num_entities, entity_dim = entities.shape
        flat_entities = entities.reshape(batch_size * num_timesteps, num_entities, entity_dim)
        flat_mask = entity_mask.reshape(batch_size * num_timesteps, num_entities)
        flat_qs = agent_qs.reshape(batch_size * num_timesteps, 1, self.num_agents)
        chunks = [
            self._forward_flat_chunk(flat_qs[start:end], flat_entities[start:end], flat_mask[start:end])
            for start, end in self._chunk_ranges(flat_entities.shape[0])
        ]
        return torch.cat(chunks, dim=0).reshape(batch_size, num_timesteps)

    def _forward_img_flat_chunk(
        self,
        flat_qs: torch.Tensor,
        flat_entities: torch.Tensor,
        flat_mask: torch.Tensor,
        within_mask: torch.Tensor,
        interact_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = flat_entities.shape[0]
        w1_within = self.hyper_w_1(
            flat_entities,
            flat_mask,
            num_agents=self.num_agents,
            attn_mask=within_mask,
        )
        w1_interact = self.hyper_w_1(
            flat_entities,
            flat_mask,
            num_agents=self.num_agents,
            attn_mask=interact_mask,
        )
        w1 = torch.cat([w1_within, w1_interact], dim=1)
        b1 = self.hyper_b_1(flat_entities, flat_mask, num_agents=self.num_agents)
        w_final = self.hyper_w_final(flat_entities, flat_mask, num_agents=self.num_agents)
        v = self.V(flat_entities, flat_mask, num_agents=self.num_agents)

        w1 = F.softmax(w1.reshape(num_rows, self.num_agents * 2, self.embed_dim), dim=-1)
        b1 = b1.reshape(num_rows, 1, self.embed_dim)
        w_final = F.softmax(w_final.reshape(num_rows, self.embed_dim, 1), dim=-2)
        v = v.reshape(num_rows, 1, 1)

        hidden = torch.bmm(flat_qs, w1) + b1
        return (torch.bmm(hidden, w_final) + v).reshape(num_rows)

    def forward_img(
        self,
        agent_qs: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        imagine_groups: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, num_timesteps, num_entities, entity_dim = entities.shape
        flat_entities = entities.reshape(batch_size * num_timesteps, num_entities, entity_dim)
        flat_mask = entity_mask.reshape(batch_size * num_timesteps, num_entities)
        flat_qs = agent_qs.reshape(batch_size * num_timesteps, 1, self.num_agents * 2)
        within_mask, interact_mask = imagine_groups
        row_indices = torch.arange(batch_size * num_timesteps, device=entities.device)
        chunks = []
        for start, end in self._chunk_ranges(flat_entities.shape[0]):
            chunk_rows = row_indices[start:end]
            batch_indices = torch.div(chunk_rows, num_timesteps, rounding_mode="floor")
            time_indices = chunk_rows.remainder(num_timesteps)
            chunks.append(
                self._forward_img_flat_chunk(
                    flat_qs[start:end],
                    flat_entities[start:end],
                    flat_mask[start:end],
                    within_mask[batch_indices, time_indices],
                    interact_mask[batch_indices, time_indices],
                )
            )
        return torch.cat(chunks, dim=0).reshape(batch_size, num_timesteps)


class MIPI(ParallelLearner):
    """Entity-based MIPI with the shared benchmark API."""

    @staticmethod
    def normalize_config(config: dict) -> dict:
        return normalize_config(config, MIPI_DEFAULT_CONFIG)

    def __init__(self, env_spec: ParallelEnvSpec, config: dict, device: str = "cpu"):
        super().__init__(env_spec=env_spec, config=self.normalize_config(config), device=device)
        config = self.config
        if not bool(config["share_parameters"]):
            raise ValueError("MIPI assumes homogeneous shared-parameter agents.")

        self.schema = EntitySchema(config["entity_schema"], obs_size=self.obs_size)
        self.batch_size = int(config["batch_size"])
        self.num_epochs = int(config["num_epochs"])
        self.gamma = float(config["gamma"])
        self.target_update_every = max(1, int(config["target_update_every"]))
        self.tau = float(config["tau"])
        self.max_grad_norm = float(config["max_grad_norm"]) if config["max_grad_norm"] is not None else None
        self.learning_starts = int(config["learning_starts"])
        self.learn_every_steps = int(config["learn_every_steps"])
        self.ba_iters = max(1, int(config["ba_iters"]))
        self.lmbda = float(config["lmbda"])
        if not 0.0 <= self.lmbda <= 1.0:
            raise ValueError("MIPI lmbda must be in [0, 1].")
        self.mi_alpha_start = float(config["mi_alpha_start"])
        self.mi_alpha_end = float(config["mi_alpha_end"])
        self.mi_alpha_anneal_steps = max(1, int(config["mi_alpha_anneal_steps"]))
        self.epsilon_start = float(config["epsilon_start"])
        self.epsilon_finish = float(config["epsilon_finish"])
        self.epsilon_anneal_steps = max(1, int(config["epsilon_anneal_steps"]))
        self.test_greedy = bool(config["test_greedy"])
        self._learn_steps = 0
        self._action_steps = 0

        self.agent_net = MIPIAgentRNN(
            self.schema.entity_dim,
            self.action_space_size,
            int(config["rnn_hidden_dim"]),
            int(config["attn_embed_dim"]),
            int(config["attn_n_heads"]),
        ).to(self.device)
        self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
        self.avg_policy_net = AveragePolicyRNN(
            self.schema.entity_dim,
            self.action_space_size,
            int(config["rnn_hidden_dim"]),
            int(config["attn_embed_dim"]),
            int(config["attn_n_heads"]),
        ).to(self.device)
        self.target_agent_net.eval()

        self.mixing_net = FlexQMixer(
            num_agents=self.max_agents,
            entity_dim=self.schema.entity_dim,
            mixing_embed_dim=int(config["mixing_embed_dim"]),
            hypernet_embed=int(config["hypernet_embed"]),
            attn_n_heads=int(config["attn_n_heads"]),
            chunk_size=int(config["mixer_chunk_size"]),
        ).to(self.device)
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)
        self.target_mixing_net.eval()

        self.optimizer = optim.RMSprop(
            list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
            lr=float(config["lr"]),
            alpha=float(config["optim_alpha"]),
            eps=float(config["optim_eps"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.avg_policy_optimizer = optim.RMSprop(
            list(self.avg_policy_net.parameters()),
            lr=float(config["lr"]),
            alpha=float(config["optim_alpha"]),
            eps=float(config["optim_eps"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.memory = deque(maxlen=int(config["buffer_size"]))
        self._episode_steps: list[dict] = []
        self._inference_hidden: dict[object, torch.Tensor] = {}
        self._agent_slot_map: dict[object, int] = {}

    def reset_episode(self) -> None:
        self._inference_hidden = {}
        self._agent_slot_map = {}

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self.agent_net.eval()
        self.target_agent_net.eval()
        self.avg_policy_net.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.agent_net.train()
        self.avg_policy_net.train()
        self.mixing_net.train()

    def _get_hidden_state(self, agent_key: object) -> torch.Tensor:
        hidden_state = self._inference_hidden.get(agent_key)
        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, self.agent_net.rnn.hidden_size, device=self.device)
        return hidden_state

    def _set_hidden_state(self, agent_key: object, hidden_state: torch.Tensor) -> None:
        self._inference_hidden[agent_key] = hidden_state.detach()

    def _actor_key(self, agent_id: object) -> object:
        slot = self._agent_slot_map.get(agent_id)
        if slot is None:
            slot = len(self._agent_slot_map)
            if slot >= self.max_agents:
                raise ValueError(f"Received more than {self.max_agents} agent ids in one episode.")
            self._agent_slot_map[agent_id] = slot
        return agent_id

    def _epsilon(self) -> float:
        progress = min(1.0, max(0.0, float(self._action_steps) / float(self.epsilon_anneal_steps)))
        return float(self.epsilon_start + progress * (self.epsilon_finish - self.epsilon_start))

    def _policy_action(self, policy_logits: torch.Tensor) -> int:
        if self._eval_mode and self.test_greedy:
            return int(torch.argmax(policy_logits).item())
        policy = torch.softmax(policy_logits, dim=-1)
        if not self._eval_mode:
            epsilon = self._epsilon()
            if random.random() < epsilon:
                return random.randrange(self.action_space_size)
        return int(torch.distributions.Categorical(probs=policy).sample().item())

    def _act_one(self, obs: np.ndarray, agent_key: object, active_count: int) -> int:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).view(1, 1, 1, -1)
        active_mask = torch.ones(1, 1, 1, dtype=torch.bool, device=self.device)
        active_counts = torch.full((1, 1), int(active_count), dtype=torch.long, device=self.device)
        entities, inactive_mask, _ = self.schema.parse(obs_tensor, active_mask, active_counts)
        _, policy_logits, hidden_state = self.agent_net(
            entities,
            inactive_mask,
            self._get_hidden_state(agent_key),
        )
        self._set_hidden_state(agent_key, hidden_state)
        del active_count
        return self._policy_action(policy_logits.squeeze(0).squeeze(0).squeeze(0))

    def act(self, state: np.ndarray, agent_index: Optional[object] = None) -> int:
        if agent_index is None:
            agent_index = 0
        return self._act_one(state, agent_index, active_count=1)

    def act_parallel(self, obs_dict: dict[object, np.ndarray]) -> dict[object, int]:
        actions_by_agent_id: dict[object, int] = {}
        active_count = len(obs_dict)
        for agent_id in _sorted_agent_ids(obs_dict.keys()):
            actions_by_agent_id[agent_id] = self._act_one(
                obs_dict[agent_id],
                self._actor_key(agent_id),
                active_count=active_count,
            )
        if not self._eval_mode:
            self._action_steps += 1
        return actions_by_agent_id

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

        self._episode_steps.append(
            {
                "obs": obs_batch,
                "actions": actions_batch,
                "rewards": rewards_batch,
                "active_mask": active_mask,
                "next_obs": next_obs_batch,
                "next_active_mask": next_active_mask,
                "done": resolve_parallel_done(transition.done_dict),
            }
        )
        if self._episode_steps[-1]["done"]:
            self.memory.append(self._finalize_episode(self._episode_steps))
            self._episode_steps = []

    def _finalize_episode(self, steps: list[dict]) -> dict:
        return {
            "obs": np.stack([step["obs"] for step in steps], axis=0),
            "actions": np.stack([step["actions"] for step in steps], axis=0),
            "rewards": np.stack([step["rewards"] for step in steps], axis=0),
            "active_mask": np.stack([step["active_mask"] for step in steps], axis=0),
            "next_obs": np.stack([step["next_obs"] for step in steps], axis=0),
            "next_active_mask": np.stack([step["next_active_mask"] for step in steps], axis=0),
            "done": np.asarray([step["done"] for step in steps], dtype=np.float32),
            "T": int(len(steps)),
        }

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
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
            return
        with torch.no_grad():
            for target_param, online_param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)
            for target_param, online_param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def _parse_batch(
        self,
        obs: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entities, inactive_mask, team_related_mask = self.schema.parse(obs, active_mask)
        central_entities, central_mask = self._central_entities(entities, inactive_mask, active_mask)
        return entities, inactive_mask, team_related_mask, central_entities, central_mask

    def _central_entities(
        self,
        local_entities: torch.Tensor,
        local_inactive_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_bool = active_mask.to(dtype=torch.bool)
        self_entities = local_entities[:, :, :, 0, :]
        self_inactive = local_inactive_mask[:, :, :, 0] | (~active_bool)
        context_entities = local_entities[:, :, :, 1:, :].reshape(
            local_entities.shape[0],
            local_entities.shape[1],
            self.max_agents * (self.schema.entities_per_observation - 1),
            self.schema.entity_dim,
        )
        context_inactive = (
            local_inactive_mask[:, :, :, 1:] | (~active_bool).unsqueeze(-1)
        ).reshape(local_entities.shape[0], local_entities.shape[1], -1)
        return torch.cat([self_entities, context_entities], dim=2), torch.cat([self_inactive, context_inactive], dim=2)

    def _agent_outputs(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
        network: MIPIAgentRNN,
        avg_network: Optional[AveragePolicyRNN] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_values, policy_logits, _ = network(entities, inactive_mask, None)
        avg_logits = None
        if avg_network is not None:
            avg_logits, _ = avg_network(entities, inactive_mask, team_related_mask, None)
        return q_values, policy_logits, avg_logits

    def _make_local_imagine_masks(self, inactive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_agents, num_entities = inactive_mask.shape
        group = torch.rand(batch_size, num_agents, num_entities, device=inactive_mask.device) < 0.5
        query_group = group[:, :, :1].unsqueeze(1)
        group = group.unsqueeze(1)
        inactive = inactive_mask
        within = inactive | (group != query_group)
        interact = inactive | (group == query_group)
        return within, interact

    def _make_central_imagine_masks(self, entity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_entities = entity_mask.shape
        group = torch.rand(batch_size, 1, num_entities, device=entity_mask.device) < 0.5
        row_group = group.unsqueeze(-1)
        col_group = group.unsqueeze(-2)
        inactive_pair = entity_mask.unsqueeze(-1) | entity_mask.unsqueeze(-2)
        same_group = row_group == col_group
        within = inactive_pair | (~same_group)
        interact = inactive_pair | same_group
        return (
            within.expand(-1, entity_mask.shape[1], -1, -1),
            interact.expand(-1, entity_mask.shape[1], -1, -1),
        )

    def _agent_outputs_img(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        within_mask, interact_mask = self._make_local_imagine_masks(inactive_mask)
        repeated_entities = entities.repeat(2, 1, 1, 1, 1)
        repeated_inactive = inactive_mask.repeat(2, 1, 1, 1)
        repeated_team_related = team_related_mask.repeat(2, 1, 1, 1)
        extra_mask = torch.cat([within_mask, interact_mask], dim=0)
        q_values, policy_logits, _ = self.agent_net(repeated_entities, repeated_inactive, None, extra_mask)
        avg_logits, _ = self.avg_policy_net(repeated_entities, repeated_inactive, repeated_team_related, None, extra_mask)
        return q_values, policy_logits, avg_logits

    def _mix_q_tot(self, chosen_q: torch.Tensor, central_entities: torch.Tensor, central_mask: torch.Tensor) -> torch.Tensor:
        return self.mixing_net(chosen_q, central_entities, central_mask)

    def _mi_alpha(self, global_step: int) -> float:
        progress = min(1.0, max(0.0, float(global_step) / float(self.mi_alpha_anneal_steps)))
        return float(self.mi_alpha_start + progress * (self.mi_alpha_end - self.mi_alpha_start))

    def _run_update(self, global_step: int) -> UpdateReport:
        memory_items = len(self.memory)
        td_losses: list[float] = []
        img_td_losses: list[float] = []
        pi_losses: list[float] = []
        img_pi_losses: list[float] = []
        pi_avg_losses: list[float] = []
        img_pi_avg_losses: list[float] = []
        mi_means: list[float] = []
        q_means: list[float] = []
        target_means: list[float] = []
        grad_norms: list[float] = []
        avg_grad_norms: list[float] = []
        total_samples_seen = 0
        alpha = self._mi_alpha(global_step)

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            tensors = self._batch_to_tensors(batch)
            obs, actions, rewards, active_mask, next_obs, next_active_mask, dones, time_mask = tensors
            total_samples_seen += int(time_mask.sum().item() * self.max_agents)

            parsed = self._parse_batch(obs, active_mask)
            entities, inactive_mask, team_related_mask, central_entities, central_mask = parsed
            next_parsed = self._parse_batch(next_obs, next_active_mask)
            next_entities, next_inactive, next_team_related, next_central_entities, next_central_mask = next_parsed

            critic_metrics = self._update_critic(
                actions,
                rewards,
                active_mask,
                next_active_mask,
                dones,
                time_mask,
                entities,
                inactive_mask,
                team_related_mask,
                central_entities,
                central_mask,
                next_entities,
                next_inactive,
                next_team_related,
                next_central_entities,
                next_central_mask,
                alpha,
            )
            td_losses.append(critic_metrics["td_loss"])
            img_td_losses.append(critic_metrics["img_td_loss"])
            q_means.append(critic_metrics["q_mean"])
            target_means.append(critic_metrics["target_mean"])
            grad_norms.append(critic_metrics["grad_norm"])

            for _ in range(self.ba_iters):
                avg_metrics = self._update_average_policy(entities, inactive_mask, team_related_mask, active_mask, time_mask)
                policy_metrics = self._update_policy(entities, inactive_mask, team_related_mask, active_mask, time_mask, alpha)
                pi_avg_losses.append(avg_metrics["pi_avg_loss"])
                img_pi_avg_losses.append(avg_metrics["img_pi_avg_loss"])
                avg_grad_norms.append(avg_metrics["grad_norm"])
                pi_losses.append(policy_metrics["pi_loss"])
                img_pi_losses.append(policy_metrics["img_pi_loss"])
                mi_means.append(policy_metrics["mi_mean"])
                grad_norms.append(policy_metrics["grad_norm"])

            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()

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
            exploration_temperature=float(self._epsilon()),
            extras={
                "td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
                "img_td_loss": float(np.mean(img_td_losses)) if img_td_losses else 0.0,
                "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
                "img_pi_loss": float(np.mean(img_pi_losses)) if img_pi_losses else 0.0,
                "pi_avg_loss": float(np.mean(pi_avg_losses)) if pi_avg_losses else 0.0,
                "img_pi_avg_loss": float(np.mean(img_pi_avg_losses)) if img_pi_avg_losses else 0.0,
                "mi_log_ratio": float(np.mean(mi_means)) if mi_means else 0.0,
                "mi_alpha": float(alpha),
                "lmbda": float(self.lmbda),
                "ba_iters": float(self.ba_iters),
                "q_mean": float(np.mean(q_means)) if q_means else 0.0,
                "target_mean": float(np.mean(target_means)) if target_means else 0.0,
                "pi_avg_grad_norm": float(np.mean(avg_grad_norms)) if avg_grad_norms else 0.0,
            },
        )
        return self._append_update_report(report)

    def _batch_to_tensors(self, batch: list[dict]) -> tuple[torch.Tensor, ...]:
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
        next_obs = torch.as_tensor(np.stack([pad_time(ep["next_obs"]) for ep in batch]), device=self.device)
        next_active_mask = torch.as_tensor(np.stack([pad_time(ep["next_active_mask"]) for ep in batch]), device=self.device)
        dones = torch.as_tensor(
            np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
            device=self.device,
            dtype=torch.float32,
        ).squeeze(-1)
        lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
        time_mask = (torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.float32)
        return obs, actions, rewards, active_mask, next_obs, next_active_mask, dones, time_mask

    def _update_critic(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        active_mask: torch.Tensor,
        next_active_mask: torch.Tensor,
        dones: torch.Tensor,
        time_mask: torch.Tensor,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
        central_entities: torch.Tensor,
        central_mask: torch.Tensor,
        next_entities: torch.Tensor,
        next_inactive: torch.Tensor,
        next_team_related: torch.Tensor,
        next_central_entities: torch.Tensor,
        next_central_mask: torch.Tensor,
        alpha: float,
    ) -> dict[str, float]:
        q_all, _, _ = self._agent_outputs(entities, inactive_mask, team_related_mask, self.agent_net)
        safe_actions = actions.clone()
        safe_actions[active_mask == 0] = 0
        chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1) * active_mask
        q_tot = self._mix_q_tot(chosen_q, central_entities, central_mask)

        active_counts = active_mask.sum(dim=2).clamp(min=1.0)
        team_rewards = (rewards * active_mask).sum(dim=2) / active_counts

        with torch.no_grad():
            next_q_target, _, _ = self._agent_outputs(
                next_entities,
                next_inactive,
                next_team_related,
                self.target_agent_net,
            )
            _, next_policy_logits, next_avg_logits = self._agent_outputs(
                next_entities,
                next_inactive,
                next_team_related,
                self.agent_net,
                self.avg_policy_net,
            )
            next_policy = torch.softmax(next_policy_logits, dim=-1)
            sampled_next_actions = torch.distributions.Categorical(
                probs=next_policy.reshape(-1, self.action_space_size)
            ).sample()
            sampled_next_actions = sampled_next_actions.reshape(next_policy.shape[:-1])
            safe_next_actions = sampled_next_actions.clone()
            safe_next_actions[next_active_mask == 0] = 0
            next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
            next_chosen_q = next_chosen_q * next_active_mask
            q_tot_next = self.target_mixing_net(next_chosen_q, next_central_entities, next_central_mask)

            log_policy = torch.log_softmax(next_policy_logits, dim=-1)
            log_avg_policy = torch.log_softmax(next_avg_logits, dim=-1)
            log_policy_taken = torch.gather(log_policy, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
            log_avg_taken = torch.gather(log_avg_policy, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
            next_counts = next_active_mask.sum(dim=2).clamp(min=1.0)
            mi_penalty = ((log_policy_taken - log_avg_taken) * next_active_mask).sum(dim=2) / next_counts
            targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next - float(alpha) * mi_penalty

        td_error = q_tot - targets
        critic_loss = ((td_error ** 2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)

        img_qvals, _, _ = self._agent_outputs_img(entities, inactive_mask, team_related_mask)
        rep_actions = actions.repeat(2, 1, 1)
        rep_active = active_mask.repeat(2, 1, 1)
        rep_actions = rep_actions.clone()
        rep_actions[rep_active == 0] = 0
        img_chosen_q = torch.gather(img_qvals, 3, rep_actions.unsqueeze(-1)).squeeze(-1) * rep_active
        within_q, interact_q = img_chosen_q.chunk(2, dim=0)
        img_agent_qs = torch.cat([within_q, interact_q], dim=2)
        img_groups = self._make_central_imagine_masks(central_mask)
        img_q_tot = self.mixing_net.forward_img(img_agent_qs, central_entities, central_mask, img_groups)
        img_td_error = img_q_tot - targets.detach()
        img_loss = ((img_td_error ** 2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)

        loss = (self.lmbda * critic_loss) + ((1.0 - self.lmbda) * img_loss)
        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.agent_net.parameters()) + list(self.mixing_net.parameters())
        grad_norm = _parameter_grad_norm(parameters)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, max_norm=self.max_grad_norm)
        self.optimizer.step()

        return {
            "td_loss": float(critic_loss.detach().item()),
            "img_td_loss": float(img_loss.detach().item()),
            "q_mean": float((q_tot.detach() * time_mask).sum().item() / time_mask.sum().clamp(min=1.0).item()),
            "target_mean": float((targets.detach() * time_mask).sum().item() / time_mask.sum().clamp(min=1.0).item()),
            "grad_norm": float(grad_norm),
        }

    def _update_average_policy(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
        active_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            _, policy_logits, _ = self._agent_outputs(entities, inactive_mask, team_related_mask, self.agent_net)
            target_policy = torch.softmax(policy_logits, dim=-1)
        avg_logits, _ = self.avg_policy_net(entities, inactive_mask, team_related_mask, None)
        log_avg_policy = torch.log_softmax(avg_logits, dim=-1)
        agent_time_mask = time_mask.unsqueeze(-1) * active_mask
        pi_avg_loss = (-(target_policy * log_avg_policy).sum(dim=-1) * agent_time_mask).sum()
        pi_avg_loss = pi_avg_loss / agent_time_mask.sum().clamp(min=1.0)

        _, img_policy_logits, img_avg_logits = self._agent_outputs_img(entities, inactive_mask, team_related_mask)
        img_target_policy = torch.softmax(img_policy_logits.detach(), dim=-1)
        img_log_avg = torch.log_softmax(img_avg_logits, dim=-1)
        img_agent_time_mask = agent_time_mask.repeat(2, 1, 1)
        img_pi_avg_loss = (-(img_target_policy * img_log_avg).sum(dim=-1) * img_agent_time_mask).sum()
        img_pi_avg_loss = img_pi_avg_loss / img_agent_time_mask.sum().clamp(min=1.0)

        loss = (self.lmbda / float(self.ba_iters)) * pi_avg_loss + (1.0 - self.lmbda) * img_pi_avg_loss
        self.avg_policy_optimizer.zero_grad()
        loss.backward()
        parameters = list(self.avg_policy_net.parameters())
        grad_norm = _parameter_grad_norm(parameters)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, max_norm=self.max_grad_norm)
        self.avg_policy_optimizer.step()
        return {
            "pi_avg_loss": float(pi_avg_loss.detach().item()),
            "img_pi_avg_loss": float(img_pi_avg_loss.detach().item()),
            "grad_norm": float(grad_norm),
        }

    def _update_policy(
        self,
        entities: torch.Tensor,
        inactive_mask: torch.Tensor,
        team_related_mask: torch.Tensor,
        active_mask: torch.Tensor,
        time_mask: torch.Tensor,
        alpha: float,
    ) -> dict[str, float]:
        q_all, policy_logits, avg_logits = self._agent_outputs(
            entities,
            inactive_mask,
            team_related_mask,
            self.agent_net,
            self.avg_policy_net,
        )
        policy = torch.softmax(policy_logits, dim=-1)
        log_policy = torch.log_softmax(policy_logits, dim=-1)
        with torch.no_grad():
            detached_q = q_all.detach()
            detached_log_avg = torch.log_softmax(avg_logits, dim=-1)
        log_ratio = log_policy - detached_log_avg
        actor_terms = policy * (float(alpha) * log_ratio - detached_q)
        agent_time_mask = time_mask.unsqueeze(-1) * active_mask
        pi_loss = (actor_terms.sum(dim=-1) * agent_time_mask).sum() / agent_time_mask.sum().clamp(min=1.0)

        img_q_vals, img_policy_logits, img_avg_logits = self._agent_outputs_img(entities, inactive_mask, team_related_mask)
        img_policy = torch.softmax(img_policy_logits, dim=-1)
        img_log_policy = torch.log_softmax(img_policy_logits, dim=-1)
        with torch.no_grad():
            img_detached_q = img_q_vals.detach()
            img_detached_log_avg = torch.log_softmax(img_avg_logits, dim=-1)
        img_log_ratio = img_log_policy - img_detached_log_avg
        img_actor_terms = img_policy * (float(alpha) * img_log_ratio - img_detached_q)
        img_agent_time_mask = agent_time_mask.repeat(2, 1, 1)
        img_pi_loss = (img_actor_terms.sum(dim=-1) * img_agent_time_mask).sum()
        img_pi_loss = img_pi_loss / img_agent_time_mask.sum().clamp(min=1.0)

        loss = self.lmbda * pi_loss + (1.0 - self.lmbda) * img_pi_loss
        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.agent_net.parameters()) + list(self.mixing_net.parameters())
        grad_norm = _parameter_grad_norm(parameters)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, max_norm=self.max_grad_norm)
        self.optimizer.step()

        mi_mean = ((policy * log_ratio).sum(dim=-1) * agent_time_mask).sum()
        mi_mean = mi_mean / agent_time_mask.sum().clamp(min=1.0)
        return {
            "pi_loss": float(pi_loss.detach().item()),
            "img_pi_loss": float(img_pi_loss.detach().item()),
            "mi_mean": float(mi_mean.detach().item()),
            "grad_norm": float(grad_norm),
        }

    def _checkpoint_state(self) -> dict:
        return {
            "epsilon_action_steps": int(self._action_steps),
            "learn_steps": int(self._learn_steps),
            "agent_state_dict": self.agent_net.state_dict(),
            "target_agent_state_dict": self.target_agent_net.state_dict(),
            "avg_policy_state_dict": self.avg_policy_net.state_dict(),
            "mixing_state_dict": self.mixing_net.state_dict(),
            "target_mixing_state_dict": self.target_mixing_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "avg_policy_optimizer_state_dict": self.avg_policy_optimizer.state_dict(),
        }

    def _load_checkpoint_state(self, checkpoint_state: dict) -> None:
        self.agent_net.load_state_dict(checkpoint_state["agent_state_dict"])
        self.target_agent_net.load_state_dict(checkpoint_state["target_agent_state_dict"])
        self.avg_policy_net.load_state_dict(checkpoint_state["avg_policy_state_dict"])
        self.mixing_net.load_state_dict(checkpoint_state["mixing_state_dict"])
        self.target_mixing_net.load_state_dict(checkpoint_state["target_mixing_state_dict"])
        optimizer_state = checkpoint_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        avg_optimizer_state = checkpoint_state.get("avg_policy_optimizer_state_dict")
        if avg_optimizer_state is not None:
            self.avg_policy_optimizer.load_state_dict(avg_optimizer_state)
        self._action_steps = int(checkpoint_state.get("epsilon_action_steps", self._action_steps))
        self._learn_steps = int(checkpoint_state.get("learn_steps", self._learn_steps))
