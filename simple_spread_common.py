"""Shared helpers for the simple-spread task family."""

from __future__ import annotations

import numpy as np


def _is_collision(agent_a, agent_b) -> bool:
    delta = np.asarray(agent_a.state.p_pos, dtype=np.float32) - np.asarray(agent_b.state.p_pos, dtype=np.float32)
    distance = float(np.sqrt(np.sum(np.square(delta))))
    return bool(distance < (float(agent_a.size) + float(agent_b.size)))


def count_collision_pairs(world) -> int:
    """Count unordered colliding agent pairs once."""
    collisions = 0
    agents = list(world.agents)
    for index, agent in enumerate(agents):
        if not bool(getattr(agent, "collide", True)):
            continue
        for other in agents[index + 1 :]:
            if not bool(getattr(other, "collide", True)):
                continue
            if _is_collision(agent, other):
                collisions += 1
    return int(collisions)


def compute_cooperative_team_reward(world, *, collision_coef: float = 1.0) -> float:
    """Combine global landmark coverage with one shared collision penalty."""
    reward = 0.0
    for landmark in world.landmarks:
        distances = [
            float(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.asarray(agent.state.p_pos, dtype=np.float32)
                            - np.asarray(landmark.state.p_pos, dtype=np.float32)
                        )
                    )
                )
            )
            for agent in world.agents
        ]
        reward -= min(distances)
    reward -= float(collision_coef) * float(count_collision_pairs(world))
    return float(reward)


class CooperativeSimpleSpreadRewardWrapper:
    """Broadcast one shared team reward for the simple-spread family."""

    def __init__(self, env, *, collision_coef: float = 1.0):
        self.env = env
        self.collision_coef = float(collision_coef)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Private attribute access is not allowed: {name}")
        return getattr(self.env, name)

    def reset(self, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        observations, _, terminations, truncations, infos = self.env.step(actions)
        shared_reward = compute_cooperative_team_reward(self.env.unwrapped.world, collision_coef=self.collision_coef)
        rewards = {agent_id: shared_reward for agent_id in self.env.possible_agents}
        return observations, rewards, terminations, truncations, infos

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
