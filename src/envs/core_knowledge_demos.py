"""Procedural core-knowledge demonstration generator (open-gap A#3, P1 recipe).

Generates interaction trajectories that *embody* Spelke-style core knowledge,
to be seeded into the bounded replay buffer via BoundedReplayBuffer.prefill
(see ROADMAP Step 3 / docs/path-to-northstar.md P1).

Three priors are covered:
  - object permanence: agent keeps approaching an object.
  - intuitive physics: agent pushes objects along the force direction.
  - number sense: agent visits a small set of distinct objects one-by-one.

These are demonstrations, not explored data: constructed to be physically
consistent so the agent can learn the prior from them.

Bounded: generation count is controlled by the caller (n_episodes / steps);
no unbounded buffers are created.
"""

from __future__ import annotations

import numpy as np

from src.envs.physics_sandbox import PhysicsSandbox
from src.memory.bounded_replay import BoundedReplayBuffer, Transition


def _make_transition(env: PhysicsSandbox, action: int, prev_obs: np.ndarray) -> Transition:
    step = env.step(action)
    return Transition(
        obs=prev_obs,
        action=action,
        reward=float(step.reward),
        next_obs=step.obs,
        done=step.terminated or step.truncated,
    )


def gen_object_permanence(n_episodes: int = 4, steps_per_ep: int = 20) -> list[Transition]:
    """Agent repeatedly approaches a single object — embodies 'objects persist'."""
    out: list[Transition] = []
    for _ in range(n_episodes):
        env = PhysicsSandbox(num_objects=1, gravity=-2.0, action_force=60.0,
                             max_episode_steps=steps_per_ep)
        obs = env.reset()
        for _ in range(steps_per_ep):
            agent = env._agent
            obj = env._objects[0]
            dx = obj.x - agent.x
            dy = obj.y - agent.y
            if abs(dx) > abs(dy):
                action = 3 if dx > 0 else 2
            else:
                action = 1 if dy > 0 else 0
            out.append(_make_transition(env, action, obs))
            obs = out[-1].next_obs
    return out


def gen_intuitive_physics(n_episodes: int = 4, steps_per_ep: int = 20) -> list[Transition]:
    """Agent pushes objects along the applied-force direction (cause to effect)."""
    out: list[Transition] = []
    for _ in range(n_episodes):
        env = PhysicsSandbox(num_objects=3, gravity=-6.0, action_force=55.0,
                            max_episode_steps=steps_per_ep)
        obs = env.reset()
        for _ in range(steps_per_ep):
            agent = env._agent
            obj = min(env._objects, key=lambda o: (o.x - agent.x) ** 2 + (o.y - agent.y) ** 2)
            dx = obj.x - agent.x
            dy = obj.y - agent.y
            if abs(dx) > abs(dy):
                action = 3 if dx > 0 else 2
            else:
                action = 1 if dy > 0 else 0
            out.append(_make_transition(env, action, obs))
            obs = out[-1].next_obs
    return out


def gen_number_sense(n_episodes: int = 4, steps_per_ep: int = 15) -> list[Transition]:
    """Agent visits a small set of distinct objects one-by-one (embedded counting)."""
    out: list[Transition] = []
    for n in (2, 3, 4):
        for _ in range(n_episodes):
            env = PhysicsSandbox(num_objects=n, gravity=-4.0, action_force=50.0,
                                max_episode_steps=steps_per_ep)
            obs = env.reset()
            for _ in range(steps_per_ep):
                agent = env._agent
                obj = env._objects[_ % len(env._objects)]
                dx = obj.x - agent.x
                dy = obj.y - agent.y
                if abs(dx) > abs(dy):
                    action = 3 if dx > 0 else 2
                else:
                    action = 1 if dy > 0 else 0
                out.append(_make_transition(env, action, obs))
                obs = out[-1].next_obs
    return out


def generate_all(demos_per_prior: int = 4) -> list[Transition]:
    """Generate all three core-knowledge demo sets."""
    out: list[Transition] = []
    out += gen_object_permanence(demos_per_prior)
    out += gen_intuitive_physics(demos_per_prior)
    out += gen_number_sense(demos_per_prior)
    return out


def seed_into(buffer: BoundedReplayBuffer, demos_per_prior: int = 4) -> int:
    """Convenience: generate demos and prefill into the buffer. Returns count."""
    demos = generate_all(demos_per_prior)
    return buffer.prefill(demos)
