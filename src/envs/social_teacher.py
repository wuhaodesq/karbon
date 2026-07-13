"""Social Teacher Environment — Demonstrator + Correction for Imitation Learning.

Wraps the PhysicsSandbox with a second agent (the "teacher") that can:
1. Demonstrate actions for the learner to imitate
2. Provide correction rewards when the learner deviates
3. Joint attention (teacher points at objects)

The teacher runs a simple scripted policy (no learning):
- Move toward the nearest object
- Push it toward the center
- Occasionally point at interesting objects

The learner observes both the physics state AND the teacher's actions,
enabling imitation and social learning.

社会教师环境：在物理沙盒中加入第二个智能体作为教师，用于模仿学习。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .physics_sandbox import EnvStep, PhysicsSandbox, _Body


class SocialTeacher:
    """Scripted teacher agent that demonstrates helpful behaviors.

    Not a trainable agent. Runs a simple heuristic policy:
    - Find nearest object → move toward it → push it
    - Occasionally point at objects for joint attention
    """

    def __init__(self, world_size: float = 2.0, speed: float = 60.0) -> None:
        self._world = world_size
        self._hw = world_size / 2
        self._speed = speed
        self._rng = np.random.RandomState()
        self._pointing_at: int | None = None  # object index being pointed at
        self._point_timer: int = 0
        self._current_target: int | None = None

    def reset(self, agent_x: float, agent_y: float, rng: np.random.RandomState) -> _Body:
        """Spawn teacher away from the learner agent."""
        self._rng = rng
        self._pointing_at = None
        self._point_timer = 0
        self._current_target = None
        # Spawn on the opposite side from the learner
        tx = -agent_x * 0.7 + self._rng.uniform(-0.3, 0.3)
        ty = -agent_y * 0.7 + self._rng.uniform(-0.3, 0.3)
        tx = max(-self._hw + 0.3, min(self._hw - 0.3, tx))
        ty = max(-self._hw + 0.3, min(self._hw - 0.3, ty))
        return _Body(
            x=tx, y=ty, radius=0.15, mass=1.5,
            color=(255, 200, 50), tag="teacher",
        )

    def act(
        self,
        teacher: _Body,
        learner: _Body,
        objects: list[_Body],
    ) -> tuple[int, dict[str, Any]]:
        """Compute teacher action and metadata.

        Returns (action, info) where info contains:
        - "teacher_action": int (0-7)
        - "pointing_at": object index or None
        - "target_object": object index being pushed
        """
        info: dict[str, Any] = {
            "teacher_action": 0,
            "pointing_at": None,
            "target_object": None,
        }

        # Pointing: every 30-50 steps, point at a random object for 10 steps
        self._point_timer -= 1
        if self._point_timer <= 0:
            self._point_timer = self._rng.randint(30, 50)
            if objects and self._rng.random() < 0.3:
                self._pointing_at = self._rng.randint(0, len(objects))
                info["pointing_at"] = self._pointing_at
        if self._point_timer > 40:
            self._pointing_at = None

        # Target selection: nearest object to teacher
        if objects:
            if self._current_target is None or self._rng.random() < 0.02:
                dists = [
                    np.sqrt((teacher.x - o.x) ** 2 + (teacher.y - o.y) ** 2)
                    for o in objects
                ]
                self._current_target = int(np.argmin(dists))
            target = objects[self._current_target]
            info["target_object"] = self._current_target

            # Move toward target
            dx = target.x - teacher.x
            dy = target.y - teacher.y
            action = _direction_to_action(dx, dy)
        else:
            action = self._rng.randint(0, 4)

        # Don't collide with learner — if too close, move away
        dx_l = teacher.x - learner.x
        dy_l = teacher.y - learner.y
        dist_l = np.sqrt(dx_l * dx_l + dy_l * dy_l)
        if dist_l < 0.5:
            action = _direction_to_action(-dx_l, -dy_l)

        info["teacher_action"] = action
        return int(action), info

    @property
    def is_pointing(self) -> bool:
        return self._pointing_at is not None

    @property
    def pointing_target(self) -> int | None:
        return self._pointing_at


class SocialTeacherWrapper:
    """Wraps PhysicsSandbox with a scripted teacher agent.

    The teacher acts first (one physics step), then the learner acts.
    The learner receives:
    - Normal observation (from PhysicsSandbox)
    - Teacher action as extra info
    - Imitation reward if learner action matches teacher action
    - Joint attention reward if learner looks at what teacher points at
    - Correction penalty if learner action is very different from teacher

    Matches the EnvStep interface expected by train.py.
    """

    def __init__(
        self,
        num_objects: int = 3,
        seed: int | None = None,
        max_episode_steps: int = 200,
        render_size: int = 64,
        imitation_reward: float = 0.3,
        attention_reward: float = 0.1,
        correction_penalty: float = 0.05,
    ) -> None:
        self._sandbox = PhysicsSandbox(
            num_objects=num_objects,
            seed=seed,
            max_episode_steps=max_episode_steps,
            render_size=render_size,
        )
        self._teacher_agent = SocialTeacher()
        self._teacher_body: _Body | None = None
        self._imitation_reward = imitation_reward
        self._attention_reward = attention_reward
        self._correction_penalty = correction_penalty
        self._teacher_action: int = 0
        self._teacher_info: dict[str, Any] = {}
        self._episode_returns: list[float] = []

    # ------------------------------------------------------------------ properties

    @property
    def action_space_n(self) -> int:
        return self._sandbox.action_space_n

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self._sandbox.observation_shape

    def summary(self) -> dict:
        returns = self._episode_returns
        return {
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "last_return": returns[-1] if returns else 0.0,
        }

    def close(self) -> None:
        self._sandbox.close()

    # ------------------------------------------------------------------ gym-like

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self._sandbox.reset(seed=seed)
        # Spawn teacher opposite the learner
        learner = self._sandbox._agent
        self._teacher_body = self._teacher_agent.reset(
            learner.x, learner.y, self._sandbox._rng,
        )
        # Add teacher to sandbox object list for rendering
        self._sandbox._teacher = self._teacher_body
        self._teacher_action = 0
        return obs

    def step(self, action: int) -> EnvStep:
        """Learner takes action, teacher also acts.

        The teacher acts BEFORE the learner in the physics step,
        so the learner sees the teacher's movement in the observation.
        """
        # Teacher acts
        if self._teacher_body is not None:
            t_action, self._teacher_info = self._teacher_agent.act(
                self._teacher_body,
                self._sandbox._agent,
                self._sandbox._objects,
            )
            self._teacher_action = t_action
            # Apply teacher action as force
            directions = [
                (0, 1), (0, -1), (-1, 0), (1, 0),
                (0, 2), (0, -2), (-2, 0), (2, 0),
            ]
            fx, fy = directions[t_action]
            force_mag = self._teacher_agent._speed
            self._teacher_body.apply_force(fx * force_mag, fy * force_mag)

        # Include teacher in the sandbox's physics collision list
        if self._teacher_body is not None:
            self._sandbox._teacher = self._teacher_body

        # Learner acts (delegate to sandbox)
        step_out = self._sandbox.step(action)

        # Compute social rewards
        social_reward = self._compute_social_reward(action)
        # Note: we can't modify EnvStep.reward directly since it's a dataclass
        # but the caller gets the info dict with social reward

        done = step_out.terminated or step_out.truncated
        if done:
            self._episode_returns.append(
                self._sandbox._current_return + social_reward
            )
            if len(self._episode_returns) > 1024:  # BOUNDS-OK: rolling window cap
                self._episode_returns = self._episode_returns[-1024:]

        # Step info enriched with teacher data
        step_info = dict(step_out.info)
        step_info.update({
            "teacher_action": self._teacher_action,
            "teacher_pointing_at": self._teacher_info.get("pointing_at"),
            "teacher_target": self._teacher_info.get("target_object"),
            "social_reward": social_reward,
        })

        return EnvStep(
            obs=step_out.obs,
            reward=step_out.reward + social_reward,
            terminated=step_out.terminated,
            truncated=step_out.truncated,
            info=step_info,
            proprio=step_out.proprio,
        )

    def _compute_social_reward(self, learner_action: int) -> float:
        """Compute imitation + attention + correction rewards."""
        reward = 0.0

        # Imitation: same action as teacher → bonus
        if learner_action == self._teacher_action:
            reward += self._imitation_reward

        # Joint attention: if teacher is pointing, reward looking toward that object
        pointing_at = self._teacher_info.get("pointing_at")
        if pointing_at is not None and pointing_at < len(self._sandbox._objects):
            obj = self._sandbox._objects[pointing_at]
            learner = self._sandbox._agent
            dx = obj.x - learner.x
            dy = obj.y - learner.y
            # Reward if learner action moves TOWARD the pointed object
            toward_actions = {_direction_to_action(dx, dy)}
            if learner_action in toward_actions:
                reward += self._attention_reward

        # Correction: if learner does opposite of teacher, small penalty
        opposite_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
        if learner_action == opposite_map.get(self._teacher_action, -1):
            reward -= self._correction_penalty

        return float(reward)


def _direction_to_action(dx: float, dy: float) -> int:
    """Map (dx, dy) direction to action index 0-3."""
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2  # right / left
    else:
        return 0 if dy > 0 else 1  # up / down
