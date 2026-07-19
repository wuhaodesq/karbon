"""2D Physics Sandbox for developmental AI.

A minimal pure-numpy 2D physics environment with:
- Agent (circle) that can apply forces
- Objects (circles) with color, mass, position, velocity
- Gravity, collision, friction
- Proprioceptive output (agent position, velocity, contact forces)

Zero external dependencies (no pymunk/brax needed).
Matches the ``EnvStep`` interface used by ``src/train.py``.

发育式 AI 的 2D 物理沙盒环境。纯 numpy 实现，零外部依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# =====================================================================
# EnvStep — matches src.envs.minigrid_wrapper.EnvStep
# =====================================================================


@dataclass
class EnvStep:
    obs: np.ndarray          # (H, W, C) uint8 rendered image
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    proprio: np.ndarray      # (6,) proprioceptive state (pos_x, pos_y, vel_x, vel_y,
                             #   contact_force_x, contact_force_y)


# =====================================================================
# Simple 2D body
# =====================================================================


@dataclass
class _Body:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.15
    mass: float = 1.0
    color: tuple[int, int, int] = (255, 0, 0)
    tag: str = "object"
    static: bool = False
    friction: float = 0.5
    restitution: float = 0.5
    temperature: float = 20.0
    _fx: float = 0.0
    _fy: float = 0.0
    _prev_vx: float = 0.0
    _prev_vy: float = 0.0

    def apply_force(self, fx: float, fy: float) -> None:
        self._fx += fx
        self._fy += fy

    @property
    def thermal_state(self) -> str:
        if self.temperature < 5:
            return "cold"
        if self.temperature > 50:
            return "hot"
        return "warm"


# =====================================================================
# PhysicsSandbox
# =====================================================================


class PhysicsSandbox:
    """2D physics sandbox with gravity, collisions, and agent forces.

    Action space: 8 discrete actions
        0-3: apply force (up, down, left, right)
        4-7: apply force × 2 (strong push)

    Observation: rendered 64×64 RGB image + (6,) proprioceptive vector.

    Bounded: world is 2×2 meters, capped episode length, max 8 objects.
    """

    def __init__(
        self,
        num_objects: int = 3,
        seed: int | None = None,
        max_episode_steps: int = 200,
        render_size: int = 64,
        gravity: float = -9.8,
        dt: float = 1.0 / 60.0,
        action_force: float = 50.0,
        object_radius: float = 0.12,
        agent_radius: float = 0.15,
        world_size: float = 2.0,
    ) -> None:
        self._num_objects = num_objects
        self._max_steps = max_episode_steps
        self._render_size = render_size
        self._gravity = gravity
        self._dt = dt
        self._action_force = action_force
        self._object_radius = object_radius
        self._agent_radius = agent_radius
        self._world = world_size
        self._hw = world_size / 2

        self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._episode_returns: list[float] = []
        self._current_return: float = 0.0
        self._episode_lengths: list[int] = []

        # World state
        self._agent: _Body = None  # type: ignore[assignment]
        self._objects: list[_Body] = []
        self._auto_reset = True
        self._prev_distances: dict[int, float] = {}

    # ------------------------------------------------------------------ properties

    @property
    def action_space_n(self) -> int:
        return 8

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self._render_size, self._render_size, 3)

    @property
    def proprio_dim(self) -> int:
        return 6  # x, y, vx, vy, contact_fx, contact_fy

    def summary(self) -> dict:
        returns = self._episode_returns
        lengths = self._episode_lengths
        return {
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "last_return": returns[-1] if returns else 0.0,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ gym-like

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._current_return = 0.0

        # Place agent near center
        self._agent = _Body(
            x=self._rng.uniform(-0.3, 0.3),
            y=self._rng.uniform(-0.3, 0.3),
            radius=self._agent_radius,
            mass=1.5,
            color=(0, 200, 0),
            tag="agent",
        )

        # Place objects randomly without overlap
        colors = [
            (220, 50, 50), (50, 50, 220), (220, 180, 0),
            (50, 200, 50), (200, 50, 200), (50, 200, 200),
            (200, 100, 50), (100, 100, 200),
        ]
        self._objects = []
        self._prev_distances = {}
        for i in range(self._num_objects):
            for _ in range(50):  # try 50 times to place without overlap
                ox = self._rng.uniform(-self._hw + 0.3, self._hw - 0.3)
                oy = self._rng.uniform(-self._hw + 0.3, self._hw - 0.3)
                too_close = False
                dx = ox - self._agent.x
                dy = oy - self._agent.y
                if np.sqrt(dx*dx + dy*dy) < self._agent_radius + self._object_radius + 0.1:
                    too_close = True
                for obj in self._objects:
                    dx2 = ox - obj.x
                    dy2 = oy - obj.y
                    if np.sqrt(dx2*dx2 + dy2*dy2) < 2 * self._object_radius + 0.05:
                        too_close = True
                        break
                if not too_close:
                    body = _Body(
                        x=ox, y=oy,
                        radius=self._object_radius,
                        mass=self._rng.uniform(0.5, 2.0),
                        color=colors[i % len(colors)],
                        tag=f"obj_{i}",
                        friction=self._rng.uniform(0.1, 1.0),
                        restitution=self._rng.uniform(0.1, 0.9),
                        temperature=self._rng.uniform(5.0, 60.0),
                    )
                    self._objects.append(body)
                    # Small random initial velocity
                    body.vx = self._rng.uniform(-0.5, 0.5)
                    body.vy = self._rng.uniform(-0.5, 0.5)
                    self._prev_distances[i] = np.sqrt(
                        (ox - self._agent.x) ** 2 + (oy - self._agent.y) ** 2
                    )
                    break

        return self._render()

    def step(self, action: int) -> EnvStep:
        action = int(action) % 8
        force_mag = self._action_force
        if action >= 4:
            force_mag *= 2.0
            action -= 4

        # Apply force to agent
        directions = [(0, force_mag), (0, -force_mag), (-force_mag, 0), (force_mag, 0)]
        fx, fy = directions[action]
        self._agent.apply_force(fx, fy)

        # Physics step (semi-implicit Euler)
        self._physics_step()

        # Compute reward
        reward = self._compute_reward()

        self._step_count += 1
        self._current_return += reward
        done = self._step_count >= self._max_steps

        if done:
            self._episode_returns.append(self._current_return)
            self._episode_lengths.append(self._step_count)
            if len(self._episode_returns) > 1024:  # BOUNDS-OK: rolling window cap
                self._episode_returns = self._episode_returns[-1024:]
                self._episode_lengths = self._episode_lengths[-1024:]
            if self._auto_reset:
                self._reset_world()
            self._current_return = 0.0
            self._step_count = 0

        return EnvStep(
            obs=self._render(),
            reward=reward,
            terminated=done,
            truncated=done,
            info={"step": self._step_count, "max_steps": self._max_steps},
            proprio=self._proprioceptive(),
        )

    # ------------------------------------------------------------------ internals

    def _reset_world(self) -> None:
        """Reset without changing object count (for auto-reset)."""
        self._step_count = 0
        self._agent = _Body(
            x=self._rng.uniform(-0.3, 0.3),
            y=self._rng.uniform(-0.3, 0.3),
            radius=self._agent_radius,
            mass=1.5,
            color=(0, 200, 0),
            tag="agent",
        )
        for obj in self._objects:
            obj.x = self._rng.uniform(-self._hw + 0.3, self._hw - 0.3)
            obj.y = self._rng.uniform(-self._hw + 0.3, self._hw - 0.3)
            obj.vx = self._rng.uniform(-0.5, 0.5)
            obj.vy = self._rng.uniform(-0.5, 0.5)
            obj._fx = 0.0
            obj._fy = 0.0

    def _physics_step(self) -> None:
        dt = self._dt

        # Integrate
        for body in [self._agent] + self._objects:
            if body.static:
                continue
            # Acceleration from forces
            ax = body._fx / body.mass
            ay = (body._fy / body.mass) + self._gravity
            body.vx += ax * dt
            body.vy += ay * dt
            body.x += body.vx * dt
            body.y += body.vy * dt
            # Damping
            body.vx *= 0.995
            body.vy *= 0.995
            body._fx = 0.0
            body._fy = 0.0

        # Wall collisions (bounce)
        for body in [self._agent] + self._objects:
            r = body.radius
            if body.x - r < -self._hw:
                body.x = -self._hw + r
                body.vx *= -0.5
            if body.x + r > self._hw:
                body.x = self._hw - r
                body.vx *= -0.5
            if body.y - r < -self._hw:
                body.y = -self._hw + r
                body.vy *= -0.5
            if body.y + r > self._hw:
                body.y = self._hw - r
                body.vy *= -0.5

        # Object-object + agent-object collisions
        all_bodies = [self._agent] + self._objects
        for i, a in enumerate(all_bodies):
            for j, b in enumerate(all_bodies):
                if j <= i:
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = a.radius + b.radius
                if dist < min_dist and dist > 1e-8:
                    # Separate
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    total_mass = a.mass + b.mass
                    a.x -= overlap * nx * b.mass / total_mass
                    a.y -= overlap * ny * b.mass / total_mass
                    b.x += overlap * nx * a.mass / total_mass
                    b.y += overlap * ny * a.mass / total_mass
                    # Elastic collision
                    dvx = a.vx - b.vx
                    dvy = a.vy - b.vy
                    vn = dvx * nx + dvy * ny
                    if vn > 0:
                        impulse = 1.5 * vn / total_mass
                        a.vx -= impulse * b.mass * nx
                        a.vy -= impulse * b.mass * ny
                        b.vx += impulse * a.mass * nx
                        b.vy += impulse * a.mass * ny

        # Floor (prevent falling through)
        for body in [self._agent] + self._objects:
            if body.y - body.radius < -self._hw:
                body.y = -self._hw + body.radius
                body.vy *= -0.3

        # Record previous-frame velocity for acceleration-based reward
        # (only meaningful AFTER this step's integration, so next _compute_reward
        # compares against pre-step velocity).
        for body in [self._agent] + self._objects:
            body._prev_vx = body.vx
            body._prev_vy = body.vy

    def _compute_reward(self) -> float:
        """Reward for interacting with objects.

        v9 redesign (break local-optimum trap): the old reward gave speed*0.05 to
        EVERY object unconditionally, so passive inertia/collisions "white-gift"
        return and the agent got stuck nudging a few objects. New design rewards
        only AGENT-CAUSED object acceleration while in contact, forcing active
        multi-object pushing to maximize return.
        - Contact: +0.15 per object touched (was 0.1)
        - Active acceleration: only when agent touches object, reward |Δv|
          (agent pushing causes instantaneous speed change; passive motion ≠ Δv)
        - Approach: keep (exploration toward distant objects)
        """
        reward = 0.0
        agent = self._agent
        contact_count = 0

        for i, obj in enumerate(self._objects):
            dx = agent.x - obj.x
            dy = agent.y - obj.y
            dist = np.sqrt(dx * dx + dy * dy)
            touch_dist = agent.radius + obj.radius

            # Contact reward
            in_contact = dist < touch_dist + 0.02
            if in_contact:
                contact_count += 1

            # Agent-caused acceleration reward: only when agent touches the object,
            # reward the object's SPEED CHANGE (|Δv|) this step. Passive inertia/
            # collisions give ~0 Δv (uniform motion); only an active push by the
            # agent produces a large instantaneous Δv. This removes the "white-gift"
            # speed reward that let the agent sit in a local optimum.
            accel = abs(obj.vx - obj._prev_vx) + abs(obj.vy - obj._prev_vy)
            if in_contact and accel > 1e-4:
                reward += accel * 0.5

            # Approach reward (reducing distance to objects)
            prev = self._prev_distances.get(i, dist)
            if dist < prev:
                reward += (prev - dist) * 0.2
            self._prev_distances[i] = dist

        reward += contact_count * 0.15

        # Small penalty for hitting walls
        wall_dist = self._hw - max(
            abs(agent.x), abs(agent.y),
            agent.radius + 0.05,
        )
        if wall_dist < 0.1:
            reward -= (0.1 - wall_dist) * 0.5

        return float(max(-0.5, min(3.0, reward)))

    def _proprioceptive(self) -> np.ndarray:
        """Return agent's proprioceptive state: (x, y, vx, vy, contact_fx, contact_fy)."""
        agent = self._agent
        # Compute contact forces from nearby objects
        cfx, cfy = 0.0, 0.0
        for obj in self._objects:
            dx = agent.x - obj.x
            dy = agent.y - obj.y
            dist = np.sqrt(dx * dx + dy * dy)
            touch_dist = agent.radius + obj.radius
            if dist < touch_dist + 0.01 and dist > 1e-8:
                nx = dx / dist
                ny = dy / dist
                cfx += nx * (touch_dist - dist) * 100.0
                cfy += ny * (touch_dist - dist) * 100.0
        return np.array(
            [agent.x, agent.y, agent.vx, agent.vy, cfx, cfy],
            dtype=np.float32,
        )

    def _render(self) -> np.ndarray:
        """Render world to (H, W, 3) uint8 image."""
        size = self._render_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Background gradient
        for i in range(size):
            shade = int(30 + 20 * i / size)
            img[i, :, :] = shade

        # World-to-pixel transform
        scale = (size - 4) / (self._world)
        offset = size / 2

        def world_to_pixel(wx: float, wy: float) -> tuple[int, int]:
            px = int(offset + wx * scale)
            py = int(offset - wy * scale)
            return max(0, min(size - 1, px)), max(0, min(size - 1, py))

        # Draw objects
        for body in self._objects:
            cx, cy = world_to_pixel(body.x, body.y)
            r = max(2, int(body.radius * scale))
            _draw_circle_filled(img, cx, cy, r, body.color)

        # Draw agent
        ax, ay = world_to_pixel(self._agent.x, self._agent.y)
        ar = max(3, int(self._agent.radius * scale))
        _draw_circle_filled(img, ax, ay, ar, self._agent.color)
        # Agent direction indicator (small line in velocity direction)
        vx_px = int(self._agent.vx * scale * 0.3)
        vy_px = int(-self._agent.vy * scale * 0.3)
        if abs(vx_px) + abs(vy_px) > 1:
            _draw_line(img, ax, ay, ax + vx_px, ay + vy_px, (255, 255, 255))

        return img


# =====================================================================
# Drawing helpers (pure numpy, no cv2 dependency)
# =====================================================================


def _draw_circle_filled(
    img: np.ndarray,
    cx: int, cy: int, radius: int,
    color: tuple[int, int, int],
) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    r2 = radius * radius
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                img[y, x] = color


def _draw_line(
    img: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    color: tuple[int, int, int],
) -> None:
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
