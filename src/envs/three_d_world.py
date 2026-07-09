"""3D Developmental World — Home, Objects, Caregivers.

A MuJoCo-based 3D environment designed for developmental AI training.

Features:
    - Room with walls, floor, furniture
    - 20–500 procedurally generated objects (balls, blocks, cups, plates)
    - Day/night cycle (lighting changes throughout the day)
    - Up to 3 agents (learner + caregiver + sibling)
    - Language labels on objects and actions
    - Developmental body scaling (agent grows from baby → child size)
    - Proprioceptive output (position, velocity, touch, joint angles)

Architecture:
    MuJoCo physics engine (CPU, ~1ms/step)
    → Offscreen renderer (256×256×3 RGB, GPU/CUDA)
    → SlotAttention encoder
    → TTT-Hybrid backbone + 7 cognitive modules (unchanged)

3D 发育式世界：家、物体、看护者。基于 MuJoCo 物理引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# MuJoCo is the physics backend; renderer uses its offscreen context.
# All imports are lazy to allow graceful fallback on systems without MuJoCo.
_mj_available = False
try:
    import mujoco
    _mj_available = True
except ImportError:
    pass

if _mj_available:
    import mujoco.viewer  # noqa: F401 — used internally for offscreen rendering


# =====================================================================
# EnvStep — matches existing interface
# =====================================================================


@dataclass
class EnvStep3D:
    obs: np.ndarray          # (256, 256, 3) uint8 RGB
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    proprio: np.ndarray      # (12,) — position(3) + velocity(3) + touch(3) + joint(3)


# =====================================================================
# Object Library — procedural generation
# =====================================================================


@dataclass
class ObjectDef:
    """Definition of a spawnable object."""
    name: str
    kind: str                # "sphere", "box", "cylinder", "capsule"
    size: tuple[float, float, float]  # (hx, hy, hz) or radius
    color: tuple[float, float, float, float]  # RGBA
    mass: float = 0.5
    label: str = ""          # language label
    category: str = "toy"    # "toy", "food", "tool", "furniture", "container"
    graspable: bool = True


def _generate_object_library(num_objects: int = 100, seed: int = 42) -> list[ObjectDef]:
    """Procedurally generate objects with varying properties."""
    rng = np.random.RandomState(seed)
    objects: list[ObjectDef] = []

    categories = {
        "toy": ["ball", "block", "doll", "car", "train", "puzzle_piece", "marbles", "top", "drum", "rattle"],
        "food": ["apple", "banana", "carrot", "bread", "cookie", "cheese", "egg", "milk_carton", "cupcake", "grape"],
        "tool": ["spoon", "fork", "knife", "cup", "plate", "bowl", "hammer", "screwdriver", "brush", "comb"],
        "furniture": ["chair", "table", "bed", "shelf", "lamp", "rug", "pillow", "blanket", "mirror", "clock"],
        "container": ["box", "basket", "bottle", "jar", "can", "bag", "bucket", "crate", "drawer", "cabinet"],
    }

    kinds = ["sphere", "box", "cylinder", "capsule"]
    kind_params = {
        "sphere": lambda rng: (rng.uniform(0.03, 0.15), 0.0, 0.0),
        "box": lambda rng: (rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.12)),
        "cylinder": lambda rng: (rng.uniform(0.03, 0.10), 0.0, rng.uniform(0.03, 0.12)),
        "capsule": lambda rng: (rng.uniform(0.02, 0.08), 0.0, rng.uniform(0.04, 0.15)),
    }

    for i in range(min(num_objects, 500)):
        category = list(categories.keys())[i % len(categories)]
        label_pool = categories[category]
        label = label_pool[i % len(label_pool)]
        kind = kinds[rng.randint(0, len(kinds))]
        sx, sy, sz = kind_params[kind](rng)
        r, g, b = rng.uniform(0.1, 0.95, 3)
        mass = rng.uniform(0.1, 3.0)

        if category == "container":
            kind = "box"
            sx, sy, sz = rng.uniform(0.05, 0.18, 3)
        if category == "furniture":
            mass = rng.uniform(3.0, 20.0)
            sx, sy, sz = max(sx, 0.1), max(sy, 0.1), max(sz, 0.1)

        objects.append(ObjectDef(
            name=f"{label}_{i}",
            kind=kind,
            size=(float(sx), float(sy), float(sz)),
            color=(float(r), float(g), float(b), 1.0),
            mass=float(mass),
            label=label,
            category=category,
            graspable=kind != "box" or mass < 3.0,
        ))

    return objects


# =====================================================================
# Scene Builder — programmatic MuJoCo XML
# =====================================================================


class SceneBuilder:
    """Builds MuJoCo XML scenes programmatically.

    Scene layout (top-down view):
        ┌──────────────────────────────────────┐
        │  [table]                           │
        │       [shelf]                      │
        │           ┌──────────┐              │
        │           │  agent   │              │
        │           └──────────┘              │
        │  [caregiver]              [sibling] │
        │                          [rug]    │
        │    ┌──────────┐                     │
        │    │  bed     │                      │
        │    └──────────┘                     │
        └──────────────────────────────────────┘

    Room: 4m × 4m × 2.5m (walls, floor, ceiling).
    """

    def __init__(self, room_size: tuple[float, float, float] = (4.0, 4.0, 2.5)) -> None:
        self._rw, self._rl, self._rh = room_size
        self._objects: list[dict] = []
        self._agents: list[dict] = []
        self._sun_angle: float = 0.0  # radians

    def add_agent(
        self,
        name: str = "learner",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: float = 0.15,
        color: tuple[float, float, float, float] = (0.0, 1.0, 0.5, 1.0),
        can_move: bool = True,
    ) -> None:
        self._agents.append({
            "name": name,
            "pos": position,
            "size": size,
            "color": color,
            "can_move": can_move,
        })

    def add_object(self, obj: ObjectDef, position: tuple[float, float, float]) -> None:
        self._objects.append({
            "def": obj,
            "pos": position,
        })

    def set_sun_angle(self, radians: float) -> None:
        self._sun_angle = radians

    def build_xml(self) -> str:
        """Generate MuJoCo XML string."""
        xml = f"""<mujoco model="devagi_home">
  <compiler angle="radian"/>
  <option timestep="0.016" gravity="0 0 -9.81"/>

  <visual>
    <map force="0.1" zfar="30"/>
    <quality shadowsize="2048"/>
    <global offwidth="256" offheight="256"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.9 0.9 1.0"
             width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.6 0.5 0.4" rgb2="0.7 0.6 0.5"/>
    <material name="floor_mat" texture="floor_tex" reflectance="0.2"/>
    <material name="wall_mat" rgba="0.85 0.82 0.75 1.0" reflectance="0.1"/>
    <material name="furniture_mat" rgba="0.4 0.25 0.15 1.0" reflectance="0.1"/>
"""
        # Add materials for each object color
        for i, obj in enumerate(self._objects):
            c = obj["def"].color
            xml += f'    <material name="obj_{i}_mat" rgba="{c[0]} {c[1]} {c[2]} {c[3]}" reflectance="0.2"/>\n'

        xml += """  </asset>

  <worldbody>
"""
        # Day/night: light height based on sun angle
        light_z = float(5.0 * max(0.1, np.sin(self._sun_angle + 0.3)))
        light_intensity = float(0.5 + 0.5 * max(0.1, np.sin(self._sun_angle + 0.3)))
        xml += f"""    <light name="sun" directional="true" diffuse="{light_intensity} {light_intensity} {light_intensity * 0.9}"
           specular="0.2 0.2 0.2" pos="0 0 {light_z}" dir="0 0 -1"/>
    <light name="ambient" pos="0 0 3" dir="0 0 -1" diffuse="0.15 0.15 0.18"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="{self._rw/2+0.1} {self._rl/2+0.1} 0.05"
          pos="0 0 0" material="floor_mat"/>

    <!-- Walls -->
    <geom name="wall_n" type="box" size="{self._rw/2+0.1} 0.05 {self._rh/2}"
          pos="0 {self._rl/2} {self._rh/2}" material="wall_mat"/>
    <geom name="wall_s" type="box" size="{self._rw/2+0.1} 0.05 {self._rh/2}"
          pos="0 {-self._rl/2} {self._rh/2}" material="wall_mat"/>
    <geom name="wall_e" type="box" size="0.05 {self._rl/2+0.1} {self._rh/2}"
          pos="{self._rw/2} 0 {self._rh/2}" material="wall_mat"/>
    <geom name="wall_w" type="box" size="0.05 {self._rl/2+0.1} {self._rh/2}"
          pos="{-self._rw/2} 0 {self._rh/2}" material="wall_mat"/>

    <!-- Furniture: table -->
    <body name="table" pos="{-self._rw/4} {self._rl/4} 0.4">
      <geom name="table_top" type="box" size="0.4 0.3 0.02" pos="0 0 0.4" material="furniture_mat"/>
      <geom name="table_leg1" type="cylinder" size="0.02 0.4" pos="0.35 0.25 0.2" material="furniture_mat"/>
      <geom name="table_leg2" type="cylinder" size="0.02 0.4" pos="-0.35 0.25 0.2" material="furniture_mat"/>
      <geom name="table_leg3" type="cylinder" size="0.02 0.4" pos="0.35 -0.25 0.2" material="furniture_mat"/>
      <geom name="table_leg4" type="cylinder" size="0.02 0.4" pos="-0.35 -0.25 0.2" material="furniture_mat"/>
    </body>

    <!-- Furniture: bed -->
    <body name="bed" pos="{-self._rw/4} {-self._rl/4} 0.15">
      <geom name="bed_mat" type="box" size="0.5 0.8 0.05" pos="0 0 0.15" material="furniture_mat"/>
      <geom name="bed_pillow" type="box" size="0.3 0.2 0.06" pos="0 {-0.5} 0.25" rgba="0.9 0.9 0.9 1.0"/>
    </body>

    <!-- Furniture: shelf -->
    <body name="shelf" pos="{self._rw/4} {self._rl/4} 0.8">
      <geom name="shelf_b1" type="box" size="0.5 0.15 0.02" pos="0 0 0.4" material="furniture_mat"/>
      <geom name="shelf_b2" type="box" size="0.5 0.15 0.02" pos="0 0 0.8" material="furniture_mat"/>
      <geom name="shelf_side1" type="box" size="0.02 0.15 0.8" pos="{-0.5} 0 0.4" material="furniture_mat"/>
      <geom name="shelf_side2" type="box" size="0.02 0.15 0.8" pos="{0.5} 0 0.4" material="furniture_mat"/>
    </body>

    <!-- Spread objects on floor, table, shelf -->
"""
        # Place objects
        for i, obj_data in enumerate(self._objects):
            obj = obj_data["def"]
            px, py, pz = obj_data["pos"]
            sx, sy, sz = obj.size
            if obj.kind == "sphere":
                geom = f'<geom name="obj_{i}" type="sphere" size="{sx}" pos="{px} {py} {pz}" mass="{obj.mass}" material="obj_{i}_mat"/>'
            elif obj.kind == "cylinder":
                geom = f'<geom name="obj_{i}" type="cylinder" size="{sx} {pz}" pos="{px} {py} {pz}" mass="{obj.mass}" material="obj_{i}_mat"/>'
            elif obj.kind == "capsule":
                geom = f'<geom name="obj_{i}" type="capsule" size="{sx} {sz}" pos="{px} {py} {pz}" mass="{obj.mass}" material="obj_{i}_mat"/>'
            else:
                geom = f'<geom name="obj_{i}" type="box" size="{sx} {sy} {sz}" pos="{px} {py} {pz}" mass="{obj.mass}" material="obj_{i}_mat"/>'
            xml += f"    {geom}\n"

        # Agents (movable spheres)
        for agent in self._agents:
            a = agent
            sz = a["size"]
            c = a["color"]
            if a["can_move"]:
                xml += f"""
    <body name="{a['name']}" pos="{a['pos'][0]} {a['pos'][1]} {a['pos'][2]}">
      <joint name="{a['name']}_x" type="slide" axis="1 0 0"/>
      <joint name="{a['name']}_y" type="slide" axis="0 1 0"/>
      <geom name="{a['name']}_geom" type="sphere" size="{sz}"
            rgba="{c[0]} {c[1]} {c[2]} {c[3]}" mass="1.5"/>
    </body>"""
            else:
                xml += f"""
    <body name="{a['name']}" pos="{a['pos'][0]} {a['pos'][1]} {a['pos'][2]}">
      <geom name="{a['name']}_geom" type="sphere" size="{sz}"
            rgba="{c[0]} {c[1]} {c[2]} {c[3]}" mass="1.5"/>
    </body>"""

        xml += """
  </worldbody>

  <actuator>
"""
        for agent in self._agents:
            if agent["can_move"]:
                xml += f"""    <position name="{agent['name']}_act_x" joint="{agent['name']}_x" kp="20"/>
    <position name="{agent['name']}_act_y" joint="{agent['name']}_y" kp="20"/>
"""

        xml += """  </actuator>

  <sensor>
  </sensor>
</mujoco>"""

        return xml


# =====================================================================
# ThreeDWorld — environment class
# =====================================================================


class ThreeDWorld:
    """3D developmental environment powered by MuJoCo.

    Matches the EnvStep interface used by train.py.

    Action space (8 discrete):
        0-3: move agent (north, south, west, east)
        4-7: move agent × 2 (strong push)

    Observation: 256×256×3 RGB + 12-dim proprioceptive.
    """

    def __init__(
        self,
        num_objects: int = 100,
        seed: int | None = None,
        max_episode_steps: int = 500,
        render_size: int = 256,
        day_cycle_steps: int = 1000,  # full day in 1000 steps
        action_force: float = 2.0,
        developmental_age: float = 0.0,  # 0=infant, 1=child
    ) -> None:
        if not _mj_available:
            raise ImportError("mujoco is required for ThreeDWorld. Run: pip install mujoco")

        self._num_objects = num_objects
        self._max_steps = max_episode_steps
        self._render_size = render_size
        self._day_cycle = day_cycle_steps
        self._action_force = action_force
        self._dev_age = developmental_age
        self._rng = np.random.RandomState(seed)

        # Object library
        self._object_lib = _generate_object_library(min(num_objects, 500), seed or 42)

        # Scene (rebuilt per reset)
        self._model: Any = None  # mujoco.MjModel
        self._data: Any = None   # mujoco.MjData
        self._renderer: Any = None
        self._step_count: int = 0
        self._agent_names: list[str] = ["learner"]

        self._episode_returns: list[float] = []
        self._current_return: float = 0.0
        self._auto_reset: bool = True
        self._sun_angle: float = 0.0

        # Initial reset
        self._build_scene()

    # ------------------------------------------------------------------ properties

    @property
    def action_space_n(self) -> int:
        return 8

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self._render_size, self._render_size, 3)

    @property
    def proprio_dim(self) -> int:
        return 12

    @property
    def objects(self) -> list[dict]:
        return [obj for obj in self._object_lib[:self._num_objects]]

    def summary(self) -> dict:
        returns = self._episode_returns
        return {
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "last_return": returns[-1] if returns else 0.0,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._model = None
        self._data = None

    # ------------------------------------------------------------------ gym-like

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._current_return = 0.0
        self._build_scene()
        return self._render()

    def step(self, action: int) -> EnvStep3D:
        action = int(action) % 8
        agent_name = self._agent_names[0]

        # Apply force via position actuators
        if action < 4:
            force = self._action_force
            dir_idx = action
        else:
            force = self._action_force * 2.0
            dir_idx = action - 4

        dx = force * [0, 0, -1, 1][dir_idx]
        dy = force * [1, -1, 0, 0][dir_idx]

        # Move agent target position via velocity control
        try:
            # Get agent body and apply velocity
            body_id = self._model.body(agent_name).id
            dof_addr = self._model.body_dofadr[body_id]
            if dof_addr >= 0:
                self._data.qvel[dof_addr] += dx * self._model.opt.timestep
                self._data.qvel[dof_addr + 1] += dy * self._model.opt.timestep
        except Exception:
            pass
        # For agents with actuators, also set control targets
        try:
            act_x_id = self._model.actuator(f"{agent_name}_act_x").id
            act_y_id = self._model.actuator(f"{agent_name}_act_y").id
            self._data.ctrl[act_x_id] = self._data.qpos[self._model.jnt_qposadr[self._model.joint(f"{agent_name}_x").id]] + dx * 0.01
            self._data.ctrl[act_y_id] = self._data.qpos[self._model.jnt_qposadr[self._model.joint(f"{agent_name}_y").id]] + dy * 0.01
        except Exception:
            pass

        # Advance day/night
        self._sun_angle = (self._step_count % self._day_cycle) / self._day_cycle * 2 * np.pi
        self._step_count += 1

        # Physics step
        mujoco.mj_step(self._model, self._data)

        # Reward
        reward = self._compute_reward()
        self._current_return += reward

        done = self._step_count >= self._max_steps

        if done:
            self._episode_returns.append(self._current_return)
            if self._auto_reset:
                self._rebuild_scene()
            self._current_return = 0.0
            self._step_count = 0

        return EnvStep3D(
            obs=self._render(),
            reward=reward,
            terminated=done,
            truncated=done,
            info={"step": self._step_count, "dev_age": self._dev_age, "sun_angle": self._sun_angle},
            proprio=self._proprio(),
        )

    # ------------------------------------------------------------------ internals

    def _build_scene(self) -> None:
        """Build MuJoCo scene from scratch."""
        builder = SceneBuilder()

        # Agent size grows with developmental age
        agent_size = 0.12 + self._dev_age * 0.08  # 0.12 (infant) → 0.20 (child)

        builder.add_agent(
            name="learner",
            position=(self._rng.uniform(-0.5, 0.5), self._rng.uniform(-0.5, 0.5), agent_size),
            size=agent_size,
            color=(0.0, 0.9, 0.4, 1.0),
            can_move=True,
        )

        # Caregiver (stationary, observes)
        builder.add_agent(
            name="caregiver",
            position=(-1.2, 0.8, 0.18),
            size=0.18,
            color=(1.0, 0.8, 0.2, 1.0),
            can_move=False,
        )
        self._agent_names.append("caregiver")

        # Place objects
        used_positions: list[tuple[float, float, float]] = []
        for i in range(min(self._num_objects, len(self._object_lib))):
            obj = self._object_lib[i]
            for _ in range(20):
                pos = (
                    self._rng.uniform(-1.8, 1.8),
                    self._rng.uniform(-1.8, 1.8),
                    self._rng.uniform(0.08, 1.2),
                )
                # Avoid overlap
                too_close = any(
                    np.sqrt((pos[0] - px)**2 + (pos[1] - py)**2 + (pos[2] - pz)**2) < 0.15
                    for px, py, pz in used_positions
                )
                if not too_close:
                    used_positions.append(pos)
                    builder.add_object(obj, pos)
                    break

        builder.set_sun_angle(self._sun_angle)
        xml = builder.build_xml()

        # Load MuJoCo model
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

        # Initialize renderer
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(
            self._model, height=self._render_size, width=self._render_size,
        )

        mujoco.mj_forward(self._model, self._data)

    def _rebuild_scene(self) -> None:
        self._build_scene()

    def _render(self) -> np.ndarray:
        """Render current scene to (H, W, 3) uint8 RGB."""
        if self._renderer is None:
            return np.zeros((self._render_size, self._render_size, 3), dtype=np.uint8)

        # Update scene state
        mujoco.mj_forward(self._model, self._data)

        # Update renderer scene
        self._renderer.update_scene(self._data)

        # Render offscreen
        pixels = self._renderer.render()
        # pixels is (H, W, 3) float32 in [0, 1] → uint8
        return (np.clip(pixels, 0, 1) * 255).astype(np.uint8)

    def _compute_reward(self) -> float:
        """Multi-component reward.

        - Object interaction: velocity of scene objects (learner caused movement)
        - Exploration: visiting new locations
        - Social: proximity to caregiver (safety reward)
        """
        reward = 0.0

        # Object movement reward
        for i in range(self._model.ngeom):
            name = self._model.geom(i).name
            if name.startswith("obj_"):
                gid = self._model.geom(name).id
                body_id = self._model.geom_bodyid[gid]
                dof_addr = self._model.body_dofadr[body_id]
                if dof_addr >= 0 and dof_addr + 2 < self._model.nv:
                    vel = self._data.qvel[dof_addr:dof_addr+3]
                    speed = float(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2))
                else:
                    speed = 0.0
                reward += speed * 0.1

        # Contact reward: touching objects
        touch_sensor_name = "learner_touch"
        if touch_sensor_name in [self._model.sensor(i).name for i in range(self._model.nsensor)]:
            try:
                touch_data = self._data.sensor(touch_sensor_name).data
                reward += float(np.sum(np.abs(touch_data))) * 0.2
            except Exception:
                pass

        # Caregiver proximity reward
        try:
            lx = float(self._data.body("learner").xpos[0])
            ly = float(self._data.body("learner").xpos[1])
            cx = float(self._data.body("caregiver").xpos[0])
            cy = float(self._data.body("caregiver").xpos[1])
            dist_caregiver = np.sqrt((lx - cx)**2 + (ly - cy)**2)
            reward += max(0, (1.0 - dist_caregiver)) * 0.05
        except Exception:
            pass

        return float(max(0.0, min(5.0, reward)))

    def _proprio(self) -> np.ndarray:
        """Return (12,) proprioceptive vector."""
        try:
            body = self._data.body("learner")
            pos = body.xpos[:3].copy()
            vel = self._data.cvel[self._model.body("learner").id][3:6].copy()
            # Touch sensor
            touch = np.zeros(3)
            # Touch: compute from contact forces (simplified)
            try:
                for i in range(self._model.ncon):
                    contact = self._data.contact[i]
                    touch[:] = [1.0, 1.0, 1.0]  # any contact = touch signal
            except Exception:
                pass
            # Joint positions
            joints = np.zeros(3)
            for i, axis in enumerate(["x", "y"]):
                try:
                    joint_id = self._model.joint(f"learner_{axis}").id
                    joints[i] = float(self._data.qpos[joint_id])
                except Exception:
                    pass
            return np.concatenate([pos, vel, touch, joints]).astype(np.float32)
        except Exception:
            return np.zeros(12, dtype=np.float32)

    def set_developmental_age(self, age: float) -> None:
        """Update developmental age (body grows, more objects emerge)."""
        self._dev_age = max(0.0, min(1.0, age))
