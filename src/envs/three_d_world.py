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
    - Grasping system (unlocks at developmental_age > 0.15)
    - Chain tasks: obstacle clearing, key-door (unlock at age > 0.3)
    - Touch sensor for contact-force rewards

Architecture:
    MuJoCo physics engine (CPU, ~1ms/step)
    → Offscreen renderer (256×256×3 RGB, GPU/CUDA)
    → SlotAttention encoder
    → TTT-Hybrid backbone + 7 cognitive modules (unchanged)

3D 发育式世界：家、物体、看护者。基于 MuJoCo 物理引擎。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _expose_exc(where: str) -> None:
    """Surface an exception instead of silently swallowing it.

    Training-critical paths MUST NOT swallow errors: a bare
    ``except Exception: pass / return 0.0`` hid the missing ``import
    math`` and killed the occluder reward since Stage 20 (op bottleneck
    root cause). We still return a safe fallback to keep the process
    alive, but ALWAYS print the full traceback so any silent degradation
    shows up immediately in the training log.
    """
    import traceback  # lazy: this helper is only hit on the error path
    traceback.print_exc()
    print(f"[env-EXC] exception in {where}: see traceback above (NOT swallowed)", flush=True)


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
    proprio: np.ndarray      # (12,) or (16,) depending on dev_age— position(3) + velocity(3) + touch(3) + joint(3)


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

    def __init__(self, room_size: tuple[float, float, float] = (4.0, 4.0, 2.5),
                 camera_pos: tuple[float, float, float] = (0.0, -1.0, 0.8),
                 camera_fovy: float = 60.0) -> None:
        self._rw, self._rl, self._rh = room_size
        self._camera_pos = camera_pos
        self._camera_fovy = camera_fovy
        self._objects: list[dict] = []
        self._agents: list[dict] = []
        self._occluders: list[dict] = []
        self._trace_marker_xml: list[str] = []
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

    def add_occluder(
        self,
        position: tuple[float, float, float],
        size: tuple[float, float, float] = (0.5, 0.05, 0.6),
    ) -> None:
        """Add a static wall that can occlude objects from the agent's view.

        Kinematic (no joint), does not participate in dynamics — it only
        blocks the line of sight for the true occlusion probe.
        """
        self._occluders.append({
            "pos": position,
            "size": size,
        })

    def add_trace_markers(self, n: int) -> None:
        """Add hidden ground markers (visual occlusion traces).

        Developmental feedback: when an object is occluded, the env moves a
        marker to the object's last-known position so the agent can learn
        "occluded -> trace -> find" from environment feedback (not rewards).
        Markers start at y=100 (hidden); the env repositions them at runtime.
        """
        for i in range(n):
            self._trace_marker_xml.append(
                f'    <geom name="trace_marker_{i}" type="cylinder" '
                f'size="0.09 0.006" pos="0 0 100" '
                f'rgba="1.0 0.85 0.1 1.0" contype="0" conaffinity="0"/>\n'
            )

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

    <!-- Walls (Stage 20z: room walls MUST collide with the agent
         (conaffinity=1) or the velocity-controlled agent walks out of the
         room forever. Furniture below stays conaffinity=0 so it never
         wedges the agent, while objects (conaffinity=1) still rest on it. -->
    <geom name="wall_n" type="box" size="{self._rw/2+0.1} 0.05 {self._rh/2}"
          pos="0 {self._rl/2} {self._rh/2}" material="wall_mat"/>
    <geom name="wall_s" type="box" size="{self._rw/2+0.1} 0.05 {self._rh/2}"
          pos="0 {-self._rl/2} {self._rh/2}" material="wall_mat"/>
    <geom name="wall_e" type="box" size="0.05 {self._rl/2+0.1} {self._rh/2}"
          pos="{self._rw/2} 0 {self._rh/2}" material="wall_mat"/>
    <geom name="wall_w" type="box" size="0.05 {self._rl/2+0.1} {self._rh/2}"
          pos="{-self._rw/2} 0 {self._rh/2}" material="wall_mat"/>

    <!-- Furniture: table (supports objects but never blocks the agent:
         conaffinity=0 means only the agent (also conaffinity=0) ignores
         them while objects (conaffinity=1) still rest on them. Stage 20z. -->
    <body name="table" pos="{-self._rw/4} {self._rl/4} 0.4">
      <geom name="table_top" type="box" size="0.4 0.3 0.02" pos="0 0 0.4" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="table_leg1" type="cylinder" size="0.02 0.4" pos="0.35 0.25 0.2" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="table_leg2" type="cylinder" size="0.02 0.4" pos="-0.35 0.25 0.2" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="table_leg3" type="cylinder" size="0.02 0.4" pos="0.35 -0.25 0.2" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="table_leg4" type="cylinder" size="0.02 0.4" pos="-0.35 -0.25 0.2" material="furniture_mat" contype="1" conaffinity="0"/>
    </body>

    <!-- Furniture: bed -->
    <body name="bed" pos="{-self._rw/4} {-self._rl/4} 0.15">
      <geom name="bed_mat" type="box" size="0.5 0.8 0.05" pos="0 0 0.15" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="bed_pillow" type="box" size="0.3 0.2 0.06" pos="0 {-0.5} 0.25" rgba="0.9 0.9 0.9 1.0" contype="1" conaffinity="0"/>
    </body>

    <!-- Furniture: shelf -->
    <body name="shelf" pos="{self._rw/4} {self._rl/4} 0.8">
      <geom name="shelf_b1" type="box" size="0.5 0.15 0.02" pos="0 0 0.4" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="shelf_b2" type="box" size="0.5 0.15 0.02" pos="0 0 0.8" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="shelf_side1" type="box" size="0.02 0.15 0.8" pos="{-0.5} 0 0.4" material="furniture_mat" contype="1" conaffinity="0"/>
      <geom name="shelf_side2" type="box" size="0.02 0.15 0.8" pos="{0.5} 0 0.4" material="furniture_mat" contype="1" conaffinity="0"/>
    </body>

    <!-- Spread objects on floor, table, shelf -->
"""
        # Occluder walls (static, block line of sight for occlusion probe)
        # Stage 20k: contype/conaffinity=0 — the wall is a VISUAL occluder
        # only. It must block line-of-sight (pure geometry test) but never
        # physically block the agent: with 4-direction actions the agent
        # cannot steer around a wall, so a colliding wall made every
        # wall-behind object physically unreachable (teacher force 1.0
        # measured rate=0.008-0.051; agent wedged against the wall at
        # ~2.0m forever). Objects already use contype=0 (line 233).
        for oi, occ in enumerate(self._occluders):
            sx, sy, sz = occ["size"]
            px, py, pz = occ["pos"]
            xml += (f'    <body name="occluder_{oi}" pos="{px} {py} {pz}">\n'
                    f'      <geom type="box" size="{sx} {sy} {sz}" pos="0 0 0" '
                    f'rgba="0.3 0.3 0.35 1.0" mass="0" '
                    f'contype="0" conaffinity="0"/>\n'
                    f'    </body>\n')

        # Place objects — each wrapped in its own body for physics tracking
        for i, obj_data in enumerate(self._objects):
            obj = obj_data["def"]
            px, py, pz = obj_data["pos"]
            sx, sy, sz = obj.size
            if obj.kind == "sphere":
                geom = f'<geom type="sphere" size="{sx}" pos="0 0 0" mass="{obj.mass}" material="obj_{i}_mat"/>'
            elif obj.kind == "cylinder":
                geom = f'<geom type="cylinder" size="{sx} {pz}" pos="0 0 0" mass="{obj.mass}" material="obj_{i}_mat"/>'
            elif obj.kind == "capsule":
                geom = f'<geom type="capsule" size="{sx} {sz}" pos="0 0 0" mass="{obj.mass}" material="obj_{i}_mat"/>'
            else:
                geom = f'<geom type="box" size="{sx} {sy} {sz}" pos="0 0 0" mass="{obj.mass}" material="obj_{i}_mat"/>'
            xml += f'    <body name="obj_{i}" pos="{px} {py} {pz}">\n      {geom}\n    </body>\n'

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
            rgba="{c[0]} {c[1]} {c[2]} {c[3]}" mass="1.5"
            conaffinity="0"/>
    </body>"""
            else:
                xml += f"""
    <body name="{a['name']}" pos="{a['pos'][0]} {a['pos'][1]} {a['pos'][2]}">
      <geom name="{a['name']}_geom" type="sphere" size="{sz}"
            rgba="{c[0]} {c[1]} {c[2]} {c[3]}" mass="1.5"
            conaffinity="0"/>
    </body>"""

        # Third-person camera tracking the learner (targetbody mode: the
        # camera stays at a fixed offset relative to the learner body, so it
        # follows movement while keeping the scene in view).
        cam_pos = self._camera_pos
        cam_fovy = self._camera_fovy
        for marker in self._trace_marker_xml:
            xml += marker
        xml += f"""
    <camera name="ego" mode="targetbody" target="learner"
            pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}" fovy="{cam_fovy}"/>

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

    Action space (8 or 12 discrete, depends on developmental age):
        0-3: move agent (north, south, west, east)
        4-7: move agent × 2 (strong push)
        8:   grasp nearest object (dev_age > 0.15)
        9:   release held object (dev_age > 0.15)
        10:  use held object as tool (push forward) (dev_age > 0.15)
        11:  rotate (visual exploration) (dev_age > 0.15)

    Chain tasks (unlock with developmental age):
        age < 0.3:  free play
        age 0.3-0.5: obstacle clearing (push barrier to reach target)
        age > 0.5:  key-door (grasp key, bring to door, reach reward)

    Observation: render_size×render_size×3 RGB + 12/16-dim proprioceptive.
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
        camera_pos: tuple[float, float, float] = (0.0, -1.0, 0.8),
        camera_fovy: float = 60.0,
        num_occluders: int = 0,  # Stage 19: true occlusion probe walls
        approach_reward_weight: float = 0.0,  # 0=off (default); see 400K regression
        occluder_trace: bool = False,  # dev feedback: marker at last-known pos
        occluder_target_reward: float = 0.0,  # situational single-target guidance
        occluder_shaping_weight: float = 0.0,  # Stage 20c: direction-to-last-known shaping
        occluder_reveal_bonus: float = 0.0,  # Stage 20d: 揭示归因奖励 (找到=因果)
        occluder_reveal_ratio: float = 0.7,  # Stage 20d: 归因阈值 end_d < ratio*start_d
        occluder_reach_reward: float = 0.0,  # Stage 20k: 绝对到达奖励 (走进触达圈, 0=off)
        occluder_reach_radius: float = 0.8,  # Stage 20k: 触达半径 (与 eval op 硬门槛同构, 判据恒用)
        occluder_reward_radius: float = 0.0,  # Stage 20p: 奖励圈半径 (课程化 2.0→0.8, 0=跟随判据)
        occluder_reach_hold: float = 0.0,  # Stage 20n: 圈内持续分 (max(0, radius-dist)*w, 0=off)
        occluder_orient_weight: float = 0.0,  # Stage 20m: L1 朝向 - 每步非对称方向塑形 (0=off)
        occluder_orient_bonus: float = 0.0,  # Stage 20m: L1 朝向 - 事件首3步对齐一次性奖励 (0=off)
        object_crossing_every: int = 0,  # Stage 20: 物体穿越墙周期 (0=off)
        object_crossing_hold_steps: int = 0,  # Stage 20: 穿越后停在墙后步数 (0=off)
        object_crossing_fixed_object: int = -1,  # Stage 20d P1: 固定穿越物体 (-1=随机)
        object_crossing_fixed_wall: int = -1,  # Stage 20d P1: 固定穿越墙 (-1=随机)
        occluder_arrival_reveal: bool = False,  # Stage 20y: 到达即揭示 - agent 进判据圈即结束事件 (与 eval far 同构)
        occluder_arrival_deadline: int = 0,  # Stage 20z: hold 结束后再等 N 步让 agent 到达 (0=默认150)
        occluder_obs_slots: int = 0,  # Stage 20e: 遮挡记忆槽注入观测 (0=off, 容量<=3)
        occluder_teacher_force: float = 0.0,  # Stage 20h: BC 教学接管率 (0=off, 1=每步接管)
        focus_op_only: bool = False,  # Stage 20b: 课程固化 - 封闭其他目标, 专训 op
    ) -> None:
        if not _mj_available:
            raise ImportError("mujoco is required for ThreeDWorld. Run: pip install mujoco")

        self._num_objects = num_objects
        self._max_steps = max_episode_steps
        self._render_size = render_size
        self._day_cycle = day_cycle_steps
        self._action_force = action_force
        self._dev_age = developmental_age
        self._camera_pos = camera_pos
        self._camera_fovy = camera_fovy
        self._num_occluders = int(num_occluders)
        self._approach_reward_weight = float(approach_reward_weight)
        self._occluder_trace = bool(occluder_trace)
        self._occluder_target_reward = float(occluder_target_reward)
        self._occluder_shaping_weight = float(occluder_shaping_weight)
        self._occluder_reveal_bonus = float(occluder_reveal_bonus)
        self._occluder_reveal_ratio = float(occluder_reveal_ratio)
        self._occluder_reach_reward = float(occluder_reach_reward)
        self._occluder_reach_radius = float(occluder_reach_radius)
        self._occluder_reward_radius = float(occluder_reward_radius) \
            if occluder_reward_radius > 0.0 else float(occluder_reach_radius)
        self._occluder_reach_hold = float(occluder_reach_hold)
        self._occluder_orient_weight = float(occluder_orient_weight)
        self._occluder_orient_bonus = float(occluder_orient_bonus)
        # Stage 20d: reveal-attribution bonus pending delivery to the next
        # reward step (capacity 1 float, consumed once in _occluder_only_reward).
        # Reveals are detected in _track_3d_developmental_signals (which runs
        # AFTER reward), so the bonus is attributed one step later — a 1-frame
        # delay, invisible to PPO's GAE.
        self._reveal_bonus_pending = 0.0
        self._object_crossing_every = int(object_crossing_every)
        self._object_crossing_hold_steps = int(object_crossing_hold_steps)
        self._occluder_arrival_reveal = bool(occluder_arrival_reveal)
        # Stage 20z: after the crossing hold expires, keep the event alive
        # this many extra steps for the agent to physically arrive (agent
        # speed ~0.05-0.08 m/step; 150 steps covers ~7-12m).
        self._arrival_deadline_steps = int(
            occluder_arrival_deadline) if occluder_arrival_deadline else 150
        self._object_crossing_fixed_object = int(object_crossing_fixed_object)
        self._object_crossing_fixed_wall = int(object_crossing_fixed_wall)
        # Stage 20e: occluder memory slots in the observation. The policy
        # is proprio-only (vision encoder off), so without these it can
        # NEVER know where a hidden object was last seen — rewards alone
        # are unlearnable (0.11 plateau across 2.4M steps = random-base).
        # Slot content: for the closest active occlusions, the normalized
        # relative offset (dx, dy, dist) to last_known in the agent's own
        # memory — the same quantity the eval metric measures.
        self._occluder_obs_slots = max(0, min(3, int(occluder_obs_slots)))
        # Stage 20h: BC teacher scaffold — teacher takes over locomotion
        # during active occlusion windows and walks toward last_known; the
        # taken action is exposed via last_teacher_action so train.py can
        # apply an imitation loss. Eval envs keep the default 0.0 -> pure.
        self._occluder_teacher_force = float(occluder_teacher_force)
        self.last_teacher_action: int = -1  # -1 = no teacher this step
        # Stage 20j threshold-gate counters (bounded ints): tracking
        # attempts vs successes while teacher=0, same d_now<ratio*d0
        # semantic as the eval op metric. train.py snaps & resets them at
        # the end of each probe window. 20h/20i note: train.py wrote
        # env.occluder_teacher_force but there was NO property -> hasattr()
        # False -> the fade schedule silently never ran; the property
        # below is the fix (external schedule now really controls the
        # internal force).
        self._gate_arrivals = 0
        self._gate_success = 0
        # Stage 20p: reward-circle gate counters (second set): arrivals use
        # the same tracking attempts; success uses the CURRENT reward
        # circle (d_now < reward_radius). train.py promotes the curriculum
        # stage when this rate clears its threshold — the reward circle
        # shrinks, the 0.8m judge (tgate) never moves.
        self._gate_rw_arrivals = 0
        self._gate_rw_success = 0
        # Stage 20m: L1 orientation counters (bounded ints): events with
        # first-3-step alignment credit vs total events (see
        # orient_stats_snapshot_and_reset).
        self._orient_events = 0
        self._orient_aligned = 0
        self._focus_op_only = bool(focus_op_only)
        # Objects parked behind a wall after crossing (bounded: num_objects).
        # obj_id -> remaining hold steps; while held the object is reported
        # as truly_occluded so the occlusion event lasts long enough for the
        # object_permanence eval metric (end<0.7*start) to measure tracking.
        self._crossing_hold: dict[int, int] = {}
        self._occ_signal_active: list[tuple[int, float, float]] = []  # (obj_id, lk_x, lk_y)
        self._occ_signal_just_occluded: list[tuple[int, float, float]] = []
        self._occ_signal_just_revealed: list[int] = []
        self._trace_geom_ids: list[int] = []  # parallel to objects
        self._rng = np.random.RandomState(seed)

        # Stage 20-ToM: caregiver gaze behavior + false-belief tracking.
        # The caregiver "watches" the nearest visible object; when the object
        # is occluded/moved away, its believed position (last_seen) stays
        # stale = a false belief the learner can observe and reason about.
        self._cg_gaze_id: int = -1
        self._cg_last_seen: tuple[float, float] | None = None
        self._cg_gaze_every: int = 20  # re-pick gaze target cadence
        self._cg_last_gaze_dir: int = 0  # last discrete gaze direction (0-7)
        self._cg_belief_stale: bool = False

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

        # --- Developmental signal trackers (Stage 8+) ---
        self._occlusion_events: list[dict] = []
        self._force_motion_pairs: list[dict] = []
        # Per-object previous positions for velocity_after (bounded:
        # num_objects entries). 2026-09-23 fm_consistency fix.
        self._obj_prev_xy: list[tuple[float, float]] = []
        self._count_trials: list[dict] = []
        self._actions: list[int] = []
        self._object_contact_order: list[int] = []
        self._contacted: set[int] = set()
        self._last_force_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._active_occlusions_3d: dict[str, dict] = {}
        # Per-object previous distance for approach reward (bounded: num_objects)
        self._prev_obj_dist: list[float] = []
        # Logic task context (set by train.py for symbolic reward)
        self._logic_bonus_action: int | None = None
        self._logic_bonus_weight: float = 0.3

        # --- Grasping state (developmental: unlocks at age > 0.15) ---
        self._held_obj_id: int | None = None  # index into _object_lib, None = empty hand
        self._weld_eq_id: int = -1  # MuJoCo equality constraint id for weld

        # --- Chain task state (developmental: unlocks at age > 0.3) ---
        self._task_type: str = "free_play"  # "free_play", "obstacle", "key_door"
        self._task_target_body: str | None = None  # body name of target object
        self._task_barrier_body: str | None = None  # body name of barrier
        self._task_key_body: str | None = None  # body name of key
        self._task_door_open: bool = False
        self._task_progress: float = 0.0  # 0..1
        self._task_reward_collected: bool = False

        # --- Grasp/carry/tool-use signal tracking (for milestone evaluation) ---
        self._grasp_carry_events: list[dict] = []
        self._tool_use_events: list[dict] = []
        self._release_events: list[dict] = []
        self._grasp_start_pos: tuple[float, float] | None = None  # agent pos when grasp started

        # Initial reset
        self._build_scene()

    # ------------------------------------------------------------------ properties

    @property
    def occluder_teacher_force(self) -> float:
        """Stage 20h/20j: live BC teacher takeover rate (0=off, 1=every step).

        Stage 20j fix: this property is the ONLY way train.py's schedule
        reaches the internal ``_occluder_teacher_force``. Before it existed
        (20h/20i) train.py assigned ``env.occluder_teacher_force`` directly,
        which silently created a dead instance attribute; ``hasattr`` was
        True-ish only after that assignment, the fade never ran, and the
        teacher stayed at the constructor value for 4.5M steps.
        """
        return self._occluder_teacher_force

    @occluder_teacher_force.setter
    def occluder_teacher_force(self, value: float) -> None:
        self._occluder_teacher_force = float(value)

    @property
    def occluder_reveal_ratio(self) -> float:
        """Stage 20j: live reveal-attribution ratio (eval keeps its own kw)."""
        return self._occluder_reveal_ratio

    @occluder_reveal_ratio.setter
    def occluder_reveal_ratio(self, value: float) -> None:
        self._occluder_reveal_ratio = float(value)

    @property
    def occluder_reward_radius(self) -> float:
        """Stage 20p: live reward-circle radius (curriculum 2.0→0.8).
        The tgate/eval JUDGE radius (occluder_reach_radius) stays fixed at
        0.8m; only this TEACHING circle shrinks as the agent masters each
        stage (measurement never polluted, reward gets the curriculum)."""
        return self._occluder_reward_radius

    @occluder_reward_radius.setter
    def occluder_reward_radius(self, value: float) -> None:
        self._occluder_reward_radius = float(value)

    @property
    def occluder_orient_weight(self) -> float:
        """Stage 20m: live L1-orientation shaping weight (train.py fades it
        when the orientation sub-goal is mastered)."""
        return self._occluder_orient_weight

    @occluder_orient_weight.setter
    def occluder_orient_weight(self, value: float) -> None:
        self._occluder_orient_weight = float(value)

    def gate_stats_snapshot_and_reset(self) -> tuple[int, int, int]:
        """Stage 20j/20p: (arrivals, 0.8m-judge successes, reward-circle
        successes) since the last probe-window end. The reward-circle count
        feeds the curriculum promoter (reward_radius 2.0→0.8); the judge
        count feeds teacher release (tgate)."""
        snap = (self._gate_arrivals, self._gate_success, self._gate_rw_success)
        self._gate_arrivals = 0
        self._gate_success = 0
        self._gate_rw_arrivals = 0
        self._gate_rw_success = 0
        return snap

    def orient_stats_snapshot_and_reset(self) -> tuple[int, int]:
        """Stage 20m: (events, first3-aligned) since last probe-window end.

        ``aligned`` counts events whose first-3-step mean displacement
        vs last_known direction cosine > 0.5 (the L1 orientation
        sub-goal: "turned the right way right after the object
        vanished"). train.py fades the orientation shaping when this
        rate clears its threshold.
        """
        snap = (self._orient_events, self._orient_aligned)
        self._orient_events = 0
        self._orient_aligned = 0
        return snap

    @property
    def action_space_n(self) -> int:
        # Always 12: grasping actions (8-11) are present from the start
        # but only have effect when dev_age > 0.15. Before that they
        # fall through to locomotion (action % 8). This ensures the
        # model has 12 output heads from day 1, so checkpoints are
        # compatible across developmental stages.
        return 12

    # Stage 20q: 8-direction locomotion (cardinal + diagonal). Actions 0-7 are
    # 8 unit directions; actions 8-11 remain grasp/release/use/rotate (gated by
    # dev_age). Previously 4-7 were a 2x-force gear of the SAME 4 cardinal dirs,
    # so effective locomotion was only 4 directions. Repurposing 4-7 as diagonals
    # gives finer angular approach control WITHOUT changing num_actions (still
    # 12) -> 8M checkpoints resume directly, only the action semantics differ.
    _EIGHT_DIRS = (
        (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0),
        (0.70710678, 0.70710678), (-0.70710678, 0.70710678),
        (-0.70710678, -0.70710678), (0.70710678, -0.70710678),
    )

    @staticmethod
    def _action_direction(action: int) -> tuple[float, float]:
        """Unit (dx, dy) for a locomotion action. Pure-logic helper so the
        8-direction mapping is testable without a MuJoCo instance. Actions
        0-7 are the 8 directions; 8-11 map via %8 to a cardinal dir (pre
        dev_age unlock)."""
        return ThreeDWorld._EIGHT_DIRS[int(action) % 8]

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self._render_size, self._render_size, 3)

    @property
    def proprio_dim(self) -> int:
        # Always 18 = base(13: pos3+vel3+touch3+joints2+yaw2) + grasp(5),
        # grasping fields are zeros until dev_age > 0.15. Ensures checkpoint
        # compatibility.
        # Stage 20e: occluder memory slots appended (3 dims per slot).
        # Stage 20f: yaw (cos,sin) in base so the policy can map the global
        # (dx,dy) last_known slots to LOCAL turn/forward actions — without
        # orientation the slots are unaddressable (op stuck at 0.11).
        return 18 + 3 * self._occluder_obs_slots

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

    def get_occlusion_signal(self) -> dict:
        """Return one-step occlusion signals for the hypothesis-deduction loop.

        Returns:
            {
              "active": [(obj_id, lk_x, lk_y), ...],   # currently occluded
              "just_occluded": [(obj_id, lk_x, lk_y), ...],  # new this step
              "just_revealed": [obj_id, ...],           # revealed this step
            }
        """
        active = [
            (int(k.split("_")[1]), float(v["last_known"][0]), float(v["last_known"][1]))
            for k, v in self._active_occlusions_3d.items()
            if v.get("truly_occluded", False)
        ]
        sig = {
            "active": active,
            "just_occluded": list(self._occ_signal_just_occluded),
            "just_revealed": list(self._occ_signal_just_revealed),
        }
        self._occ_signal_just_occluded = []
        self._occ_signal_just_revealed = []
        return sig

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
        action = int(action) % self.action_space_n
        agent_name = self._agent_names[0]

        # --- Stage 20h: BC teacher scaffold override ---
        # While an occlusion is genuinely active (object truly hidden), the
        # teacher may take over locomotion: walk the learner toward the
        # last_known position. This guarantees the agent repeatedly
        # experiences "search -> reveal" and hands train.py an imitation
        # label (last_teacher_action) for the BC auxiliary loss. Ramp-down
        # is driven from train.py via the occluder_teacher_force attribute.
        self.last_teacher_action = -1
        if self._occluder_teacher_force > 0.0 and self._active_occlusions_3d:
            try:
                ax = float(self._data.body("learner").xpos[0])
                ay = float(self._data.body("learner").xpos[1])
                best_dir, best_dot, best_dist = 0, -1.0, 1e9
                for _key, _occ in list(self._active_occlusions_3d.items()):
                    if not _occ.get("truly_occluded", False):
                        continue
                    _lk = _occ["last_known"]
                    _dist = math.hypot(_lk[0] - ax, _lk[1] - ay)
                    # Stage 20k: teacher must stop STRICTLY INSIDE the
                    # success circle, not on its edge — and the circle is
                    # min(ratio*d0, radius), NOT a fixed radius: for close
                    # starts (d0 < 0.94m with ratio 0.85) the criterion is
                    # tighter than 0.8m. Fixed stop lines fail near starts
                    # (teacher 1.0 measured rate=0.008-0.051). Dynamic
                    # margin 0.875 clears both the ratio and radius terms.
                    # Stage 20p: the stop circle follows the CURRICULUM
                    # reward_radius (2.0→0.8) — the teacher demonstrates
                    # arrival at the current stage's circle, not the final
                    # 0.8m (no jump-ahead teaching).
                    _t0 = _occ.get("agent_traj_during_occ", [(ax, ay)])
                    _d0 = math.hypot(_t0[0][0] - _lk[0], _t0[0][1] - _lk[1])
                    if _dist < min(
                            self._occluder_reveal_ratio * _d0,
                            self._occluder_reward_radius) * 0.875:
                        continue  # already there: stillness is the correct move
                    # Stage 20m#4: track the NEAREST truly-occluded object
                    # (the observation-slot #1 target), not the one with the
                    # most aligned direction. The L1 orientation signal
                    # (first-3-step credit) measures heading against the
                    # nearest lk; a teacher walking the best-dot object was
                    # misaligned with the measured target (0.16 even at
                    # force 1.0).
                    if _dist >= best_dist:
                        continue
                    best_dist = _dist
                    _tx, _ty = (_lk[0] - ax) / _dist, (_lk[1] - ay) / _dist
                    # Stage 20q: pick among all 8 directions (cardinal+diagonal)
                    # for the finest angular approach. No 2x-force gear — uniform
                    # speed keeps the demonstrated trajectory clean and matches the
                    # learner's own action set (actions 0-7 = 8 dirs now).
                    for _d in range(8):
                        _dot = _tx * self._EIGHT_DIRS[_d][0] + _ty * self._EIGHT_DIRS[_d][1]
                        if _dot > best_dot:
                            best_dot, best_dir = _dot, _d
                if best_dist < 1e9 and self._rng.rand() < self._occluder_teacher_force:
                    self.last_teacher_action = best_dir
                    action = self.last_teacher_action
                elif best_dist >= 1e9 and self._rng.rand() < self._occluder_teacher_force:
                    # Stage 20h#5: already at last_known -> teach STILLNESS.
                    # Action 11 (rotate, visual explore) is zero-force at
                    # dev_age>0.15, so BC teaches "hold position" — exactly
                    # what the eval's end_d<0.7*d0 needs at reveal time
                    # (3400K: policy roams, hits frames 38x but the reveal
                    # moment drifts -> op stuck 0.08).
                    # Stage 20h#6: was action 8 (grasp) — grasp with reach
                    # ~0.5m picks up revealed objects standing right next to
                    # the agent, corrupting last_known. 11 has no side effect.
                    self.last_teacher_action = 11
                    action = 11
            except Exception as _e:
                _expose_exc("occluder_teacher")

        # --- Grasping actions (8-11): only effective when dev_age > 0.15 ---
        # Before unlocking, actions 8-11 fall through to locomotion (action % 8)
        if action >= 8 and self._dev_age > 0.15:
            if action == 8:
                self._grasp()
            elif action == 9:
                self._release()
            elif action == 10:
                self._use_held_as_tool()
            elif action == 11:
                pass  # rotate: visual exploration
            dx, dy = 0.0, 0.0
        else:
            # Apply force via position actuators (actions 0-7 = 8 directions;
            # 8-11 mapped via %8 to a cardinal dir before dev_age unlocks).
            # Stage 20q: all 8 directions use the same base force (no gear) so
            # the agent has uniform-speed, finer-angular control to reach a
            # tight 0.8m circle precisely.
            eff_action = int(action) % 8
            ux, uy = self._EIGHT_DIRS[eff_action]
            force = self._action_force
            dx = force * ux
            dy = force * uy

        # Move agent target position via velocity control
        try:
            # Get agent body and apply velocity
            body_id = self._model.body(agent_name).id
            dof_addr = self._model.body_dofadr[body_id]
            if dof_addr >= 0:
                self._data.qvel[dof_addr] += dx * self._model.opt.timestep
                self._data.qvel[dof_addr + 1] += dy * self._model.opt.timestep
        except Exception:
            pass  # legit: agent body may not exist for this name
        # For agents with actuators, also set control targets
        try:
            act_x_id = self._model.actuator(f"{agent_name}_act_x").id
            act_y_id = self._model.actuator(f"{agent_name}_act_y").id
            self._data.ctrl[act_x_id] = self._data.qpos[self._model.jnt_qposadr[self._model.joint(f"{agent_name}_x").id]] + dx * 0.01
            self._data.ctrl[act_y_id] = self._data.qpos[self._model.jnt_qposadr[self._model.joint(f"{agent_name}_y").id]] + dy * 0.01
        except Exception:
            pass  # legit: velocity-controlled agents have no actuators

        # Advance day/night
        self._sun_angle = (self._step_count % self._day_cycle) / self._day_cycle * 2 * np.pi
        self._step_count += 1

        # Physics step
        mujoco.mj_step(self._model, self._data)

        # Sync held object position (virtual grasp)
        if self._held_obj_id is not None:
            self._sync_held_object()

        # Reward
        reward = self._compute_reward(action)
        self._current_return += reward

        done = self._step_count >= self._max_steps

        # --- Developmental signal tracking (Stage 8+) ---
        self._actions.append(action)
        self._track_3d_developmental_signals(dx, dy)

        # --- Stage 20: object crossing (dense occlusion practice) ---
        # Every object_crossing_every steps, mirror one random object across
        # an occluder wall so the agent repeatedly witnesses occlusion ->
        # reveal events (hypothesis-deduction training material).
        # Stage 20d P1: fixed target/wall curriculum — one ever-same object
        # and wall, so "which object disappears -> where to find it" is a
        # stable, learnable mapping instead of a fresh random target every
        # 50 steps (random targets gave the policy no constant concept).
        if self._object_crossing_every > 0 and self._num_occluders > 0 \
                and self._step_count % self._object_crossing_every == 0:
            try:
                # Stage 20-ToM: 50% of crossings deliberately move the object
                # the caregiver is currently watching — the caregiver's belief
                # (last_seen) then goes stale = a FALSE BELIEF the learner can
                # observe. Without this, stale_frac stayed ~0 and the ToM
                # module only ever learned ordinary gaze prediction.
                _ci = None
                if (self._cg_gaze_id >= 0 and self._cg_gaze_id not in self._crossing_hold
                        and self._rng.rand() < 0.7):
                    _ci = int(self._cg_gaze_id)
                if _ci is None:
                    _ci = int(self._object_crossing_fixed_object) if self._object_crossing_fixed_object >= 0 \
                        else int(self._rng.randint(0, self._num_objects))
                if _ci in self._crossing_hold:
                    if self._object_crossing_fixed_object >= 0:
                        _ci = (self._object_crossing_fixed_object + 1) % self._num_objects
                    else:
                        _ci = int(self._rng.randint(0, self._num_objects))
                # Stage 20y: if the (possibly wrapped) pick is STILL in hold,
                # skip this crossing entirely — re-mirroring an object whose
                # occlusion event is still active teleports its last_known,
                # making the target unreachable for any controller (the
                # teacher chases a jumping target: measured success collapse).
                # With few objects and hold > crossing_every this used to
                # re-mirror the SAME object every cycle.
                if _ci in self._crossing_hold:
                    _crossing_skipped = True
                else:
                    _crossing_skipped = False
                if not _crossing_skipped:
                    _bid = self._model.body(f"obj_{_ci}").id
                    _occ_i = int(self._object_crossing_fixed_wall) if self._object_crossing_fixed_wall >= 0 \
                        else int(self._rng.randint(0, self._num_occluders))
                    _occ_id = self._model.body(f"occluder_{_occ_i}").id
                    _ocx = float(self._data.xpos[_occ_id, 0])
                    _ocy = float(self._data.xpos[_occ_id, 1])
                    _px = float(self._data.xpos[_bid, 0])
                    _py = float(self._data.xpos[_bid, 1])
                    # Mirror position across the wall (keep z)
                    _mx = 2.0 * _ocx - _px
                    _my = 2.0 * _ocy - _py
                    # Stage 20z: never mirror the object OUTSIDE the room —
                    # the agent is confined by the room walls (4x4m, objects
                    # spawn within +-1.8), so a wall-exterior last_known is
                    # unreachable and the event can never finalize (measured:
                    # events=0 across episodes). Mirrors SceneBuilder default
                    # room (4.0, 4.0, 2.5) and the +-1.8 spawn bound.
                    _hb = 1.8  # in-room bound (same as object spawn limit)
                    if abs(_mx) > _hb or abs(_my) > _hb:
                        _crossing_skipped = True
                    # Stage 20z (A-fix): never mirror the object INTO a
                    # colliding furniture region — MuJoCo ejects overlapping
                    # bodies with huge velocity (object flies to hundreds of
                    # meters, then every later crossing mirrors it further).
                    # Furniture/caregiver zones (SceneBuilder layout):
                    #   table (-1, 1) r~0.6 | bed (-1,-1) r~1.0
                    #   shelf (1, 1) r~0.7  | caregiver (-1.2, 0.8) r~0.35
                    if not _crossing_skipped:
                        _fd = math.hypot(_mx + 1.0, _my - 1.0)   # table
                        if _fd < 0.7:
                            _crossing_skipped = True
                        _fd = math.hypot(_mx + 1.0, _my + 1.0)   # bed
                        if not _crossing_skipped and _fd < 1.05:
                            _crossing_skipped = True
                        _fd = math.hypot(_mx - 1.0, _my - 1.0)   # shelf
                        if not _crossing_skipped and _fd < 0.75:
                            _crossing_skipped = True
                        _fd = math.hypot(_mx + 1.2, _my - 0.8)   # caregiver
                        if not _crossing_skipped and _fd < 0.4:
                            _crossing_skipped = True
                if not _crossing_skipped:
                    # Mirror position across the wall (keep z)
                    self._model.body_pos[_bid] = np.array([
                        2.0 * _ocx - _px, 2.0 * _ocy - _py,
                        float(self._data.xpos[_bid, 2]),
                    ])
                    # Keep the object behind the wall for hold steps so the
                    # occlusion persists long enough to be measured by the eval
                    # metric (previously the event closed within 1-2 steps and
                    # trajectories <3 points were dropped -> op measured ~0).
                    if self._object_crossing_hold_steps > 0:
                        self._crossing_hold[_ci] = int(self._object_crossing_hold_steps)
                        # Regenerate the occlusion record fresh (it may have
                        # already been closed by an earlier reveal).
                        _key = f"occ_{_ci}"
                        if _key not in self._active_occlusions_3d:
                            self._active_occlusions_3d[_key] = {
                                "last_known": (2.0 * _ocx - _px, 2.0 * _ocy - _py),
                                "agent_traj_during_occ": [(float(self._data.body("learner").xpos[0]),
                                                           float(self._data.body("learner").xpos[1]))],
                                "truly_occluded": True,
                                "is_crossing": True,  # Stage 20z: arrival semantics for crossing events
                            }
                        self._occ_signal_just_occluded.append(
                            (_ci, 2.0 * _ocx - _px, 2.0 * _ocy - _py))
                    else:
                        # Legacy instant-crossing: snap any active occlusion record
                        _key = f"occ_{_ci}"
                        if _key in self._active_occlusions_3d:
                            _ev = self._active_occlusions_3d.pop(_key)
                            if len(_ev["agent_traj_during_occ"]) >= 3:
                                self._occlusion_events.append(_ev)
                            # Crossing = object re-appeared: emit reveal signal
                            self._occ_signal_just_revealed.append(_ci)
            except Exception as _e:
                _expose_exc("object_crossing_teleport")

        if done:
            # Finalize count trial: count objects within agent's awareness radius
            try:
                learner_id = self._model.body("learner").id if self._model else None
                ax = float(self._data.xpos[learner_id, 0]) if learner_id is not None else 0.0
                ay = float(self._data.xpos[learner_id, 1]) if learner_id is not None else 0.0
                # Count objects within 3.0 distance (wider 3D awareness)
                nearby = 0
                for i in range(self._num_objects):
                    try:
                        body_id = self._model.body(f"obj_{i}").id
                        ox = float(self._data.xpos[body_id, 0])
                        oy = float(self._data.xpos[body_id, 1])
                        if ((ax - ox)**2 + (ay - oy)**2) ** 0.5 < 3.0:
                            nearby += 1
                    except Exception:
                        continue  # legit: per-object loop, obj_ may be gone
                self._count_trials.append({
                    "true_count": self._num_objects,
                    "estimated_count": max(nearby, len(self._contacted)),
                })
            except Exception as _e:
                _expose_exc("count_finalize")

        if done:
            self._episode_returns.append(self._current_return)
            if len(self._episode_returns) > 1024:  # BOUNDS-OK: rolling window cap
                self._episode_returns = self._episode_returns[-1024:]
            if self._auto_reset:
                self._rebuild_scene()
            self._current_return = 0.0
            self._step_count = 0

        return EnvStep3D(
            obs=self._render(),
            reward=reward,
            terminated=done,
            truncated=done,
            info={
                "step": self._step_count, "dev_age": self._dev_age, "sun_angle": self._sun_angle,
                "occlusion_events": self._occlusion_events,
                "force_motion_pairs": self._force_motion_pairs,
                "count_trials": self._count_trials,
                "actions": self._actions,
                "object_contact_order": self._object_contact_order,
                "task_type": self._task_type,
                "task_progress": self._task_progress,
                "held_obj": self._held_obj_id,
                "means_ends_score": self._task_progress if self._task_reward_collected else 0.0,
                "grasp_carry_events": self._grasp_carry_events,
                "tool_use_events": self._tool_use_events,
                "release_events": self._release_events,
            },
            proprio=self._proprio(),
        )

    # ------------------------------------------------------------------ internals

    def _build_scene(self) -> None:
        """Build MuJoCo scene from scratch."""
        builder = SceneBuilder(camera_pos=self._camera_pos, camera_fovy=self._camera_fovy)

        # Occluder walls: place them between the center and random ring
        # positions so they can actually block line of sight (Stage 19).
        for _oi in range(self._num_occluders):
            ang = self._rng.uniform(0.0, 2 * np.pi)
            rad = self._rng.uniform(0.6, 1.4)
            builder.add_occluder(
                position=(float(rad * np.cos(ang)), float(rad * np.sin(ang)), 0.35),
                size=(0.55, 0.05, 0.6),
            )

        # Visual occlusion traces (developmental feedback, training only):
        # hidden ground markers moved to last-known positions during occlusion.
        if self._occluder_trace:
            builder.add_trace_markers(min(self._num_objects, 24))

        # Agent size grows with developmental age
        agent_size = 0.12 + self._dev_age * 0.08  # 0.12 (infant) → 0.20 (child)
        self._agent_size = agent_size

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

        # Record trace-marker geom ids (parallel to objects)
        self._trace_geom_ids = []
        if self._occluder_trace:
            for i in range(min(self._num_objects, 24)):
                gid = self._model.geom(f"trace_marker_{i}").id
                self._trace_geom_ids.append(gid)
                self._model.geom_pos[gid] = [0.0, 0.0, 100.0]  # hidden
        mujoco.mj_forward(self._model, self._data)

        # Reset developmental trackers on scene rebuild
        self._occlusion_events = []
        self._force_motion_pairs = []
        self._obj_prev_xy: list[tuple[float, float]] = []
        self._count_trials = []
        self._actions = []
        self._object_contact_order = []
        self._contacted = set()
        self._active_occlusions_3d = {}
        self._crossing_hold = {}
        self._occ_signal_active = []
        self._occ_signal_just_occluded = []
        self._occ_signal_just_revealed = []
        # Stage 20-ToM: reset caregiver gaze/belief state on scene rebuild.
        self._cg_gaze_id = -1
        self._cg_last_seen = None
        self._cg_last_gaze_dir = 0
        self._cg_belief_stale = False
        for gid in self._trace_geom_ids:
            self._model.geom_pos[gid] = [0.0, 0.0, 100.0]
        self._prev_obj_dist = [0.0] * self._num_objects

        # Reset grasping state
        self._held_obj_id = None
        self._grasp_start_pos = None

        # Reset signal trackers
        self._grasp_carry_events = []
        self._tool_use_events = []
        self._release_events = []

        # Set up chain task based on developmental age
        self._task_reward_collected = False
        self._task_progress = 0.0
        self._task_door_open = False
        self._setup_chain_task()

    def _rebuild_scene(self) -> None:
        self._build_scene()

    def _render(self) -> np.ndarray:
        """Render current scene to (H, W, 3) uint8 RGB."""
        if self._renderer is None:
            return np.zeros((self._render_size, self._render_size, 3), dtype=np.uint8)

        # Update scene state
        mujoco.mj_forward(self._model, self._data)

        # Update renderer scene (fixed third-person camera)
        self._renderer.update_scene(self._data, camera="ego")

        # Render offscreen — mujoco >= 3.x returns (H, W, 3) uint8 already
        pixels = self._renderer.render()
        if pixels.dtype != np.uint8:
            pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        return pixels

    # ------------------------------------------------------------------ dev signals

    def _track_3d_developmental_signals(self, dx: float, dy: float) -> None:
        """Accumulate per-step developmental signals for milestone evaluation.

        3D equivalents of PhysicsSandbox's _track_developmental_signals.
        """
        if self._model is None or self._data is None:
            return

        # --- force_motion_pairs ---
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            learner_id = self._model.body("learner").id
            ax = float(self._data.xpos[learner_id, 0])
            ay = float(self._data.xpos[learner_id, 1])
            # Movement direction
            mag = (dx**2 + dy**2) ** 0.5
            ux, uy = dx / max(mag, 0.01), dy / max(mag, 0.01)

            for i in range(self._num_objects):
                try:
                    body_id = self._model.body(f"obj_{i}").id
                    ox = float(self._data.xpos[body_id, 0])
                    oy = float(self._data.xpos[body_id, 1])
                    rel_x, rel_y = ox - ax, oy - ay
                    rel_dist = (rel_x**2 + rel_y**2) ** 0.5
                    reach = self._contact_reach(i)
                    # Object in front and close enough to be pushed
                    if ux * rel_x + uy * rel_y > 0 and rel_dist < reach:
                        # velocity_after: the object's ACTUAL per-step
                        # displacement. The old code read
                        # ``self._data.qvel[body_id, 0/1]`` — but in this
                        # kinematic-object model qvel is 1-D (shape (2,)),
                        # so the two-index access ALWAYS raised IndexError
                        # and force_motion_pairs stayed empty (2026-09-23
                        # diagnosis: 0 pairs in 12k eval steps).
                        #
                        # Honesty rule (anti-self-deception): a pair is only
                        # recorded when the object ACTUALLY MOVED. In the
                        # current kinematic 3D world objects are static
                        # (verified: 0 of 309 near-object events displaced
                        # the object), so counting proximity events would
                        # let fm_consistency report force->motion
                        # understanding that does not exist. Static objects
                        # -> empty pairs -> the component honestly reads 0
                        # until the world gains real push physics.
                        _prev = (self._obj_prev_xy[i]
                                 if i < len(self._obj_prev_xy) else (ox, oy))
                        _vx, _vy = ox - _prev[0], oy - _prev[1]
                        if (abs(_vx) + abs(_vy)) > 1e-4:
                            self._force_motion_pairs.append({
                                "force": (dx, dy),
                                "velocity_after": (_vx, _vy),
                                "object_id": i,
                            })
                except Exception:
                    continue  # legit: per-object loop, obj_ may be gone

        # --- Stage 20-ToM: caregiver gaze / false-belief tracking ---
        self._update_caregiver_gaze()

        # --- occlusion_events (3D: far objects with multi-step agent trajectory) ---
        learner_id = self._model.body("learner").id
        ax = float(self._data.xpos[learner_id, 0])
        ay = float(self._data.xpos[learner_id, 1])
        for i in range(self._num_objects):
            try:
                body_id = self._model.body(f"obj_{i}").id
                ox = float(self._data.xpos[body_id, 0])
                oy = float(self._data.xpos[body_id, 1])
                dist = ((ax - ox)**2 + (ay - oy)**2) ** 0.5
                # Stage 20: a crossing object parked behind the wall stays
                # "truly_occluded" for its remaining hold steps so the event
                # persists (eval op metric needs a multi-step trajectory).
                key = f"occ_{i}"
                held = self._crossing_hold.get(i, 0)
                _finalized = False
                if held > 0:
                    truly_occluded = True
                    if held <= 1:
                        # Hold timer expired. Legacy: judge + reveal here at a
                        # fixed cadence (usually a miss — the agent has not
                        # arrived yet). Stage 20y arrival_reveal mode: the
                        # event is NOT over — the object stays occluded until
                        # the agent finds it (arrival) or a deadline expires,
                        # matching the eval far probe semantics.
                        self._crossing_hold.pop(i, None)
                        if not self._occluder_arrival_reveal:
                            self._maybe_reveal_bonus(key)
                            self._occ_signal_just_revealed.append(i)
                    else:
                        self._crossing_hold[i] = held - 1
                else:
                    truly_occluded = self._line_of_sight_blocked(ax, ay, ox, oy)
                    # Stage 20z#2 (B-fix): in arrival_reveal mode, LOS
                    # unblocked must NOT end an event — the agent sees the
                    # object from the side while still 1-3m away, and ending
                    # there judged a miss before arrival (measured: events
                    # finalized at d=0.94/1.45m, arrival rate 0/80). Keep the
                    # event alive (as if occluded) until the agent enters the
                    # judge circle (success) or the deadline.
                    # Stage 21 (A-fix): applies to REAL LOS events too (not
                    # just crossing) — when the agent wanders past a wall and
                    # an object behind it becomes occluded, the event now
                    # persists and (a) the teacher demonstrates walking to
                    # last_known on LOS events as well, and (b) arrival pays
                    # off, teaching proactive search ("object hidden -> find
                    # it") — the far-probe behavior.
                    if (self._occluder_arrival_reveal
                            and key in self._active_occlusions_3d):
                        _evc = self._active_occlusions_3d[key]
                        _dl = _evc.get("deadline")
                        if _dl is None:
                            _dl = self._step_count + self._arrival_deadline_steps
                            _evc["deadline"] = _dl
                        if self._step_count < _dl:
                            truly_occluded = True
                # Stage 20y: arrival-triggered reveal — if the agent has
                # entered the judge circle around last_known, end the
                # occlusion NOW (train/eval isomorphism: the eval far probe
                # ends when the agent finds the object). Checked on EVERY
                # occluded step (held or LOS) so a slow agent that arrives
                # after the hold timer expired still scores the arrival.
                if (self._occluder_arrival_reveal and truly_occluded
                        and not _finalized and key in self._active_occlusions_3d):
                    try:
                        _ev = self._active_occlusions_3d[key]
                        _lk = _ev.get("last_known")
                        _tr = _ev.get("agent_traj_during_occ") or []
                        if _lk is not None and len(_tr) >= 1:
                            _d0 = math.hypot(_tr[0][0] - _lk[0], _tr[0][1] - _lk[1])
                            _dn = math.hypot(ax - _lk[0], ay - _lk[1])
                            if _dn < min(
                                    self._occluder_reveal_ratio * _d0,
                                    self._occluder_reach_radius):
                                # Arrived: finalize success NOW. Append the
                                # CURRENT position first — the tracking block
                                # below is skipped on finalize, so without
                                # this the trajectory would end at the PREVIOUS
                                # step (best_d ~0.98m > 0.8m -> judged a miss
                                # even though the agent reached 0.78m this step;
                                # measured: arrival rate 0/83 in verify).
                                _ev["agent_traj_during_occ"].append((ax, ay))
                                self._maybe_reveal_bonus(key)
                                _ev2 = self._active_occlusions_3d.pop(key)
                                if len(_ev2.get("agent_traj_during_occ") or []) >= 3:
                                    self._occlusion_events.append(_ev2)
                                if self._occluder_trace and i < len(self._trace_geom_ids):
                                    self._model.geom_pos[self._trace_geom_ids[i]] = [0.0, 0.0, 100.0]
                                self._occ_signal_just_revealed.append(i)
                                _finalized = True
                            elif held == 0 and self._occluder_arrival_reveal:
                                # Deadline logic (arrival_reveal, post-hold):
                                # event held alive past its deadline without an
                                # arrival -> finalize now (judged miss unless
                                # actually inside the circle). Prevents events
                                # hanging forever when the agent wanders off.
                                _evc2 = self._active_occlusions_3d.get(key)
                                _dl2 = _evc2.get("deadline") if _evc2 else None
                                if _dl2 is not None and self._step_count >= _dl2:
                                    _evc2["agent_traj_during_occ"].append((ax, ay))
                                    self._maybe_reveal_bonus(key)
                                    _ev3 = self._active_occlusions_3d.pop(key)
                                    if len(_ev3.get("agent_traj_during_occ") or []) >= 3:
                                        self._occlusion_events.append(_ev3)
                                    if self._occluder_trace and i < len(self._trace_geom_ids):
                                        self._model.geom_pos[self._trace_geom_ids[i]] = [0.0, 0.0, 100.0]
                                    self._occ_signal_just_revealed.append(i)
                                    _finalized = True
                    except Exception:
                        pass
                if truly_occluded or self._num_occluders == 0:
                    # Stage 20y: if arrival already finalized this event,
                    # skip tracking (the pop above removed the key; without
                    # this guard the block would resurrect a fresh event).
                    if _finalized:
                        continue
                    # Track per-object trajectory over multiple steps.
                    # NOTE: no dist>0.8 gate here (historical bug): a reaching
                    # agent (<0.8m from last_known) used to terminate the event
                    # -> the eval op denominator silently dropped exactly the
                    # successful tracking episodes. The event now persists
                    # while the object stays occluded and only finalizes when
                    # it re-becomes visible (else branch below), so "arrived
                    # and waited for the reveal" scores as tracking success.
                    if key not in self._active_occlusions_3d:
                        self._active_occlusions_3d[key] = {
                            "last_known": (ox, oy),
                            "agent_traj_during_occ": [],
                            "truly_occluded": bool(truly_occluded),
                        }
                        # Dev feedback: show ground marker at last-known pos
                        if self._occluder_trace and i < len(self._trace_geom_ids):
                            self._model.geom_pos[self._trace_geom_ids[i]] = [ox, oy, 0.01]
                        # Stage 20: occlusion signal (hypothesis-deduction input)
                        if truly_occluded:
                            self._occ_signal_just_occluded.append((i, float(ox), float(oy)))
                    self._active_occlusions_3d[key]["agent_traj_during_occ"].append((ax, ay))
                else:
                    # Object re-became visible — finalize and emit event
                    key = f"occ_{i}"
                    if key in self._active_occlusions_3d:
                        self._maybe_reveal_bonus(key)
                        ev = self._active_occlusions_3d.pop(key)
                        if self._occluder_trace and i < len(self._trace_geom_ids):
                            self._model.geom_pos[self._trace_geom_ids[i]] = [0.0, 0.0, 100.0]
                        if len(ev["agent_traj_during_occ"]) >= 3:
                            self._occlusion_events.append(ev)
                        # Stage 20: reveal signal (verification feedback)
                        self._occ_signal_just_revealed.append(i)
            except Exception:
                continue  # legit: per-object loop, obj_ may be gone

        # --- object contact tracking ---
        for i in range(self._num_objects):
            if i in self._contacted:
                continue
            try:
                body_id = self._model.body(f"obj_{i}").id
                ox = float(self._data.xpos[body_id, 0])
                oy = float(self._data.xpos[body_id, 1])
                d = ((ax - ox)**2 + (ay - oy)**2) ** 0.5
                if d < self._contact_reach(i):
                    self._contacted.add(i)
                    self._object_contact_order.append(i)
            except Exception:
                continue  # legit: per-object loop, obj_ may be gone

        # Remember object positions for next step's velocity_after
        # (bounded: one (x, y) tuple per object, 2026-09-23).
        try:
            self._obj_prev_xy = [
                (float(self._data.xpos[self._model.body(f"obj_{i}").id, 0]),
                 float(self._data.xpos[self._model.body(f"obj_{i}").id, 1]))
                for i in range(self._num_objects)
            ]
        except Exception:
            self._obj_prev_xy = []  # legit: scene rebuilt mid-call

    @staticmethod
    def _approach_ok(d_now: float, d0: float, ratio: float, radius: float) -> bool:
        """Stage 20k: op success criterion — isomorphic to the eval metric
        (best_d < min(0.7*start_d, 0.8m)). The ratio alone lets a 6m-away
        object count as "found" at 5.1m; the absolute radius floor demands
        an actual arrival for far starts while keeping the ratio's
        softening for close ones."""
        return d0 >= 1e-6 and d_now < min(ratio * d0, radius)

    def _maybe_reveal_bonus(self, key: str) -> None:
        """Stage 20d: attribute a large bonus when an occlusion REVEALS.

        The causal frame the dense distance/shaping rewards cannot teach:
        "I tracked the hidden object and it reappeared where I was looking."
        Mirrors the eval op metric exactly (end_d < ratio*start_d and >= 3
        trajectory points), so the training signal is the same quantity the
        eval measures — a true success attribution, not a shortcut.

        Delivery: sets a single pending float (capacity 1) consumed by the
        NEXT call to _occluder_only_reward (reveals fire after reward in the
        step's signal tracking). 1-frame delay is invisible to PPO's GAE.
        """
        if self._occluder_reveal_bonus <= 0.0:
            return
        ev = self._active_occlusions_3d.get(key)
        if not ev:
            return
        traj = ev.get("agent_traj_during_occ")
        if not traj or len(traj) < 3:
            return  # same validity gate as the eval metric
        lk = ev["last_known"]
        ax = float(self._data.body("learner").xpos[0])
        ay = float(self._data.body("learner").xpos[1])
        sx, sy = traj[0]
        d0 = math.hypot(sx - lk[0], sy - lk[1])
        d_now = math.hypot(ax - lk[0], ay - lk[1])
        # Stage 20j threshold-gate probe counters: every valid tracking
        # attempt counts an arrival; an approach success counts a hit.
        # Same semantic as the eval op metric (but isomorphized to the
        # absolute-reach hard floor in Stage 20k, see _approach_ok).
        self._gate_arrivals += 1
        if self._approach_ok(
                d_now, d0, self._occluder_reveal_ratio,
                self._occluder_reach_radius):
            self._gate_success += 1
            self._reveal_bonus_pending = max(
                self._reveal_bonus_pending, self._occluder_reveal_bonus)
        # Stage 20p: reward-circle gate — same arrivals, success judged by
        # the CURRENT curriculum circle (d_now < reward_radius). This is
        # the TEACHING measure; the 0.8m judge above never moves.
        self._gate_rw_arrivals += 1
        if d0 >= 1e-6 and d_now < self._occluder_reward_radius:
            self._gate_rw_success += 1
        # Stage 20m: L1 orientation payout — event ended; if the agent's
        # FIRST 3 steps of the event were aligned with last_known, credit
        # the "turned the right way" sub-goal (one-shot). Asymmetric: no
        # punishment for bad first steps, only reward for good ones.
        _n3 = ev.get("first3_n", 0)
        if _n3 >= 2:
            self._orient_events += 1
            _aligned = ev.get("first3_cos", 0.0) / _n3 > 0.5
            if _aligned:
                self._orient_aligned += 1
                if self._occluder_orient_bonus > 0.0:
                    self._reveal_bonus_pending = max(
                        self._reveal_bonus_pending, self._occluder_orient_bonus)

    def _update_caregiver_gaze(self) -> None:
        """Stage 20-ToM: caregiver gaze + stale-belief tracking.

        - Every ``_cg_gaze_every`` steps, the caregiver re-picks the nearest
          object it can SEE (dist<2.5, LOS clear) as its gaze target.
        - While the target stays visible, ``_cg_last_seen`` tracks its live
          position (the caregiver's belief stays current).
        - When the target becomes occluded (wall or teleport), ``_cg_last_seen``
          FREEZES at the old position — the caregiver now holds a FALSE belief
          the learner can observe (gaze fixed on empty/old spot).
        - The discrete gaze direction (8-dir 0-7) is the caregiver's
          "action" label for Theory-of-Mind action prediction.
        """
        try:
            cg = self._model.body("caregiver")
            cx = float(self._data.xpos[cg.id, 0])
            cy = float(self._data.xpos[cg.id, 1])

            def _obj_pos(i: int) -> tuple[float, float]:
                b = self._model.body(f"obj_{i}")
                return (float(self._data.xpos[b.id, 0]), float(self._data.xpos[b.id, 1]))

            # Re-pick gaze target periodically (or if none yet).
            if self._cg_gaze_id < 0 or self._step_count % self._cg_gaze_every == 0:
                best, bd = -1, 1e9
                for i in range(self._num_objects):
                    ox, oy = _obj_pos(i)
                    d = math.hypot(cx - ox, cy - oy)
                    if d < 2.5 and d < bd and not self._line_of_sight_blocked(cx, cy, ox, oy):
                        best, bd = i, d
                if best >= 0:
                    self._cg_gaze_id = best

            # Track last-seen when the target is visible; freeze otherwise.
            self._cg_belief_stale = False
            if self._cg_gaze_id >= 0:
                ox, oy = _obj_pos(self._cg_gaze_id)
                visible = (math.hypot(cx - ox, cy - oy) < 2.5
                           and not self._line_of_sight_blocked(cx, cy, ox, oy))
                if visible:
                    self._cg_last_seen = (ox, oy)
                else:
                    self._cg_belief_stale = True  # false belief active

            # Discrete gaze direction (8-dir) toward believed position.
            tx, ty = (self._cg_last_seen if self._cg_last_seen is not None
                      else (_obj_pos(self._cg_gaze_id) if self._cg_gaze_id >= 0 else (cx, cy)))
            dx, dy = tx - cx, ty - cy
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                ang = math.atan2(dy, dx)
                self._cg_last_gaze_dir = int(((ang + math.pi) / (2 * math.pi)) * 8) % 8
        except Exception:
            pass  # legit: caregiver/objects may be absent in odd scenes

    def read_caregiver_state(self) -> dict:
        """Stage 20-ToM: ground-truth caregiver state for ToM supervision.

        Returns a dict with:
          - pos: caregiver xy
          - gaze_id: currently watched object id (-1 none)
          - last_seen: believed object position (stale when occluded)
          - actual_target: the watched object's REAL position (differs from
            last_seen exactly when the belief is stale = false belief)
          - stale: bool, true when the caregiver holds a false belief
          - gaze_dir: discrete 8-dir gaze action (0-7)
          - visible_ids: object ids the caregiver can currently see
        """
        out: dict = {
            "pos": (-1.2, 0.8), "gaze_id": self._cg_gaze_id,
            "last_seen": self._cg_last_seen, "actual_target": None,
            "stale": bool(self._cg_belief_stale),
            "gaze_dir": int(self._cg_last_gaze_dir), "visible_ids": [],
        }
        try:
            cg = self._model.body("caregiver")
            cx = float(self._data.xpos[cg.id, 0])
            cy = float(self._data.xpos[cg.id, 1])
            out["pos"] = (cx, cy)
            vis = []
            for i in range(self._num_objects):
                b = self._model.body(f"obj_{i}")
                ox = float(self._data.xpos[b.id, 0])
                oy = float(self._data.xpos[b.id, 1])
                if math.hypot(cx - ox, cy - oy) < 2.5 and \
                        not self._line_of_sight_blocked(cx, cy, ox, oy):
                    vis.append(i)
            out["visible_ids"] = vis
            if self._cg_gaze_id >= 0:
                b = self._model.body(f"obj_{self._cg_gaze_id}")
                out["actual_target"] = (
                    float(self._data.xpos[b.id, 0]), float(self._data.xpos[b.id, 1]))
        except Exception:
            pass  # legit: objects/caregiver absent
        return out

    def _line_of_sight_blocked(
        self, ax: float, ay: float, ox: float, oy: float,
    ) -> bool:
        """True if the 2D segment (agent -> object) crosses an occluder wall.

        Occluders are modeled as thin vertical boxes (extent in x, thin in
        y, tall in z). For the line-of-sight check we use the segment-AABB
        intersection in the x-y plane (2D), which is sufficient since all
        occluders sit on the floor and the objects are at similar heights.
        """
        if self._model is None or self._data is None or self._num_occluders == 0:
            return False
        for oi in range(self._num_occluders):
            try:
                body_id = self._model.body(f"occluder_{oi}").id
                cx = float(self._data.xpos[body_id, 0])
                cy = float(self._data.xpos[body_id, 1])
            except Exception:
                continue  # legit: per-occluder loop, occluder may be gone
            # Occluder half-extents (matches SceneBuilder default 0.55 x 0.025)
            hx, hy = 0.55, 0.025
            if self._segment_hits_aabb(ax, ay, ox, oy, cx, cy, hx, hy):
                return True
        return False

    @staticmethod
    def _segment_hits_aabb(
        ax: float, ay: float, bx: float, by: float,
        cx: float, cy: float, hx: float, hy: float,
    ) -> bool:
        """Segment (A->B) vs AABB (center C, half-extents hx,hy) in 2D."""
        # Clip segment to the AABB using the slab method
        dx, dy = bx - ax, by - ay
        # AABB bounds
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        t0, t1 = 0.0, 1.0
        for p, d, lo, hi in ((ax, dx, x0, x1), (ay, dy, y0, y1)):
            if abs(d) < 1e-9:
                if p < lo or p > hi:
                    return False
            else:
                t_lo = (lo - p) / d
                t_hi = (hi - p) / d
                if t_lo > t_hi:
                    t_lo, t_hi = t_hi, t_lo
                t0 = max(t0, t_lo)
                t1 = min(t1, t_hi)
                if t0 > t1:
                    return False
        return True

    def _contact_reach(self, obj_idx: int) -> float:
        """Geometry-based contact detection radius.

        Uses the physical sizes (agent radius + object half-extent + small
        margin) instead of a fixed threshold, so large objects are correctly
        detected as contacted even though their center is far away.

        Returns reach (a 2D distance).
        """
        try:
            obj = self._object_lib[obj_idx]  # ObjectDef (size is (hx,hy,hz) or radius)
            sx, sy, _sz = obj.size
            obj_extent = max(float(sx), float(sy))
            agent_r = float(getattr(self, "_agent_size", 0.15))
            return agent_r + obj_extent + 0.15  # wider reach for better signal capture
        except Exception as _e:
            _expose_exc("_contact_reach")
            return 0.35

    # ------------------------------------------------------------------ grasping

    def _grasp(self) -> None:
        """Attempt to grasp the nearest graspable object within reach."""
        if self._held_obj_id is not None or self._model is None:
            return
        learner_id = self._model.body("learner").id
        ax = float(self._data.xpos[learner_id, 0])
        ay = float(self._data.xpos[learner_id, 1])
        best_idx = -1
        best_dist = 1e9
        for i in range(min(self._num_objects, len(self._object_lib))):
            if not self._object_lib[i].graspable:
                continue
            try:
                bid = self._model.body(f"obj_{i}").id
                ox = float(self._data.xpos[bid, 0])
                oy = float(self._data.xpos[bid, 1])
                d = ((ax - ox)**2 + (ay - oy)**2) ** 0.5
                if d < self._contact_reach(i) and d < best_dist:
                    best_dist = d
                    best_idx = i
            except Exception:
                continue  # legit: per-object loop, obj_ may be gone
        if best_idx >= 0:
            self._held_obj_id = best_idx
            self._grasp_start_pos = (ax, ay)

    def _release(self) -> None:
        """Release the currently held object."""
        if self._held_obj_id is None or self._model is None:
            return
        learner_id = self._model.body("learner").id
        ax = float(self._data.xpos[learner_id, 0])
        ay = float(self._data.xpos[learner_id, 1])
        carry_dist = 0.0
        if self._grasp_start_pos is not None:
            carry_dist = ((ax - self._grasp_start_pos[0])**2 + (ay - self._grasp_start_pos[1])**2) ** 0.5
        self._grasp_carry_events.append({
            "obj_id": self._held_obj_id,
            "carry_distance": carry_dist,
        })
        self._release_events.append({"obj_id": self._held_obj_id})
        self._held_obj_id = None
        self._grasp_start_pos = None

    def _use_held_as_tool(self) -> None:
        """Use held object as a tool: apply extra force to objects in front."""
        if self._held_obj_id is None or self._model is None:
            return
        try:
            learner_id = self._model.body("learner").id
            ax = float(self._data.xpos[learner_id, 0])
            ay = float(self._data.xpos[learner_id, 1])
            held_bid = self._model.body(f"obj_{self._held_obj_id}").id
            hx = float(self._data.xpos[held_bid, 0])
            hy = float(self._data.xpos[held_bid, 1])
            for i in range(min(self._num_objects, len(self._object_lib))):
                if i == self._held_obj_id:
                    continue
                try:
                    bid = self._model.body(f"obj_{i}").id
                    ox = float(self._data.xpos[bid, 0])
                    oy = float(self._data.xpos[bid, 1])
                    d = ((hx - ox)**2 + (hy - oy)**2) ** 0.5
                    if d < 0.25:
                        dx_push = (ox - hx) / max(d, 0.01) * self._action_force * 0.5
                        dy_push = (oy - hy) / max(d, 0.01) * self._action_force * 0.5
                        dof = self._model.body_dofadr[bid]
                        if dof >= 0:
                            self._data.qvel[dof] += dx_push * self._model.opt.timestep
                            self._data.qvel[dof + 1] += dy_push * self._model.opt.timestep
                        self._tool_use_events.append({
                            "held_obj": self._held_obj_id,
                            "affected_obj": i,
                            "distance": d,
                        })
                except Exception:
                    continue  # legit: per-object loop, affected object may be gone
        except Exception as _e:
            _expose_exc("_use_held_as_tool")

    def _sync_held_object(self) -> None:
        """Move held object to follow agent (virtual grasp without weld constraint)."""
        if self._held_obj_id is None or self._model is None:
            return
        try:
            learner_id = self._model.body("learner").id
            ax = float(self._data.xpos[learner_id, 0])
            ay = float(self._data.xpos[learner_id, 1])
            az = float(self._data.xpos[learner_id, 2])
            held_bid = self._model.body(f"obj_{self._held_obj_id}").id
            # Position held object just in front of agent
            target_x = ax + 0.12
            target_y = ay
            target_z = az
            dof = self._model.body_dofadr[held_bid]
            if dof >= 0:
                # Smoothly move toward target
                cur_x = self._data.qpos[dof]
                cur_y = self._data.qpos[dof + 1]
                self._data.qpos[dof] = cur_x + (target_x - cur_x) * 0.3
                self._data.qpos[dof + 1] = cur_y + (target_y - cur_y) * 0.3
                # Reduce velocity to prevent flinging
                self._data.qvel[dof] *= 0.3
                self._data.qvel[dof + 1] *= 0.3
        except Exception as _e:
            _expose_exc("_sync_held_object")

    # ------------------------------------------------------------------ chain tasks

    def _setup_chain_task(self) -> None:
        """Set up a chain task based on developmental age.

        Developmental progression:
        - age < 0.3: free play (no task)
        - age 0.3-0.5: obstacle clearing (push barrier to reach target)
        - age > 0.5: key-door (find key, grasp, bring to door)
        """
        if self._dev_age < 0.3:
            self._task_type = "free_play"
            return
        elif self._dev_age < 0.5:
            self._task_type = "obstacle" if self._rng.random() < 0.5 else "free_play"
        else:
            r = self._rng.random()
            if r < 0.33:
                self._task_type = "obstacle"
            elif r < 0.66:
                self._task_type = "key_door"
            else:
                self._task_type = "free_play"

        # Select a target object (reward) from existing objects
        if self._task_type == "obstacle":
            self._task_target_body = f"obj_{min(self._num_objects - 1, 0)}"
            self._task_barrier_body = f"obj_{min(self._num_objects - 1, 1)}"
        elif self._task_type == "key_door":
            self._task_target_body = f"obj_{min(self._num_objects - 1, 0)}"
            self._task_key_body = f"obj_{min(self._num_objects - 1, 1)}"
            self._task_door_open = False

    def _update_chain_task(self, dx: float, dy: float) -> float:
        """Update chain task progress and return bonus reward."""
        if self._task_type == "free_play" or self._model is None:
            return 0.0
        bonus = 0.0
        try:
            learner_id = self._model.body("learner").id
            ax = float(self._data.xpos[learner_id, 0])
            ay = float(self._data.xpos[learner_id, 1])

            if self._task_type == "obstacle" and self._task_target_body:
                # Progress: agent reaches target (barrier was in the way)
                tid = self._model.body(self._task_target_body).id
                tx = float(self._data.xpos[tid, 0])
                ty = float(self._data.xpos[tid, 1])
                dist = ((ax - tx)**2 + (ay - ty)**2) ** 0.5
                if dist < 0.3 and not self._task_reward_collected:
                    bonus = 3.0
                    self._task_reward_collected = True
                    self._task_progress = 1.0
                else:
                    self._task_progress = max(0, 1.0 - dist / 3.0)

            elif self._task_type == "key_door" and self._task_target_body:
                # Phase 1: find and grasp key
                if self._held_obj_id is not None and not self._task_door_open:
                    # Check if agent is near target (door) with key
                    tid = self._model.body(self._task_target_body).id
                    tx = float(self._data.xpos[tid, 0])
                    ty = float(self._data.xpos[tid, 1])
                    dist = ((ax - tx)**2 + (ay - ty)**2) ** 0.5
                    if dist < 0.5:
                        self._task_door_open = True
                        bonus = 1.0
                        self._task_progress = 0.5
                elif self._task_door_open and not self._task_reward_collected:
                    # Phase 2: reach target (door is open)
                    tid = self._model.body(self._task_target_body).id
                    tx = float(self._data.xpos[tid, 0])
                    ty = float(self._data.xpos[tid, 1])
                    dist = ((ax - tx)**2 + (ay - ty)**2) ** 0.5
                    if dist < 0.3:
                        bonus = 3.0
                        self._task_reward_collected = True
                        self._task_progress = 1.0
                else:
                    # Progress toward key
                    if self._task_key_body:
                        kid = self._model.body(self._task_key_body).id
                        kx = float(self._data.xpos[kid, 0])
                        ky = float(self._data.xpos[kid, 1])
                        dist_key = ((ax - kx)**2 + (ay - ky)**2) ** 0.5
                        self._task_progress = max(0, 0.3 - dist_key / 3.0)
        except Exception as _e:
            _expose_exc("_update_chain_task")
        return bonus

    def _occluder_only_reward(self) -> float:
        """Stage 20b: reward ONLY occlusion tracking (object permanence).

        Mirrors the occluder_target_reward block: while an object is in
        ``_active_occlusions_3d`` the agent is rewarded for reducing the
        distance to that object's last-known position. Used in focus_op_only
        mode so means-ends / mobility rewards cannot crowd it out.

        Stage 20c: optional direction shaping — reward velocity components
        pointing toward ``last_known``. Denser than pure distance reduction
        (which needs the agent to already be approaching), so the policy
        learns "object disappeared -> move where it was" faster. Uses only
        last_known (pre-occlusion memory); no eval-only information leaks.

        Stage 20g: symmetric — the same dot product is SUBTRACTED when the
        learner moves AWAY from last_known. 2800K diagnostics showed the
        policy freezing (entropy~2.45, ~0 movement, 20 reward vs 408 for a
        random policy) because away-movement and stillness both cost zero;
        only approach earned, so the zero-gradient policy never moved.
        Teaching-scaffold semantics: approach earns, away is discouraged,
        stillness is neutral — the optimal policy is forced to move.
        """
        if self._occluder_target_reward <= 0.0 and self._occluder_shaping_weight <= 0.0 \
                and self._occluder_orient_weight <= 0.0 and self._occluder_orient_bonus <= 0.0:
            return 0.0
        try:
            bonus = self._reveal_bonus_pending  # Stage 20d: reveal attribution
            self._reveal_bonus_pending = 0.0    # consumed once
            ax = float(self._data.body("learner").xpos[0])
            ay = float(self._data.body("learner").xpos[1])
            r = bonus
            # Stage 20m#2: directional terms (orientation shaping, symmetric
            # shaping) act on the NEAREST active occlusion ONLY — the agent
            # can only walk one direction at a time, and the observation
            # slots feed it the nearest lk first. Summing dots over every
            # concurrent occlusion diluted the signal: with 3-8 objects the
            # summed gradient averaged toward zero direction (measured
            # first-3-step alignment stuck at random ~0.16-0.22 regardless
            # of teacher force).
            _nearest_key = None
            _nearest_dist = float("inf")
            for _key in self._active_occlusions_3d:
                _occ = self._active_occlusions_3d[_key]
                _lk = _occ.get("last_known")
                if _lk is None:
                    continue
                _d = math.hypot(_lk[0] - ax, _lk[1] - ay)
                if _d < _nearest_dist:
                    _nearest_dist = _d
                    _nearest_key = _key
            for key, occ in list(self._active_occlusions_3d.items()):
                lk = occ["last_known"]
                dist = math.hypot(ax - lk[0], ay - lk[1])
                # Stage 20m: L1 orientation — first-3-steps direction credit.
                # "Turned the right way right after the object vanished" is
                # the simplest, earliest-learnable sub-goal (head_cos was
                # 0.28: the policy's first move barely correlates with lk).
                # Accumulate displacement-vs-lk cos for the first 3 steps of
                # the event; the event-end bonus (in _maybe_reveal_bonus)
                # pays out if the mean alignment clears 0.5. Only reward
                # alignment, never penalize misalignment (asymmetric).
                # Stage 20m#2: only the NEAREST occlusion accumulates —
                # the agent cannot align with several last_knowns at once.
                # Stage 20m#4: orientation is measured against the NEAREST
                # active lk at that step (what the observation slots feed),
                # NOT against this event's lk. Multi-object concurrency
                # makes "which event settles" unpredictable (teacher walked
                # to A, B settled) — coupling L1 to settlement kept
                # alignment at random ~0.16 even at teacher 1.0. L1 now
                # measures "did the agent turn toward where it should go"
                # (agent perspective); arrival at the settled object stays
                # with the tgate/reach signals.
                traj = occ.get("agent_traj_during_occ", [])
                if (self._occluder_orient_bonus > 0.0
                        and key == _nearest_key
                        and len(traj) >= 3
                        and len(traj) <= 5
                        and not occ.get("first3_done", False)):
                    px, py = traj[-2]
                    mx, my = ax - px, ay - py
                    mlen = math.hypot(mx, my)
                    dxl, dyl = lk[0] - px, lk[1] - py
                    dl = math.hypot(dxl, dyl)
                    if mlen > 1e-6 and dl > 1e-6:
                        c = (mx * dxl + my * dyl) / (mlen * dl)
                        occ["first3_cos"] = occ.get("first3_cos", 0.0) + c
                        occ["first3_n"] = occ.get("first3_n", 0) + 1
                    if len(traj) >= 5:
                        occ["first3_done"] = True
                # Stage 20m: L1 orientation — per-step asymmetric shaping:
                # moving TOWARD lk earns, moving away is neutral (unlike the
                # 20g symmetric term, which punished wrong direction and
                # drove the policy to average directions under noise).
                # 20m#2: nearest occlusion only (same reason as above).
                if (self._occluder_orient_weight > 0.0
                        and key == _nearest_key):
                    dx, dy = lk[0] - ax, lk[1] - ay
                    dlen = math.hypot(dx, dy)
                    if dlen > 1e-6:
                        bid = self._model.body("learner").id
                        dof = self._model.body_dofadr[bid]
                        if dof >= 0:
                            vx = float(self._data.qvel[dof])
                            vy = float(self._data.qvel[dof + 1])
                            vlen = math.hypot(vx, vy) + 1e-9
                            dot = (vx * dx + vy * dy) / (vlen * dlen)
                            r += max(0.0, dot) * self._occluder_orient_weight
                # Stage 20k: absolute-reach credit — walking INTO the contact
                # radius around last_known earns a one-shot reward. The dense
                # distance/heading terms only reward "getting closer" from
                # wherever you are; this is the only term that demands the
                # actual arrival the eval op metric measures (best_d < 0.8m).
                # Stage 20p: the circle is the CURRICULUM reward_radius
                # (2.0→0.8) — teaching circle, judge stays 0.8m.
                if (self._occluder_reach_reward > 0.0
                        and not occ.get("reached", False)
                        and dist < self._occluder_reward_radius):
                    r += self._occluder_reach_reward
                    occ["reached"] = True
                # Stage 20n: in-circle hold credit — staying inside the
                # reach radius keeps earning (closer = more). Without it,
                # arriving costs the per-step heading points (dot~0 when
                # still) while the one-shot reach reward is smaller than
                # the heading stream — the reward math made arrival
                # OPTIMALLY WRONG (measured: policy farmed heading points,
                # far stuck at 0.03). This makes "arrive and stay" the
                # dominant strategy: heading~1.5/step vs hold up to
                # 0.8*w/step + 20 one-shot.
                if (self._occluder_reach_hold > 0.0
                        and dist < self._occluder_reward_radius):
                    r += max(0.0, self._occluder_reward_radius - dist) \
                        * self._occluder_reach_hold
                prev = occ.get("prev_agent_dist", dist)
                occ["prev_agent_dist"] = dist
                if dist < prev:
                    r += (prev - dist) * self._occluder_target_reward
                if (self._occluder_shaping_weight > 0.0
                        and key == _nearest_key):
                    dx, dy = lk[0] - ax, lk[1] - ay
                    dlen = math.hypot(dx, dy)
                    if dlen > 1e-6:
                        # learner velocity toward last_known (component)
                        bid = self._model.body("learner").id
                        dof = self._model.body_dofadr[bid]
                        if dof >= 0:
                            vx = float(self._data.qvel[dof])
                            vy = float(self._data.qvel[dof + 1])
                            vlen = math.hypot(vx, vy) + 1e-9
                            dot = (vx * dx + vy * dy) / (vlen * dlen)
                            # Stage 20g: symmetric — approach earns,
                            # away is penalized (negative dot).
                            r += dot * self._occluder_shaping_weight
            return float(r)
        except Exception as _e:
            _expose_exc("_occluder_only_reward")
            return 0.0

    def _compute_reward(self, action: int = -1) -> float:
        """Multi-component reward.

        - Object interaction: velocity of scene objects (learner caused movement)
        - Contact: touching objects (from MuJoCo contacts, not dead sensor)
        - Social: proximity to caregiver (safety reward)
        - Chain task: goal-directed bonus (when dev_age > 0.3)
        """
        reward = 0.0

        # --- Stage 20b curriculum lock: focus ONLY on object permanence ---
        # While enabled, every non-op reward component is gated off so the
        # ONLY meaningful extrinsic gradient is approaching last_known during
        # occlusion (occluder_target_reward). Means-ends / object mobility /
        # contact / caregiver rewards are frozen; the agent still gets
        # intrinsic curiosity, so behavior doesn't collapse, but the policy
        # can no longer trade op against them.
        if self._focus_op_only:
            reward = self._occluder_only_reward()
            # Stage 20w: raise the per-step clamp so reach_reward can actually
            # dominate. Previously clamped to 5.0 — reach=20 was silently
            # truncated to 5, barely above the shaping stream. Now the clamp
            # is configurable and defaults to 10x the old cap.
            _clamp = float(getattr(self, '_op_reward_clamp', 50.0))
            reward = max(0.0, min(_clamp, reward))
            # Chain task & logic bonus stay so hypothesis-deduction circuits
            # keep a thin extrinsic tie to the loop (not a competing goal).
            if self._dev_age > 0.3:
                reward += self._update_chain_task(0.0, 0.0)
            if self._logic_bonus_action is not None and action >= 0 and action % 8 == self._logic_bonus_action:
                reward += self._logic_bonus_weight
            return float(max(0.0, min(10.0, reward)))

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

        # Contact reward: compute from MuJoCo contacts directly (fixed dead sensor)
        try:
            learner_bid = self._model.body("learner").id
            contact_count = 0
            for i in range(self._data.ncon):
                contact = self._data.contact[i]
                if contact.geom1 == learner_bid or contact.geom2 == learner_bid:
                    contact_count += 1
                # Also check if contact involves a body that is the learner
                g1bid = self._model.geom_bodyid[contact.geom1]
                g2bid = self._model.geom_bodyid[contact.geom2]
                if g1bid == learner_bid or g2bid == learner_bid:
                    contact_count += 1
            if contact_count > 0:
                reward += min(contact_count * 0.05, 0.5)
        except Exception as _e:
            _expose_exc("contact_reward")

        # Caregiver proximity reward
        try:
            lx = float(self._data.body("learner").xpos[0])
            ly = float(self._data.body("learner").xpos[1])
            cx = float(self._data.body("caregiver").xpos[0])
            cy = float(self._data.body("caregiver").xpos[1])
            dist_caregiver = np.sqrt((lx - cx)**2 + (ly - cy)**2)
            reward += max(0, (1.0 - dist_caregiver)) * 0.05
        except Exception as _e:
            _expose_exc("caregiver_reward")

        # Approach reward: reducing distance to any object (physics_sandbox
        # parity). 400K regression: weight>0 in the 3D scene (many objects)
        # drowns goal-directed signals -> object_permanence/means_ends/ToM
        # collapse. Disabled by default (weight=0.0).
        if self._approach_reward_weight > 0.0:
            try:
                lx = float(self._data.body("learner").xpos[0])
                ly = float(self._data.body("learner").xpos[1])
                for i in range(self._num_objects):
                    body_id = self._model.body(f"obj_{i}").id
                    ox = float(self._data.xpos[body_id, 0])
                    oy = float(self._data.xpos[body_id, 1])
                    dist = math.hypot(lx - ox, ly - oy)
                    prev = self._prev_obj_dist[i]
                    self._prev_obj_dist[i] = dist
                    if dist < prev:
                        reward += (prev - dist) * self._approach_reward_weight
            except Exception as _e:
                _expose_exc("approach_reward")

        # Occluder-target reward (situational, single-target): while an
        # object is occluded (in _active_occlusions_3d), reward the agent
        # for moving toward that object's last-known position. Unlike the
        # blanket approach reward (all objects, all steps), this fires only
        # for the occluded object during the occlusion window — a
        # context-limited behavior-guidance signal aligned 1:1 with the
        # object_permanence eval metric. Default off (weight=0.0).
        if self._occluder_target_reward > 0.0:
            try:
                ax = float(self._data.body("learner").xpos[0])
                ay = float(self._data.body("learner").xpos[1])
                for key, occ in list(self._active_occlusions_3d.items()):
                    lk = occ["last_known"]
                    dist = math.hypot(ax - lk[0], ay - lk[1])
                    prev = occ.get("prev_agent_dist", dist)
                    occ["prev_agent_dist"] = dist
                    if dist < prev:
                        reward += (prev - dist) * self._occluder_target_reward
            except Exception as _e:
                _expose_exc("occluder_target_reward")

        reward = max(0.0, min(5.0, reward))

        # --- Chain task bonus (developmental: unlocks at age > 0.3) ---
        if self._dev_age > 0.3:
            reward += self._update_chain_task(0.0, 0.0)

        # --- Logic bonus: reward agent for following symbolic rules ---
        if self._logic_bonus_action is not None and action >= 0 and action % 8 == self._logic_bonus_action:
            reward += self._logic_bonus_weight

        return float(max(0.0, min(10.0, reward)))

    def _proprio(self) -> np.ndarray:
        """Return proprioceptive vector (12 or 16 dim depending on dev age)."""
        try:
            body = self._data.body("learner")
            pos = body.xpos[:3].copy()
            vel = self._data.cvel[self._model.body("learner").id][3:6].copy()
            # Touch sensor
            touch = np.zeros(3)
            # Touch: compute from contact forces (simplified)
            try:
                for i in range(self._data.ncon):
                    contact = self._data.contact[i]
                    touch[:] = [1.0, 1.0, 1.0]  # any contact = touch signal
            except Exception:
                pass  # legit: ncon may be 0 in a fresh scene
            # Joint positions
            joints = np.zeros(3)
            for i, axis in enumerate(["x", "y"]):
                try:
                    joint_id = self._model.joint(f"learner_{axis}").id
                    joints[i] = float(self._data.qpos[joint_id])
                except Exception:
                    pass  # legit: joint may be absent until body scaffold
            # Stage 20f: heading (yaw) in global frame — cos/sin so the
            # orientation is a continuous, periodic-safe feature. Without it
            # the policy cannot convert global last_known offsets (slots)
            # into local turn/forward actions.
            yaw_xy = np.zeros(2, dtype=np.float32)
            try:
                q = self._data.body("learner").xquat  # (w,x,y,z)
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                yaw_xy[0] = math.cos(yaw)
                yaw_xy[1] = math.sin(yaw)
            except Exception:
                pass  # legit: body may be absent during scaffold build
            base = np.concatenate([pos, vel, touch, joints, yaw_xy]).astype(np.float32)
            # Add grasping state when unlocked
            if self._dev_age > 0.15:
                is_holding = np.array([1.0 if self._held_obj_id is not None else 0.0], dtype=np.float32)
                if self._held_obj_id is not None and self._model is not None:
                    try:
                        hbid = self._model.body(f"obj_{self._held_obj_id}").id
                        held_pos = self._data.xpos[hbid][:3].copy().astype(np.float32)
                    except Exception:
                        held_pos = np.zeros(3, dtype=np.float32)  # legit: obj_ may be gone after rebuild
                else:
                    held_pos = np.zeros(3, dtype=np.float32)
                grasp_state = np.concatenate([is_holding, held_pos]).astype(np.float32)
                out = np.concatenate([base, grasp_state])
            else:
                out = base
            # Stage 20e: occluder memory slots — normalized relative offset
            # (dx, dy, dist / room scale ~4.0) to the closest active
            # last_known positions; unused slots stay 0. Same signal in
            # training and eval (last_known is the agent's own memory).
            slots = self._occluder_obs_slots
            if slots > 0 and self._active_occlusions_3d:
                ax_, ay_ = float(pos[0]), float(pos[1])
                cands = []
                for _key, _occ in self._active_occlusions_3d.items():
                    _lk = _occ.get("last_known")
                    if _lk is None:
                        continue
                    _d = math.hypot(_lk[0] - ax_, _lk[1] - ay_)
                    cands.append((_d, _lk))
                cands.sort(key=lambda t: t[0])
                slot_vec = np.zeros(3 * slots, dtype=np.float32)
                for s in range(min(slots, len(cands))):
                    _d, _lk = cands[s]
                    slot_vec[3 * s] = float((_lk[0] - ax_) / 4.0)
                    slot_vec[3 * s + 1] = float((_lk[1] - ay_) / 4.0)
                    slot_vec[3 * s + 2] = float(_d / 4.0)
            else:
                slot_vec = np.zeros(3 * slots, dtype=np.float32)
            return np.concatenate([out, slot_vec]).astype(np.float32)
        except Exception as _e:
            _expose_exc("_proprio")
            dim = 16 if self._dev_age > 0.15 else 12
            return np.zeros(dim, dtype=np.float32)

    def set_developmental_age(self, age: float) -> None:
        """Update developmental age (body grows, more objects emerge)."""
        self._dev_age = max(0.0, min(1.0, age))
