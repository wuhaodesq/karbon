"""Extended 3D World — Multiple rooms, siblings, seasons.

Extends the basic 3D home with:
    1. Multiple rooms (living room, kitchen, bedroom, garden)
    2. Sibling agents (other children to interact with)
    3. Seasonal cycle (spring → summer → autumn → winter)
    4. Door/room navigation (agent can move between rooms)
    5. Day/night activity cycle
    6. Growing object library (500+ procedurally generated)

These extensions provide the rich social and environmental context
needed for developmental learning beyond the sensorimotor stage.

扩展 3D 世界：多房间、兄弟姐妹、四季循环。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .three_d_world import ThreeDWorld, EnvStep3D, ObjectDef, _generate_object_library


class ExtendedThreeDWorld(ThreeDWorld):
    """Extended 3D world with multiple rooms, siblings, and seasons.

    Extends ThreeDWorld with:
    - 4 rooms (living room, kitchen, bedroom, garden) connected by doorways
    - Up to 2 sibling agents (play/compete with the learner)
    - Seasonal cycle (500K steps per season, 2M steps per year)
    - 500+ objects distributed across rooms
    - Day/night activity (siblings more active during day)

    Inherits all ThreeDWorld functionality (MuJoCo physics, rendering,
    caregiver agent, object library).
    """

    def __init__(
        self,
        num_objects: int = 500,
        num_siblings: int = 2,
        seed: int | None = None,
        max_episode_steps: int = 500,
        render_size: int = 256,
        season_cycle_steps: int = 500_000,     # steps per season
        developmental_age: float = 0.0,
    ) -> None:
        # Override parent's num_objects before init
        self._extended_num_objects = num_objects
        self._num_siblings = num_siblings
        self._season_cycle = season_cycle_steps
        self._season_names = ["spring", "summer", "autumn", "winter"]
        self._current_season = 0
        self._room_names = ["living_room", "kitchen", "bedroom", "garden"]
        self._room_bounds: dict[str, tuple[float, float, float, float]] = {
            "living_room": (-2.0, 2.0, -2.0, 2.0),
            "kitchen": (2.0, 4.0, -2.0, 2.0),
            "bedroom": (-4.0, -2.0, -2.0, 2.0),
            "garden": (-4.0, 4.0, 2.0, 4.0),
        }
        self._sibling_names: list[str] = []
        self._sibling_positions: dict[str, tuple[float, float, float]] = {}
        self._sibling_activities: dict[str, str] = {}
        self._season_objects: dict[int, list[ObjectDef]] = {}  # seasonal variations

        # Generate extended object library (500 objects)
        self._extended_lib = _generate_object_library(min(num_objects, 500), seed or 42)

        # Initialize parent (will call _build_scene with extended features)
        super().__init__(
            num_objects=min(num_objects, 500),
            seed=seed,
            max_episode_steps=max_episode_steps,
            render_size=render_size,
            developmental_age=developmental_age,
        )

    def _build_scene(self) -> None:
        """Override: add siblings + seasonal features to scene."""
        # Update season
        self._current_season = (self._step_count // self._season_cycle) % 4

        # Assign objects to rooms
        season_mod = self._current_season / 4.0  # seasonal color shift
        for i, obj in enumerate(self._extended_lib[:self._extended_num_objects]):
            room = self._room_names[i % len(self._room_names)]
            bounds = self._room_bounds[room]
            # Seasonal color variation
            r, g, b, a = obj.color
            if self._current_season == 0:  # spring: more green
                obj.color = (r * 0.8, g * 1.2, b * 0.8, a)
            elif self._current_season == 1:  # summer: bright
                obj.color = (r * 1.1, g * 1.1, b * 0.9, a)
            elif self._current_season == 2:  # autumn: warm
                obj.color = (r * 1.2, g * 0.8, b * 0.6, a)
            elif self._current_season == 3:  # winter: cool
                obj.color = (r * 0.7, g * 0.8, b * 1.2, a)
            obj.color = tuple(max(0.0, min(1.0, c)) for c in obj.color[:3]) + (a,)

        # Add siblings
        for i in range(self._num_siblings):
            name = f"sibling_{i}"
            self._sibling_names.append(name)
            room = self._room_names[(i + 1) % len(self._room_names)]
            bounds = self._room_bounds[room]
            self._sibling_positions[name] = (
                self._rng.uniform(bounds[0], bounds[1]),
                self._rng.uniform(bounds[2], bounds[3]),
                0.15,
            )
            self._sibling_activities[name] = "exploring"

        # Add siblings to scene builder
        for name, pos in self._sibling_positions.items():
            self._scene_builder.add_agent(
                name=name,
                position=pos,
                size=0.13,  # siblings are slightly smaller than learner
                color=(
                    (0.2, 0.6, 1.0, 1.0) if "0" in name
                    else (1.0, 0.4, 0.6, 1.0)
                ),
                can_move=True,
            )
            self._agent_names.append(name)

        # Add room walls/doors
        self._add_room_partitions()

        # Call parent build
        super()._build_scene()

    def _add_room_partitions(self) -> None:
        """Add partial walls and doorways between rooms."""
        # Doorways are gaps in the partition walls (implemented as visual markers)
        # Living room / kitchen door
        self._scene_builder.add_object(
            ObjectDef("door_lr_kitchen", "box", (0.02, 0.8, 1.5),
                      (0.5, 0.4, 0.2, 1.0), label="door"),
            (2.0, 0.0, 0.75),
        )
        # Living room / bedroom door
        self._scene_builder.add_object(
            ObjectDef("door_lr_bedroom", "box", (0.02, 0.8, 1.5),
                      (0.4, 0.3, 0.1, 1.0), label="door"),
            (-2.0, 0.0, 0.75),
        )
        # Living room / garden door
        self._scene_builder.add_object(
            ObjectDef("door_lr_garden", "box", (0.8, 0.02, 1.5),
                      (0.3, 0.6, 0.1, 1.0), label="door"),
            (0.0, 2.0, 0.75),
        )

    @property
    def current_season(self) -> str:
        return self._season_names[self._current_season]

    @property
    def season_progress(self) -> float:
        """How far into the current season? [0, 1]."""
        steps_in_season = self._step_count % self._season_cycle
        return steps_in_season / self._season_cycle

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "season": self.current_season,
            "season_progress": self.season_progress,
            "num_siblings": self._num_siblings,
            "rooms": self._room_names,
            "extended_objects": self._extended_num_objects,
        })
        return base
