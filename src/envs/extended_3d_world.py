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

from .three_d_world import (
    ThreeDWorld,
    EnvStep3D,
    ObjectDef,
    _expose_exc,
    _generate_object_library,
)


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
        **kwargs: Any,
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
            # Stage 23 M2-p2: forward the full ThreeDWorld kwarg set
            # (occluder system / reward ladder / slots / focus mode) so the
            # extended world shares the SAME obs/proprio contract as the
            # checkpoint's model (27-dim proprio with 3 slots, etc.).
            **kwargs,
        )

    def _build_scene(self) -> None:
        """Stage 23 M2-p2: seasonal bookkeeping; siblings are injected into
        the parent's scene builder via the ``_extend_scene`` hook.

        The old override referenced ``self._scene_builder`` before it existed
        (the parent creates the builder inside its own ``_build_scene``) —
        the extended world was dead code since that refactor; the M2-p2 smoke
        (2026-10-22) caught the AttributeError.
        """
        self._current_season = (self._step_count // self._season_cycle) % 4
        self._sibling_names.clear()
        self._sibling_positions.clear()
        self._sibling_activities.clear()
        for i in range(self._num_siblings):
            name = f"sibling_{i}"
            room = self._room_names[(i + 1) % len(self._room_names)]
            bounds = self._room_bounds[room]
            self._sibling_names.append(name)
            self._sibling_positions[name] = (
                float(np.clip(self._rng.uniform(bounds[0], bounds[1]), -1.8, 1.8)),
                float(np.clip(self._rng.uniform(bounds[2], bounds[3]), -1.8, 1.8)),
                0.15,
            )
            self._sibling_activities[name] = "exploring"
        super()._build_scene()

    def _extend_scene(self, builder) -> None:
        """Inject siblings into the parent's scene BEFORE model compile."""
        for name, pos in self._sibling_positions.items():
            try:
                builder.add_agent(
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
            except Exception as _e:
                _expose_exc("extended.add_sibling")

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
