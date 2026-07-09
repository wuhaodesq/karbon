"""Homeostatic Drive System — Internal motivational states.

Replaces simple "reward = curiosity * constant" with proper biological-style
internal drives that the agent must maintain.

Five drives:
    1. Curiosity (求知) — must explore novel states. Depletes slowly, refills on novelty.
    2. Competence (能力) — must achieve goals. Depletes on failure, refills on success.
    3. Social (社交) — must interact with caregiver. Depletes when alone.
    4. Safety (安全) — must avoid danger. Depletes near threats (walls, falling).
    5. Rest (休息) — must periodically stop. Depletes with activity, refills when still.

Each drive:
    - Has a "level" (0 = depleted/critical, 1 = satisfied/refilled)
    - Decays over time (creates internal pressure to act)
    - Is replenished by specific behaviors (creates goal-directed behavior)
    - Produces intrinsic reward proportional to improvement

This replaces RND/RSSM curiosity with a complete motivational architecture.

The agent feels:
    - "I need to explore" (curiosity low) → seeks novel states
    - "I need to succeed" (competence low) → focuses on task
    - "I'm lonely" (social low) → approaches caregiver
    - "I'm scared" (safety low) → avoids walls
    - "I'm tired" (rest low) → reduces movement

稳态驱动力系统：五个内在驱动力替代简单的 curiosity reward。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DriveState:
    name: str
    level: float = 1.0           # 0 = depleted, 1 = full
    decay_rate: float = 0.001    # per step
    refill_rate: float = 0.1     # per satisfying event
    urgency_threshold: float = 0.3  # below this → urgent
    weight: float = 1.0          # importance in overall motivation
    history: list[float] = field(default_factory=list)  # last 100 levels


class HomeostaticDrives(nn.Module):
    """Internal drive system producing intrinsic motivation.

    Each drive produces an intrinsic reward: reward = weight * Δlevel
    (improving a depleted drive is rewarding).

    The agent learns to balance all drives — exploration, achievement,
    social bonding, safety, rest — just as biological organisms do.
    """

    def __init__(
        self,
        curiosity_decay: float = 0.002,
        competence_decay: float = 0.001,
        social_decay: float = 0.003,
        safety_decay: float = 0.001,
        rest_decay: float = 0.005,
        history_length: int = 100,
    ) -> None:
        super().__init__()
        self.drives: dict[str, DriveState] = {
            "curiosity": DriveState(
                name="curiosity", level=1.0,
                decay_rate=curiosity_decay, refill_rate=0.15,
                urgency_threshold=0.3, weight=1.5,
            ),
            "competence": DriveState(
                name="competence", level=1.0,
                decay_rate=competence_decay, refill_rate=0.2,
                urgency_threshold=0.25, weight=1.2,
            ),
            "social": DriveState(
                name="social", level=1.0,
                decay_rate=social_decay, refill_rate=0.15,
                urgency_threshold=0.3, weight=1.0,
            ),
            "safety": DriveState(
                name="safety", level=1.0,
                decay_rate=safety_decay, refill_rate=0.25,
                urgency_threshold=0.15, weight=1.3,
            ),
            "rest": DriveState(
                name="rest", level=1.0,
                decay_rate=rest_decay, refill_rate=0.05,
                urgency_threshold=0.4, weight=0.8,
            ),
        }
        self._history_len = history_length
        self._step_count = 0

    # ------------------------------------------------------------------ tick

    def tick(
        self,
        novelty: float = 0.0,        # RSSM surprise at current state
        success: bool = False,       # did agent achieve a goal?
        caregiver_proximity: float = 0.0,  # distance to caregiver [0=close, 1=far]
        danger_level: float = 0.0,   # proximity to wall/danger [0=safe, 1=danger]
        movement_level: float = 0.0, # agent speed [0=still, 1=fast]
    ) -> dict[str, float]:
        """Update all drives and compute total intrinsic reward.

        Returns dict with per-drive rewards and total.
        """
        rewards: dict[str, float] = {}
        self._step_count += 1

        for name, d in self.drives.items():
            old_level = d.level

            # Decay (all drives deplete over time)
            d.level = max(0.0, d.level - d.decay_rate)

            # Refill based on behavior
            if name == "curiosity":
                d.level = min(1.0, d.level + novelty * d.refill_rate * 10)
            elif name == "competence":
                if success:
                    d.level = min(1.0, d.level + d.refill_rate)
            elif name == "social":
                closeness = 1.0 - caregiver_proximity  # invert: 0=far, 1=close
                d.level = min(1.0, d.level + closeness * d.refill_rate * 0.3)
            elif name == "safety":
                safety = 1.0 - danger_level
                d.level = min(1.0, d.level + safety * d.refill_rate * 0.2)
            elif name == "rest":
                stillness = 1.0 - movement_level
                d.level = min(1.0, d.level + stillness * d.refill_rate)

            # Reward = improvement in drive level (positive if refilled)
            delta = d.level - old_level
            rewards[name] = delta * d.weight

            # History
            d.history.append(d.level)
            if len(d.history) > self._history_len:
                d.history.pop(0)

        rewards["total"] = sum(rewards.values())
        return rewards

    # ------------------------------------------------------------------ query

    def most_urgent_drive(self) -> str:
        """Which drive needs attention most urgently?"""
        best_name = "curiosity"
        best_urgency = 0.0
        for name, d in self.drives.items():
            urgency = (1.0 - d.level) * d.weight
            if urgency > best_urgency:
                best_urgency = urgency
                best_name = name
        return best_name

    def drive_levels(self) -> dict[str, float]:
        return {name: d.level for name, d in self.drives.items()}

    def is_homeostatic(self) -> bool:
        """Are all drives above urgency threshold? (agent is 'content')"""
        return all(
            d.level > d.urgency_threshold
            for d in self.drives.values()
        )

    def dominant_motivation(self) -> str:
        """Return a human-readable description of the agent's current state."""
        levels = self.drive_levels()
        sorted_drives = sorted(levels.items(), key=lambda x: x[1])
        lowest = sorted_drives[0]
        if lowest[1] < 0.3:
            return f"I need {lowest[0]}"
        if lowest[1] < 0.6:
            return f"I could use more {lowest[0]}"
        return "I feel content"

    # ------------------------------------------------------------------ diagnostics

    @property
    def capacity(self) -> int:
        return 5  # five drives

    def __len__(self) -> int:
        return 5

    def summary(self) -> dict:
        return {
            "drive_levels": self.drive_levels(),
            "most_urgent": self.most_urgent_drive(),
            "is_homeostatic": self.is_homeostatic(),
            "dominant_motivation": self.dominant_motivation(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "drives": {
                name: {"level": d.level, "history": d.history[-20:]}
                for name, d in self.drives.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step_count = int(state.get("step_count", 0))
        for name, d in self.drives.items():
            if name in state.get("drives", {}):
                d.level = float(state["drives"][name]["level"])
