"""Emotion System — Basic affective states from experience.

Models four basic emotion dimensions that arise organically from the agent's
interaction with the world (not hand-coded emotional responses):

    1. Pleasure (愉悦) — positive when gaining reward, achieving goals
    2. Frustration (挫败) — positive when effort exceeds expected outcome
    3. Surprise (惊喜) — positive when RSSM prediction error is high
    4. Fear (恐惧) — positive when near danger and uncertain

Each emotion is a scalar [0, 1] that:
    - Arises from specific experiential signals (reward, surprise, effort, danger)
    - Decays over time (emotions are transient)
    - Modulates learning (frustration → increase exploration, fear → inhibit action)
    - Is visible to LLM (Phase 9) for richer self-expression

This is NOT a separate trainable module — it's a signal processor that
converts raw experience → emotion states → modulates policy.

The developmental significance:
    - Infant: pleasure = basic reward, fear = loud noise / falling
    - Child: frustration = "I can't do it", surprise = "that was unexpected!"
    - Adult: complex emotions require Theory of Mind + social context

情感系统：从经验中自然产生的四个基本情感维度。
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
class EmotionState:
    """The agent's current emotional state."""
    pleasure: float = 0.5       # [0, 1]
    frustration: float = 0.0    # [0, 1]
    surprise: float = 0.0        # [0, 1]
    fear: float = 0.0            # [0, 1]
    intensity: float = 0.0       # overall emotional arousal [0, 1]
    dominant: str = "neutral"

    def to_dict(self) -> dict[str, float]:
        return {
            "pleasure": self.pleasure,
            "frustration": self.frustration,
            "surprise": self.surprise,
            "fear": self.fear,
            "intensity": self.intensity,
        }

    def describe(self) -> str:
        """Natural language description for LLM integration."""
        parts = []
        if self.pleasure > 0.7:
            parts.append("I feel happy")
        elif self.pleasure < 0.3:
            parts.append("I feel unhappy")

        if self.frustration > 0.6:
            parts.append("I'm frustrated")
        if self.surprise > 0.7:
            parts.append("I'm surprised")
        elif self.surprise > 0.4:
            parts.append("that was unexpected")
        if self.fear > 0.6:
            parts.append("I'm scared")
        elif self.fear > 0.3:
            parts.append("I'm nervous")

        if not parts:
            return "I feel neutral"
        return ". ".join(parts) + "."


class EmotionSystem(nn.Module):
    """Generates emotional states from raw experiential signals.

    Signals → Emotions:
        reward / goal achieved          → pleasure ↑
        effort > expected outcome       → frustration ↑
        RSSM prediction error           → surprise ↑
        danger proximity × uncertainty  → fear ↑

    Decay: all emotions decay toward baseline over time.

    Policy modulation:
        frustration ↑ → exploration ↑ (try something different)
        fear ↑ → action inhibition (be cautious)
        pleasure ↑ → exploitation ↑ (keep doing what works)
        surprise ↑ → attention ↑ (focus on this state)
    """

    def __init__(
        self,
        pleasure_decay: float = 0.01,
        frustration_decay: float = 0.02,
        surprise_decay: float = 0.05,
        fear_decay: float = 0.03,
        history_length: int = 100,
    ) -> None:
        super().__init__()
        self._pleasure_decay = pleasure_decay
        self._frustration_decay = frustration_decay
        self._surprise_decay = surprise_decay
        self._fear_decay = fear_decay

        self._state = EmotionState()
        self._history: list[EmotionState] = []
        self._max_history = history_length

        # Average reward accumulator (for pleasure baseline)
        self._reward_ema: float = 0.0
        self._reward_alpha: float = 0.01

        # Effort accumulator (steps per episode)
        self._steps_this_episode: int = 0

    # ------------------------------------------------------------------ update

    def update(
        self,
        reward: float,
        surprise: float,              # RSSM prediction error
        danger_level: float,          # 0=safe, 1=dangerous
        success: bool = False,
        episode_done: bool = False,
    ) -> EmotionState:
        """Update emotional state from one step of experience.

        Returns the new emotion state (also stored internally).
        """
        # Pleasure: driven by reward and success
        self._reward_ema = (
            (1 - self._reward_alpha) * self._reward_ema
            + self._reward_alpha * reward
        )
        if success:
            self._state.pleasure = min(1.0, self._state.pleasure + 0.3)
        else:
            self._state.pleasure = min(1.0, self._state.pleasure + reward * 0.1)

        # Frustration: effort exceeding reward expectation
        self._steps_this_episode += 1
        effort = min(1.0, self._steps_this_episode / 500)
        if effort > 0.3 and self._reward_ema < 0.01 and not success:
            self._state.frustration = min(1.0, self._state.frustration + 0.05)
        elif success:
            self._state.frustration = max(0.0, self._state.frustration - 0.3)

        # Surprise: RSSM prediction error
        self._state.surprise = min(1.0, self._state.surprise + surprise * 5)

        # Fear: danger × uncertainty
        uncertainty = surprise
        self._state.fear = min(1.0, self._state.fear + danger_level * uncertainty * 0.1)

        # Decay all emotions toward baseline
        self._state.pleasure = max(0.0, self._state.pleasure - self._pleasure_decay)
        self._state.frustration = max(0.0, self._state.frustration - self._frustration_decay)
        self._state.surprise = max(0.0, self._state.surprise - self._surprise_decay)
        self._state.fear = max(0.0, self._state.fear - self._fear_decay)

        # Overall intensity
        self._state.intensity = float(np.mean([
            self._state.pleasure,
            self._state.frustration,
            self._state.surprise,
            self._state.fear,
        ]))

        # Dominant emotion
        dom = max(
            [("pleasure", self._state.pleasure), ("frustration", self._state.frustration),
             ("surprise", self._state.surprise), ("fear", self._state.fear)],
            key=lambda x: x[1],
        )
        self._state.dominant = dom[0] if dom[1] > 0.3 else "neutral"

        # History
        self._history.append(EmotionState(
            pleasure=self._state.pleasure,
            frustration=self._state.frustration,
            surprise=self._state.surprise,
            fear=self._state.fear,
            intensity=self._state.intensity,
            dominant=self._state.dominant,
        ))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Reset episode counter
        if episode_done:
            self._steps_this_episode = 0

        return self._state

    # ------------------------------------------------------------------ policy modulation

    def modulate_exploration(self, base_epsilon: float = 0.1) -> float:
        """Modify exploration epsilon based on emotional state.

        - Frustrated → explore more (try different strategies)
        - Pleased → exploit more (keep doing what works)
        - Fearful → cautious (reduce random exploration)
        """
        eps = base_epsilon
        eps += self._state.frustration * 0.15          # +15% when frustrated
        eps -= self._state.pleasure * 0.05             # -5% when happy
        eps -= self._state.fear * 0.1                   # -10% when scared
        return max(0.01, min(0.5, eps))

    def modulate_learning_rate(self, base_lr: float = 3e-4) -> float:
        """Modify learning rate based on surprise.

        High surprise → learn faster (something unexpected happened).
        """
        return base_lr * (1.0 + self._state.surprise * 2.0)

    def should_approach_caregiver(self) -> bool:
        """Seek caregiver when scared or frustrated (attachment behavior)."""
        return self._state.fear > 0.6 or self._state.frustration > 0.7

    # ------------------------------------------------------------------ query

    @property
    def state(self) -> EmotionState:
        return self._state

    def describe(self) -> str:
        return self._state.describe()

    @property
    def capacity(self) -> int:
        return 4  # four emotion dimensions

    def __len__(self) -> int:
        return 4

    def summary(self) -> dict:
        return {
            "emotions": self._state.to_dict(),
            "dominant": self._state.dominant,
            "description": self.describe(),
            "modulated_exploration": self.modulate_exploration(),
            "steps_this_episode": self._steps_this_episode,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "pleasure": self._state.pleasure,
            "frustration": self._state.frustration,
            "surprise": self._state.surprise,
            "fear": self._state.fear,
            "reward_ema": self._reward_ema,
            "steps_this_episode": self._steps_this_episode,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state.pleasure = float(state.get("pleasure", 0.5))
        self._state.frustration = float(state.get("frustration", 0.0))
        self._state.surprise = float(state.get("surprise", 0.0))
        self._state.fear = float(state.get("fear", 0.0))
        self._reward_ema = float(state.get("reward_ema", 0.0))
        self._steps_this_episode = int(state.get("steps_this_episode", 0))
