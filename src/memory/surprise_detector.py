"""Surprise Detector — ensembles multiple novelty signals for episodic memory gating.

Stage 13 deliverable: decides which transitions are worth keeping in long-term
episodic storage. Combines 4 independent signals:
    1. RND prediction error (intrinsic novelty)
    2. RSSM reconstruction error (world model uncertainty)
    3. BoundedCoverage hash novelty (state visitation rarity)
    4. TD error magnitude (value prediction surprise)

The ensemble produces a normalized surprise score per transition.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class SurpriseDetector:
    """Ensemble surprise detector for episodic memory gating.

    Combines multiple novelty/uncertainty signals into a single surprise
    score, normalized by running mean to maintain a stable threshold.
    """

    def __init__(
        self,
        rnd_weight: float = 0.3,
        rssm_weight: float = 0.3,
        coverage_weight: float = 0.2,
        td_weight: float = 0.2,
        smoothing: float = 0.9,
    ) -> None:
        self.rnd_weight = float(rnd_weight)
        self.rssm_weight = float(rssm_weight)
        self.coverage_weight = float(coverage_weight)
        self.td_weight = float(td_weight)
        self._smoothing = float(smoothing)
        self._running_avg = 0.0
        self._running_std = 0.0
        self._count = 0

    def compute(
        self,
        rnd_reward: float | torch.Tensor,
        rssm_recon: float | torch.Tensor,
        coverage_novelty: float,
        td_error: float | torch.Tensor | None = None,
    ) -> float:
        """Compute normalized ensemble surprise score.

        Returns a float; higher = more surprising = more worth remembering.
        """
        r = float(rnd_reward) if isinstance(rnd_reward, torch.Tensor) else float(rnd_reward)
        rs = float(rssm_recon) if isinstance(rssm_recon, torch.Tensor) else float(rssm_recon)
        td = float(td_error) if isinstance(td_error, torch.Tensor) else (td_error or 0.0)

        raw = (self.rnd_weight * r + self.rssm_weight * rs
               + self.coverage_weight * coverage_novelty + self.td_weight * td)

        self._count += 1
        if self._count == 1:
            self._running_avg = raw
            self._running_std = abs(raw) + 1e-8
        else:
            self._running_avg = self._smoothing * self._running_avg + (1 - self._smoothing) * raw
            self._running_std = self._smoothing * self._running_std + (1 - self._smoothing) * abs(raw - self._running_avg)

        if self._running_std < 1e-8:
            return raw
        return float(raw / (self._running_avg + self._running_std + 1e-8))

    def state_dict(self) -> dict:
        return {
            "running_avg": self._running_avg,
            "running_std": self._running_std,
            "count": self._count,
            "smoothing": self._smoothing,
        }

    def load_state_dict(self, state: dict) -> None:
        self._running_avg = float(state.get("running_avg", 0.0))
        self._running_std = float(state.get("running_std", 0.0))
        self._count = int(state.get("count", 0))
        self._smoothing = float(state.get("smoothing", self._smoothing))

    def summary(self) -> dict:
        return {
            "running_avg": round(self._running_avg, 4),
            "running_std": round(self._running_std, 4),
            "count": self._count,
        }
