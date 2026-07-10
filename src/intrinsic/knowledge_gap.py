"""Knowledge Gap Detection.

Phase 4+ intrinsic motivation improvement: detect what the agent
doesn't know yet and direct curiosity toward knowledge gaps.

Core idea:
    1. Track per-concept (per-slot) prediction accuracy via EMA.
    2. Concepts with sustained high prediction error = knowledge gaps.
    3. Boost curiosity rewards for states involving gap concepts.
    4. This creates a "learning progress"-driven exploration that
       naturally seeks out the hardest-to-predict concepts.

Bounded: per-slot EMA tracking, fixed number of slots. No growing data.

知识缺口检测：追踪每个概念（槽位）的预测准确度，
高预测误差的概念 = 知识缺口 → 引导探索方向。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KnowledgeGapConfig:
    """Configuration for :class:`KnowledgeGapDetector`.

    - ``num_slots``: number of Slot Attention slots (concept count).
    - ``ema_decay``: per-slot error EMA decay rate.
    - ``gap_threshold``: relative threshold — slots with error >
      gap_threshold * mean_error are considered "gaps".
    - ``boost_factor``: multiply curiosity reward for gap-related states.
    """

    num_slots: int = 7
    ema_decay: float = 0.99
    gap_threshold: float = 1.5
    boost_factor: float = 2.0


class KnowledgeGapDetector(nn.Module):
    """Track per-concept prediction gaps to guide curiosity.

    Uses Slot Attention outputs as concept proxies. Each slot represents
    an object/concept, and the world model prediction error per concept
    tells the agent which concepts it understands poorly.

    Bounded: fixed num_slots, fixed-size EMA buffers.
    """

    def __init__(
        self,
        config: KnowledgeGapConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or KnowledgeGapConfig()
        cfg = self.config

        self._total_updates = 0
        # Per-slot prediction error EMA
        self.register_buffer("_slot_errors", torch.zeros(cfg.num_slots))
        # Per-slot visit count (for confidence)
        self.register_buffer("_slot_visits", torch.zeros(cfg.num_slots))

    @property
    def capacity(self) -> int:
        return self.config.num_slots

    def __len__(self) -> int:
        return self._total_updates

    # -------------------------------------------------------- update

    def update(
        self,
        slot_states: torch.Tensor,       # (B, num_slots, d_model)
        prediction_errors: torch.Tensor,  # (B,) per-sample RSSM error
    ) -> None:
        """Update per-slot error tracking.

        Args:
            slot_states: Slot Attention output, each slot is a concept proxy.
            prediction_errors: per-sample world model prediction error.
        """
        cfg = self.config
        bsz, n_slots, d_model = slot_states.shape

        if n_slots != cfg.num_slots:
            return

        # Each slot's activation magnitude = how "present" that concept is
        slot_activation = slot_states.abs().mean(dim=-1)  # (B, num_slots)
        slot_activation = slot_activation / (slot_activation.sum(dim=-1, keepdim=True) + 1e-8)

        # Distribute prediction error across slots proportionally
        error_expanded = prediction_errors.unsqueeze(-1)  # (B, 1)
        slot_error_contribution = (slot_activation * error_expanded).mean(dim=0)  # (num_slots,)

        # EMA update
        self._slot_errors = (
            cfg.ema_decay * self._slot_errors
            + (1 - cfg.ema_decay) * slot_error_contribution
        )
        self._slot_visits = cfg.ema_decay * self._slot_visits + (1 - cfg.ema_decay) * slot_activation.mean(dim=0)

        self._total_updates += 1

    # -------------------------------------------------------- query

    def get_gap_slots(self) -> list[int]:
        """Return slots currently identified as knowledge gaps."""
        cfg = self.config
        if self._total_updates < 10:
            return list(range(cfg.num_slots))  # all are gaps early on

        mean_err = self._slot_errors.mean()
        if mean_err < 1e-8:
            return []

        gap_threshold = mean_err * cfg.gap_threshold
        return [
            i for i in range(cfg.num_slots)
            if self._slot_errors[i].item() > gap_threshold.item()
        ]

    def get_gap_boost(
        self,
        slot_states: torch.Tensor,  # (B, num_slots, d_model)
    ) -> torch.Tensor:
        """Compute curiosity boost factor for each sample based on gap slots.

        Returns:
            (B,) tensor of boost factors (≥ 1.0).
        """
        gap_slots = self.get_gap_slots()
        if not gap_slots:
            return torch.ones(slot_states.shape[0], device=slot_states.device)

        # Sum activation over gap slots
        gap_activation = slot_states[:, gap_slots, :].abs().mean(dim=-1).sum(dim=-1)  # (B,)
        total_activation = slot_states.abs().mean(dim=-1).sum(dim=-1)  # (B,)

        # Fraction of total activation that belongs to gap slots
        gap_fraction = gap_activation / (total_activation + 1e-8)

        # Boost factor: 1.0 + (config.boost_factor - 1.0) * gap_fraction
        return 1.0 + (self.config.boost_factor - 1.0) * gap_fraction

    def get_slot_error_ranks(self) -> list[tuple[int, float]]:
        """Return slots sorted by highest error (biggest gaps first)."""
        errors = self._slot_errors.tolist()
        ranked = sorted(enumerate(errors), key=lambda x: -x[1])
        return ranked

    # -------------------------------------------------------- diagnostics

    def summary(self) -> dict[str, Any]:
        gap_slots = self.get_gap_slots()
        return {
            "total_updates": self._total_updates,
            "num_gap_slots": len(gap_slots),
            "gap_slot_ids": gap_slots,
            "slot_errors": self._slot_errors.tolist(),
            "mean_error": float(self._slot_errors.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "total_updates": self._total_updates,
            "slot_errors": self._slot_errors,
            "slot_visits": self._slot_visits,
        }

    def load_state_dict(self, state: dict) -> None:
        self._total_updates = int(state.get("total_updates", 0))
        self._slot_errors = state.get("slot_errors", torch.zeros(self.config.num_slots))
        self._slot_visits = state.get("slot_visits", torch.zeros(self.config.num_slots))
