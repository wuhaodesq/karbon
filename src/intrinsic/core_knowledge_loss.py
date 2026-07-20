"""Core-Knowledge auxiliary losses (open-gap A#4, P2 recipe).

Differentiable auxiliary losses that penalize behavior violating Spelke-style
core knowledge, wired into the PPO total loss (ROADMAP Step 3 / P1+P2). These
are *soft* inductive biases — engineering-controllable, no symbol-interface
problem, no need to wait on academic breakthroughs.

Three priors are covered as auxiliary losses:
  - object permanence: predict an object still exists when no causal event
    removed it (reward an internal "existence belief" staying high).
  - intuitive physics: when the agent applies a force, the contacted object's
    motion direction should align with the force (cause -> effect).
  - number sense: the agent's internal count estimate for a small set should
    match the true count (supervised signal on a count head).

Design: each loss is computed from *observable* env quantities (agent/object
positions & velocities, which PhysicsSandbox exposes) plus lightweight internal
state the agent must learn to maintain. To keep it bounded and differentiable,
the "internal belief" terms are read from a small buffer the trainer maintains
(per-env, fixed capacity) rather than from the env.

All losses are returned as scalar tensors on `device`; the trainer sums them
with a coefficient and adds to the PPO loss (same pattern as EWC penalty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreKnowledgeLossConfig:
    coef_object_permanence: float = 0.1
    coef_intuitive_physics: float = 0.1
    coef_number_sense: float = 0.1
    # how many recent (force, motion) pairs to keep per env for physics loss
    physics_window: int = 8


class CoreKnowledgeAuxLoss(nn.Module):
    """Computes the three core-knowledge auxiliary losses from a per-step record.

    The trainer feeds, each step, a ``record`` dict (per-env list) with:
      - "agent_pos":  (B, 2)
      - "object_pos":  (B, K, 2)  (K = max objects, padded with NaN)
      - "object_vel":  (B, K, 2)
      - "force":       (B, 2)     applied agent force this step
      - "existence_belief": (B, K) agent's internal belief an object exists
      - "count_est":   (B,)       agent's internal count estimate
      - "true_count":  (B,)       true object count
      - "removed_mask":(B, K)     1 if object was causally removed this step

    All tensors on the same device. Losses are masked over padding (NaN) and
    envs with no valid signal.
    """

    def __init__(self, config: CoreKnowledgeLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or CoreKnowledgeLossConfig()

    @staticmethod
    def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over masked entries; returns 0 if no valid entries."""
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        return (t * m).sum() / denom

    def object_permanence_loss(self, rec: dict[str, torch.Tensor]) -> torch.Tensor:
        belief = rec["existence_belief"]          # (B, K)
        removed = rec["removed_mask"]              # (B, K)
        # Penalize dropping belief on objects NOT causally removed.
        # Target: belief should stay ~1 unless removed.
        target = 1.0 - removed                     # (B, K)
        mask = (~torch.isnan(belief)) & (removed == 0)  # only non-removed count
        err = (belief - target) ** 2
        return self._masked_mean(err, mask)

    def intuitive_physics_loss(self, rec: dict[str, torch.Tensor]) -> torch.Tensor:
        force = rec["force"]                       # (B, 2)
        ovel = rec["object_vel"]                   # (B, K, 2)
        # For each env, take the max-speed object as the "contacted" one.
        speed = ovel.norm(dim=-1)                  # (B, K)
        _, idx = speed.max(dim=1, keepdim=True)    # (B, 1)
        contacted_vel = torch.gather(ovel, 1, idx.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)  # (B, 2)
        fn = force.norm(dim=-1).clamp_min(1e-6)
        vn = contacted_vel.norm(dim=-1).clamp_min(1e-6)
        cos = (force * contacted_vel).sum(dim=-1) / (fn * vn)  # (B,)
        # penalize misalignment (cos < 0.5): loss = max(0, 0.5 - cos)
        align = torch.clamp(0.5 - cos, min=0.0)
        mask = (fn > 1e-3) & (vn > 1e-3)
        return self._masked_mean(align, mask)

    def number_sense_loss(self, rec: dict[str, torch.Tensor]) -> torch.Tensor:
        est = rec["count_est"]                     # (B,)
        true = rec["true_count"]                   # (B,)
        mask = true > 0
        err = (est - true).abs() / true.clamp_min(1.0)
        return self._masked_mean(err, mask)

    def forward(self, rec: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        lp = self.object_permanence_loss(rec)
        lph = self.intuitive_physics_loss(rec)
        ln = self.number_sense_loss(rec)
        total = (
            self.config.coef_object_permanence * lp
            + self.config.coef_intuitive_physics * lph
            + self.config.coef_number_sense * ln
        )
        return {
            "object_permanence": lp,
            "intuitive_physics": lph,
            "number_sense": ln,
            "total": total,
        }
