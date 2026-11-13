"""Goal executive — Stage 27 M1: learned selection over memory slots.

The agent COMMITS to one of its own perceptual goals (the active-occlusion
last-known slots it already sees in its observation) or to "none". The
selection is sampled every step and trained jointly with PPO (joint logp:
``logp_action + logp_exec``), so the goal-choice itself receives credit from
the eventual settlement reward (one-shot, B10-safe).

Bounded by construction: capacity = number of observable slots (<= 3), no new
observation inputs, no unbounded buffers.

中文: 目标级执行层 — 从自身感知槽位中选择一个目标 (或 none), 采样与 PPO
联合训练; 有界 (容量=槽位数<=3), 无新观测输入; 结算=一次性 (B10 安全)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_SLOTS = 3
SLOT_FEATURES = 3  # (dx/4, dy/4, dist/4) — matches _proprio's slot layout


def slot_features_from_proprio(
    proprio: torch.Tensor, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (features, valid_mask) for the executive head from proprio.

    The observation slots are the LAST ``3 * num_slots`` dims of the proprio
    vector (``_proprio`` appends them after the body/grasp block), sorted
    nearest-first by the env. A slot is "valid" when its dist component is
    non-zero (empty slots are zero-filled).

    Args:
        proprio: (..., proprio_dim) float tensor.
        num_slots: bounded slot count (0..3).

    Returns:
        feats: (..., num_slots, 3); valid: (..., num_slots) float (0/1).
    """
    if proprio is None:
        raise ValueError("proprio is required for slot_features_from_proprio")
    k = int(num_slots)
    if k <= 0:
        shape = tuple(proprio.shape[:-1])
        return (
            proprio.new_zeros(*shape, 0, SLOT_FEATURES),
            proprio.new_zeros(*shape, 0),
        )
    need = k * SLOT_FEATURES
    if proprio.shape[-1] < need:
        raise ValueError(
            f"proprio dim {proprio.shape[-1]} < slots need {need}"
        )
    tail = proprio[..., -need:]
    feats = tail.reshape(*tail.shape[:-1], k, SLOT_FEATURES)
    valid = (feats[..., 2] > 1e-6).to(proprio.dtype)
    return feats, valid


class GoalExecutive(nn.Module):
    """Small categorical head: (hidden, slot feats, mask) -> (K+1) logits.

    Args:
        hidden_size: policy trunk hidden width.
        num_slots: candidate count (bounded by MAX_SLOTS).
        slot_dim: features per slot (3: normalized dx, dy, dist).
        hidden: MLP width.

    The last logit (index ``num_slots``) is "none" (no commitment now).
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = MAX_SLOTS,
        slot_dim: int = SLOT_FEATURES,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_slots = max(0, min(MAX_SLOTS, int(num_slots)))
        self.slot_dim = int(slot_dim)
        in_dim = self.hidden_size + self.num_slots * self.slot_dim + self.num_slots
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), self.num_slots + 1),
        )
        # Zero-init the output layer: uniform choice at start (pure exploration
        # until the settlement signal shapes the head).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def capacity(self) -> int:
        return self.num_slots

    def __len__(self) -> int:
        return self.num_slots

    @property
    def num_choices(self) -> int:
        return self.num_slots + 1

    @property
    def none_index(self) -> int:
        return self.num_slots

    def forward(
        self,
        hidden: torch.Tensor,
        slot_feats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits (..., num_choices).

        Invalid slots are masked to ``-inf`` so they can never be sampled;
        the "none" choice is always available.
        """
        batch_shape = hidden.shape[:-1]
        feats = slot_feats.reshape(*batch_shape, self.num_slots * self.slot_dim)
        mask = valid_mask.reshape(*batch_shape, self.num_slots)
        x = torch.cat([hidden.float(), feats.float(), mask.float()], dim=-1)
        logits = self.mlp(x)
        if self.num_slots > 0:
            valid_slots = logits[..., : self.num_slots]
            neg = torch.full_like(valid_slots, float("-inf"))
            valid_slots = torch.where(mask > 0.5, valid_slots, neg)
            logits = torch.cat([valid_slots, logits[..., self.num_slots:]], dim=-1)
        return logits
