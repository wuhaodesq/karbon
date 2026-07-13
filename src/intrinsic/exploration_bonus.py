"""Count-based exploration bonus — 3D deadlock guard.

The 3D training deadlock: when the environment reward is sparse /
near-constant, the value head eventually fits it -> advantages collapse
to ~0 -> the policy gradient vanishes and the agent stops learning.
RND-style intrinsic curiosity can patch this only while its predictor
error stays > 0; once the predictor converges on visited states the
bonus decays to 0 and the deadlock returns.

This module adds a *state-dependent* exploration bonus:

    bonus(s) = coef / sqrt(count(s) + 1)

- ``count(s)``: visitation count of a downsampled, hashed observation,
  stored in a FIXED-capacity tensor (Axiom 1: no unbounded growth).
- bonus is highest for novel / rarely-visited states and decays toward
  0 as a state is revisited, but never reaches 0 for any finite count.
- Crucially it VARIES across states AND with visitation history,
  which the value head (which only sees the current obs) cannot
  predict -> it leaves a persistent residual in the advantages, so the
  policy always has an exploration signal even when env reward is
  sparse. (A flat *constant* floor would be a no-op: advantages are
  invariant to adding a constant to every reward, because the value
  head fits the constant too.)

Used as a floor on the intrinsic term in ``src/train.py``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplorationBonus(nn.Module):
    """Bounded count-based exploration bonus (GAE deadlock guard)."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        capacity: int = 1 << 16,
        coef: float = 0.1,
        grid: int = 8,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"obs_shape must be (H,W,C), got {obs_shape}")
        self.capacity = int(capacity)  # Axiom 1: declared fixed size
        self.coef = float(coef)
        self.grid = int(grid)
        c = int(obs_shape[2])
        d = c * grid * grid
        # Deterministic mixing weights (Knuth multiplicative hash) so hashing
        # is stable without carrying RNG state.
        self._w = (
            (torch.arange(d) * 2654435761 + 1) % (self.capacity - 1)
        ).long() + 1
        self._counts = torch.zeros(self.capacity, dtype=torch.long)  # BOUNDS-OK: fixed

    def __len__(self) -> int:
        return int(self._counts.sum().item())

    @torch.no_grad()
    def _hash(self, obs_u8: torch.Tensor) -> torch.Tensor:
        if obs_u8.dim() == 3:
            obs_u8 = obs_u8.unsqueeze(0)
        x = obs_u8.float().permute(0, 3, 1, 2) / 255.0  # (B,C,H,W)
        g = F.interpolate(x, size=(self.grid, self.grid), mode="area")  # (B,C,g,g)
        q = (g * 255.0).byte().long().reshape(obs_u8.shape[0], -1)  # (B, d)
        h = (q * self._w[: q.shape[1]]).sum(dim=1) % self.capacity
        return h.long()

    @torch.no_grad()
    def bonus(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Exploration bonus per observation. Returns a (B,) tensor."""
        h = self._hash(obs_u8)
        counts = self._counts[h].float()
        return self.coef / torch.sqrt(counts + 1.0)

    @torch.no_grad()
    def update(self, obs_u8: torch.Tensor) -> None:
        h = self._hash(obs_u8)
        self._counts.index_add_(0, h, torch.ones(h.shape[0], dtype=torch.long))

    def state_dict(self) -> dict[str, Any]:
        return {"counts": self._counts.clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # type: ignore[override]
        if "counts" in state and state["counts"].shape == self._counts.shape:
            self._counts.copy_(state["counts"])
