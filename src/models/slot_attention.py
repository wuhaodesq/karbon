"""Slot Attention — Object-Centric Representation Learning.

Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.

Decomposes a visual scene into a fixed number of *slot* vectors, each
representing one object (position, color, shape, velocity).  The slots compete
via iterative softmax-normalised attention and are updated by a GRU.

Why this matters for developmental AI:
    - A newborn does not see "64×64×3 pixels" — it segments the world into
      objects.  Slot Attention provides exactly that inductive bias.
    - Each slot is a *symbol-like* entity whose embedding can be fed directly
      into the TTT-Hybrid backbone as a sequence of "tokens" (one per object).

Bounded: ``num_slots`` is fixed at construction (Axiom 1). Total parameters ~ 0.5M.
VRAM: ~0.1 GB for 7 slots × 128 dim.

槽注意力机制：把场景分解为固定数量的物体槽位。每个槽位对应一个物体的表征。
~500K 参数, ~0.1 GB VRAM.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Slot Attention (Locatello et al. 2020, adapted)
# =====================================================================


class SlotAttention(nn.Module):
    """Iterative slot attention for object-centric scene decomposition.

    Pipeline:
        CNN encoder → (B, d_model, H', W') feature grid
        → flatten → (B, H'×W', d_model) input features
        → learnable slot init + 3 iterations of:
            1. attention:  slots attend to input features (competition via softmax over slots)
            2. aggregate:  weighted sum of input features
            3. update:     GRU(slot, aggregate) with residual connection

    Output: (B, num_slots, d_model) — can be fed directly to HybridBackbone
    as a sequence where each "token" is one object.

    Parameters ~ 4 * d_model² + CNN ≈ 500K.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_slots: int = 7,
        slot_dim: int | None = None,
        num_iterations: int = 3,
        input_channels: int = 3,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        self._num_slots = num_slots
        self._num_iters = num_iterations
        self._slot_dim = slot_dim or d_model
        slot_dim = self._slot_dim

        # Lightweight CNN encoder: 64×64×3 → 8×8×hidden → d_model
        self._cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 32×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),              # 16×16×64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),             # 8×8×128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, 1),                              # 8×8×d_model
        )

        # Learnable initial slot vectors (one per slot, broadcast across batch)
        self._slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self._slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Projections for attention
        self._q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self._k_proj = nn.Linear(d_model, slot_dim, bias=False)
        self._v_proj = nn.Linear(d_model, slot_dim, bias=False)

        # GRU for slot update
        self._gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for residual update after GRU
        self._mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, d_model),
        )

        # Layer norms
        self._norm_inputs = nn.LayerNorm(d_model)
        self._norm_slots = nn.LayerNorm(slot_dim)
        self._norm_pre_ff = nn.LayerNorm(d_model)

        # Grid position embeddings (learnable)
        grid_h = (input_size + 3) // 4 // 4 // 4  # after three stride-2 convs: 64→32→16→8
        grid_w = grid_h
        self._grid_pos = nn.Parameter(torch.randn(1, grid_h * grid_w, d_model) * 0.02)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Decompose observation into object slots.

        Args:
            obs_u8: (B, H, W, 3) uint8 image in [0, 255].

        Returns:
            (B, num_slots, d_model) slot embeddings — one per object.
        """
        B = obs_u8.shape[0]

        # Normalize: (B, H, W, 3) uint8 → (B, 3, H, W) float in [0, 1]
        x = obs_u8.float() / 255.0
        x = x.permute(0, 3, 1, 2)

        # CNN encode: (B, 3, 64, 64) → (B, d_model, H', W')
        feats = self._cnn(x)                     # (B, d_model, H', W')
        _, d, h, w = feats.shape
        feats = feats.reshape(B, d, h * w)       # (B, d_model, N)
        feats = feats.permute(0, 2, 1)            # (B, N, d_model)

        # Add grid position embeddings
        feats = feats + self._grid_pos[:, : h * w, :]
        feats = self._norm_inputs(feats)          # (B, N, d_model)

        # Initialize slots from learned parameters
        slots = self._slots_mu + torch.randn(
            B, self._num_slots, self._q_proj.out_features,
            device=obs_u8.device,
        ) * torch.exp(self._slots_log_sigma * 0.5)

        # Key, value projections (done once per forward)
        k = self._k_proj(feats)     # (B, N, slot_dim)
        v = self._v_proj(feats)     # (B, N, slot_dim)

        # Iterative attention
        for _ in range(self._num_iters):
            slots_prev = slots
            slots = self._norm_slots(slots)

            # Compute attention: q(slots) @ k(feats)^T
            q = self._q_proj(slots)                            # (B, num_slots, slot_dim)
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k)   # (B, num_slots, N)
            attn_logits = attn_logits / (self._slot_dim ** 0.5)

            # Softmax over slots (competition): each pixel belongs to ONE slot
            attn = F.softmax(attn_logits, dim=1)                # (B, num_slots, N)

            # Weighted aggregation
            updates = torch.einsum("bsn,bnd->bsd", attn, v)    # (B, num_slots, slot_dim)

            # GRU update
            slots = self._gru(
                updates.reshape(B * self._num_slots, -1),
                slots_prev.reshape(B * self._num_slots, -1),
            )
            slots = slots.reshape(B, self._num_slots, -1)

            # Residual MLP
            slots = slots + self._mlp(self._norm_pre_ff(slots))

        # Optional: sort slots by attention mass for stability
        # (not strictly necessary but helps interpretability)

        return slots  # (B, num_slots, d_model)
