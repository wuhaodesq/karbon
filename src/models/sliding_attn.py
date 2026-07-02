"""Causal Sliding-Window Attention (SWA).

Each query position ``t`` attends to keys/values in the window
``[max(0, t - window_size + 1), t]``. Purely causal — no future leakage.

Implementation is a plain PyTorch masked softmax (O(T·W) work, O(T²) memory
for the mask *matrix* but the mask is boolean and small). Fine for research
scales; the Hybrid backbone can swap in a flash-attn kernel later.

滑窗因果注意力：每个 query 只看 [t-W+1, t] 的 keys/values。
纯 PyTorch 实现，Stage 2 后半段可换 flash-attn 加速。

**Bounded guarantees**: window_size and n_heads are declared at construction.
No per-step growth. Fits Axiom 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sliding_causal_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Build an additive attention mask of shape (T, T).

    Entries are 0 where attention is allowed, -inf where forbidden.
    A key at position ``j`` is visible to query at ``i`` iff
    ``i - window_size + 1 <= j <= i`` (i.e. causal and within window).

    构建 (T, T) 的加性 mask，允许位置=0，禁止位置=-inf。
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
    allowed = (j <= i) & (j >= i - window_size + 1)
    # Additive mask: 0 where allowed, -inf where forbidden.
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class SlidingWindowAttention(nn.Module):
    """Multi-head causal sliding-window self-attention.

    Args:
        d_model: model dimension (input == output).
        n_heads: number of attention heads.
        window_size: local window size W. Each query attends to the last W keys
            (including itself).
        dropout: attention dropout probability.
        bias: whether the qkv/output projections carry a bias.

    Shape convention:
        input  : (B, T, d_model)
        output : (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout_p = dropout

        # Fused qkv projection for a small speed win
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Cached mask (rebuilt on device/dtype/length change)
        self.register_buffer("_cached_mask", torch.empty(0), persistent=False)
        self._cached_len: int = 0
        self._cached_device: torch.device | None = None

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        needs_rebuild = (
            self._cached_len != seq_len
            or self._cached_device is None
            or self._cached_device != device
        )
        if needs_rebuild:
            self._cached_mask = build_sliding_causal_mask(seq_len, self.window_size, device)
            self._cached_len = seq_len
            self._cached_device = device
        return self._cached_mask  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run causal sliding-window attention.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected (B, T, {self.d_model}), got {tuple(x.shape)}")
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: (B, H, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add mask (broadcast over B, H): (T, T) → (1, 1, T, T)
        mask = self._get_mask(T, x.device).to(scores.dtype)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.d_model)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size}, dropout={self.dropout_p}"
        )
