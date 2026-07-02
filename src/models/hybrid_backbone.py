"""Hybrid Backbone: TTT-Linear + Sliding-Window Attention + FFN.

Stage 2's core deliverable. A single ``HybridBlock`` interleaves:

1. **TTT-Linear** — in-context adaptation over the sequence, O(T) time,
   bounded inner state (single d_h×d_h matrix per batch element).
2. **Sliding-Window Attention** — local precise retrieval, O(T·W).
3. **FFN** — standard 2-layer MLP with GELU.

Each sub-layer wrapped with LayerNorm + Residual (pre-norm style).

The optional third component — **TTT-MLP** — is left as a future extension in
Stage 2 late/early Stage 3. For now the block is the "fast+precise" pair which
is already the interesting Hybrid experiment.

Hybrid 骨干：TTT-Linear（快速在线适应）+ 滑窗注意力（精确近距检索）+ FFN。
每子层带 pre-norm + residual。TTT-MLP 慢层留待 Stage 2b/3 加入。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .sliding_attn import SlidingWindowAttention
from .ttt_linear import TTTLinear


class FFN(nn.Module):
    """Standard 2-layer MLP with GELU."""

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class HybridBlock(nn.Module):
    """A single block: TTT-Linear → SWA → FFN, each with pre-norm + residual.

    Args:
        d_model: model dimension.
        n_heads: number of SWA heads.
        swa_window_size: sliding-window size.
        ttt_mini_batch: mini-batch segment size for TTT-Linear.
        ttt_detach_every: BPTT chunk boundary for TTT-Linear.
        ffn_hidden_mult: FFN hidden = d_model * mult.
        dropout: dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        swa_window_size: int = 32,
        ttt_mini_batch: int = 16,
        ttt_detach_every: int | None = None,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Pre-norms
        self.norm_ttt = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Sub-layers
        self.ttt = TTTLinear(
            d_in=d_model,
            d_h=d_model,
            mini_batch=ttt_mini_batch,
            detach_every_n_segments=ttt_detach_every,
        )
        self.attn = SlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=swa_window_size,
            dropout=dropout,
        )
        self.ffn = FFN(d_model, hidden_mult=ffn_hidden_mult, dropout=dropout)

        # Residual dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) TTT-Linear with pre-norm + residual
        x = x + self.drop(self.ttt(self.norm_ttt(x)))
        # 2) Sliding-Window Attention
        x = x + self.drop(self.attn(self.norm_attn(x)))
        # 3) FFN
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class SinusoidalPositionEmbedding(nn.Module):
    """Deterministic sinusoidal position embedding.

    Not learned. Simple to serialize / migrate across hardware.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal position embedding")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len={self.max_len}")
        return x + self.pe[:T].unsqueeze(0)  # (1, T, d_model)


class HybridBackbone(nn.Module):
    """Stack of :class:`HybridBlock`s with token embedding and sinusoidal PE.

    This is the sequence encoder for Stage 2 experiments. It is unopinionated
    about the downstream head (policy/value, LM head, etc.).

    Args:
        vocab_size: input token vocabulary. Set to 0 to skip token embedding
            (i.e., pass pre-embedded ``x`` directly to :meth:`encode`).
        d_model: model dim (throughout).
        n_layers: number of HybridBlocks.
        n_heads: SWA heads.
        swa_window_size: SWA window.
        ttt_mini_batch: TTT-Linear mini-batch size.
        ttt_detach_every: TTT-Linear BPTT chunk boundary.
        max_seq_len: capacity for the position embedding.
        ffn_hidden_mult: FFN hidden multiplier.
        dropout: dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        *,
        vocab_size: int = 0,
        n_heads: int = 4,
        swa_window_size: int = 32,
        ttt_mini_batch: int = 16,
        ttt_detach_every: int | None = None,
        max_seq_len: int = 1024,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        if vocab_size > 0:
            self.tok_embed: nn.Module | None = nn.Embedding(vocab_size, d_model)
        else:
            self.tok_embed = None

        self.pos_embed = SinusoidalPositionEmbedding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                HybridBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    swa_window_size=swa_window_size,
                    ttt_mini_batch=ttt_mini_batch,
                    ttt_detach_every=ttt_detach_every,
                    ffn_hidden_mult=ffn_hidden_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, tokens_or_x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence.

        If ``vocab_size > 0``, ``tokens_or_x`` should be integer token IDs of
        shape ``(B, T)`` and will be embedded. Otherwise it must be a
        pre-embedded float tensor of shape ``(B, T, d_model)``.
        """
        if self.tok_embed is not None:
            if tokens_or_x.dtype not in (torch.long, torch.int, torch.int64, torch.int32):
                raise ValueError(
                    "vocab_size > 0 but got non-integer input; pass token IDs"
                )
            x = self.tok_embed(tokens_or_x)
        else:
            if tokens_or_x.dim() != 3 or tokens_or_x.shape[-1] != self.d_model:
                raise ValueError(
                    f"expected pre-embedded (B, T, {self.d_model}) input"
                )
            x = tokens_or_x

        x = self.pos_embed(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"vocab_size={self.vocab_size}, params={self.num_parameters()}"
        )
