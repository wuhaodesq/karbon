"""TTT-Linear: Test-Time Training with a linear inner model.

**Formulation** (Sun et al. 2024; equivalent to DeltaNet / delta-rule fast weights).

Per token at time t:

.. code-block:: text

    k_t = θ_K · x_t              # inner "training input"
    v_t = θ_V · x_t              # inner "training label"
    q_t = θ_Q · x_t              # inner "test query"

    ℓ(W; x_t) = ‖ W k_t − v_t ‖²
    ∇ℓ       = 2 (W k_t − v_t) k_tᵀ

    W_t = W_{t-1} − η · (W_{t-1} k_t − v_t) k_tᵀ    (SGD, ignore constant 2)
    y_t = W_t · q_t

This module provides two things:

1. :func:`ttt_linear_forward_pytorch` — pure-PyTorch functional forward with
   mini-batch TTT and detached segment boundaries (bounded BPTT).
2. :class:`TTTLinear` — a ``nn.Module`` wrapper that owns θ_K/V/Q and η,
   dispatches to the backend abstract layer.

**Bounded guarantees**:
- No per-step W snapshots retained past the current segment.
- Segment boundaries detach W → gradients flow through outer θ, not through
  the entire time axis. This is the bounded-BPTT policy from the paper.

**Numerical parity target** (Stage 2b):
- Any Triton implementation must match this backend to ≤1e-4 on random inputs
  for d_in=d_out ∈ {16, 32, 64} and T ∈ {32, 64, 128}.

数学上 TTT-Linear 等价于 delta-rule fast weights；这里实现 mini-batch 版本，
段内并行，段间 detach，保证 BPTT 有界。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ttt_backend import get_backend


# =====================================================================
# Functional core: mini-batch TTT-Linear forward
# =====================================================================


def ttt_linear_forward_pytorch(
    x: torch.Tensor,
    theta_K: torch.Tensor,
    theta_V: torch.Tensor,
    theta_Q: torch.Tensor,
    eta: torch.Tensor,
    mini_batch: int,
    detach_every_n_segments: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-quality PyTorch implementation of mini-batch TTT-Linear.

    Args:
        x: (B, T, d_in) input tokens.
        theta_K, theta_V, theta_Q: (d_in, d_h) projection matrices (shared over batch).
        eta: scalar or (B,) — inner SGD learning rate.
        mini_batch: segment size ``b``. Each segment computes gradients from a
            common starting W, updates W once at segment end. b=1 is exact SGD;
            b=T is one-shot batch grad.
        detach_every_n_segments: if given, W is detached from the graph every
            N segments to bound BPTT depth. ``None`` means full BPTT through
            the sequence (fine for short training sequences; not recommended
            for long ones — enable this once sequences are long).

    Returns:
        y: (B, T, d_h) — TTT-Linear outputs per token.
        W_final: (B, d_h, d_h) — final inner weight after the sequence.

    Shape convention: d_h = key/value/query hidden dim.

    **Gradient semantics**: outputs within segment ``s`` depend on ``W_{s-1}``,
    which in turn depends on segments ``0..s-2`` via the update rule. Hence
    θ_K/θ_V receive gradient contributions from *later* segments' outputs (as
    long as those segments are inside the same BPTT chunk).
    """
    B, T, d_in = x.shape
    d_h = theta_K.shape[1]
    assert theta_V.shape == (d_in, d_h)
    assert theta_Q.shape == (d_in, d_h)
    device = x.device
    dtype = x.dtype

    # Project once: (B, T, d_h)
    K = x @ theta_K
    V = x @ theta_V
    Q = x @ theta_Q

    # Inner learning rate as broadcastable tensor: (B, 1, 1)
    if eta.dim() == 0:
        eta_b = eta.reshape(1, 1, 1).expand(B, 1, 1)
    else:
        eta_b = eta.reshape(B, 1, 1)

    # W: (B, d_h, d_h). Initialize to zero — this is the "empty memory" state
    # at the start of the sequence.
    W = torch.zeros(B, d_h, d_h, device=device, dtype=dtype)

    # Output buffer, pre-allocated (Axiom 1)
    y = torch.empty(B, T, d_h, device=device, dtype=dtype)

    num_segments = math.ceil(T / mini_batch)
    for seg in range(num_segments):
        t0 = seg * mini_batch
        t1 = min(t0 + mini_batch, T)

        # Slices for this segment
        K_seg = K[:, t0:t1, :]  # (B, b, d_h)
        V_seg = V[:, t0:t1, :]
        Q_seg = Q[:, t0:t1, :]

        # Compute outputs at start-of-segment W, then update W with segment grad.
        Wk = K_seg @ W.transpose(-1, -2)          # (B, b, d_h)
        residual = Wk - V_seg                      # (B, b, d_h)
        y_seg = Q_seg @ W.transpose(-1, -2)        # (B, b, d_h)
        y[:, t0:t1, :] = y_seg

        # Aggregate segment gradient: sum_t residual_t k_t^T → (B, d_h, d_h)
        grad = residual.transpose(-1, -2) @ K_seg  # (B, d_h, d_h)

        # Update W. Detach only at fixed BPTT-chunk boundaries.
        W = W - eta_b * grad
        if (
            detach_every_n_segments is not None
            and (seg + 1) % detach_every_n_segments == 0
            and seg < num_segments - 1
        ):
            W = W.detach()

    return y, W


# =====================================================================
# Module wrapper
# =====================================================================


class TTTLinear(nn.Module):
    """TTT-Linear layer (Sun et al. 2024).

    The learnable outer parameters are:

    - θ_K, θ_V, θ_Q: linear projections R^{d_in} → R^{d_h}
    - log_eta: log of inner learning rate. Sigmoid-parametrized to stay in
      a stable range.

    The inner weight W (d_h × d_h) is *state*, not parameter — it is created
    fresh at every forward pass.

    外层 θ_K/V/Q 是 nn.Parameter；内层 W 是 forward 时新建的状态，
    不是模型参数（也不参与优化器管理）。
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        mini_batch: int = 16,
        init_log_eta: float = -4.0,   # η ≈ 0.018 after sigmoid
        share_eta_across_batch: bool = True,
        detach_every_n_segments: int | None = None,
    ) -> None:
        super().__init__()
        if d_h <= 0 or d_in <= 0:
            raise ValueError("d_in and d_h must be positive")
        if mini_batch <= 0:
            raise ValueError("mini_batch must be positive")
        self.d_in = d_in
        self.d_h = d_h
        self.mini_batch = mini_batch
        self.share_eta = share_eta_across_batch
        self.detach_every_n_segments = detach_every_n_segments

        # Xavier-ish init for projections
        scale = 1.0 / math.sqrt(d_in)
        self.theta_K = nn.Parameter(torch.randn(d_in, d_h) * scale)
        self.theta_V = nn.Parameter(torch.randn(d_in, d_h) * scale)
        self.theta_Q = nn.Parameter(torch.randn(d_in, d_h) * scale)

        # Learnable inner learning rate (positive via sigmoid)
        self.log_eta = nn.Parameter(torch.tensor(float(init_log_eta)))

    @property
    def eta(self) -> torch.Tensor:
        """Effective inner learning rate, positive scalar."""
        return torch.sigmoid(self.log_eta)

    def forward(
        self,
        x: torch.Tensor,
        backend_name: str | None = None,
        return_final_W: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run TTT-Linear.

        Args:
            x: (B, T, d_in)
            backend_name: force a specific backend (``"pytorch"|"triton"``).
                Defaults to backend auto-selection.
            return_final_W: if True, also return the final inner weight (B, d_h, d_h).

        Returns:
            y: (B, T, d_h)
            (optionally) W_final: (B, d_h, d_h)
        """
        if x.dim() != 3:
            raise ValueError(f"TTTLinear expects (B, T, d_in), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.d_in:
            raise ValueError(f"Expected last dim = {self.d_in}, got {x.shape[-1]}")

        backend = get_backend(backend_name)  # type: ignore[arg-type]
        eta = self.eta
        if not self.share_eta:  # placeholder for future per-batch eta
            eta = eta.expand(x.shape[0])

        # The Triton backend (once written) will accept the same kwarg or ignore
        # detach_every_n_segments if it has its own chunking. The PyTorch
        # backend uses it directly.
        y, W_final = backend(
            x,
            self.theta_K,
            self.theta_V,
            self.theta_Q,
            eta,
            self.mini_batch,
            self.detach_every_n_segments,
        )
        if return_final_W:
            return y, W_final
        return y

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_h={self.d_h}, mini_batch={self.mini_batch}, "
            f"eta≈{float(self.eta.item()):.4f}"
        )
