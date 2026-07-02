"""TTT-MLP: Test-Time Training with a 2-layer MLP inner model.

Extension of TTT-Linear (Sun et al. 2024). Instead of a linear map ``W k``,
the inner model is a two-layer MLP with a non-linear activation:

.. code-block:: text

    f_θ(k) = W2 · φ(W1 · k)             where φ = GELU

    ℓ(θ; x_t) = ‖ f_θ(k_t) − v_t ‖²
    θ_t = θ_{t-1} − η · ∇_θ ℓ(θ_{t-1}; x_t)

TTT-MLP has strictly greater expressive power than TTT-Linear (universal
approximator inside), at the cost of more inner state (two matrices instead
of one).

Bounded guarantees identical to TTT-Linear:
- Inner state = fixed pair (W1, W2), never grows.
- Mini-batch segments with optional BPTT-chunk boundaries.

TTT-MLP: 内层从线性升级为 2 层 MLP，表达力大幅增强，仍严格有界。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ttt_mlp_forward_pytorch(
    x: torch.Tensor,
    theta_K: torch.Tensor,
    theta_V: torch.Tensor,
    theta_Q: torch.Tensor,
    eta: torch.Tensor,
    mini_batch: int,
    detach_every_n_segments: int | None = None,
    inner_hidden_mult: int = 2,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Reference-quality mini-batch TTT-MLP forward.

    Inner model: ``f(k) = W2 · GELU(W1 · k)``
        - W1 shape: (d_hidden, d_h)
        - W2 shape: (d_h, d_hidden), where d_hidden = inner_hidden_mult * d_h
        - No biases (kept simple, and easily comparable to TTT-Linear).

    Args:
        x: (B, T, d_in)
        theta_K, theta_V, theta_Q: (d_in, d_h)
        eta: () or (B,)
        mini_batch: segment size b
        detach_every_n_segments: BPTT chunk boundary
        inner_hidden_mult: d_hidden = mult * d_h

    Returns:
        y: (B, T, d_h)
        (W1_final, W2_final): each (B, d_hidden, d_h) and (B, d_h, d_hidden)
    """
    B, T, d_in = x.shape
    d_h = theta_K.shape[1]
    d_hidden = d_h * inner_hidden_mult
    assert theta_V.shape == (d_in, d_h)
    assert theta_Q.shape == (d_in, d_h)
    device = x.device
    dtype = x.dtype

    # Project once
    K = x @ theta_K   # (B, T, d_h)
    V = x @ theta_V
    Q = x @ theta_Q

    # eta broadcastable
    if eta.dim() == 0:
        eta_b = eta.reshape(1, 1, 1).expand(B, 1, 1)
    else:
        eta_b = eta.reshape(B, 1, 1)

    # Inner weights initialized small (not zero — MLP with zero W1 has zero
    # gradients through GELU'(0) ≠ 0 but is degenerate). Use fan-in scaled
    # random init and treat these as *state*, not parameters.
    gen = torch.Generator(device=device).manual_seed(0)
    scale1 = 1.0 / math.sqrt(d_h)
    scale2 = 1.0 / math.sqrt(d_hidden)
    W1 = torch.randn(d_hidden, d_h, generator=gen, device=device, dtype=dtype) * scale1
    W2 = torch.randn(d_h, d_hidden, generator=gen, device=device, dtype=dtype) * scale2
    W1 = W1.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, d_hidden, d_h)
    W2 = W2.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, d_h, d_hidden)

    y = torch.empty(B, T, d_h, device=device, dtype=dtype)
    num_segments = math.ceil(T / mini_batch)

    for seg in range(num_segments):
        t0 = seg * mini_batch
        t1 = min(t0 + mini_batch, T)

        K_seg = K[:, t0:t1, :]  # (B, b, d_h)
        V_seg = V[:, t0:t1, :]
        Q_seg = Q[:, t0:t1, :]

        # Inner forward on segment (all at same starting weights)
        # z1 = W1 @ k → (B, b, d_hidden)
        z1 = K_seg @ W1.transpose(-1, -2)
        h = F.gelu(z1)
        # f(k) = W2 @ h → (B, b, d_h)
        fk = h @ W2.transpose(-1, -2)
        residual = fk - V_seg  # (B, b, d_h)

        # Outputs use q_t through same inner model (fixed W within segment)
        q_z1 = Q_seg @ W1.transpose(-1, -2)
        q_h = F.gelu(q_z1)
        y_seg = q_h @ W2.transpose(-1, -2)
        y[:, t0:t1, :] = y_seg

        # --- Gradients (manually, avoids autograd overhead in the inner loop) ---
        # For loss L = 0.5 * ||f(k) - v||^2 per token (sum over t and per-elem):
        #   dL/dW2 = residual · h^T       shape: (d_h, d_hidden)
        #   dL/dh  = W2^T · residual      shape: (d_hidden,)
        #   dL/dz1 = dL/dh * GELU'(z1)
        #   dL/dW1 = dL/dz1 · k^T
        #
        # Batched over (B, b) with sum over b:
        #   grad_W2 = residual^T (B, d_h, b) @ h (B, b, d_hidden) → (B, d_h, d_hidden)
        grad_W2 = residual.transpose(-1, -2) @ h  # (B, d_h, d_hidden)

        # grad_h per token: residual @ W2 → (B, b, d_hidden)
        grad_h = residual @ W2  # (B, b, d_hidden)
        gelu_grad = _gelu_derivative(z1)  # (B, b, d_hidden)
        grad_z1 = grad_h * gelu_grad  # (B, b, d_hidden)

        # grad_W1: sum over b of grad_z1_t · k_t^T
        # (B, d_hidden, b) @ (B, b, d_h) → (B, d_hidden, d_h)
        grad_W1 = grad_z1.transpose(-1, -2) @ K_seg

        # Segment SGD step (one update per segment on the aggregated grad)
        W1 = W1 - eta_b * grad_W1
        W2 = W2 - eta_b * grad_W2

        # Optional BPTT chunk boundary
        if (
            detach_every_n_segments is not None
            and (seg + 1) % detach_every_n_segments == 0
            and seg < num_segments - 1
        ):
            W1 = W1.detach()
            W2 = W2.detach()

    return y, (W1, W2)


def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of the exact GELU:

        φ(x)  = 0.5 x (1 + erf(x / √2))
        φ'(x) = 0.5 (1 + erf(x / √2)) + x · pdf(x)      where pdf(x) = e^{-x²/2} / √(2π)

    Numerically stable; matches ``torch.autograd.grad(GELU)`` to fp32 tolerance.
    """
    sqrt_2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    cdf_part = 0.5 * (1.0 + torch.erf(x / sqrt_2))
    pdf_part = torch.exp(-x.pow(2) / 2.0) / sqrt_2pi
    return cdf_part + x * pdf_part


class TTTMLP(nn.Module):
    """TTT-MLP layer.

    Outer parameters:
        θ_K, θ_V, θ_Q: (d_in, d_h)
        log_eta: scalar

    Inner state (created per forward): (W1, W2). Fixed size, no growth.
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        mini_batch: int = 16,
        inner_hidden_mult: int = 2,
        init_log_eta: float = -4.0,
        share_eta_across_batch: bool = True,
        detach_every_n_segments: int | None = None,
    ) -> None:
        super().__init__()
        if d_h <= 0 or d_in <= 0:
            raise ValueError("d_in and d_h must be positive")
        if mini_batch <= 0:
            raise ValueError("mini_batch must be positive")
        if inner_hidden_mult < 1:
            raise ValueError("inner_hidden_mult must be >= 1")

        self.d_in = d_in
        self.d_h = d_h
        self.mini_batch = mini_batch
        self.inner_hidden_mult = inner_hidden_mult
        self.share_eta = share_eta_across_batch
        self.detach_every_n_segments = detach_every_n_segments

        scale = 1.0 / math.sqrt(d_in)
        self.theta_K = nn.Parameter(torch.randn(d_in, d_h) * scale)
        self.theta_V = nn.Parameter(torch.randn(d_in, d_h) * scale)
        self.theta_Q = nn.Parameter(torch.randn(d_in, d_h) * scale)
        self.log_eta = nn.Parameter(torch.tensor(float(init_log_eta)))

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.log_eta)

    def forward(
        self,
        x: torch.Tensor,
        return_final_W: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"TTTMLP expects (B, T, d_in), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected last dim {self.d_in}, got {x.shape[-1]}")

        eta = self.eta
        if not self.share_eta:
            eta = eta.expand(x.shape[0])

        y, W_final = ttt_mlp_forward_pytorch(
            x,
            self.theta_K,
            self.theta_V,
            self.theta_Q,
            eta,
            self.mini_batch,
            detach_every_n_segments=self.detach_every_n_segments,
            inner_hidden_mult=self.inner_hidden_mult,
        )
        if return_final_W:
            return y, W_final
        return y

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_h={self.d_h}, "
            f"inner_hidden={self.d_h * self.inner_hidden_mult}, "
            f"mini_batch={self.mini_batch}, eta≈{float(self.eta.item()):.4f}"
        )
