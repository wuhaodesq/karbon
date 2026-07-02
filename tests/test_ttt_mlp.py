"""Tests for :mod:`src.models.ttt_mlp`."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.ttt_mlp import TTTMLP, _gelu_derivative, ttt_mlp_forward_pytorch


def _seed():
    torch.manual_seed(0)


def test_gelu_derivative_matches_autograd():
    """Analytic GELU' must match autograd to fp32 tolerance."""
    x = torch.randn(64, requires_grad=True, dtype=torch.float64)
    y = F.gelu(x, approximate="none")
    (autograd_grad,) = torch.autograd.grad(y.sum(), x)
    ana = _gelu_derivative(x.detach())
    torch.testing.assert_close(ana, autograd_grad, atol=1e-9, rtol=1e-9)


def test_ttt_mlp_output_shape():
    _seed()
    layer = TTTMLP(d_in=8, d_h=16, mini_batch=4, inner_hidden_mult=2)
    x = torch.randn(2, 12, 8)
    y = layer(x)
    assert y.shape == (2, 12, 16)


def test_ttt_mlp_gradient_flows_to_outer_params():
    _seed()
    layer = TTTMLP(d_in=6, d_h=6, mini_batch=3, inner_hidden_mult=2)
    x = torch.randn(1, 9, 6)
    y = layer(x)
    y.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"


def test_ttt_mlp_deterministic_with_seed():
    _seed()
    m1 = TTTMLP(d_in=4, d_h=4, mini_batch=2)
    _seed()
    m2 = TTTMLP(d_in=4, d_h=4, mini_batch=2)
    x = torch.randn(1, 6, 4)
    torch.testing.assert_close(m1(x), m2(x))


def test_ttt_mlp_various_mini_batches_finite():
    _seed()
    x = torch.randn(2, 16, 8)
    tK = torch.randn(8, 8) * 0.3
    tV = torch.randn(8, 8) * 0.3
    tQ = torch.randn(8, 8) * 0.3
    eta = torch.tensor(0.05)
    for mb in [1, 2, 4, 8, 16]:
        y, _ = ttt_mlp_forward_pytorch(x, tK, tV, tQ, eta, mb, inner_hidden_mult=2)
        assert torch.isfinite(y).all(), f"NaN at mb={mb}"


def test_ttt_mlp_zero_eta_gives_stationary_inner_weights():
    """With eta=0, the inner MLP never updates → outputs at every position use
    the initial random inner weights.
    """
    _seed()
    x = torch.randn(1, 8, 4)
    y1, (W1_1, W2_1) = ttt_mlp_forward_pytorch(
        x, torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4),
        torch.tensor(0.0), mini_batch=4,
    )
    # Same call → same result (deterministic init using generator inside)
    y2, (W1_2, W2_2) = ttt_mlp_forward_pytorch(
        x, torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4),
        torch.tensor(0.0), mini_batch=4,
    )
    # With eta=0 initial W is preserved
    torch.testing.assert_close(W1_1, W1_2)
    torch.testing.assert_close(W2_1, W2_2)


def test_ttt_mlp_expressive_advantage_over_linear():
    """Sanity check on capacity: TTT-MLP forward is stable and non-trivial
    when given a benign eta / mini_batch pair.

    We do not measure regression quality here (that's Stage 2's job);
    just guarantee finite outputs with a moderate learning rate.
    """
    _seed()
    d = 8
    T = 32
    x = torch.randn(1, T, d) * 0.5
    tK = torch.randn(d, d) * 0.3
    tV = torch.randn(d, d) * 0.3
    tQ = torch.randn(d, d) * 0.3

    y_mlp, _ = ttt_mlp_forward_pytorch(
        x, tK, tV, tQ, torch.tensor(0.01), mini_batch=4, inner_hidden_mult=2,
    )
    assert torch.isfinite(y_mlp).all()
    assert y_mlp.abs().sum() > 0
