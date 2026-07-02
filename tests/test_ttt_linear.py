"""Tests for TTT-Linear backend abstraction and PyTorch implementation.

The parity-vs-Triton test is defined here but skipped when Triton is
unavailable (i.e., on Windows / CPU-only). It becomes live automatically once
a CUDA + Triton environment is in use.

The remaining tests verify:
- shape correctness
- gradient flow through outer parameters
- deterministic behavior with fixed seeds
- mini-batch size does not blow up memory or produce NaNs
- the delta-rule identity in the b=1 limit
"""

from __future__ import annotations

import pytest
import torch

from src.models.ttt_backend import get_backend
from src.models.ttt_linear import TTTLinear, ttt_linear_forward_pytorch


def _seeded():
    torch.manual_seed(0)


# =====================================================================
# Basic shape / grad
# =====================================================================


def test_ttt_linear_output_shape():
    _seeded()
    layer = TTTLinear(d_in=8, d_h=16, mini_batch=4)
    x = torch.randn(2, 12, 8)
    y = layer(x)
    assert y.shape == (2, 12, 16)


def test_ttt_linear_returns_final_W():
    _seeded()
    layer = TTTLinear(d_in=8, d_h=8, mini_batch=4)
    x = torch.randn(2, 12, 8)
    y, W_final = layer(x, return_final_W=True)
    assert y.shape == (2, 12, 8)
    assert W_final.shape == (2, 8, 8)


def test_ttt_linear_gradient_flows_to_outer_params():
    _seeded()
    layer = TTTLinear(d_in=6, d_h=6, mini_batch=3)
    x = torch.randn(1, 9, 6)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no gradient on {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient on {name}"


def test_ttt_linear_deterministic_with_seed():
    _seeded()
    layer1 = TTTLinear(d_in=4, d_h=4, mini_batch=2)
    x = torch.randn(1, 6, 4)
    y1 = layer1(x)
    _seeded()
    layer2 = TTTLinear(d_in=4, d_h=4, mini_batch=2)
    y2 = layer2(x)
    torch.testing.assert_close(y1, y2)


# =====================================================================
# Mini-batch semantics
# =====================================================================


def test_ttt_linear_various_mini_batch_sizes():
    _seeded()
    x = torch.randn(2, 16, 8)
    theta_K = torch.randn(8, 8) * 0.3
    theta_V = torch.randn(8, 8) * 0.3
    theta_Q = torch.randn(8, 8) * 0.3
    eta = torch.tensor(0.05)

    for mb in [1, 2, 4, 8, 16]:
        y, W = ttt_linear_forward_pytorch(x, theta_K, theta_V, theta_Q, eta, mb)
        assert y.shape == (2, 16, 8)
        assert torch.isfinite(y).all(), f"NaN/inf at mini_batch={mb}"
        assert torch.isfinite(W).all()


def test_ttt_linear_zero_eta_gives_zero_output():
    """With eta=0 and initial W=0, W never updates, so W·q is always 0."""
    _seeded()
    x = torch.randn(1, 8, 4)
    y, W = ttt_linear_forward_pytorch(
        x,
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.tensor(0.0),
        mini_batch=4,
    )
    torch.testing.assert_close(y, torch.zeros_like(y))
    torch.testing.assert_close(W, torch.zeros_like(W))


def test_ttt_linear_first_segment_output_is_zero():
    """Within the first segment, W is still 0, so y_t = W · q_t = 0 for all t in seg 0."""
    _seeded()
    x = torch.randn(1, 8, 4)
    y, _ = ttt_linear_forward_pytorch(
        x,
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.tensor(0.5),
        mini_batch=4,
    )
    # First 4 tokens' outputs are y = W_start · q, W_start = 0
    torch.testing.assert_close(y[:, :4, :], torch.zeros_like(y[:, :4, :]))
    # Later tokens should generally not be zero
    assert y[:, 4:, :].abs().sum() > 0


# =====================================================================
# Backend selector
# =====================================================================


def test_backend_selector_default_pytorch_on_cpu(monkeypatch):
    monkeypatch.delenv("DEVAGI_TTT_BACKEND", raising=False)
    be = get_backend()
    # On this CPU-only test host, we should get the PyTorch backend.
    # Since backends are plain callables here, verify by calling with a small input.
    x = torch.randn(1, 4, 4)
    y, _ = be(
        x,
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.tensor(0.1),
        2,
    )
    assert y.shape == (1, 4, 4)


def test_backend_selector_force_pytorch():
    be = get_backend("pytorch")
    x = torch.randn(1, 4, 4)
    y, _ = be(
        x,
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.tensor(0.1),
        2,
    )
    assert y.shape == (1, 4, 4)


def test_backend_selector_bad_name_raises(monkeypatch):
    monkeypatch.delenv("DEVAGI_TTT_BACKEND", raising=False)
    with pytest.raises(ValueError):
        get_backend("nonesuch")  # type: ignore[arg-type]


# =====================================================================
# Parity: PyTorch vs Triton (skipped when Triton unavailable)
# =====================================================================


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _triton_available(), reason="Triton + CUDA required")
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("T", [32, 64, 128])
def test_triton_pytorch_parity(d, T):
    """Stage 2b acceptance: |Triton − PyTorch| ≤ 1e-4."""
    from src.models.ttt_linear_triton import ttt_linear_forward_triton  # type: ignore

    _seeded()
    x = torch.randn(2, T, d, device="cuda")
    tK = torch.randn(d, d, device="cuda") * 0.2
    tV = torch.randn(d, d, device="cuda") * 0.2
    tQ = torch.randn(d, d, device="cuda") * 0.2
    eta = torch.tensor(0.02, device="cuda")

    y_pt, W_pt = ttt_linear_forward_pytorch(x, tK, tV, tQ, eta, 16)
    y_tr, W_tr = ttt_linear_forward_triton(x, tK, tV, tQ, eta, 16)  # type: ignore

    torch.testing.assert_close(y_pt, y_tr, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(W_pt, W_tr, atol=1e-4, rtol=1e-4)


# =====================================================================
# Regression: TTT-Linear as delta-rule (b=1) recovers the expected update
# =====================================================================


def test_ttt_linear_b1_matches_manual_delta_rule():
    """With mini_batch=1, TTT-Linear updates identically to the manual delta rule:
        W_{t+1} = W_t - eta (W_t k_t - v_t) k_t^T
    """
    _seeded()
    torch.set_default_dtype(torch.float64)
    try:
        d = 4
        T = 5
        x = torch.randn(1, T, d)
        tK = torch.randn(d, d) * 0.3
        tV = torch.randn(d, d) * 0.3
        tQ = torch.randn(d, d) * 0.3
        eta = 0.05

        # --- Reference manual loop ---
        K = (x @ tK).squeeze(0)
        V = (x @ tV).squeeze(0)
        Q = (x @ tQ).squeeze(0)
        W = torch.zeros(d, d, dtype=torch.float64)
        y_ref = torch.zeros(T, d, dtype=torch.float64)
        for t in range(T):
            y_ref[t] = W @ Q[t]
            residual = W @ K[t] - V[t]
            grad = torch.outer(residual, K[t])
            W = W - eta * grad
        W_ref = W

        # --- ttt_linear_forward_pytorch with mini_batch=1 ---
        y, W_final = ttt_linear_forward_pytorch(
            x, tK, tV, tQ, torch.tensor(eta), mini_batch=1,
        )
        torch.testing.assert_close(y.squeeze(0), y_ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(W_final.squeeze(0), W_ref, atol=1e-9, rtol=1e-9)
    finally:
        torch.set_default_dtype(torch.float32)
