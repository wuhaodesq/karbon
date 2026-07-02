"""Tests for :mod:`src.models.sliding_attn`."""

from __future__ import annotations

import pytest
import torch

from src.models.sliding_attn import SlidingWindowAttention, build_sliding_causal_mask


# =====================================================================
# Mask
# =====================================================================


def test_sliding_mask_shape():
    m = build_sliding_causal_mask(seq_len=8, window_size=3, device=torch.device("cpu"))
    assert m.shape == (8, 8)


def test_sliding_mask_causality():
    """Upper triangle (above diagonal) must be -inf; queries never see the future."""
    T, W = 6, 3
    m = build_sliding_causal_mask(T, W, torch.device("cpu"))
    for i in range(T):
        for j in range(i + 1, T):
            assert m[i, j].item() == float("-inf"), f"causality broken at ({i},{j})"


def test_sliding_mask_window_lower_bound():
    """Query at t should only see keys in [t-W+1, t]. Below the window → -inf."""
    T, W = 8, 3
    m = build_sliding_causal_mask(T, W, torch.device("cpu"))
    for i in range(T):
        for j in range(0, T):
            in_window = (i - W + 1) <= j <= i
            if in_window:
                assert m[i, j].item() == 0.0, f"({i},{j}) should be visible"
            else:
                assert m[i, j].item() == float("-inf"), f"({i},{j}) should be masked"


# =====================================================================
# Module
# =====================================================================


def test_swa_output_shape():
    torch.manual_seed(0)
    m = SlidingWindowAttention(d_model=16, n_heads=4, window_size=4)
    x = torch.randn(2, 10, 16)
    y = m(x)
    assert y.shape == (2, 10, 16)


def test_swa_d_model_divisibility_check():
    with pytest.raises(ValueError):
        SlidingWindowAttention(d_model=17, n_heads=4, window_size=4)


def test_swa_window_size_positive_check():
    with pytest.raises(ValueError):
        SlidingWindowAttention(d_model=16, n_heads=4, window_size=0)


def test_swa_deterministic_with_seed():
    torch.manual_seed(0)
    m1 = SlidingWindowAttention(d_model=8, n_heads=2, window_size=3)
    torch.manual_seed(0)
    m2 = SlidingWindowAttention(d_model=8, n_heads=2, window_size=3)
    x = torch.randn(1, 6, 8)
    torch.testing.assert_close(m1(x), m2(x))


def test_swa_gradient_flows():
    torch.manual_seed(0)
    m = SlidingWindowAttention(d_model=8, n_heads=2, window_size=3)
    x = torch.randn(1, 6, 8, requires_grad=True)
    y = m(x)
    y.pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_swa_future_masking_by_perturbation():
    """Changing input at position ``i`` must not affect output at position ``j < i``.

    因果 mask 验证：扰动 t=k 处不能影响 t<k 处的输出。
    """
    torch.manual_seed(0)
    m = SlidingWindowAttention(d_model=8, n_heads=2, window_size=4)
    m.eval()
    x = torch.randn(1, 8, 8)
    y1 = m(x)
    # Perturb only the last token
    x2 = x.clone()
    x2[:, -1, :] += 5.0
    y2 = m(x2)
    # Outputs at all earlier positions must be identical
    torch.testing.assert_close(y1[:, :-1, :], y2[:, :-1, :])


def test_swa_window_bound_by_perturbation():
    """Perturbing input at t=0 must not affect output at t=W (outside window).

    窗口有效性：扰动 t=0 处不影响 t>=W 的输出（超窗）。
    """
    torch.manual_seed(0)
    W = 3
    m = SlidingWindowAttention(d_model=8, n_heads=2, window_size=W)
    m.eval()
    x = torch.randn(1, 8, 8)
    y1 = m(x)
    x2 = x.clone()
    x2[:, 0, :] += 10.0
    y2 = m(x2)
    # For t >= W, output should be unaffected by change at t=0
    torch.testing.assert_close(y1[:, W:, :], y2[:, W:, :], atol=1e-5, rtol=1e-5)


def test_swa_mask_cache_rebuilds_on_length_change():
    torch.manual_seed(0)
    m = SlidingWindowAttention(d_model=8, n_heads=2, window_size=3)
    # Two different lengths — must not crash and must produce correct shapes
    y1 = m(torch.randn(1, 6, 8))
    y2 = m(torch.randn(1, 12, 8))
    assert y1.shape == (1, 6, 8)
    assert y2.shape == (1, 12, 8)
