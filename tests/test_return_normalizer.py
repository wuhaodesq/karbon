"""Tests for ReturnNormalizer (PopArt-lite) — 2026-09-27 tuning."""

from __future__ import annotations

import torch

from src.train import ReturnNormalizer, _normalize_advantages


def test_fast_alpha_tracks_scale_faster_than_old():
    """After a 10x scale shift, alpha=0.1 re-normalizes its own data to
    ~unit scale within ~50 updates; the old alpha=0.01 is still far off."""
    torch.manual_seed(0)
    x = 1000.0 + torch.randn(64) * 50.0
    fast = ReturnNormalizer(alpha=0.1)
    slow = ReturnNormalizer(alpha=0.01)
    for _ in range(50):
        fast.update(x)
        slow.update(x)
    z_fast = float(fast.normalize(x).abs().mean())
    z_slow = float(slow.normalize(x).abs().mean())
    assert z_fast < 1.0
    assert z_slow > 3.0


def test_normalize_denormalize_roundtrip():
    n = ReturnNormalizer(alpha=0.5)
    n.update(torch.tensor([2.0, 4.0, 6.0, 8.0]))
    x = torch.tensor([1.0, 5.0, 9.0])
    assert torch.allclose(n.denormalize(n.normalize(x)), x, atol=1e-4)


def test_zero_variance_advantages_guard_stays_bounded():
    adv = torch.full((16,), 3.0)
    out = _normalize_advantages(adv)
    assert float(out.abs().max()) < 1e-3  # no noise amplification
