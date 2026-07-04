"""Tests for language encoder + multimodal fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.language_encoder import (
    FiLMLayer,
    InstructionConditionedActorCritic,
    LanguageEncoder,
    MultimodalFusion,
)


# =====================================================================
# FiLMLayer
# =====================================================================


def test_film_shape():
    film = FiLMLayer(d_vis=64, d_lang=64)
    v = torch.randn(4, 64)
    l = torch.randn(4, 64)
    out = film(v, l)
    assert out.shape == (4, 64)


def test_film_starts_as_identity():
    """At init, FiLM should be γ=1, β=0 → output == input."""
    film = FiLMLayer(d_vis=32, d_lang=32)
    v = torch.randn(4, 32)
    l = torch.randn(4, 32)
    out = film(v, l)
    torch.testing.assert_close(out, v)


def test_film_gradient():
    film = FiLMLayer(d_vis=16, d_lang=16)
    v = torch.randn(2, 16)
    l = torch.randn(2, 16)
    out = film(v, l)
    out.sum().backward()
    assert film.gamma.weight.grad is not None
    assert film.beta.weight.grad is not None


# =====================================================================
# MultimodalFusion
# =====================================================================


def test_fusion_film_only_shape():
    fusion = MultimodalFusion(d_model=64, use_cross_attention=False)
    v = torch.randn(4, 64)
    l = torch.randn(4, 64)
    out = fusion(v, l)
    assert out.shape == (4, 64)


def test_fusion_with_cross_attention_shape():
    fusion = MultimodalFusion(d_model=64, use_cross_attention=True, n_heads=4)
    v = torch.randn(4, 64)
    l = torch.randn(4, 64)
    out = fusion(v, l)
    assert out.shape == (4, 64)


def test_fusion_gradient():
    fusion = MultimodalFusion(d_model=32, use_cross_attention=True, n_heads=4)
    v = torch.randn(2, 32)
    l = torch.randn(2, 32)
    out = fusion(v, l)
    out.sum().backward()
    assert fusion.film.gamma.weight.grad is not None


# =====================================================================
# LanguageEncoder (offline fallback)
# =====================================================================


def test_language_encoder_offline_fallback(monkeypatch):
    """If CLIP can't load, LanguageEncoder should raise (no silent failure)."""
    # Mock both timm and clip to fail
    import sys
    # Remove timm if present
    monkeypatch.setitem(sys.modules, "timm", None)
    monkeypatch.setitem(sys.modules, "clip", None)

    with pytest.raises(RuntimeError, match="No CLIP backend"):
        LanguageEncoder(d_model=64)


# =====================================================================
# InstructionConditionedActorCritic (vision-only fallback)
# =====================================================================


def test_instruction_ac_vision_only(monkeypatch):
    """Without language encoder, should work as vision-only agent."""
    m = InstructionConditionedActorCritic(
        obs_shape=(7, 7, 3),
        num_actions=7,
        d_model=64,
        n_layers=1,
        n_heads=4,
        swa_window=4,
        ttt_mini_batch=2,
        use_language_encoder=False,
    )
    obs = torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (4, 7)
    assert value.shape == (4,)
    assert torch.isfinite(logits).all()


def test_instruction_ac_with_language_falls_back(monkeypatch):
    """If language encoder can't load, should fall back to vision-only."""
    import src.models.language_encoder as le_mod
    # Mock LanguageEncoder to raise
    original_init = le_mod.LanguageEncoder.__init__
    def mock_init(self, *args, **kwargs):
        raise RuntimeError("simulated offline")
    monkeypatch.setattr(le_mod.LanguageEncoder, "__init__", mock_init)

    m = InstructionConditionedActorCritic(
        obs_shape=(7, 7, 3),
        num_actions=7,
        d_model=64,
        n_layers=1,
        n_heads=4,
        swa_window=4,
        ttt_mini_batch=2,
        use_language_encoder=True,  # will fail, fall back
    )
    assert m.use_language is False
    obs = torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()


def test_instruction_ac_large_batch_no_nan():
    """Large batch should not produce NaN (regression for seq-as-batch bug)."""
    m = InstructionConditionedActorCritic(
        obs_shape=(7, 7, 3),
        num_actions=7,
        d_model=64,
        n_layers=1,
        n_heads=4,
        swa_window=4,
        ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (512, 7, 7, 3), dtype=torch.uint8)
    logits, value = m(obs)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_instruction_ac_deterministic():
    torch.manual_seed(0)
    m1 = InstructionConditionedActorCritic(
        obs_shape=(7, 7, 3), num_actions=7,
        d_model=32, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    torch.manual_seed(0)
    m2 = InstructionConditionedActorCritic(
        obs_shape=(7, 7, 3), num_actions=7,
        d_model=32, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (2, 7, 7, 3), dtype=torch.uint8)
    torch.testing.assert_close(m1(obs)[0], m2(obs)[0])
