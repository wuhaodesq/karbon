"""Tests for :mod:`src.models.hybrid_backbone`."""

from __future__ import annotations

import pytest
import torch

from src.models import FFN, HybridBackbone, HybridBlock, SinusoidalPositionEmbedding


# =====================================================================
# FFN
# =====================================================================


def test_ffn_shape_and_gradient():
    m = FFN(d_model=16, hidden_mult=2)
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 4, 16)
    y.pow(2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None


# =====================================================================
# Position embedding
# =====================================================================


def test_sinusoidal_pe_shape():
    pe = SinusoidalPositionEmbedding(d_model=16, max_len=32)
    x = torch.zeros(1, 10, 16)
    y = pe(x)
    assert y.shape == (1, 10, 16)


def test_sinusoidal_pe_rejects_over_max_len():
    pe = SinusoidalPositionEmbedding(d_model=8, max_len=6)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 8, 8))


def test_sinusoidal_pe_requires_even_dim():
    with pytest.raises(ValueError):
        SinusoidalPositionEmbedding(d_model=7)


# =====================================================================
# HybridBlock
# =====================================================================


def test_hybrid_block_shape():
    torch.manual_seed(0)
    block = HybridBlock(
        d_model=16, n_heads=4, swa_window_size=4, ttt_mini_batch=4
    )
    x = torch.randn(2, 12, 16)
    y = block(x)
    assert y.shape == (2, 12, 16)


def test_hybrid_block_gradient_flows_to_all_subparams():
    torch.manual_seed(0)
    block = HybridBlock(
        d_model=16, n_heads=4, swa_window_size=4, ttt_mini_batch=4
    )
    x = torch.randn(1, 8, 16)
    y = block(x)
    y.pow(2).mean().backward()
    missing = []
    for name, p in block.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            missing.append(name)
    # It's OK if a normalization bias has small/zero grad in a rare init,
    # but every module class must have at least one param with grad.
    param_names = {n.split(".", 1)[0] for n in dict(block.named_parameters())}
    for mod_prefix in ("ttt", "attn", "ffn"):
        assert any(
            n.startswith(mod_prefix) and p.grad is not None
            and p.grad.abs().sum() > 0
            for n, p in block.named_parameters()
        ), f"no grad reaching sub-module {mod_prefix}; missing={missing}, params={param_names}"


def test_hybrid_block_causality():
    """Perturbing input at t=k must not change output at t<k (through any sub-layer)."""
    torch.manual_seed(0)
    block = HybridBlock(
        d_model=16, n_heads=4, swa_window_size=4, ttt_mini_batch=4
    ).eval()
    x = torch.randn(1, 10, 16)
    y1 = block(x)
    x2 = x.clone()
    x2[:, -1, :] += 3.0
    y2 = block(x2)
    torch.testing.assert_close(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5, rtol=1e-5)


# =====================================================================
# HybridBackbone
# =====================================================================


def test_backbone_with_token_embedding():
    torch.manual_seed(0)
    model = HybridBackbone(
        d_model=16,
        n_layers=2,
        vocab_size=32,
        n_heads=4,
        swa_window_size=4,
        ttt_mini_batch=4,
        max_seq_len=16,
    )
    tokens = torch.randint(0, 32, (2, 12))
    y = model(tokens)
    assert y.shape == (2, 12, 16)


def test_backbone_without_token_embedding_accepts_embeddings():
    torch.manual_seed(0)
    model = HybridBackbone(
        d_model=8,
        n_layers=1,
        vocab_size=0,
        n_heads=2,
        swa_window_size=3,
        ttt_mini_batch=3,
        max_seq_len=16,
    )
    x = torch.randn(1, 6, 8)
    y = model(x)
    assert y.shape == (1, 6, 8)


def test_backbone_rejects_wrong_input_type():
    model = HybridBackbone(
        d_model=8, n_layers=1, vocab_size=16,
        n_heads=2, swa_window_size=3, ttt_mini_batch=3, max_seq_len=16,
    )
    with pytest.raises(ValueError):
        model(torch.randn(1, 6, 8))  # float when vocab_size>0


def test_backbone_gradient_flows_end_to_end():
    torch.manual_seed(0)
    model = HybridBackbone(
        d_model=16, n_layers=2, vocab_size=32,
        n_heads=4, swa_window_size=4, ttt_mini_batch=4, max_seq_len=16,
    )
    tokens = torch.randint(0, 32, (2, 12))
    y = model(tokens)
    y.pow(2).mean().backward()

    # Verify every top-level submodule got at least one non-zero grad
    grouped: dict[str, bool] = {}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            grouped[head] = True
    for module_name in ("tok_embed", "blocks", "norm_out"):
        assert grouped.get(module_name, False), f"no grad reaching {module_name}"


def test_backbone_deterministic_with_seed():
    torch.manual_seed(0)
    m1 = HybridBackbone(
        d_model=8, n_layers=1, vocab_size=16,
        n_heads=2, swa_window_size=3, ttt_mini_batch=3, max_seq_len=16,
    )
    torch.manual_seed(0)
    m2 = HybridBackbone(
        d_model=8, n_layers=1, vocab_size=16,
        n_heads=2, swa_window_size=3, ttt_mini_batch=3, max_seq_len=16,
    )
    tokens = torch.randint(0, 16, (1, 6))
    torch.testing.assert_close(m1(tokens), m2(tokens))


def test_backbone_param_count_scales_linearly_with_layers():
    def count(n_layers):
        return HybridBackbone(
            d_model=32, n_layers=n_layers, vocab_size=64,
            n_heads=4, swa_window_size=4, ttt_mini_batch=4, max_seq_len=32,
        ).num_parameters()

    # Doubling layers should roughly double *block* params (which dominate).
    c1 = count(1)
    c2 = count(2)
    c4 = count(4)
    # Non-block params (embeddings + final norm) are constant → gap between
    # (c2-c1) and (c4-c2) should be equal.
    assert (c4 - c2) == 2 * (c2 - c1)


def test_backbone_toy_parameter_count_reasonable():
    """A minimal Hybrid backbone should be under 1M params for smoke settings."""
    m = HybridBackbone(
        d_model=64, n_layers=2, vocab_size=64,
        n_heads=4, swa_window_size=16, ttt_mini_batch=8, max_seq_len=128,
    )
    assert 10_000 < m.num_parameters() < 1_000_000
