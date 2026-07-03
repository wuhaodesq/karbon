"""Tests for Stage 2 config and HybridActorCritic wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.platform import configs_dir
from src.train import ActorCritic, HybridActorCritic
from src.utils import load_config, validate_config


OBS_SHAPE = (7, 7, 3)


# =====================================================================
# Config
# =====================================================================


def test_stage2_config_exists():
    assert (configs_dir() / "stage2_hybrid.yaml").exists()


def test_stage2_config_loads_with_cloud_5090():
    cfg = load_config("stage2_hybrid.yaml", "cloud_5090")
    cfg.setdefault("stage", 2)
    assert cfg["stage"] == 2
    assert cfg["model"]["use_hybrid_backbone"] is True
    assert cfg["model"]["hybrid_n_layers"] > 0


def test_stage2_config_validates():
    cfg = load_config("stage2_hybrid.yaml", "cloud_5090")
    cfg.setdefault("stage", 2)
    schema = validate_config(cfg)
    assert schema.model.use_hybrid_backbone is True
    assert schema.model.hybrid_n_layers > 0


def test_stage2_carries_forward_stage1_blocks():
    """Stage 2 must inherit RND + Replay + Coverage config."""
    cfg = load_config("stage2_hybrid.yaml", "cloud_5090")
    cfg.setdefault("stage", 2)
    for key in ("intrinsic", "replay", "coverage"):
        assert key in cfg, f"stage 2 missing {key}"


def test_stage2_bad_hybrid_dropout_rejected():
    cfg = load_config("stage2_hybrid.yaml", "cloud_5090")
    cfg.setdefault("stage", 2)
    cfg["model"]["hybrid_dropout"] = 1.5
    with pytest.raises(Exception) as ei:
        validate_config(cfg)
    assert "dropout" in str(ei.value)


# =====================================================================
# HybridActorCritic
# =====================================================================


def test_hybrid_actor_critic_shape():
    """HybridActorCritic should produce (B, num_actions) logits and (B,) values
    regardless of batch size, treating each obs as an independent length-1 sequence."""
    torch.manual_seed(0)
    m = HybridActorCritic(
        obs_shape=OBS_SHAPE,
        num_actions=7,
        d_model=64,
        n_layers=2,
        n_heads=4,
        swa_window=8,
        ttt_mini_batch=4,
    )
    # Small batch
    obs = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (4, 7)
    assert value.shape == (4,)
    # Large batch — must not produce NaN (regression for the seq-as-batch bug)
    obs_large = torch.randint(0, 255, (512, *OBS_SHAPE), dtype=torch.uint8)
    logits_large, value_large = m(obs_large)
    assert logits_large.shape == (512, 7)
    assert value_large.shape == (512,)
    assert torch.isfinite(logits_large).all(), "NaN in large-batch logits"
    assert torch.isfinite(value_large).all(), "NaN in large-batch values"


def test_hybrid_actor_critic_gradient_flows():
    torch.manual_seed(0)
    m = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=32, n_layers=2, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (3, *OBS_SHAPE), dtype=torch.uint8)
    logits, value = m(obs)
    loss = logits.pow(2).mean() + value.pow(2).mean()
    loss.backward()

    grouped: dict[str, bool] = {}
    for name, p in m.named_parameters():
        top = name.split(".", 1)[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            grouped[top] = True
    for expected in ("encoder", "backbone", "policy_head", "value_head"):
        assert grouped.get(expected, False), f"no grad reaching {expected}"


def test_hybrid_actor_critic_d_model_snapped_up_to_head_multiple():
    """If requested d_model isn't divisible by n_heads, snap up."""
    m = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=63, n_heads=4, n_layers=1, swa_window=4, ttt_mini_batch=2,
    )
    assert m.d_model % 4 == 0
    assert m.d_model >= 63


def test_hybrid_actor_critic_deterministic_with_seed():
    torch.manual_seed(0)
    m1 = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=32, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    torch.manual_seed(0)
    m2 = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=32, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (2, *OBS_SHAPE), dtype=torch.uint8)
    torch.manual_seed(1)
    l1, v1 = m1(obs)
    torch.manual_seed(1)
    l2, v2 = m2(obs)
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(v1, v2)


def test_hybrid_param_count_reasonable():
    """d_model=128, 3 layers should be under ~1M params."""
    m = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=128, n_layers=3, n_heads=4, swa_window=16, ttt_mini_batch=8,
    )
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 5_000_000, f"unexpected param count: {n}"


def test_hybrid_vs_baseline_output_shape_parity():
    """Both models produce (B, num_actions) logits and (B,) values."""
    torch.manual_seed(0)
    baseline = ActorCritic(OBS_SHAPE, num_actions=7, hidden=64)
    hybrid = HybridActorCritic(
        obs_shape=OBS_SHAPE, num_actions=7,
        d_model=64, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (5, *OBS_SHAPE), dtype=torch.uint8)
    lb, vb = baseline(obs)
    lh, vh = hybrid(obs)
    assert lb.shape == lh.shape
    assert vb.shape == vh.shape
