"""Tests for the pretrained vision encoder module."""

from __future__ import annotations

import pytest
import torch

from src.models.vision_encoder import (
    CNNEncoder,
    VisionEncoder,
    build_encoder,
    list_available_vision_models,
)


OBS_SHAPE = (7, 7, 3)


# =====================================================================
# CNNEncoder (fallback)
# =====================================================================


def test_cnn_encoder_shape():
    enc = CNNEncoder(OBS_SHAPE, d_model=64)
    obs = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    feats = enc(obs)
    assert feats.shape == (4, 64)


def test_cnn_encoder_gradient():
    enc = CNNEncoder(OBS_SHAPE, d_model=32)
    obs = torch.randint(0, 255, (2, *OBS_SHAPE), dtype=torch.uint8)
    feats = enc(obs)
    feats.sum().backward()
    for p in enc.parameters():
        assert p.grad is not None


# =====================================================================
# VisionEncoder model registry
# =====================================================================


def test_list_available_models_nonempty():
    models = list_available_vision_models()
    assert "dinov2_vits14" in models
    assert len(models) >= 3


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown vision model"):
        VisionEncoder(model_name="nonesuch_model")


# =====================================================================
# build_encoder factory
# =====================================================================


def test_build_encoder_defaults_to_cnn():
    """When use_vision_encoder is False (or missing), should return CNNEncoder."""
    config = {"model": {"hidden_size": 64}}
    enc = build_encoder(config, OBS_SHAPE, torch.device("cpu"))
    assert isinstance(enc, CNNEncoder)


def test_build_encoder_vision_requested_but_falls_back(monkeypatch):
    """If vision encoder can't load (no internet), should fall back to CNN."""
    # Mock torch.hub.load to simulate offline failure
    import src.models.vision_encoder as ve_mod
    original_load = ve_mod.torch.hub.load
    def mock_load(*args, **kwargs):
        raise RuntimeError("simulated offline")
    monkeypatch.setattr(ve_mod.torch.hub, "load", mock_load)

    config = {"model": {"use_vision_encoder": True, "vision_model": "dinov2_vits14"}}
    enc = build_encoder(config, OBS_SHAPE, torch.device("cpu"))
    # Should fall back to CNN (not raise)
    assert isinstance(enc, CNNEncoder)


def test_build_encoder_cnn_config_has_d_model():
    config = {"model": {"hidden_size": 128}}
    enc = build_encoder(config, OBS_SHAPE, torch.device("cpu"))
    assert enc.d_model == 128


# =====================================================================
# Config schema for vision encoder
# =====================================================================


def test_vision_encoder_config_validates():
    from src.utils import validate_config

    cfg = {
        "preset": "cloud_5090",
        "device_preferred": "cuda",
        "stage": 3,
        "model": {
            "hidden_size": 384,
            "use_hybrid_backbone": True,
            "use_vision_encoder": True,
            "vision_model": "dinov2_vits14",
            "vision_freeze": True,
            "vision_target_size": 224,
        },
        "memory": {"gpu_budget_gb": 22, "cpu_ram_budget_gb": 48,
                     "replay_gpu_capacity": 1, "replay_cpu_capacity": 1,
                     "skill_gpu_capacity": 1, "wm_rollout_max_steps": 1},
        "env": {"id": "MiniGrid-Empty-5x5-v0", "num_envs": 1},
        "train": {"batch_size": 4, "seq_len": 32, "learning_rate": 1e-4,
                  "total_steps": 100, "log_every_steps": 50, "ckpt_every_steps": 100},
        "monitor": {"sample_interval_s": 5, "slope_alarm_gb_per_hour": 0.2,
                    "empty_cache_every_steps": 100},
    }
    schema = validate_config(cfg)
    assert schema.model.use_vision_encoder is True
    assert schema.model.vision_model == "dinov2_vits14"


def test_vision_encoder_bad_target_size_rejected():
    from src.utils import validate_config, ConfigValidationError

    cfg = {
        "preset": "cloud_5090",
        "device_preferred": "cuda",
        "stage": 3,
        "model": {
            "hidden_size": 384,
            "use_hybrid_backbone": True,
            "use_vision_encoder": True,
            "vision_target_size": 7,  # too small
        },
        "memory": {"gpu_budget_gb": 22, "cpu_ram_budget_gb": 48,
                     "replay_gpu_capacity": 1, "replay_cpu_capacity": 1,
                     "skill_gpu_capacity": 1, "wm_rollout_max_steps": 1},
        "env": {"id": "MiniGrid-Empty-5x5-v0", "num_envs": 1},
        "train": {"batch_size": 4, "seq_len": 32, "learning_rate": 1e-4,
                  "total_steps": 100, "log_every_steps": 50, "ckpt_every_steps": 100},
        "monitor": {"sample_interval_s": 5, "slope_alarm_gb_per_hour": 0.2,
                    "empty_cache_every_steps": 100},
    }
    with pytest.raises(ConfigValidationError, match="vision_target_size"):
        validate_config(cfg)


# =====================================================================
# HybridActorCritic with vision encoder fallback
# =====================================================================


def test_hybrid_actor_critic_with_vision_falls_back_to_cnn(monkeypatch):
    """HybridActorCritic with use_vision_encoder=True should gracefully
    fall back to inline CNN when the pretrained model can't be loaded."""
    import src.models.vision_encoder as ve_mod
    def mock_load(*args, **kwargs):
        raise RuntimeError("simulated offline")
    monkeypatch.setattr(ve_mod.torch.hub, "load", mock_load)

    from src.train import HybridActorCritic

    torch.manual_seed(0)
    m = HybridActorCritic(
        obs_shape=OBS_SHAPE,
        num_actions=7,
        d_model=64,
        n_layers=1,
        n_heads=4,
        swa_window=4,
        ttt_mini_batch=2,
        use_vision_encoder=True,  # will try DINOv2, fail, fall back to inline CNN
        vision_model_name="dinov2_vits14",
    )
    # Should have fallen back to inline CNN encoder
    assert m.use_vision is False
    # Should still produce valid output
    obs = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()
