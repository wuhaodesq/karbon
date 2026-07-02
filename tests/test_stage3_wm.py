"""Tests for Stage 3 config + RSSM integration in the trainer."""

from __future__ import annotations

import pytest

from src.platform import configs_dir
from src.utils import load_config, validate_config


def test_stage3_config_exists():
    assert (configs_dir() / "stage3_world_model.yaml").exists()


def test_stage3_config_loads():
    cfg = load_config("stage3_world_model.yaml", "cloud_5090")
    cfg.setdefault("stage", 3)
    assert cfg["stage"] == 3
    assert "world_model" in cfg
    wm = cfg["world_model"]
    assert wm["z_dim"] > 0
    assert wm["h_dim"] > 0
    assert wm["max_rollout_steps"] > 0


def test_stage3_carries_stage_1_and_2_blocks():
    cfg = load_config("stage3_world_model.yaml", "cloud_5090")
    for key in ("intrinsic", "replay", "coverage", "world_model"):
        assert key in cfg
    assert cfg["model"]["use_hybrid_backbone"] is True


def test_stage3_config_validates():
    cfg = load_config("stage3_world_model.yaml", "cloud_5090")
    cfg.setdefault("stage", 3)
    schema = validate_config(cfg)
    assert isinstance(schema.world_model, dict)
    assert schema.world_model["z_dim"] > 0


def test_stage3_max_rollout_bounded_positive():
    """Enforce that WM rollout length is a positive bounded integer (Axiom 1)."""
    cfg = load_config("stage3_world_model.yaml", "cloud_5090")
    assert 1 <= cfg["world_model"]["max_rollout_steps"] <= 100


def test_stage3_wm_lr_reasonable():
    cfg = load_config("stage3_world_model.yaml", "cloud_5090")
    lr = cfg["world_model"]["lr"]
    assert 1e-6 < lr < 1e-1


# =====================================================================
# End-to-end sanity: trainer sees the WM config and builds RSSM
# =====================================================================


def test_stage3_train_module_imports_rssm():
    """The train module must import RSSM as a hard dependency for Stage 3+."""
    from src import train
    assert hasattr(train, "RSSM")
    assert hasattr(train, "RSSMConfig")


def test_stage3_default_config_map_wired():
    from src.train import _DEFAULT_STAGE_CONFIGS
    assert _DEFAULT_STAGE_CONFIGS[3] == "stage3_world_model.yaml"
