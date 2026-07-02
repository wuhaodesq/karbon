"""Tests for Stage 6 config + Online EWC + Generative Replay + Sleep integration."""

from __future__ import annotations

import pytest

from src.platform import configs_dir
from src.utils import load_config, validate_config


def test_stage6_config_exists():
    assert (configs_dir() / "stage6_consolidation.yaml").exists()


def test_stage6_config_loads():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    cfg.setdefault("stage", 6)
    assert cfg["stage"] == 6
    assert "continual" in cfg
    ct = cfg["continual"]
    assert ct["ewc_lambda"] > 0
    assert 0 < ct["ewc_gamma"] <= 1.0


def test_stage6_ewc_hyperparams_sane():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    ct = cfg["continual"]
    assert ct["ewc_anchor_mode"] in ("replace", "ema")
    assert 0.0 < ct["ewc_anchor_ema_alpha"] <= 1.0
    assert ct["ewc_consolidate_every_steps"] > 0
    assert ct["ewc_consolidate_num_batches"] > 0


def test_stage6_generative_replay_hyperparams():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    ct = cfg["continual"]
    assert ct["gr_latent_dim"] > 0
    assert ct["gr_batch_size"] > 0
    assert ct["gr_update_every_steps"] > 0
    assert ct["gr_lr"] > 0


def test_stage6_sleep_periods_sane():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    ct = cfg["continual"]
    assert ct["sleep_warmup_steps"] > 0
    assert ct["sleep_replay_trim_every"] > 0


def test_stage6_carries_all_prior_blocks():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    for key in (
        "intrinsic", "replay", "coverage", "world_model",
        "skills", "curriculum", "continual",
    ):
        assert key in cfg


def test_stage6_config_validates():
    cfg = load_config("stage6_consolidation.yaml", "cloud_5090")
    cfg.setdefault("stage", 6)
    schema = validate_config(cfg)
    assert isinstance(schema.continual, dict)
    assert schema.continual["ewc_lambda"] > 0


def test_stage6_train_module_imports_all_stage6_modules():
    from src import train
    for name in (
        "OnlineEWC", "OnlineEWCConfig",
        "GenerativeReplayVAE", "GenerativeReplayConfig",
        "SleepConsolidationLoop", "ConsolidationConfig",
    ):
        assert hasattr(train, name), f"train.py missing {name}"


def test_stage6_default_config_map_wired():
    from src.train import _DEFAULT_STAGE_CONFIGS
    assert _DEFAULT_STAGE_CONFIGS[6] == "stage6_consolidation.yaml"
