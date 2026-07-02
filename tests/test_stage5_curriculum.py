"""Tests for Stage 5 config + Auto Curriculum integration in the trainer."""

from __future__ import annotations

import pytest

from src.platform import configs_dir
from src.utils import load_config, validate_config


def test_stage5_config_exists():
    assert (configs_dir() / "stage5_curriculum.yaml").exists()


def test_stage5_config_loads():
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    cfg.setdefault("stage", 5)
    assert cfg["stage"] == 5
    assert "curriculum" in cfg
    cu = cfg["curriculum"]
    assert cu["max_tasks"] > 0
    assert cu["lp_window_size"] > 0
    assert 0.0 <= cu["exploration_epsilon"] <= 1.0


def test_stage5_tasks_have_env_ids():
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    tasks = cfg["curriculum"]["tasks"]
    assert len(tasks) > 0
    for t in tasks:
        assert "id" in t
        assert "env_id" in t
        assert t["env_id"].startswith("MiniGrid")


def test_stage5_tasks_within_max():
    """All declared tasks must fit within max_tasks capacity (Axiom 1)."""
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    tasks = cfg["curriculum"]["tasks"]
    assert len(tasks) <= cfg["curriculum"]["max_tasks"]


def test_stage5_carries_all_prior_blocks():
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    for key in ("intrinsic", "replay", "coverage", "world_model", "skills", "curriculum"):
        assert key in cfg


def test_stage5_config_validates():
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    cfg.setdefault("stage", 5)
    schema = validate_config(cfg)
    assert isinstance(schema.curriculum, dict)


def test_stage5_train_module_imports_curriculum():
    from src import train
    assert hasattr(train, "AutoCurriculum")
    assert hasattr(train, "TaskTemplate")


def test_stage5_default_config_map_wired():
    from src.train import _DEFAULT_STAGE_CONFIGS
    assert _DEFAULT_STAGE_CONFIGS[5] == "stage5_curriculum.yaml"


def test_stage5_switch_interval_reasonable():
    cfg = load_config("stage5_curriculum.yaml", "cloud_5090")
    assert cfg["curriculum"]["switch_every_steps"] > 100
    assert cfg["curriculum"]["report_every_steps"] > 0
