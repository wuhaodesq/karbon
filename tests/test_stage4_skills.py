"""Tests for Stage 4 config + Skill Library integration in the trainer."""

from __future__ import annotations

import pytest

from src.platform import configs_dir
from src.utils import load_config, validate_config


def test_stage4_config_exists():
    assert (configs_dir() / "stage4_skills.yaml").exists()


def test_stage4_config_loads():
    cfg = load_config("stage4_skills.yaml", "cloud_5090")
    cfg.setdefault("stage", 4)
    assert cfg["stage"] == 4
    assert "skills" in cfg
    sk = cfg["skills"]
    assert sk["gpu_capacity"] > 0
    assert sk["cpu_capacity"] >= sk["gpu_capacity"]
    assert sk["skill_rank"] > 0


def test_stage4_carries_all_prior_blocks():
    cfg = load_config("stage4_skills.yaml", "cloud_5090")
    for key in ("intrinsic", "replay", "coverage", "world_model", "skills"):
        assert key in cfg
    assert cfg["model"]["use_hybrid_backbone"] is True


def test_stage4_config_validates():
    cfg = load_config("stage4_skills.yaml", "cloud_5090")
    cfg.setdefault("stage", 4)
    schema = validate_config(cfg)
    assert isinstance(schema.skills, dict)


def test_stage4_similarity_threshold_in_range():
    cfg = load_config("stage4_skills.yaml", "cloud_5090")
    thr = cfg["skills"]["merge_similarity_threshold"]
    assert 0.0 < thr <= 1.0


def test_stage4_score_weights_present():
    cfg = load_config("stage4_skills.yaml", "cloud_5090")
    for key in ("score_alpha", "score_beta", "score_gamma"):
        assert key in cfg["skills"]


def test_stage4_train_module_imports_skills():
    from src import train
    assert hasattr(train, "BoundedSkillLibrary")
    assert hasattr(train, "SkillLibraryBudget")


def test_stage4_default_config_map_wired():
    from src.train import _DEFAULT_STAGE_CONFIGS
    assert _DEFAULT_STAGE_CONFIGS[4] == "stage4_skills.yaml"
