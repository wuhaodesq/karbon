"""Tests for Stage 1 config + BoundedCoverage."""

from __future__ import annotations

import numpy as np
import pytest

from src.platform import configs_dir
from src.utils import load_config, validate_config


def test_stage1_config_exists():
    assert (configs_dir() / "stage1_curiosity.yaml").exists()


def test_stage1_config_loads_with_cloud_5090():
    cfg = load_config("stage1_curiosity.yaml", "cloud_5090")
    assert cfg["stage"] == 1
    assert "intrinsic" in cfg
    assert "replay" in cfg
    assert "coverage" in cfg
    # Preset overrides still present
    assert cfg["model"]["hidden_size"] > 0
    assert cfg["env"]["id"].startswith("MiniGrid")


def test_stage1_config_validates_against_schema():
    """Stage 1 sub-blocks (intrinsic/replay/coverage) must not break schema."""
    cfg = load_config("stage1_curiosity.yaml", "cloud_5090")
    cfg.setdefault("stage", 1)
    schema = validate_config(cfg)
    assert schema.stage == 1
    # Optional sub-blocks are dicts (permissively validated)
    assert isinstance(schema.intrinsic, dict)
    assert isinstance(schema.replay, dict)
    assert isinstance(schema.coverage, dict)
    assert schema.intrinsic["embed_dim"] > 0


def test_stage1_intrinsic_hyperparams_sane():
    cfg = load_config("stage1_curiosity.yaml", "cloud_5090")
    intr = cfg["intrinsic"]
    assert 0 < intr["reward_coef"] <= 1.0
    assert intr["embed_dim"] >= 16
    assert intr["lr"] > 0


def test_stage1_replay_hyperparams_sane():
    cfg = load_config("stage1_curiosity.yaml", "cloud_5090")
    rp = cfg["replay"]
    assert rp["hot_capacity"] > 0
    assert rp["warm_capacity"] >= rp["hot_capacity"]  # warm should be at least as big as hot
    assert rp["cold_max_shards"] > 0
    assert rp["min_size_to_sample"] <= rp["hot_capacity"] + rp["warm_capacity"]


def test_stage1_coverage_hyperparams_sane():
    cfg = load_config("stage1_curiosity.yaml", "cloud_5090")
    cov = cfg["coverage"]
    assert cov["num_buckets"] > 0
    # Power-of-two for cheap masking
    n = cov["num_buckets"]
    assert (n & (n - 1)) == 0, f"num_buckets={n} not a power of 2"


# =====================================================================
# BoundedCoverage
# =====================================================================


from src.train import BoundedCoverage  # noqa: E402


def test_bounded_coverage_capacity_enforced():
    c = BoundedCoverage(num_buckets=64)
    assert c.capacity == 64
    for _ in range(1000):
        arr = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
        c.touch(arr)
    assert len(c) <= c.capacity


def test_bounded_coverage_stable_on_repeated_state():
    c = BoundedCoverage(num_buckets=64)
    arr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    for _ in range(50):
        c.touch(arr)
    # Only one unique bucket touched
    assert len(c) == 1
    assert c.summary()["visits"] == 50


def test_bounded_coverage_rejects_non_positive():
    with pytest.raises(ValueError):
        BoundedCoverage(num_buckets=0)


def test_bounded_coverage_state_dict_roundtrip():
    c1 = BoundedCoverage(num_buckets=64)
    for i in range(20):
        arr = np.array([[i, i + 1, i + 2]], dtype=np.uint8)
        c1.touch(arr)
    state = c1.state_dict()

    c2 = BoundedCoverage(num_buckets=64)
    c2.load_state_dict(state)
    assert len(c2) == len(c1)
    assert c2.summary()["visits"] == c1.summary()["visits"]


def test_bounded_coverage_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    c = BoundedCoverage(num_buckets=16)
    assert isinstance(c, BoundedComponent)
