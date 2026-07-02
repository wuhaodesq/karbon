"""Config preset system tests."""

from __future__ import annotations

import pytest
import yaml

from src.platform import configs_dir
from src.utils import load_config


PRESETS = ["local_smoke", "cloud_24g", "cloud_5090", "home_64g"]


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_file_exists(preset):
    p = configs_dir() / "_presets" / f"{preset}.yaml"
    assert p.exists(), f"preset file missing: {p}"
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for required in ("preset", "device_preferred", "model", "memory", "env", "train", "monitor"):
        assert required in data, f"preset {preset} missing key: {required}"


def test_stage0_baseline_config_exists():
    p = configs_dir() / "stage0_baseline.yaml"
    assert p.exists()


@pytest.mark.parametrize("preset", PRESETS)
def test_load_config_merges_preset_and_stage(preset):
    cfg = load_config("stage0_baseline.yaml", preset)
    assert cfg["preset"] == preset
    # Stage 0 overlays PPO hyperparams
    assert "ppo_clip" in cfg["train"]
    # Preset supplies env / model
    assert "id" in cfg["env"]
    assert "hidden_size" in cfg["model"]
    # Meta paths populated
    assert "_meta" in cfg


def test_local_smoke_has_low_budget():
    cfg = load_config("stage0_baseline.yaml", "local_smoke")
    assert cfg["memory"]["cpu_ram_budget_gb"] <= 4.0
    assert cfg["memory"]["gpu_budget_gb"] == 0.0
    assert cfg["env"]["num_envs"] <= 4


def test_home_64g_has_high_budget():
    cfg = load_config("stage0_baseline.yaml", "home_64g")
    assert cfg["memory"]["gpu_budget_gb"] >= 32.0
