"""Tests for :mod:`src.utils.config_schema`."""

from __future__ import annotations

import copy

import pytest

from src.utils import (
    ConfigValidationError,
    load_config,
    validate_and_dump,
    validate_config,
)


PRESETS = ["local_smoke", "cloud_24g", "cloud_5090", "home_64g"]


@pytest.fixture()
def base_cfg() -> dict:
    cfg = load_config("stage0_baseline.yaml", "cloud_24g")
    cfg.setdefault("stage", 0)
    return cfg


# =====================================================================
# Happy paths
# =====================================================================


@pytest.mark.parametrize("preset", PRESETS)
def test_all_presets_validate(preset):
    cfg = load_config("stage0_baseline.yaml", preset)
    cfg.setdefault("stage", 0)
    schema = validate_config(cfg)
    assert schema.preset == preset
    assert schema.model.hidden_size > 0


def test_dump_roundtrip_stable(base_cfg):
    dumped = validate_and_dump(base_cfg)
    # Re-validate the dumped version
    schema2 = validate_config(dumped)
    assert schema2.preset == base_cfg["preset"]


# =====================================================================
# Rejects unknown keys
# =====================================================================


def test_unknown_top_level_key_rejected(base_cfg):
    base_cfg["mystery_field"] = 42
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        validate_config(base_cfg)


def test_unknown_subsection_key_rejected(base_cfg):
    base_cfg["train"]["typo_field"] = 1
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        validate_config(base_cfg)


# =====================================================================
# Rejects bad values
# =====================================================================


def test_negative_hidden_size_rejected(base_cfg):
    base_cfg["model"]["hidden_size"] = -1
    with pytest.raises(ConfigValidationError, match="hidden_size"):
        validate_config(base_cfg)


def test_bad_backend_rejected(base_cfg):
    base_cfg["model"]["ttt_backend"] = "nonesuch"
    with pytest.raises(ConfigValidationError, match="ttt_backend"):
        validate_config(base_cfg)


def test_out_of_range_learning_rate_rejected(base_cfg):
    base_cfg["train"]["learning_rate"] = 5.0
    with pytest.raises(ConfigValidationError, match="learning_rate"):
        validate_config(base_cfg)


def test_negative_num_envs_rejected(base_cfg):
    base_cfg["env"]["num_envs"] = 0
    with pytest.raises(ConfigValidationError, match="num_envs"):
        validate_config(base_cfg)


def test_bad_device_rejected(base_cfg):
    base_cfg["device_preferred"] = "gpu"
    with pytest.raises(ConfigValidationError, match="device_preferred"):
        validate_config(base_cfg)


def test_missing_section_rejected(base_cfg):
    del base_cfg["monitor"]
    with pytest.raises(ConfigValidationError, match="missing section"):
        validate_config(base_cfg)


def test_wrong_type_rejected(base_cfg):
    base_cfg["model"] = "not-a-dict"
    with pytest.raises(ConfigValidationError, match="must be a mapping"):
        validate_config(base_cfg)


def test_bad_gamma_rejected(base_cfg):
    base_cfg["train"]["gamma"] = 1.5
    with pytest.raises(ConfigValidationError, match="gamma"):
        validate_config(base_cfg)


def test_zero_slope_alarm_rejected(base_cfg):
    base_cfg["monitor"]["slope_alarm_gb_per_hour"] = 0.0
    with pytest.raises(ConfigValidationError, match="slope_alarm"):
        validate_config(base_cfg)


# =====================================================================
# Non-dict input
# =====================================================================


def test_non_dict_input_rejected():
    with pytest.raises(ConfigValidationError):
        validate_config("not-a-dict")  # type: ignore[arg-type]
