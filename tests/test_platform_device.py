"""Platform abstraction tests: device / paths / memory probe."""

from __future__ import annotations

import os

import pytest
import torch

from src.platform import (
    ckpt_dir,
    data_dir,
    get_device,
    get_device_info,
    logs_dir,
    project_root,
    reset_cache,
    snapshot,
    stage_ckpt_path,
    stage_log_dir,
)


def test_project_root_is_repo_root():
    root = project_root()
    # PLAN.md must exist at project root
    assert (root / "PLAN.md").exists(), f"project_root() = {root}"


def test_get_device_returns_torch_device():
    reset_cache()
    dev = get_device()
    assert isinstance(dev, torch.device)
    info = get_device_info()
    assert info.kind in {"cuda", "xpu", "mps", "cpu"}


def test_forced_cpu_via_env(monkeypatch):
    reset_cache()
    monkeypatch.setenv("DEVAGI_DEVICE", "cpu")
    reset_cache()
    dev = get_device()
    assert dev.type == "cpu"
    reset_cache()


def test_paths_default_to_project_root():
    root = project_root()
    assert data_dir().is_relative_to(root) or data_dir() == root / "data"
    assert logs_dir().is_relative_to(root) or logs_dir() == root / "logs"
    assert ckpt_dir().is_relative_to(root) or ckpt_dir() == root / "checkpoints"


def test_paths_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DEVAGI_DATA_DIR", str(tmp_path / "custom_data"))
    # cache-less function; call again
    assert data_dir() == (tmp_path / "custom_data").resolve()


def test_stage_paths_compose_correctly():
    p = stage_ckpt_path(stage=0, step=42)
    assert p.name == "ckpt_stage0_000000042.pt"

    d = stage_log_dir(stage=1, run_id="20260101_000000_deadbee")
    assert d.exists()
    assert d.name == "20260101_000000_deadbee"
    assert d.parent.name == "stage1"


def test_memory_snapshot_returns_sane_values():
    reset_cache()
    snap = snapshot()
    assert snap.kind in {"cpu", "cuda"}
    assert snap.used_bytes >= 0
    assert snap.total_bytes > 0 or snap.kind == "cuda"  # CUDA total may be missing under some drivers
    assert snap.process_rss_bytes > 0
