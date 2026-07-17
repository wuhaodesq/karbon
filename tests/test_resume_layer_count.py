"""Tests for resume layer-count inference.

The resume path must build the model with the SAME number of backbone blocks
as the checkpoint. Otherwise a 3-layer checkpoint loaded into a 2-layer model
raises a state_dict size mismatch and the model is reinitialized RANDOMLY
(the "spurious growth + crashed mean_return" symptom). `_ckpt_layer_count`
infers the layer count from the checkpoint so train.py can build to match.

测试 resume 层数与检查点对齐：防止 3 层权重误载入 2 层模型导致随机初始化。"""

from __future__ import annotations

from pathlib import Path

import torch

from src.train import HybridActorCritic, _ckpt_layer_count


def _save_ckpt(path: Path, model: torch.nn.Module) -> None:
    torch.save({"model_state": model.state_dict(), "step": 0, "stage": 2}, path)


def test_ckpt_layer_count_3layer():
    model = HybridActorCritic(
        obs_shape=(8, 8, 3), num_actions=4, d_model=16, n_layers=3, n_heads=4,
    )
    p = Path("test_ckpt_3l.pt")
    try:
        _save_ckpt(p, model)
        assert _ckpt_layer_count(p) == 3
    finally:
        p.unlink(missing_ok=True)


def test_ckpt_layer_count_2layer():
    model = HybridActorCritic(
        obs_shape=(8, 8, 3), num_actions=4, d_model=16, n_layers=2, n_heads=4,
    )
    p = Path("test_ckpt_2l.pt")
    try:
        _save_ckpt(p, model)
        assert _ckpt_layer_count(p) == 2
    finally:
        p.unlink(missing_ok=True)


def test_ckpt_layer_count_missing_file_returns_0():
    assert _ckpt_layer_count(Path("does_not_exist_xyz.pt")) == 0


def test_ckpt_layer_count_no_model_state_returns_0():
    p = Path("test_ckpt_noms.pt")
    try:
        torch.save({"step": 0}, p)
        assert _ckpt_layer_count(p) == 0
    finally:
        p.unlink(missing_ok=True)
