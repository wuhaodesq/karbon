"""Tests for model growth, knowledge distillation, and curriculum gate."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.model_growth import (
    CurriculumGate,
    GrowthConfig,
    GrowthRecord,
    KnowledgeDistiller,
    ModelGrower,
)


# =====================================================================
# ModelGrower
# =====================================================================


def test_model_grower_initial_state():
    mg = ModelGrower(GrowthConfig(initial_params=7_000_000, max_params=100_000_000))
    assert mg.current_params == 7_000_000
    assert mg.can_grow is True
    assert mg.num_growths == 0


def test_model_grower_should_grow_plateau():
    """LP low + coverage high → should grow."""
    mg = ModelGrower(GrowthConfig(
        initial_params=7_000_000, max_params=100_000_000,
        grow_threshold_lp=0.1, grow_threshold_coverage=0.3,
        min_steps_between_growths=100,
    ))
    assert mg.should_grow(step=200, learning_progress=0.02, coverage_ratio=0.5) is True


def test_model_grower_should_not_grow_still_learning():
    """LP high → should not grow."""
    mg = ModelGrower(GrowthConfig(initial_params=7_000_000, max_params=100_000_000))
    assert mg.should_grow(step=200, learning_progress=0.5, coverage_ratio=0.8) is False


def test_model_grower_should_not_grow_low_coverage():
    """Coverage low → should explore, not grow."""
    mg = ModelGrower(GrowthConfig(
        initial_params=7_000_000, grow_threshold_coverage=0.3,
    ))
    assert mg.should_grow(step=200, learning_progress=0.02, coverage_ratio=0.1) is False


def test_model_grower_cooldown():
    """Should not grow within cooldown period."""
    mg = ModelGrower(GrowthConfig(
        initial_params=7_000_000, min_steps_between_growths=1000,
    ))
    assert mg.should_grow(step=500, learning_progress=0.01, coverage_ratio=0.8) is False
    assert mg.should_grow(step=1001, learning_progress=0.01, coverage_ratio=0.8) is True


def test_model_grower_grow_increases_params():
    mg = ModelGrower(GrowthConfig(
        initial_params=7_000_000, max_params=100_000_000, grow_factor=2.0,
    ))
    record = mg.grow(step=1000, trigger="lp_plateau")
    assert record is not None
    assert record.new_params == 14_000_000
    assert mg.current_params == 14_000_000
    assert mg.num_growths == 1


def test_model_grower_caps_at_max():
    mg = ModelGrower(GrowthConfig(
        initial_params=7_000_000, max_params=10_000_000, grow_factor=2.0,
    ))
    mg.grow(step=1000)
    assert mg.can_grow is False  # 14M > 10M cap


def test_model_grower_state_dict_roundtrip():
    mg = ModelGrower(GrowthConfig(initial_params=7_000_000, max_params=100_000_000))
    mg.grow(step=1000, trigger="lp_plateau")
    state = mg.state_dict()

    mg2 = ModelGrower(GrowthConfig())
    mg2.load_state_dict(state)
    assert mg2.current_params == mg.current_params
    assert mg2.num_growths == 1


def test_model_grower_summary():
    mg = ModelGrower(GrowthConfig(initial_params=7_000_000, max_params=100_000_000))
    mg.grow(step=1000)
    s = mg.summary()
    assert s["current_params"] == 10_500_000  # 7M * 1.5
    assert s["num_growths"] == 1
    assert s["can_grow"] is True


# =====================================================================
# KnowledgeDistiller
# =====================================================================


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_distiller_distill_simple():
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    distiller = KnowledgeDistiller(distill_steps=10, distill_lr=1e-3)

    data = [(torch.randn(4, 10), torch.randn(4, 5)) for _ in range(20)]
    result = distiller.distill(model, opt, data)

    assert result["steps"] == 10
    assert "final_loss" in result
    assert result["total_distillations"] == 1


def test_distiller_empty_data():
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    distiller = KnowledgeDistiller()

    result = distiller.distill(model, opt, [])
    assert result["steps"] == 0


def test_distiller_clear_ratio():
    distiller = KnowledgeDistiller(clear_ratio=0.3)
    assert distiller.clear_ratio == 0.3


def test_distiller_summary():
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    distiller = KnowledgeDistiller(distill_steps=5)
    data = [(torch.randn(2, 10), torch.randn(2, 5))]
    distiller.distill(model, opt, data)
    s = distiller.summary()
    assert s["total_distillations"] == 1
    assert s["distill_steps"] == 5


# =====================================================================
# CurriculumGate
# =====================================================================


def test_gate_learn():
    gate = CurriculumGate()
    assert gate.decide(learning_progress=0.5, coverage_ratio=0.3, task_return=0.5) == "learn"


def test_gate_explore():
    gate = CurriculumGate(lp_threshold=0.1, coverage_threshold=0.3)
    assert gate.decide(learning_progress=0.02, coverage_ratio=0.1, task_return=0.3) == "explore"


def test_gate_grow():
    gate = CurriculumGate(lp_threshold=0.1, coverage_threshold=0.3)
    assert gate.decide(learning_progress=0.02, coverage_ratio=0.5, task_return=0.3) == "grow"


def test_gate_switch():
    gate = CurriculumGate(mastery_threshold=0.8, coverage_threshold=0.3)
    assert gate.decide(learning_progress=0.5, coverage_ratio=0.8, task_return=0.9) == "switch"


def test_gate_summary():
    gate = CurriculumGate()
    s = gate.summary()
    assert "lp_threshold" in s
    assert "coverage_threshold" in s
    assert "mastery_threshold" in s
