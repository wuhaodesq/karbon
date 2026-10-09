"""Tests for Stage 21 M4 — DeliberationGate (budget-neutral control)."""

from __future__ import annotations

from src.models.deliberation import DeliberationGate


def test_modulate_bounds_and_type():
    g = DeliberationGate(window=8, gain=0.5, floor=0.02, cap=0.5)
    eps = g.modulate(0.1, 0.5)
    assert 0.02 <= eps <= 0.5


def test_stuck_state_gets_more_than_rolling_state():
    g = DeliberationGate(window=16, gain=0.5, floor=0.0, cap=1.0)
    # establish a balanced history first (pressure ~0.5)
    for _ in range(16):
        g.modulate(0.1, 0.5)
    eps_stuck = g.modulate(0.1, 1.0)   # failing recently
    eps_roll = g.modulate(0.1, 0.0)    # succeeding recently
    assert eps_stuck > eps_roll


def test_budget_neutral_average_factor():
    """Centering by the rolling mean keeps the long-run average ~1."""
    g = DeliberationGate(window=1000, gain=0.5, floor=0.0, cap=1.0)
    eps = 0.1
    # alternating stuck/rolling -> mean pressure ~0.5
    for i in range(600):
        g.modulate(eps, 1.0 if i % 2 == 0 else 0.0)
    sm = g.summary
    assert abs(sm["avg_factor"] - 1.0) < 0.05, sm


def test_clamps_respected_for_extreme_inputs():
    g = DeliberationGate(window=8, gain=0.5, floor=0.05, cap=0.3)
    for _ in range(4):
        g.modulate(0.25, 0.0)
    for _ in range(4):
        hi = g.modulate(0.25, 1.0)
        assert hi <= 0.3
    lo = g.modulate(0.25, 1.0)
    assert lo >= 0.05
