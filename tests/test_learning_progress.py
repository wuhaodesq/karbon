"""Tests for :mod:`src.intrinsic.learning_progress`."""

from __future__ import annotations

import pytest

from src.intrinsic import LearningProgressTracker, LPConfig


def _mk(window=8, minsamp=4, smoothing=0.0):
    return LearningProgressTracker(LPConfig(
        window_size=window,
        min_samples_for_signal=minsamp,
        smoothing=smoothing,
    ))


def test_config_validation():
    with pytest.raises(ValueError):
        LearningProgressTracker(LPConfig(window_size=3))    # too small
    with pytest.raises(ValueError):
        LearningProgressTracker(LPConfig(window_size=7))    # odd


def test_lp_positive_when_error_decreases():
    lp = _mk(window=8, minsamp=4)
    # First half high error, second half low
    for e in [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]:
        lp.push(task_id=0, error=e)
    val = lp.learning_progress(0)
    assert val > 0.5, f"expected positive LP, got {val}"


def test_lp_negative_when_error_increases():
    lp = _mk()
    for e in [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]:
        lp.push(task_id=0, error=e)
    val = lp.learning_progress(0)
    assert val < -0.5, f"expected negative LP, got {val}"


def test_lp_zero_when_stationary():
    lp = _mk()
    for _ in range(8):
        lp.push(0, 0.5)
    assert abs(lp.learning_progress(0)) < 1e-6


def test_lp_returns_zero_with_insufficient_data():
    lp = _mk(minsamp=4)
    for e in [1.0, 0.5]:
        lp.push(0, e)
    assert lp.learning_progress(0) == 0.0


def test_ring_buffer_capacity_bounded():
    lp = _mk(window=4)
    for e in range(100):
        lp.push(0, float(e))
    assert lp.sample_count(0) == 4    # capped at window_size
    # Only the last 4 values retained → learning_progress reflects those
    val = lp.learning_progress(0)
    # errors ended increasing → LP should be negative
    assert val < 0


def test_priorities_span_multiple_tasks():
    lp = _mk()
    # Task A: strongly learning
    for e in [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
        lp.push(0, e)
    # Task B: no progress
    for _ in range(8):
        lp.push(1, 0.5)
    prios = lp.priorities()
    # A should get higher priority than B
    assert prios[0] > prios[1]


def test_normalized_priorities_sum_to_one():
    lp = _mk()
    for _ in range(8):
        lp.push(0, 0.5)
        lp.push(1, 0.5)
        lp.push(2, 0.5)
    norm = lp.normalize_priorities()
    total = sum(norm.values())
    assert abs(total - 1.0) < 1e-6


def test_normalized_priorities_uniform_when_empty():
    lp = _mk()
    for t in (0, 1, 2):
        lp.push(t, 0.0)
    # All zero LP → epsilon-based equal, sums to 1
    norm = lp.normalize_priorities()
    vals = list(norm.values())
    for v in vals:
        assert abs(v - vals[0]) < 1e-3


def test_smoothing_dampens_swings():
    lp1 = _mk(window=8, minsamp=4, smoothing=0.0)
    lp2 = _mk(window=8, minsamp=4, smoothing=0.9)
    for e in [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
        lp1.push(0, e)
        lp2.push(0, e)
    raw = lp1.learning_progress(0)
    smoothed = lp2.learning_progress(0)
    # With smoothing=0.9, smoothed magnitude should be strictly less than raw
    assert abs(smoothed) < abs(raw)


def test_state_dict_roundtrip():
    lp = _mk()
    for e in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        lp.push(0, e)
    state = lp.state_dict()
    lp2 = _mk()
    lp2.load_state_dict(state)
    assert lp.learning_progress(0) == lp2.learning_progress(0)


def test_reset_specific_task():
    lp = _mk()
    for _ in range(8):
        lp.push(0, 0.5)
        lp.push(1, 0.5)
    lp.reset(task_id=0)
    assert 0 not in lp.known_tasks()
    assert 1 in lp.known_tasks()


def test_snapshot_shape():
    lp = _mk()
    for _ in range(5):
        lp.push(0, 0.5)
    snap = lp.snapshot()
    assert 0 in snap["tasks"]
    assert "capacity" in snap
    assert "total_samples" in snap
