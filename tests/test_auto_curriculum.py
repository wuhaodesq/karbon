"""Tests for :mod:`src.curriculum.auto_curriculum`."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from src.curriculum import AutoCurriculum, AutoCurriculumConfig, TaskTemplate


def _mk(max_tasks=5, eps=0.0, window=8, minsamp=4):
    return AutoCurriculum(
        AutoCurriculumConfig(
            max_tasks=max_tasks,
            lp_window_size=window,
            lp_min_samples=minsamp,
            exploration_epsilon=eps,
            smoothing=0.0,
        ),
        rng=random.Random(0),
    )


def test_add_task_bounded_by_max():
    curr = _mk(max_tasks=3)
    for i in range(10):
        curr.add_task(TaskTemplate(id=i, spec={"i": i}))
    assert len(curr) == 3
    assert len(curr) <= curr.capacity


def test_evicts_oldest():
    curr = _mk(max_tasks=2)
    curr.add_task(TaskTemplate(id=0, spec={}))
    curr.add_task(TaskTemplate(id=1, spec={}))
    evicted = curr.add_task(TaskTemplate(id=2, spec={}))
    assert evicted is not None and evicted.id == 0
    assert set(curr.known_tasks()) == {1, 2}


def test_sample_requires_at_least_one_task():
    curr = _mk()
    with pytest.raises(RuntimeError):
        curr.sample_task()


def test_report_error_unknown_task_raises():
    curr = _mk()
    curr.add_task(TaskTemplate(id=0, spec={}))
    with pytest.raises(KeyError):
        curr.report_error(999, 0.5)


def test_lp_driven_sampling_biases_toward_high_progress_task():
    """Populate two tasks:
    - task A: strongly decreasing error → high |LP|
    - task B: flat error → low |LP|
    Expect A sampled more often than B (over many draws).
    """
    curr = _mk(max_tasks=2, eps=0.0)
    curr.add_task(TaskTemplate(id=0, spec={}, tag="A"))
    curr.add_task(TaskTemplate(id=1, spec={}, tag="B"))
    # Task A: high→low
    for e in [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
        curr.report_error(0, e)
    # Task B: flat
    for _ in range(8):
        curr.report_error(1, 0.5)

    counts: Counter[int] = Counter()
    for _ in range(500):
        counts[curr.sample_task().id] += 1
    assert counts[0] > counts[1], f"expected A>B, got {counts}"


def test_uniform_sampling_when_all_lp_zero():
    """Before sufficient data, sampling should be uniform-ish."""
    curr = _mk(max_tasks=3, eps=0.0, window=8, minsamp=8)
    for i in range(3):
        curr.add_task(TaskTemplate(id=i, spec={}))
    counts: Counter[int] = Counter()
    for _ in range(600):
        counts[curr.sample_task().id] += 1
    # Uniform expected ~200 each; allow generous slack
    for i in range(3):
        assert 100 <= counts[i] <= 300, f"non-uniform: {counts}"


def test_exploration_epsilon_forces_uniform_share():
    curr = _mk(max_tasks=2, eps=1.0)  # 100% uniform
    curr.add_task(TaskTemplate(id=0, spec={}, tag="A"))
    curr.add_task(TaskTemplate(id=1, spec={}, tag="B"))
    # Feed A strong LP; expect it's ignored due to eps=1
    for e in [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
        curr.report_error(0, e)

    counts: Counter[int] = Counter()
    for _ in range(400):
        counts[curr.sample_task().id] += 1
    # With eps=1, both should be ~200 each
    ratio = counts[0] / max(1, counts[1])
    assert 0.7 <= ratio <= 1.3, f"eps=1 did not enforce uniform: {counts}"


def test_summary_shape():
    curr = _mk(max_tasks=3)
    curr.add_task(TaskTemplate(id=0, spec={}))
    curr.add_task(TaskTemplate(id=1, spec={}))
    for _ in range(5):
        curr.report_error(0, 0.5)
    s = curr.summary()
    assert s["num_tasks"] == 2
    assert set(s["lp_by_task"].keys()) == {0, 1}


def test_state_dict_roundtrip():
    curr = _mk(max_tasks=3)
    curr.add_task(TaskTemplate(id=0, spec={"x": 1}, tag="A"))
    curr.add_task(TaskTemplate(id=1, spec={"x": 2}, tag="B"))
    for _ in range(4):
        curr.report_error(0, 0.5)
    state = curr.state_dict()

    curr2 = _mk(max_tasks=3)
    curr2.load_state_dict(state)
    assert set(curr2.known_tasks()) == {0, 1}
    assert curr2.get_task(0).tag == "A"


def test_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    curr = _mk()
    assert isinstance(curr, BoundedComponent)
