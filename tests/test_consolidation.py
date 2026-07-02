"""Tests for :mod:`src.continual.consolidation`."""

from __future__ import annotations

import logging

from src.continual import (
    ConsolidationConfig,
    SleepConsolidationLoop,
)


def test_no_tasks_registered_returns_empty():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=10))
    fired = loop.tick(step=10)
    assert fired == []


def test_warmup_gate():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=100, replay_trim_every=10))
    counter = {"n": 0}
    loop.set_replay_trim(lambda: counter.update(n=counter["n"] + 1))
    for step in range(50):
        loop.tick(step)
    assert counter["n"] == 0


def test_replay_trim_fires_on_period():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=5))
    counter = {"n": 0}
    loop.set_replay_trim(lambda: counter.update(n=counter["n"] + 1))
    for step in range(21):  # 0..20
        loop.tick(step)
    # Fires at step 5, 10, 15, 20 → 4 times
    assert counter["n"] == 4


def test_disabled_task_never_fires():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=0))
    counter = {"n": 0}
    loop.set_replay_trim(lambda: counter.update(n=counter["n"] + 1))
    for step in range(100):
        loop.tick(step)
    assert counter["n"] == 0


def test_multiple_tasks_fire_at_own_periods():
    loop = SleepConsolidationLoop(ConsolidationConfig(
        warmup_steps=0,
        replay_trim_every=3,
        skills_merge_every=5,
        ttt_distill_every=0,           # disabled
        ewc_consolidate_every=7,
    ))
    counts = {"r": 0, "s": 0, "e": 0}
    loop.set_replay_trim(lambda: counts.update(r=counts["r"] + 1))
    loop.set_skills_merge(lambda: counts.update(s=counts["s"] + 1))
    loop.set_ewc_consolidate(lambda: counts.update(e=counts["e"] + 1))
    for step in range(1, 21):
        loop.tick(step)
    # r fires at 3,6,9,12,15,18 → 6
    # s fires at 5,10,15,20 → 4
    # e fires at 7,14 → 2
    assert counts["r"] == 6
    assert counts["s"] == 4
    assert counts["e"] == 2


def test_task_exception_does_not_crash(caplog):
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=1))

    def _boom():
        raise RuntimeError("expected")

    loop.set_replay_trim(_boom)
    caplog.set_level(logging.ERROR)
    fired = loop.tick(step=1)
    # Task fired (registered), even though it raised
    assert fired == ["replay_trim"]
    # Log captured
    assert any("Consolidation task replay_trim failed" in r.message for r in caplog.records)


def test_counters_bounded_state():
    """The counters dataclass has a fixed number of fields; no unbounded growth."""
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=1))
    loop.set_replay_trim(lambda: None)
    for step in range(1, 101):
        loop.tick(step)
    counters = loop.counters()
    # Fixed structure — attribute count same as dataclass definition
    assert counters.replay_trim_runs == 100
    # No dict growth: only the dataclass fields exist
    expected_fields = {
        "replay_trim_runs",
        "skills_merge_runs",
        "ttt_distill_runs",
        "ewc_consolidate_runs",
        "last_wall_time",
        "total_wall_seconds",
    }
    assert set(counters.__dict__.keys()) == expected_fields


def test_state_dict_roundtrip():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=1))
    loop.set_replay_trim(lambda: None)
    for step in range(1, 6):
        loop.tick(step)
    state = loop.state_dict()

    loop2 = SleepConsolidationLoop(ConsolidationConfig())
    loop2.load_state_dict(state)
    assert loop2.counters().replay_trim_runs == 5
    assert loop2.config.replay_trim_every == 1


def test_summary_shape():
    loop = SleepConsolidationLoop(ConsolidationConfig(warmup_steps=0, replay_trim_every=1))
    loop.set_replay_trim(lambda: None)
    for step in range(1, 4):
        loop.tick(step)
    s = loop.summary()
    assert set(s.keys()) >= {"replay_trim_runs", "skills_merge_runs",
                             "ttt_distill_runs", "ewc_consolidate_runs",
                             "total_wall_seconds"}
