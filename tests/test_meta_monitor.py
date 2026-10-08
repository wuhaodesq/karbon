"""Tests for Stage 21 MetaMonitor (online recursive metacognition)."""

from __future__ import annotations

import torch

from src.models.meta_monitor import MetaMonitor


def test_probe_returns_probability():
    m = MetaMonitor()
    p = m.on_probe(100)
    assert 0.0 <= p <= 1.0


def test_outcome_trains_and_updates_counter():
    m = MetaMonitor(lr=1e-2)
    for i in range(40):
        m.on_probe(i * 10)
        m.on_outcome(i * 10 + 1, 1.0)
    assert m._updates == 40
    # consistent labels -> the head should learn to predict ~success
    p = m.on_probe(1000)
    assert p > 0.5


def test_skill_stats_after_enough_samples():
    m = MetaMonitor(window=8, skill_window=200)
    # outcomes alternate in blocks; recent-window should carry signal
    ok = 1.0
    for i in range(300):
        if i % 25 == 0:
            ok = 1.0 - ok
        m.on_probe(i)
        m.on_outcome(i, ok)
    s = m.skill
    assert s["n"] >= 100
    assert -1.0 <= s["skill"] <= 1.0
    assert 0.0 <= s["majority"] <= 1.0


def test_skill_small_n_returns_zeros():
    m = MetaMonitor()
    m.on_probe(0)
    m.on_outcome(1, 1.0)
    s = m.skill
    assert s["n"] < 100 and s["skill"] == 0.0


def test_state_dict_roundtrip_updates():
    m = MetaMonitor()
    for i in range(5):
        m.on_probe(i)
        m.on_outcome(i, 1.0)
    sd = m.state_dict()
    m2 = MetaMonitor()
    m2.load_state_dict(sd)
    assert m2._updates == 5


def test_bounded_windows():
    m = MetaMonitor(window=4, skill_window=16)
    for i in range(100):
        m.on_probe(i)
        m.on_outcome(i, float(i % 2))
    assert len(m._outcomes) <= 4
    assert len(m._correct) <= 16
    assert len(m._preds) <= 16
    assert len(m._outcome_win) <= 16
    assert len(m._probe_steps) <= 256
