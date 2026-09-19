"""Tests for developmental memory components (autobiographical recency)."""

from __future__ import annotations

from src.models.developmental_memory import AutobiographicalMemory


def test_recency_eviction_displaces_legacy_high_importance():
    """2026-09-19: identity is recent life, not the all-time top-reward events.
    Legacy huge-importance events must be displaceable by recent modest ones,
    otherwise exploration events never survive (openness pinned at 0.00)."""
    mem = AutobiographicalMemory(
        max_events=3, promotion_threshold=0.0, half_life_steps=1000,
    )
    for i in range(3):
        mem.add_event(step=0, description=f"legacy {i}",
                      importance=100.0, episode_id=i)
    for i in range(3):
        mem.add_event(step=10_000, description=f"recent {i}",
                      importance=8.0, episode_id=100 + i)
    descs = [e.description for e in mem._events]
    assert len(mem) == 3
    assert all(d.startswith("recent") for d in descs)


def test_autobiographical_memory_bounded():
    mem = AutobiographicalMemory(max_events=5, promotion_threshold=0.0)
    for i in range(20):
        mem.add_event(step=i * 1000, description=str(i),
                      importance=1.0, episode_id=i)
    assert len(mem) <= 5  # Axiom 1


def test_promotion_threshold_filters_low_importance():
    mem = AutobiographicalMemory(max_events=5, promotion_threshold=5.0)
    assert mem.add_event(step=0, description="boring", importance=1.0,
                         episode_id=0) is None
    assert mem.add_event(step=0, description="big", importance=8.0,
                         episode_id=1) is not None
    assert len(mem) == 1
