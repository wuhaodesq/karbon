"""Tests for the knowledge ledger (symbolic state -> task preference)."""

from __future__ import annotations

from src.curriculum.knowledge_ledger import KnowledgeLedger


def test_ledger_bounded_capacity():
    led = KnowledgeLedger(capacity=3)
    for i in range(10):
        led.record_episode(i, 1.0)
    assert len(led) <= 3  # Axiom 1


def test_ledger_empty_preference_is_none():
    led = KnowledgeLedger()
    assert led.preference([0, 1, 2]) is None


def test_ledger_prefers_unmastered_unfamiliar_task():
    led = KnowledgeLedger(ret_scale=100.0)
    # Task 0: mastered (high return, familiar states)
    for _ in range(20):
        led.record_episode(0, 90.0)
        led.record_margin(0, 0.9)
    # Task 1: not mastered, unfamiliar states
    for _ in range(20):
        led.record_episode(1, 10.0)
        led.record_margin(1, -0.5)
    pref = led.preference([0, 1])
    assert pref is not None
    assert pref[1] > pref[0]


def test_ledger_unseen_task_gets_exploration_weight():
    led = KnowledgeLedger()
    led.record_episode(0, 50.0)
    pref = led.preference([0, 1])  # task 1 never observed
    assert pref is not None
    assert pref[1] > pref[0]


def test_ledger_mastered_task_keeps_floor():
    led = KnowledgeLedger(ret_scale=50.0, floor=0.05)
    for _ in range(30):
        led.record_episode(0, 200.0)
        led.record_margin(0, 1.0)
    pref = led.preference([0])
    assert pref is not None
    assert 0.0 < pref[0] <= 1.0  # never fully banned


def test_ledger_state_roundtrip():
    led = KnowledgeLedger()
    led.record_episode(3, 42.0)
    led.record_margin(3, 0.3)
    state = led.state_dict()

    led2 = KnowledgeLedger()
    led2.load_state_dict(state)
    assert len(led2) == 1
    assert led2.practice_headroom(3) == led.practice_headroom(3)
