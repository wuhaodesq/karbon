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


def test_novelty_boost_prefers_high_intrinsic_task():
    led = KnowledgeLedger(novelty_weight=0.5)
    for t in (0, 1):
        led.record_episode(t, 50.0)  # equal mastery/familiarity
    # Task 1 keeps paying curiosity; task 0 is stale.
    for _ in range(20):
        led.record_intrinsic(1, 5.0)
        led.record_intrinsic(0, 0.0)
    pref = led.preference([0, 1])
    assert pref is not None
    assert pref[1] > pref[0]


def test_novelty_off_by_default():
    led = KnowledgeLedger()  # novelty_weight=0
    for t in (0, 1):
        led.record_episode(t, 50.0)
    for _ in range(10):
        led.record_intrinsic(1, 9.0)
    p_off = led.preference([0, 1])
    led2 = KnowledgeLedger(novelty_weight=0.0)
    for t in (0, 1):
        led2.record_episode(t, 50.0)
    p_base = led2.preference([0, 1])
    assert p_off == p_base  # intrinsic data ignored when weight=0


def test_novelty_state_roundtrip():
    led = KnowledgeLedger(novelty_weight=0.7)
    led.record_episode(3, 40.0)
    for _ in range(5):
        led.record_intrinsic(3, 2.0)
    sd = led.state_dict()
    led2 = KnowledgeLedger()
    led2.load_state_dict(sd)
    assert led2._novelty_weight == 0.7
    assert led2._tasks[3]["ema_int"] > 0.0


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


def test_relative_mastery_rich_reward_regime():
    """B13: with rich returns (1200-2400) the ZPD must still favor the
    harder task (absolute ret_scale=100 marked everything mastered ->
    uniform sampling; the zone-of-proximal-development signal was dead)."""
    led = KnowledgeLedger(ret_scale=100.0)
    for _ in range(20):
        led.record_episode(0, 2000.0)   # easier
        led.record_episode(2, 1200.0)   # harder
    pref = led.preference([0, 2])
    assert pref is not None
    assert pref[1] > pref[0]


def test_lp_selection_prefers_improving_task():
    """B15: LP mode must practice the IMPROVING task, not the
    high-but-flat one (the level criterion over-practiced low-payout
    tasks and the op band drifted)."""
    led = KnowledgeLedger(lp_selection=True)
    for _ in range(40):
        led.record_episode(0, 2000.0)             # flat high
    for i in range(40):
        led.record_episode(1, 800.0 + 10.0 * i)   # improving 800 -> 1190
    pref = led.preference([0, 1])
    assert pref is not None
    assert pref[1] > pref[0], pref


def test_lp_selection_stalled_task_gets_floor():
    led = KnowledgeLedger(lp_selection=True)
    for _ in range(40):
        led.record_episode(0, 1000.0)   # perfectly flat
    for i in range(40):
        led.record_episode(2, 900.0 + 8.0 * i)  # improving
    pref = led.preference([0, 2])
    assert pref is not None
    assert pref[0] < 0.3  # stalled -> near floor
    assert pref[1] > pref[0]


def test_lp_selection_unseen_gets_exploration():
    led = KnowledgeLedger(lp_selection=True)
    for _ in range(20):
        led.record_episode(0, 1000.0)  # stalled -> floor weight
    pref = led.preference([0, 3])  # task 3 unseen -> full exploration
    assert pref is not None
    assert pref[1] > pref[0], pref


def test_default_mode_unchanged():
    led = KnowledgeLedger()  # lp_selection=False
    assert led._lp_selection is False
    for _ in range(10):
        led.record_episode(0, 100.0)
    pref = led.preference([0])
    assert pref is not None


def test_lp_resume_from_legacy_state():
    """Resume from a pre-LP state dict (no fast/slow EMAs) must not crash
    and must still record returns (live KeyError 2026-10-27)."""
    legacy = {"capacity": 16, "alpha": 0.2, "ret_scale": 100.0,
              "floor": 0.05,
              "tasks": {"0": {"n": 12.0, "ema_ret": 900.0,
                              "ema_margin": 0.0, "margin_n": 0.0,
                              "ema_int": 0.0}}}
    led = KnowledgeLedger(lp_selection=True)
    led.load_state_dict(legacy)
    led.record_episode(0, 1000.0)  # must not raise
    p = led.preference([0])
    assert p is not None
