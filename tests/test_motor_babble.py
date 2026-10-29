"""Tests for the motor-babbling novelty drive (Stage 23 M3)."""

from __future__ import annotations

from src.models.motor_babble import MotorBabble


def test_constant_actions_saturate_novelty():
    mb = MotorBabble()
    for s in range(3):  # warmup: n-gram needs n=3 actions
        mb.step(5, s)
    after_warmup = mb.step(5, 3)
    for s in range(4, 200):
        last = mb.step(5, s)
    # repeated identical n-grams earn progressively less
    assert after_warmup > 0.05
    assert last < after_warmup * 0.2


def test_varied_actions_earn_more_than_constant():
    const = MotorBabble()
    varied = MotorBabble()
    import random
    rng = random.Random(0)
    c_sum = v_sum = 0.0
    for s in range(300):
        c_sum += const.step(0, s)
        v_sum += varied.step(rng.randint(0, 11), s)
    assert v_sum > c_sum * 1.5


def test_coefficient_schedule_decays_to_floor():
    mb = MotorBabble(coef=0.3, floor=0.02, tau=1000.0)
    c0 = mb.coef_at(0)
    c_late = mb.coef_at(50_000)
    assert c0 > c_late
    assert abs(c_late - 0.02) < 0.005


def test_reactivation_boost_on_collapse():
    mb = MotorBabble(window=256, min_unique_frac=0.3)
    for s in range(400):
        mb.step(3, s)  # single action -> collapsed repertoire
    sm = mb.summary
    assert sm["boost"] == 2.0
    assert mb._boost_windows > 0


def test_bounded_ngram_table():
    mb = MotorBabble(max_entries=16)
    import random
    rng = random.Random(1)
    for s in range(5000):
        mb.step(rng.randint(0, 11), s)
    assert len(mb._counts) <= 16  # Axiom 1
    assert mb.capacity == 16


def test_state_roundtrip():
    mb = MotorBabble()
    for s in range(50):
        mb.step(s % 4, s)
    sd = mb.state_dict()
    mb2 = MotorBabble()
    mb2.load_state_dict(sd)
    assert mb2._steps == 50
    assert len(mb2._counts) == len(mb._counts)
