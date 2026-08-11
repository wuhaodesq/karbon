"""Tests for the developmental milestone scale (open-gap C#8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.developmental_milestones import (
    DevelopmentalEvaluator,
    MILESTONES,
    estimate_cognitive_age,
)


def _states_with_physics_ok() -> list[dict]:
    """Agent applies force along +x and object moves along +x."""
    return [{
        "force_motion_pairs": [
            {"force": (1.0, 0.0), "velocity_after": (0.8, 0.1), "object_id": 0},
            {"force": (0.0, 1.0), "velocity_after": (0.1, 0.7), "object_id": 1},
            {"force": (-1.0, 0.0), "velocity_after": (-0.9, 0.0), "object_id": 2},
        ],
        "occlusion_events": [],
        "count_trials": [],
        "object_contact_order": [0, 1, 2],
        "means_ends_score": 1.0,
    }]


def _states_with_occlusion_ok() -> list[dict]:
    """During occlusion agent moves toward last-known position."""
    return [{
        "force_motion_pairs": [],
        "occlusion_events": [{
            "last_known": (5.0, 5.0),
            "agent_traj_during_occ": [(0.0, 0.0), (2.0, 2.0), (4.5, 4.8)],
        }],
        "count_trials": [],
    }]


def _states_with_count_ok() -> list[dict]:
    return [{
        "force_motion_pairs": [],
        "occlusion_events": [],
        "count_trials": [
            {"true_count": 3, "estimated_count": 3},
            {"true_count": 5, "estimated_count": 4},
            {"true_count": 2, "estimated_count": 2},
        ],
    }]


def test_scale_has_six_milestones():
    assert len(MILESTONES) == 6
    ages = [m.age_years for m in MILESTONES]
    assert ages == sorted(ages)


def test_intuitive_physics_detects_causality():
    rep = estimate_cognitive_age(_states_with_physics_ok())
    assert rep.passed["intuitive_physics"]
    assert rep.scores["intuitive_physics"] >= 0.6


def test_object_permanence_pass():
    rep = estimate_cognitive_age(_states_with_occlusion_ok())
    assert rep.passed["object_permanence"]


def test_number_sense_pass():
    rep = estimate_cognitive_age(_states_with_count_ok())
    assert rep.passed["number_sense"]


def test_estimated_age_is_max_passed():
    rep = estimate_cognitive_age(
        _states_with_physics_ok()
        + _states_with_occlusion_ok()
        + _states_with_count_ok()
    )
    # passed: 1y(obj), 1.5y(means_ends), 2.5y(physics), 3.5y(count), 4.0y(tom) -> max = 4.0
    assert rep.estimated_age == 4.0


def test_empty_states_gives_zero_age():
    rep = estimate_cognitive_age([])
    assert rep.estimated_age == 0.0


def test_report_summary_runs():
    rep = estimate_cognitive_age(_states_with_physics_ok())
    assert "estimated cognitive age" in rep.summary()


# ----------------------------------------------------------------------
# Systematic reasoning (Stage 18 eval-fix regression tests)
# ----------------------------------------------------------------------


def _systematic_ok_states(num_actions=12) -> list[dict]:
    """Low-entropy actions + consistent force direction + rules present."""
    import math
    pairs = []
    for i in range(12):
        ang = i % 4 * math.pi / 2  # all in the same 8-bucket range
        pairs.append({
            "force": (math.cos(ang), math.sin(ang)),
            "velocity_after": (0.9, 0.1),
            "object_id": i % 3,
        })
    return [{
        "force_motion_pairs": pairs,
        "occlusion_events": [],
        "count_trials": [],
        "actions": [0] * 60,          # zero entropy
        "rule_count": 30,             # discovered rules (maxes the rule term)
        "num_actions": num_actions,   # 12-action space (env-provided)
    }]


def test_systematic_reasoning_can_pass_with_rule_count():
    """With rule_count supplied + 12-action normalization, systematic
    behavior must be able to pass the 0.6 threshold."""
    rep = estimate_cognitive_age(_systematic_ok_states())
    assert rep.scores["systematic_reasoning"] >= 0.6
    assert rep.passed["systematic_reasoning"]


def test_systematic_drops_without_rule_count():
    """No rule_count -> rule term is 0 -> milestone must NOT pass
    (eval script bug: it never passed rule_count, so the milestone was
    capped below threshold for the whole of Stage 18)."""
    states = _systematic_ok_states()
    states[0].pop("rule_count")
    rep = estimate_cognitive_age(states)
    assert rep.scores["systematic_reasoning"] < 0.6


def test_systematic_uniform_12_actions_stays_low():
    """Uniform random policy in a 12-action space must score low
    (regression: with the old hardcoded ln(8) normalization this term
    was always ~0 regardless of behavior)."""
    import math
    acts = [i % 12 for i in range(120)]
    pairs = []
    for i in range(120):
        ang = (i % 8) * math.pi / 4
        pairs.append({
            "force": (math.cos(ang), math.sin(ang)),
            "velocity_after": (0.1, 0.1),
            "object_id": i % 4,
        })
    states = [{"actions": acts, "force_motion_pairs": pairs,
               "rule_count": 0, "num_actions": 12}]
    rep = estimate_cognitive_age(states)
    assert rep.scores["systematic_reasoning"] < 0.2
