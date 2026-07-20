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
            {"force": (1.0, 0.0), "velocity_after": (0.8, 0.1)},
            {"force": (0.0, 1.0), "velocity_after": (0.1, 0.7)},
            {"force": (-1.0, 0.0), "velocity_after": (-0.9, 0.0)},
        ],
        "occlusion_events": [],
        "count_trials": [],
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
    # passed: 1y(obj), 2.5y(physics), 3.5y(count) -> max = 3.5
    assert rep.estimated_age == 3.5


def test_empty_states_gives_zero_age():
    rep = estimate_cognitive_age([])
    assert rep.estimated_age == 0.0


def test_report_summary_runs():
    rep = estimate_cognitive_age(_states_with_physics_ok())
    assert "estimated cognitive age" in rep.summary()
