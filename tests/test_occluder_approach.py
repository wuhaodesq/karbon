"""Stage 20k: absolute-reach op criterion tests.

The training gate / reveal-bonus criterion is min(ratio*d0, radius) —
isomorphic to the eval op metric (best_d < min(0.7*start_d, 0.8m)).
These are pure-logic tests on the static helper (no MuJoCo instance).
"""
import pytest

from src.envs.three_d_world import ThreeDWorld

RATIO = 0.85
RADIUS = 0.8


def test_far_object_requires_absolute_arrival():
    """6m-away object: ratio alone would credit 5.1m (0.85*6=5.1);
    the 0.8m radius floor demands an actual arrival."""
    assert not ThreeDWorld._approach_ok(5.1, 6.0, RATIO, RADIUS)
    assert ThreeDWorld._approach_ok(0.7, 6.0, RATIO, RADIUS)


def test_close_start_uses_ratio_softening():
    """1.0m-away start: min(0.85*1.0, 0.8) = 0.8 -> ratio wins (still < 1m)."""
    assert ThreeDWorld._approach_ok(0.79, 1.0, RATIO, RADIUS)
    assert not ThreeDWorld._approach_ok(0.81, 1.0, RATIO, RADIUS)


def test_degenerate_d0_never_success():
    assert not ThreeDWorld._approach_ok(0.1, 0.0, RATIO, RADIUS)


def test_zero_radius_fails_everything():
    assert not ThreeDWorld._approach_ok(0.0, 6.0, RATIO, 0.0)


def test_eight_directions_are_distinct_and_unit():
    """Stage 20q: actions 0-7 must be 8 distinct unit directions (4 cardinal
    + 4 diagonal), not a 4-dir set with a 2x-force gear."""
    dirs = [ThreeDWorld._action_direction(a) for a in range(8)]
    assert len(set(dirs)) == 8
    for (ux, uy) in dirs:
        assert abs((ux * ux + uy * uy) - 1.0) < 1e-6
    # cardinal
    assert ThreeDWorld._action_direction(0) == (0.0, 1.0)
    assert ThreeDWorld._action_direction(1) == (0.0, -1.0)
    assert ThreeDWorld._action_direction(2) == (-1.0, 0.0)
    assert ThreeDWorld._action_direction(3) == (1.0, 0.0)
    # diagonal (action 4 = +x+y)
    assert abs(ThreeDWorld._action_direction(4)[0] - 0.70710678) < 1e-5
    assert abs(ThreeDWorld._action_direction(4)[1] - 0.70710678) < 1e-5


def test_grasp_actions_map_to_cardinal_before_unlock():
    """Actions 8-11 fall through to locomotion (action % 8) before dev_age
    unlocks grasping -> must resolve to cardinal dirs 0-3."""
    assert ThreeDWorld._action_direction(8) == (0.0, 1.0)   # 8 % 8 = 0
    assert ThreeDWorld._action_direction(11) == (1.0, 0.0)  # 11 % 8 = 3
