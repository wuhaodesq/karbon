"""Tests for the C#8 evaluation harness data contract.

These exercise the way ``scripts/eval/run_developmental_eval.py`` collects
``env_states`` during rollout (one ``info`` dict *per step*, so each entry's
signal lists are usually small/empty and must be aggregated across steps by
``DevelopmentalEvaluator._aggregate``). No torch needed — this only validates
the data contract between the rollout collector and the milestone scorer.
"""

from __future__ import annotations

from src.eval.developmental_milestones import DevelopmentalEvaluator


def _rollout_like_states() -> list[dict]:
    """Mimic what the eval script appends each step: each entry is one step's
    ``info``, so signals are spread across many small dicts (NOT pre-aggregated
    into one big list like the standalone milestone tests use)."""
    states: list[dict] = []
    # Step 1: agent pushes an object to the right (force +x, vel +x)
    states.append({
        "force_motion_pairs": [{"force": (1.0, 0.0), "velocity_after": (0.9, 0.05)}],
        "occlusion_events": [],
        "count_trials": [],
    })
    # Step 2: another correct push up
    states.append({
        "force_motion_pairs": [{"force": (0.0, 1.0), "velocity_after": (0.0, 0.8)}],
        "occlusion_events": [],
        "count_trials": [],
    })
    # Step 3: occlusion begins, agent near last-known
    states.append({
        "force_motion_pairs": [],
        "occlusion_events": [{
            "last_known": (5.0, 5.0),
            "agent_traj_during_occ": [(0.0, 0.0), (3.0, 3.0)],
        }],
        "count_trials": [],
    })
    # Step 4: episode ends, count trial recorded
    states.append({
        "force_motion_pairs": [],
        "occlusion_events": [],
        "count_trials": [{"true_count": 4, "estimated_count": 4}],
    })
    return states


def test_rollout_aggregation_scores_across_steps():
    rep = DevelopmentalEvaluator().evaluate(_rollout_like_states())
    # intuitive physics: 2/2 correct pushes -> pass
    assert rep.passed["intuitive_physics"]
    # object permanence: agent moved toward last-known -> pass
    assert rep.passed["object_permanence"]
    # number sense: exact count -> pass
    assert rep.passed["number_sense"]
    # all three real milestones passed -> estimated age = max(1.0, 2.5, 3.5)
    assert rep.estimated_age == 3.5


def test_rollout_with_no_signals_is_zero():
    states = [{"force_motion_pairs": [], "occlusion_events": [], "count_trials": []}
              for _ in range(10)]
    rep = DevelopmentalEvaluator().evaluate(states)
    assert rep.estimated_age == 0.0
