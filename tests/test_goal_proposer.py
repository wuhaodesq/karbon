"""Tests for the Stage 24 M1 GoalProposer (bounded, deterministic)."""
import pytest

from src.models.goal_proposer import Goal, GoalProposer, KINDS


def _drives(**over):
    base = {"curiosity": 1.0, "competence": 1.0, "social": 1.0,
            "safety": 1.0, "rest": 1.0}
    base.update(over)
    return base


def test_propose_bounds_and_kinds():
    gp = GoalProposer(seed=1)
    goals = gp.propose(_drives(curiosity=0.1, competence=0.2,
                               social=0.3, safety=0.4, rest=0.5),
                       lp_slopes={"t0": 0.4, "t1": 0.1, "t2": -0.2},
                       pending_hypotheses=3)
    assert 1 <= len(goals) <= GoalProposer.MAX_PROPOSALS
    assert all(isinstance(g, Goal) for g in goals)
    assert all(g.kind in KINDS for g in goals)
    assert all(0.0 <= g.utility <= 1.0 for g in goals)
    assert goals == sorted(goals, key=lambda g: g.utility, reverse=True)


def test_deficit_prioritized():
    gp = GoalProposer(seed=2)
    goals = gp.propose(_drives(curiosity=0.05))
    assert goals[0].kind == "explore"
    assert goals[0].source == "curiosity"


def test_lp_and_hypothesis_goals():
    gp = GoalProposer(seed=3)
    goals = gp.propose(_drives(), lp_slopes={"t9": 0.8}, pending_hypotheses=2)
    keys = {g.key for g in goals}
    assert "practice:t9" in keys
    assert "test:hypothesis" in keys


def test_anti_monopoly_suppresses_repeat():
    gp = GoalProposer(seed=4, monopoly_k=4, explore_eps=0.0)
    a = Goal("explore", "novel_zone", 0.9, "curiosity")
    b = Goal("practice", "current_task", 0.5, "competence")
    picks = [gp.arbitrate([a, b]).key for _ in range(4)]
    assert picks == [a.key] * 4
    nxt = gp.arbitrate([a, b])
    assert nxt is not None and nxt.key == b.key


def test_feedback_flips_arbitration():
    gp = GoalProposer(seed=5, explore_eps=0.0, monopoly_k=100)
    a = Goal("explore", "novel_zone", 0.60, "curiosity")
    b = Goal("practice", "current_task", 0.55, "competence")
    assert gp.arbitrate([a, b]).key == a.key
    for _ in range(30):
        gp.record_outcome(a.key, achieved=False)
        gp.record_outcome(b.key, achieved=True)
    goals = gp.propose(_drives(curiosity=0.4, competence=0.4))
    by_key = {g.key: g.utility for g in goals}
    assert by_key[b.key] > by_key[a.key]


def test_utility_table_bounded():
    gp = GoalProposer(seed=6, capacity=8)
    for i in range(200):
        gp.record_outcome(f"practice:task{i}", achieved=(i % 2 == 0))
    assert len(gp) <= gp.capacity == 8


def test_state_roundtrip_and_determinism():
    gp = GoalProposer(seed=7, monopoly_k=3)
    a = Goal("explore", "novel_zone", 0.9, "curiosity")
    b = Goal("practice", "current_task", 0.4, "competence")
    for _ in range(5):
        gp.arbitrate([a, b])
    gp.record_outcome(a.key, True)
    state = gp.state_dict()

    gp2 = GoalProposer(seed=99)
    gp2.load_state_dict(state)
    assert gp2.last_choice == gp.last_choice
    assert len(gp2) == len(gp)
    assert gp2.arbitrate([a, b]).key == gp.arbitrate([a, b]).key

    g1 = GoalProposer(seed=11).propose(_drives(curiosity=0.1))
    g2 = GoalProposer(seed=11).propose(_drives(curiosity=0.1))
    assert [x.key for x in g1] == [x.key for x in g2]


def test_empty_drives_returns_empty():
    gp = GoalProposer(seed=8)
    assert gp.propose({}) == []
    assert gp.arbitrate([]) is None
