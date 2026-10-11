"""Tests for Stage 22 M2 — concept-level rules with unification-lite.

M2 criterion: multi-hop concept reasoning (2-hop via the super level,
3-hop via implies composition) succeeds while the instance-level baseline
(R1/R3 only, no abstraction) cannot join the parent chain — SOUND
(unsupported queries empty), bounded (facts/answers/ops).
"""

from __future__ import annotations

from src.models.concept_rules import ConceptProver


def _world() -> ConceptProver:
    p = ConceptProver()
    p.add_concept_rules()
    p.add_fact("member", ("i1", "c1"))
    p.add_fact("parent", ("c1", "s1"))
    p.add_fact("prop", ("c1", "red"))
    p.add_fact("prop", ("s1", "heavy"))
    p.add_fact("implies", ("heavy", "hard_to_lift"))
    return p


def test_one_hop_inheritance():
    p = _world()
    ans = p.query("prop", ("i1", "?p"))
    assert ("red",) in ans


def test_two_hop_via_super_abstraction():
    p = _world()
    ans = p.query("prop", ("i1", "?p"))
    assert ("heavy",) in ans  # member->super->prop chain


def test_three_hop_with_implies():
    p = _world()
    ans = p.query("prop", ("i1", "?p"))
    assert ("hard_to_lift",) in ans  # + implies composition


def test_baseline_without_abstraction_cannot_join_parent():
    """The instance-level baseline (no R2) cannot reach super props."""
    p = ConceptProver()
    p.add_rule([("member", ("?x", "?c")), ("prop", ("?c", "?p"))],
               ("prop", ("?x", "?p")))
    p.add_rule([("prop", ("?x", "?p")), ("implies", ("?p", "?q"))],
               ("prop", ("?x", "?q")))
    p.add_fact("member", ("i1", "c1"))
    p.add_fact("parent", ("c1", "s1"))
    p.add_fact("prop", ("c1", "red"))
    p.add_fact("prop", ("s1", "heavy"))
    assert ("red",) in p.query("prop", ("i1", "?p"))
    assert ("heavy",) not in p.query("prop", ("i1", "?p"))


def test_soundness_unsupported_query_empty():
    p = _world()
    assert p.query("prop", ("i9", "?p")) == []
    assert p.query("prop", ("i1", "green")) == []


def test_bounded_facts_and_answers():
    p = ConceptProver(max_facts=3, max_answers=2)
    assert p.add_fact("a", ("1",))
    assert p.add_fact("a", ("2",))
    assert p.add_fact("a", ("3",))
    assert not p.add_fact("a", ("4",))  # Axiom 1
    ans = p.query("a", ("?x",))
    assert len(ans) <= 2


def test_op_budget_exposed():
    p = _world()
    p.query("prop", ("i1", "?p"))
    assert p.last_ops > 0
