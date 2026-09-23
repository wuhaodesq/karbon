"""Tests for SymbolBackend public premise injection + query shapes.

2026-09-23: added with the force->motion / formal-reasoning work. Covers
add_fact/add_rule (capacity-bounded), fallback modus ponens, and the
query-branch selection that keeps ground/wildcard lookups on the
fact+rule path (the kanren branch only serves enumerating queries).
"""

from __future__ import annotations

from src.models.symbol_backend import SymbolBackend


def test_add_fact_roundtrip():
    be = SymbolBackend(max_facts=16, max_rules=8)
    assert be.add_fact("at", ("a",))
    res = be.query("at", ("a",))
    assert len(res.answers) >= 1


def test_add_fact_wildcard_lookup():
    be = SymbolBackend(max_facts=16, max_rules=8)
    be.add_fact("at", ("a",))
    be.add_fact("at", ("b",))
    res = be.query("at", ("_",))
    assert len(res.answers) >= 2


def test_add_rule_modus_ponens():
    be = SymbolBackend(max_facts=16, max_rules=8)
    be.add_fact("occluded", ("o1",))
    assert be.add_rule([("occluded", ("o1",))], ("track", ("o1",)), 0.9)
    res = be.query("track", ("o1",))
    assert len(res.answers) >= 1


def test_missing_antecedent_stays_empty():
    be = SymbolBackend(max_facts=16, max_rules=8)
    be.add_fact("visible", ("k",))
    be.add_rule([("visible", ("k",)), ("near", ("a", "k"))],
                ("grasp", ("k",)), 0.8)
    res = be.query("grasp", ("k",))
    assert len(res.answers) == 0  # soundness: no false positive


def test_fact_capacity_bounded():
    be = SymbolBackend(max_facts=2, max_rules=8)
    assert be.add_fact("p", ("1",))
    assert be.add_fact("p", ("2",))
    assert not be.add_fact("p", ("3",))  # Axiom 1


def test_rule_capacity_bounded():
    be = SymbolBackend(max_facts=8, max_rules=1)
    assert be.add_rule([("a", ("x",))], ("b", ("x",)))
    assert not be.add_rule([("a", ("y",))], ("b", ("y",)))


def test_enumerating_query_with_wildcard():
    """Enumeration is expressed with a wildcard argument (the query branch
    requires matching arity; an empty args tuple does not enumerate)."""
    be = SymbolBackend(max_facts=16, max_rules=8)
    be.add_fact("at", ("a",))
    be.add_fact("at", ("b",))
    res = be.query("at", ("_",))
    assert len(res.answers) >= 2
