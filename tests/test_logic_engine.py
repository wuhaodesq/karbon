"""Tests for the symbolic logic engine."""

from __future__ import annotations

import pytest
import torch

from src.models.logic_engine import (
    LogicEngine,
    QuantifiedRule,
    Quantifier,
    Variable,
    VariableType,
)


D_MODEL = 64


# =====================================================================
# Variable
# =====================================================================


def test_variable_match_high_similarity():
    emb = torch.randn(D_MODEL)
    var = Variable(name="key", var_type=VariableType.OBJECT, category_embedding=emb)
    # Same embedding → similarity ~1.0
    sim = var.match(emb)
    assert sim > 0.9


def test_variable_match_low_similarity():
    var = Variable(name="key", var_type=VariableType.OBJECT, category_embedding=torch.randn(D_MODEL))
    sim = var.match(torch.randn(D_MODEL))
    assert sim < 0.5


def test_variable_binds_on_match():
    emb = torch.randn(D_MODEL)
    var = Variable(name="key", var_type=VariableType.OBJECT, category_embedding=emb)
    var.match(emb)
    assert len(var.bindings) == 0  # match() doesn't bind; unify() does


# =====================================================================
# LogicEngine — variables
# =====================================================================


def test_engine_define_variable():
    engine = LogicEngine(d_model=D_MODEL)
    var = engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    assert var.name == "key"
    assert engine.get_variable("key") is not None


def test_engine_variable_capacity_bounded():
    engine = LogicEngine(d_model=D_MODEL, max_variables=4)
    for i in range(10):
        engine.define_variable(f"var_{i}", VariableType.OBJECT, torch.randn(D_MODEL))
    assert len(engine._variables) <= 4  # Axiom 1


def test_engine_unify_finds_matching_variable():
    engine = LogicEngine(d_model=D_MODEL, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, emb)
    # Query with the same embedding → should match
    matches = engine.unify(emb)
    assert len(matches) == 1
    assert matches[0][0].name == "key"
    assert matches[0][1] > 0.5


def test_engine_unify_no_match():
    engine = LogicEngine(d_model=D_MODEL, match_threshold=0.9)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    matches = engine.unify(torch.randn(D_MODEL))
    assert len(matches) == 0


# =====================================================================
# LogicEngine — rules
# =====================================================================


def test_engine_add_rule():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    rule = engine.add_rule(
        quantifier=Quantifier.UNIVERSAL,
        variable_name="key",
        condition="see(X)",
        action=3,
        confidence=0.8,
    )
    assert rule.id == 0
    assert len(engine) == 1


def test_engine_rule_capacity_bounded():
    engine = LogicEngine(d_model=D_MODEL, max_rules=8)
    engine.define_variable("X", VariableType.ABSTRACT, torch.randn(D_MODEL))
    for i in range(20):
        engine.add_rule(
            Quantifier.UNIVERSAL, "X", f"cond_{i}", i % 7, confidence=0.5,
        )
    assert len(engine) <= engine.capacity  # Axiom 1


def test_engine_rule_repr():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    rule = engine.add_rule(
        Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9,
        proof_verified=True,
    )
    text = str(rule)
    assert "∀" in text
    assert "key" in text
    assert "✓" in text  # proof_verified


# =====================================================================
# LogicEngine — forward chaining
# =====================================================================


def test_forward_chain_derives_new_rule():
    """Rule A + Rule B → derived Rule C."""
    engine = LogicEngine(d_model=D_MODEL, max_rules=16, match_threshold=0.5)

    # Both rules use the same variable category → can chain
    emb = torch.randn(D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, emb)

    # Rule A: IF see(item) THEN pick_up(item)
    engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X)", 3, confidence=0.9)
    # Rule B: IF see(item) THEN use(item) (different action)
    engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X) → use(X)", 5, confidence=0.8)

    new_rules = engine.forward_chain()
    assert len(new_rules) > 0
    # The derived rule should have proof_verified=True
    assert all(r.proof_verified for r in new_rules)
    # The derived rule should have a derivation chain
    assert all(len(r.derivation_chain) == 2 for r in new_rules)


def test_forward_chain_no_chain_different_categories():
    """Rules with different variable categories should NOT chain."""
    engine = LogicEngine(d_model=D_MODEL, max_rules=16, match_threshold=0.9)

    # Two very different embeddings
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    engine.define_variable("wall", VariableType.OBJECT, torch.randn(D_MODEL) * 10)

    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)
    engine.add_rule(Quantifier.UNIVERSAL, "wall", "see(X)", 0, confidence=0.9)

    new_rules = engine.forward_chain()
    assert len(new_rules) == 0  # different categories → no chaining


def test_forward_chain_avoids_duplicates():
    """Running forward_chain twice should not produce duplicate derivations."""
    engine = LogicEngine(d_model=D_MODEL, max_rules=32, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, emb)
    engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X)", 3, confidence=0.9)
    engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X) → use(X)", 5, confidence=0.8)

    first = engine.forward_chain()
    second = engine.forward_chain()
    # Second run should find no NEW derivations (already done)
    assert len(second) == 0


# =====================================================================
# LogicEngine — reasoning
# =====================================================================


def test_reason_finds_matching_rule():
    engine = LogicEngine(d_model=D_MODEL, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, emb)
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)

    rule, info = engine.reason(emb)
    assert rule is not None
    assert rule.action == 3
    assert len(info["unified_variables"]) == 1
    assert info["unified_variables"][0]["name"] == "key"


def test_reason_returns_none_no_match():
    engine = LogicEngine(d_model=D_MODEL, match_threshold=0.9)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)

    rule, info = engine.reason(torch.randn(D_MODEL))
    assert rule is None
    assert len(info["unified_variables"]) == 0


def test_reason_prefers_higher_confidence():
    engine = LogicEngine(d_model=D_MODEL, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, emb)
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.5)
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 5, confidence=0.9)

    rule, _ = engine.reason(emb)
    assert rule.action == 5  # higher confidence rule wins


# =====================================================================
# LogicEngine — proof checking
# =====================================================================


def test_verify_proof_valid_chain():
    engine = LogicEngine(d_model=D_MODEL, max_rules=16, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, emb)
    r1 = engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X)", 3, confidence=0.9)
    r2 = engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X) → use(X)", 5, confidence=0.8)
    derived = engine.forward_chain()
    if derived:
        assert engine.verify_proof(derived[0].id) is True


def test_verify_proof_invalid_missing_chain_rule():
    engine = LogicEngine(d_model=D_MODEL, max_rules=16, match_threshold=0.5)
    emb = torch.randn(D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, emb)
    r1 = engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X)", 3, confidence=0.9)
    # Manually create a rule with a non-existent chain
    r2 = engine.add_rule(
        Quantifier.UNIVERSAL, "item", "derived", 5, confidence=0.8,
        proof_verified=True, derivation_chain=[999],  # 999 doesn't exist
    )
    assert engine.verify_proof(r2.id) is False


def test_verify_proof_self_referencing():
    engine = LogicEngine(d_model=D_MODEL)
    emb = torch.randn(D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, emb)
    r = engine.add_rule(
        Quantifier.UNIVERSAL, "item", "derived", 5, confidence=0.8,
        proof_verified=True, derivation_chain=[0],  # references itself (id=0)
    )
    assert engine.verify_proof(r.id) is False


def test_verify_proof_non_derived_trivially_valid():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("item", VariableType.OBJECT, torch.randn(D_MODEL))
    r = engine.add_rule(Quantifier.UNIVERSAL, "item", "see(X)", 3, confidence=0.9)
    # Empirical rule (not derived) → trivially valid
    assert engine.verify_proof(r.id) is True


# =====================================================================
# LogicEngine — diagnostics + persistence
# =====================================================================


def test_get_rules_text():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)
    texts = engine.get_rules_text()
    assert len(texts) == 1
    assert "∀" in texts[0]
    assert "key" in texts[0]


def test_get_variables_text():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    texts = engine.get_variables_text()
    assert len(texts) == 1
    assert "key" in texts[0]


def test_summary():
    engine = LogicEngine(d_model=D_MODEL)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)
    s = engine.summary()
    assert s["num_rules"] == 1
    assert s["num_variables"] == 1
    assert s["mean_confidence"] == 0.9


def test_state_dict_roundtrip():
    engine = LogicEngine(d_model=D_MODEL, max_rules=16)
    engine.define_variable("key", VariableType.OBJECT, torch.randn(D_MODEL))
    engine.add_rule(Quantifier.UNIVERSAL, "key", "see(X)", 3, confidence=0.9)
    state = engine.state_dict()

    engine2 = LogicEngine(d_model=D_MODEL, max_rules=16)
    engine2.load_state_dict(state)
    assert len(engine2) == 1
    assert engine2.get_variable("key") is not None


def test_engine_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    engine = LogicEngine(d_model=D_MODEL, max_rules=8)
    assert isinstance(engine, BoundedComponent)
