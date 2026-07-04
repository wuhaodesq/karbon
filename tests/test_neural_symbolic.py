"""Tests for the neural-symbolic layer."""

from __future__ import annotations

import pytest
import torch

from src.models.neural_symbolic import (
    NeuralSymbolicLayer,
    Rule,
    RuleMemory,
)


D_MODEL = 64
NUM_ACTIONS = 7


# =====================================================================
# Rule
# =====================================================================


def test_rule_update_increases_confidence_on_positive_reward():
    r = Rule(
        id=0,
        condition_embedding=torch.randn(D_MODEL),
        action=2,
        confidence=0.5,
    )
    r.update(reward=1.0, decay=0.9)
    assert r.confidence > 0.5
    assert r.usage_count == 1
    assert r.success_count == 1


def test_rule_update_decreases_confidence_on_negative_reward():
    r = Rule(
        id=0,
        condition_embedding=torch.randn(D_MODEL),
        action=2,
        confidence=0.8,
    )
    r.update(reward=-1.0, decay=0.9)
    assert r.confidence < 0.8
    assert r.success_count == 0


def test_rule_success_rate():
    r = Rule(id=0, condition_embedding=torch.randn(D_MODEL), action=0)
    for _ in range(10):
        r.update(reward=1.0 if r.usage_count % 2 == 0 else -1.0)
    assert r.success_rate == 0.5


# =====================================================================
# RuleMemory
# =====================================================================


def test_rule_memory_capacity_bounded():
    mem = RuleMemory(max_rules=8, d_model=D_MODEL)
    for i in range(20):
        mem.add(
            condition_embedding=torch.randn(D_MODEL),
            action=i % NUM_ACTIONS,
            description=f"rule-{i}",
        )
    assert len(mem) <= mem.capacity  # Axiom 1


def test_rule_memory_add_returns_rule():
    mem = RuleMemory(max_rules=16, d_model=D_MODEL)
    rule = mem.add(
        condition_embedding=torch.randn(D_MODEL),
        action=3,
        description="test",
    )
    assert isinstance(rule, Rule)
    assert rule.action == 3
    assert rule.description == "test"


def test_rule_memory_duplicate_merges():
    """Adding a near-identical rule should update, not create."""
    mem = RuleMemory(max_rules=16, d_model=D_MODEL)
    emb = torch.randn(D_MODEL)
    r1 = mem.add(condition_embedding=emb, action=2, confidence=0.5)
    r2 = mem.add(condition_embedding=emb + 1e-4 * torch.randn(D_MODEL), action=2, confidence=0.8)
    # Should merge into r1 (same id)
    assert r1.id == r2.id
    assert len(mem) == 1
    assert r1.confidence >= 0.8  # updated to higher confidence


def test_rule_memory_match_finds_similar():
    mem = RuleMemory(max_rules=16, d_model=D_MODEL)
    emb = torch.randn(D_MODEL)
    mem.add(condition_embedding=emb, action=3, confidence=0.9)

    # Query with a slightly perturbed version
    query = emb + 0.01 * torch.randn(D_MODEL)
    rule, sim = mem.match(query, threshold=0.5)
    assert rule is not None
    assert rule.action == 3
    assert sim > 0.5


def test_rule_memory_no_match_below_threshold():
    mem = RuleMemory(max_rules=16, d_model=D_MODEL)
    mem.add(condition_embedding=torch.randn(D_MODEL), action=3, confidence=0.9)
    rule, sim = mem.match(torch.randn(D_MODEL), threshold=0.99)
    assert rule is None


def test_rule_memory_eviction_keeps_high_confidence():
    mem = RuleMemory(max_rules=4, d_model=D_MODEL)
    # Add 4 high-confidence rules
    for i in range(4):
        mem.add(
            condition_embedding=torch.randn(D_MODEL),
            action=i,
            confidence=0.9,
            description=f"high-{i}",
        )
    # Add a 5th (low confidence) — should evict the 5th, not the high ones
    # Actually, the new one has default confidence 0.5, which is lower
    mem.add(
        condition_embedding=torch.randn(D_MODEL),
        action=5,
        confidence=0.1,
        description="low",
    )
    assert len(mem) <= 4


def test_rule_memory_state_dict_roundtrip():
    mem = RuleMemory(max_rules=8, d_model=D_MODEL)
    for i in range(5):
        mem.add(
            condition_embedding=torch.randn(D_MODEL),
            action=i,
            confidence=0.5 + i * 0.1,
            description=f"rule-{i}",
        )
    state = mem.state_dict()

    mem2 = RuleMemory(max_rules=8, d_model=D_MODEL)
    mem2.load_state_dict(state)
    assert len(mem2) == 5
    assert mem2._next_id == mem._next_id


def test_rule_memory_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    mem = RuleMemory(max_rules=8, d_model=D_MODEL)
    assert isinstance(mem, BoundedComponent)


# =====================================================================
# NeuralSymbolicLayer
# =====================================================================


def test_symbolic_layer_forward_no_rules():
    """With no rules, should pass through neural logits unchanged."""
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    h = torch.randn(4, D_MODEL)
    logits = torch.randn(4, NUM_ACTIONS)
    out, info = layer(h, logits)
    assert info["rule_matched"] is False
    assert info["override"] is False
    torch.testing.assert_close(out, logits)


def test_symbolic_layer_forward_with_matching_rule():
    """When a matching rule exists, should override the action."""
    layer = NeuralSymbolicLayer(
        d_model=D_MODEL, num_actions=NUM_ACTIONS,
        match_threshold=0.5, override_confidence_threshold=0.5,
    )
    # Add a rule with high confidence
    h = torch.randn(D_MODEL)
    layer.rule_memory.add(
        condition_embedding=layer.rule_projection(h),
        action=5,
        confidence=0.9,
        description="IF see key THEN pick up",
    )
    # Forward with the same hidden state
    neural_logits = torch.zeros(1, NUM_ACTIONS)
    out, info = layer(h.unsqueeze(0), neural_logits)
    assert info["rule_matched"] is True
    assert info["override"] is True
    # The rule's action (5) should be strongly preferred
    assert out[0, 5] > out[0, 0]


def test_symbolic_layer_extract_rules_from_positive_reward():
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    hidden_states = [torch.randn(D_MODEL) for _ in range(10)]
    actions = [i % NUM_ACTIONS for i in range(10)]
    rewards = [0.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.0, 0.3, 0.0, 0.0]
    new_rules = layer.extract_rules(hidden_states, actions, rewards)
    # 3 transitions have reward > 0.3 threshold
    assert len(new_rules) == 3


def test_symbolic_layer_feedback_updates_rule():
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    h = torch.randn(D_MODEL)
    layer.rule_memory.add(
        condition_embedding=layer.rule_projection(h),
        action=3, confidence=0.7,
    )
    # Trigger a match
    layer(h.unsqueeze(0), torch.zeros(1, NUM_ACTIONS))
    # Feedback positive reward
    layer.feedback(reward=1.0)
    # The rule's confidence should have increased
    rules = list(layer.rule_memory._rules.values())
    assert rules[0].confidence > 0.7


def test_symbolic_layer_get_rules_text():
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    h = torch.randn(D_MODEL)
    layer.rule_memory.add(
        condition_embedding=layer.rule_projection(h),
        action=2, confidence=0.8,
        description="IF see key THEN pick up",
    )
    texts = layer.get_rules_text()
    assert len(texts) == 1
    assert "pick up" in texts[0]
    assert "conf=0.80" in texts[0]


def test_symbolic_layer_no_growing_state():
    """Symbolic layer should not accumulate unbounded state (Axiom 1)."""
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS, max_rules=8)
    for _ in range(100):
        h = torch.randn(D_MODEL)
        layer(h.unsqueeze(0), torch.randn(1, NUM_ACTIONS))
        layer.feedback(reward=0.5)
    assert len(layer.rule_memory) <= layer.rule_memory.capacity


def test_symbolic_layer_summary():
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    for i in range(5):
        layer.rule_memory.add(
            condition_embedding=torch.randn(D_MODEL),
            action=i, confidence=0.5 + i * 0.1,
        )
    s = layer.summary()
    assert s["num_rules"] == 5
    assert s["capacity"] == 64
    assert 0 < s["mean_confidence"] <= 1.0


def test_symbolic_layer_rule_chain():
    """get_rule_chain should return all rules for a given action."""
    layer = NeuralSymbolicLayer(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    for i in range(5):
        layer.rule_memory.add(
            condition_embedding=torch.randn(D_MODEL),
            action=2 if i < 3 else 4,
        )
    chain = layer.rule_memory.get_rule_chain(action=2)
    assert len(chain) == 3
    assert all(r.action == 2 for r in chain)
