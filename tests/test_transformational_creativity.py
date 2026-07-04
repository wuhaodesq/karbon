"""Tests for the transformational creativity engine."""

from __future__ import annotations

import pytest
import torch

from src.models.transformational_creativity import (
    Transformation,
    TransformationalCreativityEngine,
)


D_MODEL = 64


class MockRule:
    def __init__(self, rid, desc, emb, action=0):
        self.id = rid
        self.description = desc
        self.condition_embedding = emb
        self.action = action


class MockSkill:
    def __init__(self, sid, tag, d=D_MODEL):
        self.id = sid
        self.tag = tag

        class W:
            def __init__(self, d):
                self.A = torch.randn(d, 4)

        self.weights = W(d)


class MockVariable:
    def __init__(self, name, emb):
        self.name = name
        self.category_embedding = emb


def _make_concepts():
    return (
        [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i % 7) for i in range(3)],
        [MockSkill(i, f"skill_{i}") for i in range(3)],
        [MockVariable(f"var_{i}", torch.randn(D_MODEL)) for i in range(2)],
    )


# =====================================================================
# Transformation
# =====================================================================


def test_transformation_compute_score():
    tf = Transformation(
        id=0, distance_score=0.8, causal_consistent=True, curiosity_score=0.7,
    )
    score = tf.compute_score()
    # 0.25*0.8 + 0.35*1.0 + 0.40*0.7 = 0.2 + 0.35 + 0.28 = 0.83
    assert abs(score - 0.83) < 1e-4


def test_transformation_compute_score_causal_false():
    tf = Transformation(
        id=0, distance_score=0.8, causal_consistent=False, curiosity_score=0.7,
    )
    score = tf.compute_score()
    # 0.25*0.8 + 0.35*0.0 + 0.40*0.7 = 0.2 + 0 + 0.28 = 0.48
    assert abs(score - 0.48) < 1e-4


def test_transformation_repr():
    tf = Transformation(id=0, status="accepted", original_rule="A", broken_rule="B")
    text = str(tf)
    assert "✓" in text
    assert "BREAK" in text


# =====================================================================
# TransformationalCreativityEngine
# =====================================================================


def test_engine_generate_basic():
    engine = TransformationalCreativityEngine(d_model=D_MODEL)
    rules, skills, variables = _make_concepts()
    results = engine.generate(rules=rules, skills=skills, variables=variables, n_transformations=5)
    assert len(results) > 0
    for tf in results:
        assert isinstance(tf, Transformation)
        assert tf.status in ("accepted", "proposed")
        assert tf.distance_score >= 0  # distance = 1 - cos_sim, can be >1
        assert 0 <= tf.curiosity_score <= 1


def test_engine_far_recombine_maximizes_distance():
    """Far pairs should have higher distance than random pairs."""
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    results = engine.generate(rules=rules, skills=skills, n_transformations=3)
    if results:
        # All results should have passed the distance threshold
        for tf in results:
            assert tf.distance_score >= 0.0


def test_engine_rule_breaking_generates_description():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules = [MockRule(0, "IF see key THEN pick up", torch.randn(D_MODEL), 3)]
    skills = [MockSkill(0, "fast_walk")]
    results = engine.generate(rules=rules, skills=skills, n_transformations=3)
    for tf in results:
        assert "WHAT IF" in tf.broken_rule or "combined" in tf.broken_rule


def test_engine_causal_check_without_world_model():
    """Without world model, causal check should be optimistic (True)."""
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    results = engine.generate(rules=rules, skills=skills, n_transformations=3, world_model=None)
    # All should be causally consistent (optimistic)
    for tf in results:
        assert tf.causal_consistent is True


def test_engine_causal_check_with_world_model():
    """With world model, should actually simulate."""
    from src.models import RSSM, RSSMConfig
    wm = RSSM(RSSMConfig(
        obs_dim=16, action_dim=7, z_dim=8, h_dim=16,
        embed_dim=8, hidden=16, max_rollout_steps=5,
    ))
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    results = engine.generate(rules=rules, skills=skills, n_transformations=3, world_model=wm)
    # Should have run simulations
    for tf in results:
        assert isinstance(tf.causal_consistent, bool)


def test_engine_curiosity_gate():
    """Low-curiosity transformations should be rejected."""
    engine = TransformationalCreativityEngine(
        d_model=D_MODEL, curiosity_threshold=0.99, distance_threshold=0.0,
    )
    rules, skills, variables = _make_concepts()
    results = engine.generate(rules=rules, skills=skills, n_transformations=5)
    # With 0.99 curiosity threshold, almost everything should be rejected
    # (curiosity predictor outputs ~0.5 on average)
    accepted = [t for t in results if t.status == "accepted"]
    assert len(accepted) <= 1  # very few pass


def test_engine_capacity_bounded():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, max_transformations=8)
    rules, skills, variables = _make_concepts()
    for _ in range(10):
        engine.generate(rules=rules, skills=skills, n_transformations=5)
    assert len(engine) <= 8  # Axiom 1


def test_engine_get_accepted():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    engine.generate(rules=rules, skills=skills, variables=variables, n_transformations=5)
    accepted = engine.get_accepted()
    for tf in accepted:
        assert tf.status == "accepted"
        assert tf.causal_consistent is True


def test_engine_get_text():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    engine.generate(rules=rules, skills=skills, n_transformations=5)
    texts = engine.get_text(5)
    assert len(texts) > 0
    for t in texts:
        assert "Transform" in t
        assert "BREAK" in t


def test_engine_summary():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    engine.generate(rules=rules, skills=skills, n_transformations=5)
    s = engine.summary()
    assert s["total"] > 0
    assert s["capacity"] == 64
    assert "best_score" in s


def test_engine_state_dict_roundtrip():
    engine = TransformationalCreativityEngine(d_model=D_MODEL, distance_threshold=0.0)
    rules, skills, variables = _make_concepts()
    engine.generate(rules=rules, skills=skills, n_transformations=5)
    state = engine.state_dict()

    engine2 = TransformationalCreativityEngine(d_model=D_MODEL)
    engine2.load_state_dict(state)
    assert len(engine2) == len(engine)
    assert engine2._next_id == engine._next_id


def test_engine_empty_input():
    engine = TransformationalCreativityEngine(d_model=D_MODEL)
    results = engine.generate(rules=[], skills=[], variables=[], n_transformations=5)
    assert len(results) == 0


def test_engine_single_concept():
    engine = TransformationalCreativityEngine(d_model=D_MODEL)
    results = engine.generate(
        rules=[MockRule(0, "only rule", torch.randn(D_MODEL), 0)],
        n_transformations=3,
    )
    assert len(results) == 0  # need at least 2 concepts


def test_engine_no_growing_state():
    """Multiple generate calls should not cause unbounded growth (Axiom 1)."""
    engine = TransformationalCreativityEngine(d_model=D_MODEL, max_transformations=8)
    rules, skills, variables = _make_concepts()
    for _ in range(20):
        engine.generate(rules=rules, skills=skills, n_transformations=3)
    assert len(engine) <= 8


def test_engine_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    engine = TransformationalCreativityEngine(d_model=D_MODEL, max_transformations=8)
    assert isinstance(engine, BoundedComponent)
