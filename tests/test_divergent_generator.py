"""Tests for the divergent generator (combinational creativity)."""

from __future__ import annotations

import pytest
import torch

from src.models.divergent_generator import CreativeIdea, DivergentGenerator


D_MODEL = 64


# =====================================================================
# CreativeIdea
# =====================================================================


def test_creative_idea_overall_score():
    idea = CreativeIdea(
        id=0, description="test",
        novelty_score=0.8, coherence_score=0.6, feasibility_score=0.7,
    )
    # 0.4*0.8 + 0.3*0.6 + 0.3*0.7 = 0.32 + 0.18 + 0.21 = 0.71
    assert abs(idea.overall_score - 0.71) < 1e-4


def test_creative_idea_repr():
    idea = CreativeIdea(id=0, description="combine A and B")
    text = str(idea)
    assert "Idea #0" in text
    assert "combine A and B" in text


# =====================================================================
# DivergentGenerator
# =====================================================================


# Mock rule objects
class MockRule:
    def __init__(self, rid, desc, emb, action=0):
        self.id = rid
        self.description = desc
        self.condition_embedding = emb
        self.action = action


class MockSkill:
    def __init__(self, sid, tag, d_model=D_MODEL):
        self.id = sid
        self.tag = f"skill_{tag}"

        class MockWeights:
            def __init__(self, d):
                self.A = torch.randn(d, 4)

        self.weights = MockWeights(d_model)


def test_divergent_generator_generate_with_rules():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [
        MockRule(0, "IF see key THEN pick up", torch.randn(D_MODEL), action=3),
        MockRule(1, "IF see door THEN open", torch.randn(D_MODEL), action=5),
        MockRule(2, "IF see wall THEN turn", torch.randn(D_MODEL), action=1),
    ]
    ideas = gen.generate(rules=rules, n_ideas=9)
    assert len(ideas) > 0
    # Each idea should have a description and scores
    for idea in ideas:
        assert len(idea.description) > 0
        assert 0 <= idea.novelty_score <= 1
        assert 0 <= idea.coherence_score <= 1
        assert 0 <= idea.feasibility_score <= 1


def test_divergent_generator_generate_with_skills():
    gen = DivergentGenerator(d_model=D_MODEL)
    skills = [
        MockSkill(0, "walk"),
        MockSkill(1, "turn"),
        MockSkill(2, "pickup"),
    ]
    ideas = gen.generate(skills=skills, n_ideas=6)
    assert len(ideas) > 0


def test_divergent_generator_generate_cross_domain():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [
        MockRule(0, "IF see key THEN pick up", torch.randn(D_MODEL), action=3),
        MockRule(1, "IF see door THEN open", torch.randn(D_MODEL), action=5),
    ]
    skills = [
        MockSkill(0, "walk"),
        MockSkill(1, "turn"),
    ]
    ideas = gen.generate(rules=rules, skills=skills, n_ideas=9)
    assert len(ideas) > 0
    # Should have some cross-domain ideas
    cross_domain = [i for i in ideas if i.source_rules and i.source_skills]
    assert len(cross_domain) > 0


def test_divergent_generator_generate_empty():
    gen = DivergentGenerator(d_model=D_MODEL)
    ideas = gen.generate(rules=[], skills=[], n_ideas=10)
    assert len(ideas) == 0


def test_divergent_generator_filter():
    gen = DivergentGenerator(d_model=D_MODEL)
    ideas = [
        CreativeIdea(id=0, description="A", novelty_score=0.8, coherence_score=0.7, feasibility_score=0.6),
        CreativeIdea(id=1, description="B", novelty_score=0.2, coherence_score=0.8, feasibility_score=0.9),
        CreativeIdea(id=2, description="C", novelty_score=0.9, coherence_score=0.6, feasibility_score=0.5),
        CreativeIdea(id=3, description="D", novelty_score=0.1, coherence_score=0.1, feasibility_score=0.1),
    ]
    filtered = gen.filter(ideas, top_k=2, min_novelty=0.3, min_coherence=0.3)
    # Should keep ideas 0 and 2 (both pass filters, 0 has higher score)
    assert len(filtered) == 2
    # Idea 1 has low novelty (0.2 < 0.3) → filtered out
    # Idea 3 has everything low → filtered out
    assert filtered[0].overall_score >= filtered[1].overall_score


def test_divergent_generator_filter_removes_low_novelty():
    gen = DivergentGenerator(d_model=D_MODEL)
    ideas = [
        CreativeIdea(id=0, description="A", novelty_score=0.1, coherence_score=0.9, feasibility_score=0.9),
        CreativeIdea(id=1, description="B", novelty_score=0.8, coherence_score=0.5, feasibility_score=0.5),
    ]
    filtered = gen.filter(ideas, min_novelty=0.3)
    assert len(filtered) == 1
    assert filtered[0].id == 1  # only high-novelty idea passes


def test_divergent_generator_history_bounded():
    gen = DivergentGenerator(d_model=D_MODEL, max_history=8)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i % 7) for i in range(5)]
    for _ in range(10):
        gen.generate(rules=rules, n_ideas=6)
    assert len(gen) <= 8  # Axiom 1


def test_divergent_generator_get_best_ideas():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i % 7) for i in range(4)]
    gen.generate(rules=rules, n_ideas=12)
    best = gen.get_best_ideas(n=3)
    assert len(best) <= 3
    # Should be sorted by overall_score descending
    for i in range(len(best) - 1):
        assert best[i].overall_score >= best[i + 1].overall_score


def test_divergent_generator_get_ideas_text():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i) for i in range(3)]
    gen.generate(rules=rules, n_ideas=6)
    texts = gen.get_ideas_text(3)
    assert len(texts) > 0
    assert all("Idea" in t for t in texts)


def test_divergent_generator_summary():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i) for i in range(3)]
    gen.generate(rules=rules, n_ideas=6)
    s = gen.summary()
    assert s["total_ideas"] > 0
    assert s["capacity"] > 0
    assert "mean_novelty" in s
    assert "best_score" in s


def test_divergent_generator_state_dict_roundtrip():
    gen = DivergentGenerator(d_model=D_MODEL)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i) for i in range(3)]
    gen.generate(rules=rules, n_ideas=6)
    state = gen.state_dict()

    gen2 = DivergentGenerator(d_model=D_MODEL)
    gen2.load_state_dict(state)
    assert len(gen2) == len(gen)
    assert gen2._next_id == gen._next_id


def test_divergent_generator_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    gen = DivergentGenerator(d_model=D_MODEL, max_history=8)
    assert isinstance(gen, BoundedComponent)


def test_divergent_generator_novelty_scorer_gradient():
    """The novelty scorer should be trainable (for future improvement)."""
    gen = DivergentGenerator(d_model=D_MODEL)
    emb_a = torch.randn(D_MODEL)
    emb_b = torch.randn(D_MODEL)
    combined = torch.cat([emb_a, emb_b], dim=-1).unsqueeze(0)
    score = gen.novelty_scorer(combined)
    score.backward()
    for p in gen.novelty_scorer.parameters():
        assert p.grad is not None


def test_divergent_generator_no_growing_state():
    """Generating many ideas should not cause unbounded growth (Axiom 1)."""
    gen = DivergentGenerator(d_model=D_MODEL, max_history=8)
    rules = [MockRule(i, f"rule_{i}", torch.randn(D_MODEL), i) for i in range(3)]
    for _ in range(50):
        gen.generate(rules=rules, n_ideas=5)
        gen.get_best_ideas(3)
    assert len(gen) <= 8
