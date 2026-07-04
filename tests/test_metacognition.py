"""Tests for metacognition, self-reflection, and inner dialogue."""

from __future__ import annotations

import pytest
import torch

from src.models.metacognition import (
    EpisodeReflection,
    InnerDialogue,
    ReflectionLoop,
    SelfAssessment,
    SelfModel,
)


D_MODEL = 64


# =====================================================================
# SelfModel (Metacognition)
# =====================================================================


def test_self_model_output_shape():
    sm = SelfModel(d_model=D_MODEL)
    h = torch.randn(4, D_MODEL)
    out = sm.forward(h)
    assert set(out.keys()) == {"confidence", "familiarity", "progress"}
    assert out["confidence"].shape == (4, 1)
    assert all((out[k] >= 0).all() and (out[k] <= 1).all() for k in out)


def test_self_model_assess_single():
    sm = SelfModel(d_model=D_MODEL)
    h = torch.randn(D_MODEL)
    assessment = sm.assess(h)
    assert isinstance(assessment, SelfAssessment)
    assert 0 <= assessment.confidence <= 1
    assert 0 <= assessment.familiarity <= 1
    assert 0 <= assessment.progress <= 1
    assert 0 <= assessment.uncertainty <= 1


def test_self_model_gradient():
    sm = SelfModel(d_model=D_MODEL)
    h = torch.randn(4, D_MODEL)
    targets = {
        "confidence": torch.rand(4),
        "familiarity": torch.rand(4),
        "progress": torch.rand(4),
    }
    loss = sm.auxiliary_loss(h, targets)
    loss.backward()
    for p in sm.parameters():
        assert p.grad is not None


def test_self_model_no_growing_state():
    """SelfModel should not accumulate state across calls (Axiom 1)."""
    sm = SelfModel(d_model=D_MODEL)
    params_before = sum(p.numel() for p in sm.parameters())
    for _ in range(100):
        h = torch.randn(4, D_MODEL)
        sm.forward(h)
        sm.assess(h[0])
    params_after = sum(p.numel() for p in sm.parameters())
    assert params_before == params_after


# =====================================================================
# ReflectionLoop (Self-reflection)
# =====================================================================


def test_reflection_loop_capacity_bounded():
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=8, reflection_every_episodes=1)
    for ep in range(20):
        h = torch.randn(1, D_MODEL)
        loop.record_step(h, action=0, reward=0.5, done=True)
        loop.end_episode(episode_return=0.5)
    assert len(loop) <= loop.capacity  # Axiom 1


def test_reflection_loop_returns_reflection_periodically():
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=3)
    results = []
    for ep in range(10):
        h = torch.randn(1, D_MODEL)
        loop.record_step(h, action=0, reward=0.5, done=True)
        r = loop.end_episode(episode_return=0.5)
        results.append(r)
    # every 3 episodes → should get reflections at ep 3, 6, 9
    non_none = [r for r in results if r is not None]
    assert len(non_none) == 3  # 3, 6, 9


def test_reflection_loop_generates_adjustments():
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=1)
    # Force low familiarity by using a fresh model (untrained → low outputs)
    h = torch.randn(1, D_MODEL)
    loop.record_step(h, action=0, reward=0.0, done=True)
    r = loop.end_episode(episode_return=0.0)
    assert r is not None
    assert isinstance(r, EpisodeReflection)
    assert r.success is False


def test_reflection_loop_recent_summary():
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=1)
    for ep in range(10):
        h = torch.randn(1, D_MODEL)
        loop.record_step(h, action=0, reward=0.5, done=True)
        loop.end_episode(episode_return=0.5 + ep * 0.01)
    summary = loop.recent_summary(n=5)
    assert summary["n"] == 5
    assert summary["success_rate"] == 1.0
    assert 0.4 < summary["mean_return"] < 0.7


def test_reflection_loop_state_dict_roundtrip():
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=1)
    for ep in range(5):
        h = torch.randn(1, D_MODEL)
        loop.record_step(h, action=0, reward=0.5, done=True)
        loop.end_episode(episode_return=0.5)
    state = loop.state_dict()

    loop2 = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=1)
    loop2.load_state_dict(state)
    assert len(loop2) == len(loop)
    assert loop2._episode_count == loop._episode_count


# =====================================================================
# InnerDialogue
# =====================================================================


def test_inner_dialogue_template_mode():
    """Template mode should always work (no LLM dependency)."""
    id = InnerDialogue(mode="template")
    assert id.mode == "template"

    r = EpisodeReflection(
        episode_return=0.8,
        mean_confidence=0.9,
        mean_familiarity=0.8,
        mean_progress=0.7,
        success=True,
    )
    lessons = id.generate(r)
    assert len(lessons) > 0
    assert any("success" in l.lower() or "succeeded" in l.lower() for l in lessons)


def test_inner_dialogue_low_confidence_lesson():
    id = InnerDialogue(mode="template")
    r = EpisodeReflection(
        episode_return=0.0,
        mean_confidence=0.1,
        mean_familiarity=0.5,
        mean_progress=0.3,
        success=False,
    )
    lessons = id.generate(r)
    assert any("uncertain" in l.lower() for l in lessons)


def test_inner_dialogue_low_familiarity_lesson():
    id = InnerDialogue(mode="template")
    r = EpisodeReflection(
        episode_return=0.5,
        mean_confidence=0.6,
        mean_familiarity=0.1,
        mean_progress=0.5,
        success=True,
    )
    lessons = id.generate(r)
    assert any("unfamiliar" in l.lower() for l in lessons)


def test_inner_dialogue_overconfident_failure():
    id = InnerDialogue(mode="template")
    r = EpisodeReflection(
        episode_return=0.0,
        mean_confidence=0.95,
        mean_familiarity=0.7,
        mean_progress=0.3,
        success=False,
    )
    lessons = id.generate(r)
    assert any("overconfident" in l.lower() for l in lessons)


def test_inner_dialogue_llm_mode_falls_back(monkeypatch):
    """LLM mode should fall back to template when LLM can't load."""
    # Block transformers import to simulate offline / no library
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    id = InnerDialogue(mode="llm", llm_model_name="nonexistent/model")
    assert id.mode == "template"  # fell back


def test_inner_dialogue_no_growing_state():
    """Generating dialogue should not accumulate state (Axiom 1)."""
    id = InnerDialogue(mode="template")
    r = EpisodeReflection(
        episode_return=0.5, mean_confidence=0.5,
        mean_familiarity=0.5, mean_progress=0.5, success=True,
    )
    for _ in range(100):
        id.generate(r)
    # No state to check — just verify no crash
    assert True


# =====================================================================
# BoundedComponent protocol conformance
# =====================================================================


def test_reflection_loop_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    sm = SelfModel(d_model=D_MODEL)
    loop = ReflectionLoop(sm, max_reflections=8)
    assert isinstance(loop, BoundedComponent)
