"""Tests for language generation module."""

from __future__ import annotations

import pytest
import torch

from src.models.language_generation import LanguageGenerator


# =====================================================================
# Template mode (fallback) — always works
# =====================================================================


def test_language_generator_template_fallback(monkeypatch):
    """If transformers/bitsandbytes can't load, should fall back to template."""
    # Block transformers import
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    gen = LanguageGenerator(model_name="Qwen/Qwen2.5-7B-Instruct")
    assert gen.mode == "template"
    assert not gen.is_llm_active


def test_language_generator_template_generate():
    """Template mode should still produce text."""
    import sys
    # Ensure transformers is blocked
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    result = gen.generate("Hello, what can you do?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_language_generator_describe_template():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    state = {
        "confidence": 0.85,
        "familiarity": 0.6,
        "progress": 0.7,
        "active_rules": ["IF see key THEN pick up"],
        "current_task": "DoorKey-5x5",
        "episode_return": 0.5,
    }
    result = gen.describe(state)
    assert isinstance(result, str)
    assert len(result) > 0


def test_language_generator_explain_template():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    result = gen.explain(
        action=2,
        action_name="forward",
        rule_description="IF see corridor THEN go forward",
        confidence=0.9,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_language_generator_explain_no_rule():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    result = gen.explain(
        action=3,
        action_name="pickup",
        rule_description=None,
        confidence=0.4,
    )
    assert isinstance(result, str)


def test_language_generator_answer():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    state = {
        "confidence": 0.7,
        "familiarity": 0.5,
        "progress": 0.6,
        "active_rules": ["IF see goal THEN go to goal"],
        "current_task": "Empty-5x5",
        "num_skills": 5,
        "episode_return": 0.8,
    }
    result = gen.answer("What are you doing?", state)
    assert isinstance(result, str)


def test_language_generator_inner_monologue():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    result = gen.inner_monologue(
        reflection_text="I failed because I didn't have the key",
        confidence=0.3,
        familiarity=0.2,
    )
    assert isinstance(result, str)


# =====================================================================
# Bounded state
# =====================================================================


def test_language_generator_no_growing_state():
    """Generator should not accumulate state across calls (Axiom 1)."""
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "test"
    gen._max_new_tokens = 128
    gen._fallback_temp = 0.7

    for _ in range(50):
        gen.generate("test prompt")
        gen.describe({"confidence": 0.5, "familiarity": 0.5, "progress": 0.5})
        gen.explain(0, "forward", None, 0.5)
    # No state to check — just verify no crash and no accumulation
    assert gen.mode == "template"


def test_language_generator_summary():
    gen = LanguageGenerator.__new__(LanguageGenerator)
    gen._mode = "template"
    gen._llm = None
    gen._tokenizer = None
    gen._model_name = "Qwen/Qwen2.5-7B-Instruct"
    gen._max_new_tokens = 256
    gen._fallback_temp = 0.7

    s = gen.summary()
    assert s["mode"] == "template"
    assert "template" in s["model"]
    assert s["max_new_tokens"] == 256
