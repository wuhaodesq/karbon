"""Tests for the independent evaluator's metric plumbing (no envs needed)."""

from __future__ import annotations

from src.eval.independent_evaluator import IndependentEvaluator


class _RuleMemoryStub:
    def __init__(self, summary: dict):
        self._summary = summary

    def summary(self) -> dict:
        return self._summary


class _SymbolicStub:
    def __init__(self, summary: dict):
        self.rule_memory = _RuleMemoryStub(summary)


def test_symbolic_uses_rule_memory_usage_keys():
    """Regression: RuleMemory.summary() reports total_success/total_usage,
    but the scorer used to read total_matches/total_queries and therefore
    silently reported 0.000 forever (380/380 stage-20 evals)."""
    stub = _SymbolicStub({"total_usage": 10, "total_success": 7})
    assert abs(IndependentEvaluator._measure_symbolic(stub) - 0.7) < 1e-6


def test_symbolic_zero_usage_is_zero():
    stub = _SymbolicStub({"total_usage": 0, "total_success": 0})
    assert IndependentEvaluator._measure_symbolic(stub) == 0.0


def test_symbolic_legacy_keys_still_supported():
    stub = _SymbolicStub({"total_matches": 3, "total_queries": 6})
    assert abs(IndependentEvaluator._measure_symbolic(stub) - 0.5) < 1e-6


def test_symbolic_missing_layer_is_zero():
    assert IndependentEvaluator._measure_symbolic(None) == 0.0
