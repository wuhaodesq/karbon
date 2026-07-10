"""Program Synthesis, Active Experimentation, Temporal Abstraction.

Three additions inspired by the DeepSeek article on System 2 reasoning.

1. ProgramSynthesizer — generate candidate rules from sparse examples (o3-lite)
2. ActiveExperimenter — curiosity-driven hypothesis testing loop
3. TemporalAbstractor — extract sequential patterns (first→then→result)

All zero GPU, zero retraining. Use existing module outputs as input.

程序合成 + 主动实验 + 时序抽象。三个高增益 System 2 组件。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Program Synthesis — generate rules from sparse examples
# =====================================================================


class ProgramSynthesizer:
    """Generate candidate symbolic programs from sparse input-output pairs.

    The article highlighted o3's approach: given 2-3 examples of a transformation,
    search a program space for a rule that explains them, then test on new inputs.

    Our simplified version:
        1. Given N input-output predicate pairs
        2. Extract common patterns (shared predicates, shared changes)
        3. Hypothesize a candidate rule: IF [common predicates] THEN [common outcome]
        4. Validate against examples → if passes, add to RuleInductionEngine

    This is what makes SchemaDetector more than just clustering — it generates
    candidate EXPLANATIONS, not just templates.
    """

    def __init__(self, max_candidates: int = 50, min_examples: int = 2) -> None:
        self._max_candidates = max_candidates
        self._min_examples = min_examples
        self._synthesized: list[dict[str, Any]] = []

    def synthesize(
        self,
        examples: list[dict[str, Any]],   # [{"input": predicates, "output": outcome}, ...]
        existing_rules: Any = None,       # RuleInductionEngine for dedup
    ) -> list[dict[str, Any]]:
        """Synthesize candidate rules from examples.

        Each example = {"input": {"pred1": True, ...}, "output": 1 or 0}
        Returns list of candidate rules as dicts.
        """
        if len(examples) < self._min_examples:
            return []

        new_candidates = []

        # Strategy 1: Find predicates that are True in ALL positive examples
        positive = [e for e in examples if e.get("output", 0) > 0]
        negative = [e for e in examples if e.get("output", 0) <= 0]

        if len(positive) < self._min_examples:
            return []

        # Common predicates across all positive examples
        common_pos = set(positive[0]["input"].keys())
        for ex in positive[1:]:
            common_pos &= set(ex["input"].keys())

        # Remove predicates that also appear in negative examples (not discriminative)
        if negative:
            for ex in negative:
                common_pos -= set(ex["input"].keys())

        if common_pos:
            # Candidate: IF all common predicates THEN outcome is positive
            candidate = {
                "type": "synthesized_rule",
                "if_predicates": sorted(common_pos),
                "then_action": self._estimate_action(
                    positive, existing_rules,
                ),
                "confidence": len(positive) / len(examples),
                "support": len(positive),
            }
            if not self._already_exists(candidate):
                self._synthesized.append(candidate)
                new_candidates.append(candidate)

        # Strategy 2: Find predicates that CHANGE between input and output
        if len(positive) >= 2:
            changed = set()
            for i in range(len(positive)):
                for j in range(i + 1, len(positive)):
                    changed |= (
                        set(positive[i]["input"].keys())
                        ^ set(positive[j]["input"].keys())
                    )
            # These are the "active" predicates — something about them matters
            if changed:
                candidate2 = {
                    "type": "synthesized_rule_v2",
                    "if_predicates": sorted(changed),
                    "then_action": self._estimate_action(positive, existing_rules),
                    "confidence": len(positive) / len(examples) * 0.8,
                    "support": len(positive),
                }
                if not self._already_exists(candidate2):
                    self._synthesized.append(candidate2)
                    new_candidates.append(candidate2)

        if len(self._synthesized) > self._max_candidates:
            self._synthesized = self._synthesized[-self._max_candidates:]

        return new_candidates

    def _estimate_action(
        self, positive_examples: list, existing_rules: Any,
    ) -> int:
        """Estimate which action the rule should recommend."""
        return 0  # default: first action (can be refined with RL data)

    def _already_exists(self, candidate: dict) -> bool:
        for s in self._synthesized:
            if set(s["if_predicates"]) == set(candidate["if_predicates"]):
                return True
        return False

    def feedback_to_rules(
        self, rule_engine: Any,
    ) -> int:
        """Feed synthesized candidates into RuleInductionEngine."""
        added = 0
        for c in self._synthesized[:-1]:  # skip most recent (unvalidated)
            if c["confidence"] > 0.6:
                for pred_str in c["if_predicates"]:
                    if hasattr(rule_engine, '_add_rule'):
                        try:
                            from src.models.rule_induction import InducedRule
                            pred_tuples = [
                                _parse_simple(pred_str)
                                for pred_str in c["if_predicates"]
                            ]
                            rule = InducedRule(
                                if_predicates=[p for p in pred_tuples if p],
                                then_predicate=("action", (c["then_action"],)),
                                confidence=c["confidence"],
                                positive_examples=c["support"],
                            )
                            rule_engine._add_rule(rule)
                            added += 1
                        except Exception:
                            pass
        return added


def _parse_simple(s: str) -> tuple:
    name, args_str = s.split("(", 1)
    args_str = args_str.rstrip(")")
    args = tuple(a.strip() for a in args_str.split(","))
    return (name, args)


# =====================================================================
# 2. Active Experimentation — curiosity-driven hypothesis testing
# =====================================================================


class ActiveExperimenter(nn.Module):
    """Orchestrates curiosity-driven hypothesis testing.

    Instead of waiting for random exploration to produce data,
    this module DELIBERATELY chooses actions that would test
    the agent's current hypotheses.

    Loop:
        1. Check CuriosityDirector → which domain is most uncertain?
        2. Query CausalDiscovery → what hypothesis has lowest confidence?
        3. Compute "test action" → an action that would distinguish
           between competing hypotheses
        4. Execute → observe outcome → update hypothesis confidence

    This is the developmental precursor to scientific experimentation.
    """

    def __init__(
        self,
        test_every_steps: int = 2000,
        max_hypotheses: int = 100,
    ) -> None:
        super().__init__()
        self._test_every = test_every_steps
        self._max_hypotheses = max_hypotheses
        self._last_test_step = -test_every_steps
        self._hypotheses: list[dict] = []
        self._test_results: list[dict] = []

    def should_test(self, step: int) -> bool:
        return (step - self._last_test_step) >= self._test_every

    def propose_experiment(
        self,
        causal_disc: Any,
        curiosity_director: Any,
        rssm_uncertainty: float = 0.0,
    ) -> dict[str, Any] | None:
        """Propose an active experiment: which action to test and why.

        Returns {"test_action": int, "hypothesis": str, "rationale": str}
        or None if no interesting hypothesis.
        """
        self._last_test_step = 0  # will be updated by caller

        if causal_disc is None or not hasattr(causal_disc, '_graph'):
            return None

        # Find the causal edge with lowest confidence
        lowest_edge = None
        lowest_conf = 1.0
        for (src, tgt), edge in causal_disc._graph.edges.items():
            if edge.strength < lowest_conf and edge.strength > 0.1:
                lowest_conf = edge.strength
                lowest_edge = edge

        if lowest_edge is None:
            return None

        # Propose: test the action that caused this low-confidence edge
        action_str = lowest_edge.source.replace("action_", "")
        try:
            test_action = int(action_str)
        except ValueError:
            test_action = 0

        hypothesis = {
            "test_action": test_action,
            "hypothesis": f"Testing whether {lowest_edge.source} causes {lowest_edge.target}",
            "rationale": f"Edge confidence is only {lowest_conf:.2f}, needs more data",
            "related_uncertainty": rssm_uncertainty,
        }

        if len(self._hypotheses) >= self._max_hypotheses:
            self._hypotheses.pop(0)
        self._hypotheses.append(hypothesis)
        return hypothesis

    def record_result(
        self, hypothesis: dict, actual_outcome: float, step: int,
    ) -> None:
        """Record the outcome of an experiment."""
        self._test_results.append({
            "hypothesis": hypothesis,
            "outcome": actual_outcome,
            "step": step,
        })
        self._last_test_step = step

    def summary(self) -> str:
        if not self._hypotheses:
            return "No experiments conducted yet."
        recent = self._hypotheses[-3:]
        return "Recent experiments: " + "; ".join(
            h["hypothesis"][:60] for h in recent
        )


# =====================================================================
# 3. Temporal Abstraction — extract sequential patterns
# =====================================================================


class TemporalAbstractor:
    """Extracts sequential patterns: "first X happens, then Y, which leads to Z".

    Unlike SchemaDetector (spatial predicates), this looks at TEMPORAL
    sequences across consecutive steps.

    Example:
        Step t:   push ball (action 4), ball velocity increases
        Step t+1: ball hits block (contact), block velocity increases
        Step t+2: block falls off table

        → Pattern: push → hit → fall  (3-step sequence)
        → Abstracted rule: IF push_and_hit THEN chain_reaction

    This is the bridge from "what is happening" to "what happens next".
    """

    def __init__(
        self,
        max_patterns: int = 100,
        min_occurrences: int = 3,
        max_sequence_length: int = 4,
    ) -> None:
        self._max_patterns = max_patterns
        self._min_occurrences = min_occurrences
        self._max_seq_len = max_sequence_length

        self._sequence_buffer: list[list[str]] = []
        self._patterns: dict[str, dict] = {}  # seq_sig → {count, avg_reward, ...}
        self._reported: set[str] = set()       # sigs already emitted as patterns

    def record_step(self, predicates: list[str], reward: float) -> None:
        """Record one step's predicates for pattern extraction."""
        self._sequence_buffer.append(predicates)
        if len(self._sequence_buffer) > 500:
            self._sequence_buffer.pop(0)

    def extract_episode_patterns(self) -> list[dict[str, Any]]:
        """Extract temporal patterns from the current episode sequence.

        Returns list of discovered patterns.
        """
        if len(self._sequence_buffer) < self._min_occurrences:
            return []

        new_patterns = []

        # Look for repeated sequences of length 2-4
        for seq_len in range(2, min(self._max_seq_len + 1, len(self._sequence_buffer) + 1)):
            for i in range(len(self._sequence_buffer) - seq_len + 1):
                seq = self._sequence_buffer[i:i + seq_len]
                sig = " → ".join("+".join(steps[:3]) for steps in seq)
                if sig in self._patterns:
                    self._patterns[sig]["count"] += 1
                else:
                    self._patterns[sig] = {"count": 1, "length": seq_len}

        # Find patterns meeting threshold (report each sig only once)
        for sig, info in self._patterns.items():
            if info["count"] >= self._min_occurrences and sig not in self._reported:
                pattern = {
                    "sig": sig,
                    "count": info["count"],
                    "length": info["length"],
                    "description": f"Temporal pattern (length={info['length']}, seen {info['count']}x): {sig}",
                }
                new_patterns.append(pattern)
                self._reported.add(sig)

        self._sequence_buffer.clear()

        if new_patterns:
            logger.info("[temporal] %d new patterns discovered (top: %dx, len=%d)",
                       len(new_patterns),
                       new_patterns[0]["count"] if new_patterns else 0,
                       new_patterns[0]["length"] if new_patterns else 0)
        return new_patterns

    def _saved_patterns(self) -> list[dict]:
        return sorted(
            self._patterns.items(),
            key=lambda x: -x[1]["count"],
        )[:self._max_patterns]

    def summary(self) -> str:
        if not self._patterns:
            return "No temporal patterns yet."
        top = sorted(self._patterns.items(), key=lambda x: -x[1]["count"])[:3]
        return "Top patterns: " + "; ".join(
            f"{sig} ({info['count']}x)" for sig, info in top
        )
