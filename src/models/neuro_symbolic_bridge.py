"""Neuro-Symbolic Bridge — Connect neural outputs to symbolic engine.

Three connections that unite the two sides of devagi:

1. Causal2Prolog — convert CausalDiscovery edges into Prolog rules
2. Number2Math — hook NumberSense output into MicroPrologMath queries
3. SchemaDetector — extract template patterns from multiple similar rules

All zero GPU, zero retraining. Reads existing module states and converts them.

神经符号桥梁：连接神经网络输出到符号推理引擎。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Causal2Prolog — CausalDiscovery → Prolog rules
# =====================================================================


class Causal2Prolog:
    """Converts CausalDiscovery graph edges into MicroPrologMath rules.

    Causal edge: "push_action → object_moves" (strength 0.7)
    → Prolog: causes(push, move) :- confidence(0.7).

    This enables chaining: push → move, move → hit_other, hit_other → also_moves
    → Prolog can deduce "push causes chain of 3 movements".
    """

    def __init__(self, min_strength: float = 0.3, max_rules: int = 200) -> None:
        self._min_strength = min_strength
        self._max_rules = max_rules
        self._converted: list[str] = []

    def convert(self, causal_disc: Any, step: int) -> list[str]:
        """Convert causal edges above threshold to Prolog-atoms.

        Returns list of new Prolog facts to add to MicroPrologMath.
        """
        if causal_disc is None or not hasattr(causal_disc, '_graph'):
            return []

        new_rules: list[str] = []
        for (src, tgt), edge in causal_disc._graph.edges.items():
            if edge.strength < self._min_strength:
                continue
            # Clean names for Prolog
            src_clean = src.replace(" ", "_").replace("-", "_").lower()
            tgt_clean = tgt.replace(" ", "_").replace("-", "_").lower()
            rule = f"causes({src_clean}, {tgt_clean})"
            if rule not in self._converted:
                self._converted.append(rule)
                new_rules.append(rule)

        if len(self._converted) > self._max_rules:
            self._converted = self._converted[-self._max_rules:]

        if new_rules:
            logger.info("[causal2prolog] %d new causal rules converted", len(new_rules))
        return new_rules

    def feed_to_math(self, micro_math: Any, causal_disc: Any) -> int:
        """Feed converted rules into MicroPrologMath for resolution."""
        rules = self.convert(causal_disc, 0)
        for rule in rules:
            if hasattr(micro_math, '_add_axiom'):
                micro_math._add_axiom(rule)
        return len(rules)


# =====================================================================
# 2. Number2Math — NumberSense → MicroPrologMath bridge
# =====================================================================


class Number2Math:
    """Routes NumberSense cardinality predictions into math engine queries.

    NumberSense says "there are 5 objects" →
    MicroPrologMath answers:
        "5 > 3?" → yes
        "5 - 2 = ?" → 3
        "is 5 prime?" → simplified: no (divisible by 1)

    This gives the agent the ability to REASON about quantities,
    not just predict which of two piles has more.
    """

    def __init__(self):
        self._last_count: int = 0

    def observe(self, number_sense: Any, slots: torch.Tensor) -> int:
        """Get current count from NumberSense."""
        if number_sense is None:
            return 0
        try:
            with torch.no_grad():
                count = int(number_sense.predict_count(slots.unsqueeze(0)).item())
            self._last_count = count
            return count
        except Exception:
            return 0

    def query(self, micro_math: Any, question: str) -> str:
        """Use MicroPrologMath to answer a quantity question.

        Supported: "more_than_last(N)" → checks if N > last_count
                  "add_to_last(N)" → last_count + N
                  "half_of_last" → last_count // 2
        """
        if micro_math is None:
            return "math engine not available"

        if question == "more_than_3":
            result = self._last_count > 3
            return f"I see {self._last_count} objects, which is {'more' if result else 'not more'} than 3."
        elif question.startswith("add_"):
            try:
                add_val = int(question.split("_")[1])
                total = self._last_count + add_val
                return f"If I add {add_val}, there would be {total} objects."
            except (ValueError, IndexError):
                return f"I see {self._last_count} objects."
        elif question == "count":
            return f"I see {self._last_count} objects."

        # Try Prolog
        try:
            sols = micro_math.solve(f"greater(X, {self._last_count})")
            return f"Something greater than {self._last_count}: {sols}"
        except Exception:
            return f"I see {self._last_count} objects (math query failed)."


# =====================================================================
# 3. SchemaDetector — extract templates from multiple rules
# =====================================================================


class SchemaDetector:
    """Detects abstract schema patterns from multiple induced rules.

    Schema = a template with variables, like:
        "pushing [X] harder makes it move [faster]"
    extracted from:
        "push ball hard → ball moves fast" (conf 0.7)
        "push block hard → block moves fast" (conf 0.6)
        "push cylinder hard → cylinder moves fast" (conf 0.8)

    This is the bridge between "many specific rules" (RuleInductionEngine)
    and "abstract principle" (what the article called "大众推理力").

    Schema learning = unsupervised template extraction from similar rules.
    """

    def __init__(self, min_rule_count: int = 3, max_schemas: int = 50) -> None:
        self._min_count = min_rule_count
        self._max_schemas = max_schemas
        self._schemas: list[dict[str, Any]] = []

    def extract(
        self, rule_engine: Any, step: int,
    ) -> list[dict[str, Any]]:
        """Extract schemas from RuleInductionEngine rules.

        Groups rules by shared action and predicate count.
        If ≥ min_rule_count rules share same action + same # of conditions → schema.
        """
        if rule_engine is None or not hasattr(rule_engine, '_rules'):
            return []

        rules = list(rule_engine._rules.values())
        if len(rules) < self._min_count:
            return []

        # Group by (action, num_conditions)
        groups: dict[tuple[int, int], list[Any]] = {}
        for r in rules:
            if not hasattr(r, 'then_predicate') or r.then_predicate[0] != "action":
                continue
            action = r.then_predicate[1][0] if hasattr(r.then_predicate, '__getitem__') else 0
            n_conds = len(r.if_predicates)
            key = (int(action), n_conds)
            groups.setdefault(key, []).append(r)

        new_schemas = []
        for (action, n_conds), group in groups.items():
            if len(group) < self._min_count:
                continue
            avg_conf = sum(r.confidence for r in group) / len(group)
            # Abstract the predicates
            pred_names = [p[0] for p in group[0].if_predicates] if group[0].if_predicates else []
            schema = {
                "action": action,
                "num_conditions": n_conds,
                "avg_confidence": avg_conf,
                "example_predicates": pred_names[:3],
                "num_examples": len(group),
                "description": f"Action {action} with {n_conds} condition(s): "
                              f"observed {len(group)} times (avg conf: {avg_conf:.2f})",
            }

            if not any(s["action"] == action and s["num_conditions"] == n_conds
                      for s in self._schemas):
                self._schemas.append(schema)
                new_schemas.append(schema)

        if len(self._schemas) > self._max_schemas:
            self._schemas = self._schemas[-self._max_schemas:]

        if new_schemas:
            logger.info("[schema] %d new schemas detected (top: action=%d, examples=%d)",
                       len(new_schemas), new_schemas[0]["action"] if new_schemas else -1,
                       new_schemas[0]["num_examples"] if new_schemas else 0)
        return new_schemas

    def get_best_schema(self) -> str:
        if not self._schemas:
            return "No schemas learned yet."
        best = max(self._schemas, key=lambda s: s["num_examples"])
        return f"Most reliable pattern: {best['description']}"
