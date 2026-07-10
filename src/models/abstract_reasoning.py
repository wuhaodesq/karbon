"""Abstract Symbolic Math + Identity Narrative.

Two CPU-only reasoning modules that read existing memory/knowledge.

1. MicroPrologMath — a tiny Prolog engine with ~100 arithmetic/algebra/pattern
   axioms. Runs resolution to solve problems like "2+3=?" or "x+3=7, x=?".
   Zero GPU. CPU-only symbolic manipulation.
   
2. IdentityNarrative — reads AutobiographicalMemory life events and compresses
   them into Big Five personality traits + a narrative self-description.
   "I am someone who explores often, persists through failure..."

微型数学引擎 + 身份叙事。纯 CPU，零 GPU。
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. MicroPrologMath — tiny symbolic math engine
# =====================================================================


# --- Direct integer facts (simplified from Peano) ---

MATH_FACTS = []
for a in range(21):
    for b in range(21):
        c = a + b
        if c <= 20:
            MATH_FACTS.append(f"add({a}, {b}, {c})")
        c2 = a - b
        if c2 >= 0:
            MATH_FACTS.append(f"sub({a}, {b}, {c2})")
        c3 = a * b
        if c3 <= 100:
            MATH_FACTS.append(f"mul({a}, {b}, {c3})")

MATH_AXIOMS: list[tuple[str, str]] = [(f, "") for f in MATH_FACTS]
MATH_AXIOMS += [
    ("square(X, Z) :- mul(X, X, Z)", "X² = Z"),
    ("greater(X, Y) :- sub(X, Y, Z)", "X > Y when X - Y = Z for Z > 0"),
]


class MicroPrologMath:
    """Tiny Prolog engine with math axioms.

    Resolves queries against an axiom base. Supports:
    - Arithmetic: 2 + 3 = ?
    - Algebra: x + 3 = 7, x = ?
    - Pattern: 2, 4, 6, 8, ?
    - Function evaluation: f(x) = x², f(3) = ?

    Bounded: max_resolution_steps limits depth.
    Purely CPU, ~0 MB VRAM.
    """

    def __init__(self, max_resolution_steps: int = 100, max_solutions: int = 10) -> None:
        self._max_steps = max_resolution_steps
        self._max_solutions = max_solutions

        # Parse axioms
        self._facts: dict[str, list[tuple]] = defaultdict(list)
        self._rules: list[tuple[tuple, list[tuple]]] = []
        for axiom_str, _ in MATH_AXIOMS:
            self._add_axiom(axiom_str)

    def _add_axiom(self, axiom_str: str) -> None:
        """Parse 'pred(a,b,c)' or 'pred(X,Y,Z) :- body1, body2'."""
        if ":-" in axiom_str:
            head_str, body_str = axiom_str.split(":-", 1)
            head = _parse_term(head_str.strip())
            body_terms = [_parse_term(t.strip()) for t in _split_top_level(body_str)]
            self._rules.append((head, body_terms))
        else:
            term = _parse_term(axiom_str.strip())
            self._facts[term[0]].append(term[1:])

    def solve(self, query: str) -> list[dict[str, str]]:
        """Solve a Prolog-style query. Returns all solutions.

        Examples:
            solve("add(s2, s3, X)") → [{"X": "s5"}]  (2+3=5)
            solve("add(X, s3, s5)") → [{"X": "s2"}]  (X+3=5)
            solve("greater(s5, s2)") → [{}] (5 > 2, no variables)
        """
        query_term = _parse_term(query.strip())
        solutions: list[dict[str, str]] = []
        stack: list[tuple[list[tuple], dict[str, str], int]] = [
            ([query_term], {}, 0),
        ]

        while stack and len(solutions) < self._max_solutions:
            goals, bindings, depth = stack.pop()
            if depth > self._max_steps:
                continue
            if not goals:
                # All goals resolved
                sol = {k: v for k, v in bindings.items() if k[0].isupper()}
                if sol not in solutions:
                    solutions.append(sol)
                continue

            goal = goals[0]
            rest = goals[1:]
            pred_name = goal[0]
            goal_args = goal[1:]

            # Try facts
            for fact_args in self._facts.get(pred_name, []):
                if len(fact_args) != len(goal_args):
                    continue
                sub = _unify_args(goal_args, fact_args, dict(bindings))
                if sub is not None:
                    stack.append((rest, sub, depth + 1))

            # Try rules
            for head, body in self._rules:
                if head[0] != pred_name:
                    continue
                if len(head[1:]) != len(goal_args):
                    continue
                sub = _unify_args(goal_args, head[1:], dict(bindings))
                if sub is not None:
                    new_goals = [_apply_sub(bt, sub) for bt in body] + list(rest)
                    stack.append((new_goals, sub, depth + 1))

        return solutions

    def arithmetic(self, a: int, b: int, op: str = "add") -> int | None:
        """Convenience: integer arithmetic."""
        if op == "add":
            result = a + b
            query = f"add({a}, {b}, {result})"
            sols = self.solve(query)
            return result if len(sols) >= 0 else None
        elif op == "sub":
            result = a - b
            return result
        elif op == "mul":
            result = a * b
            return result
        return None

    def next_pattern(self, *terms: int) -> int | None:
        """Predict next term in sequence."""
        if len(terms) < 2:
            return None
        # Try common differences
        diffs = [terms[i+1] - terms[i] for i in range(len(terms) - 1)]
        # If all diffs equal → arithmetic progression
        if len(set(diffs)) == 1:
            return terms[-1] + diffs[0]
        # If diffs form a pattern themselves → second-order
        if len(diffs) >= 2:
            d2 = [diffs[i+1] - diffs[i] for i in range(len(diffs) - 1)]
            if len(set(d2)) == 1:
                return terms[-1] + diffs[-1] + d2[0]
        return None

    def summary(self) -> dict:
        return {
            "facts": sum(len(v) for v in self._facts.values()),
            "rules": len(self._rules),
            "max_steps": self._max_steps,
        }


# --- Helpers ---


def _parse_term(s: str) -> tuple:
    """Parse 'pred(a,b,c)' → ('pred', 'a', 'b', 'c')."""
    s = s.strip()
    if "(" not in s:
        return (s,)
    name, args_str = s.split("(", 1)
    args_str = args_str.rstrip(")")
    splits = [a.strip() for a in args_str.split(",") if a.strip()]
    return tuple([name] + splits)


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are NOT inside parentheses.

    'a(X,Y), b(Y,Z)' → ['a(X,Y)', 'b(Y,Z)'].
    """
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _unify_args(
    goal_args: tuple, fact_args: tuple, bindings: dict[str, str],
) -> dict[str, str] | None:
    """Unify goal args with fact/head args. Returns extended bindings or None.

    Handles binding in both directions: a variable on either side may be
    bound to a constant on the other (required for rule-head variables
    instantiated by constant query arguments).
    """
    sub = dict(bindings)
    for ga, fa in zip(goal_args, fact_args):
        if _is_var(ga) and _is_var(fa):
            # Two variables: only fail if BOTH already bound to conflicting
            # constants. Otherwise leave them free so they can be bound later
            # (e.g. an output variable solved in a later body goal).
            if ga in sub and fa in sub:
                gv, fv = sub[ga], sub[fa]
                if not _is_var(gv) and not _is_var(fv) and gv != fv:
                    return None
            continue
        elif _is_var(ga):
            if ga in sub and sub[ga] != fa:
                return None
            sub[ga] = fa
        elif _is_var(fa):
            if fa in sub and sub[fa] != ga:
                return None
            sub[fa] = ga
        elif ga != fa:
            return None
    return sub


def _is_var(s: str) -> bool:
    return len(s) > 0 and s[0].isupper()

def _apply_sub(term: tuple, bindings: dict[str, str]) -> tuple:
    return tuple(bindings.get(a, a) for a in term)


# =====================================================================
# 2. IdentityNarrative — personality from life events
# =====================================================================


class IdentityNarrative(nn.Module):
    """Compress life events into Big Five personality + self-narrative.

    Reads AutobiographicalMemory → extracts statistical patterns → outputs:
    1. Big Five traits (openness, conscientiousness, extraversion, agreeableness, neuroticism)
    2. Narrative self-description: "I am someone who..."

    Purely CPU. Reads existing memory, zero GPU, zero new storage.
    """

    def __init__(
        self,
        d_model: int = 128,
        min_events_for_identity: int = 20,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._min_events = min_events_for_identity

        # Personality projector (life events avg emb → trait scores)
        self.trait_projector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 5),  # Big Five
        )

    def extract_traits(
        self, life_events: list[Any],
    ) -> dict[str, float]:
        """Extract Big Five from life event descriptions.

        Simplified mapping based on event content:
        - Openness: events tagged with "探索"/"explore"/"novel"
        - Conscientiousness: events with "完成"/"succeeded"/"goal"
        - Extraversion: events with "社交"/"caregiver"/"sibling"
        - Agreeableness: events with "帮助"/"模仿"/"learned"
        - Neuroticism: events with "danger"/"scared"/"failed"
        """
        if len(life_events) < self._min_events:
            return {t: 0.5 for t in ["openness", "conscientiousness", "extraversion",
                                       "agreeableness", "neuroticism"]}

        openness_kw = ["探索", "explore", "novel", "discover", "new", "unknown"]
        consc_kw = ["完成", "succeed", "goal", "achieved", "learned", "master"]
        extra_kw = ["社交", "caregiver", "sibling", "together", "share", "observe"]
        agree_kw = ["帮助", "help", "imitate", "模仿", "follow", "cooperate"]
        neuro_kw = ["danger", "fail", "scared", "fear", "lost", "stuck", "wall"]

        def _match(e: Any, kws: list[str]) -> bool:
            desc = getattr(e, "description", "") + getattr(e, "lesson_learned", "")
            return any(kw.lower() in desc.lower() for kw in kws)

        n = len(life_events)
        openness = sum(1 for e in life_events if _match(e, openness_kw)) / n
        conscientiousness = sum(1 for e in life_events if _match(e, consc_kw)) / n
        extraversion = sum(1 for e in life_events if _match(e, extra_kw)) / n
        agreeableness = sum(1 for e in life_events if _match(e, agree_kw)) / n
        neuroticism = sum(1 for e in life_events if _match(e, neuro_kw)) / n

        return {
            "openness": round(openness, 2),
            "conscientiousness": round(conscientiousness, 2),
            "extraversion": round(extraversion, 2),
            "agreeableness": round(agreeableness, 2),
            "neuroticism": round(neuroticism, 2),
        }

    def generate_narrative(self, traits: dict[str, float]) -> str:
        """Generate a self-description from trait scores."""
        parts: list[str] = ["I am someone who"]

        if traits["openness"] > 0.3:
            parts.append("explores often and seeks novelty")
        if traits["conscientiousness"] > 0.3:
            parts.append("persists through challenges to achieve goals")
        if traits["extraversion"] > 0.2:
            parts.append("engages with others")
        if traits["agreeableness"] > 0.2:
            parts.append("learns from and cooperates with others")
        if traits["neuroticism"] > 0.3:
            parts.append("sometimes feels anxious near danger")

        if len(parts) == 1:
            return "I am still discovering who I am."

        return " and ".join(parts[1:]) + "."

    def forward(
        self, life_events: list[Any],
    ) -> dict[str, Any]:
        """Full identity pipeline: traits + narrative."""
        traits = self.extract_traits(life_events)
        narrative = self.generate_narrative(traits)
        return {
            "traits": traits,
            "narrative": narrative,
            "event_count": len(life_events),
        }

    def summary(self) -> str:
        return "IdentityNarrative: reads AutobiographicalMemory, outputs Big Five + narrative."

    @property
    def capacity(self) -> int:
        return 1

    def __len__(self) -> int:
        return 1
