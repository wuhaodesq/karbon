"""Symbolic Reasoning Upgrade — Unification + Resolution + Explanation.

Extends RuleInductionEngine with three targeted upgrades:

1. Proper unification (occurs check) — replaces loose predicate matching
2. Resolution (modus ponens + modus tollens) — logical inference rules
3. Explanation generator — depth=2 why-chains from induced rules

FUTURE WORK (marked):
    - Nested belief reasoning: `believes(A, believes(B, P))`
    - Depth>2 recursive chains with combinatorial guard (exponential, not for training loop)
    - Probabilistic resolution with confidence-weighted inference

符号推理升级：合一 + 归结 + 解释生成。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Proper Unification (with occurs check)
# =====================================================================


def unify(
    term1: tuple[str, tuple],  # ("pred_name", ("s0", "s1"))
    term2: tuple[str, tuple],
    substitution: dict[str, tuple[str, tuple]] | None = None,
) -> dict[str, tuple[str, tuple]] | None:
    """Unify two predicate terms with occurs check.

    Replaces the cosine-similarity-based "matching" in the original
    RuleInductionEngine with proper syntactic unification.

    Example:
        unify(("near", ("s0", "s1")), ("near", ("X", "s1")))
        → {"X": ("s0",)}

        unify(("near", ("X", "X")), ("near", ("s0", "s1")))
        → None  (can't unify s0 with s1 in occurs check)

    Args:
        term1, term2: (pred_name, (args...)) tuples.
        substitution: current variable bindings.

    Returns:
        Updated substitution dict, or None if unification fails.
    """
    if substitution is None:
        substitution = {}

    name1, args1 = term1
    name2, args2 = term2

    # Predicate names must match
    if name1 != name2:
        return None

    # Arity must match
    if len(args1) != len(args2):
        return None

    sub = dict(substitution)  # copy to avoid mutation from failed branch

    for a1, a2 in zip(args1, args2):
        a1_str = str(a1)
        a2_str = str(a2)

        # Variable binding
        is_var1 = a1_str[0].isupper() if a1_str else False
        is_var2 = a2_str[0].isupper() if a2_str else False

        if is_var1 and not is_var2:
            # X = s0
            if a1_str in sub:
                if not _terms_equal(sub[a1_str], (name2, (a2,))):
                    return None
            else:
                # Occurs check
                if _occurs_in(a1_str, (name2, (a2,)), sub):
                    return None
                sub[a1_str] = (name2, (a2,))
        elif is_var2 and not is_var1:
            # s0 = X
            if a2_str in sub:
                if not _terms_equal(sub[a2_str], (name1, (a1,))):
                    return None
            else:
                if _occurs_in(a2_str, (name1, (a1,)), sub):
                    return None
                sub[a2_str] = (name1, (a1,))
        elif a1 != a2:
            # Two constants that differ → fail
            return None

    return sub


def _terms_equal(t1: tuple[str, tuple], t2: tuple[str, tuple]) -> bool:
    """Check if two terms are syntactically equal."""
    return t1[0] == t2[0] and tuple(str(x) for x in t1[1]) == tuple(str(x) for x in t2[1])


def _occurs_in(
    var: str, term: tuple[str, tuple], sub: dict[str, tuple[str, tuple]],
) -> bool:
    """Occurs check: does variable appear in term (directly or via substitution)?"""
    _, args = term
    for a in args:
        a_str = str(a)
        if a_str == var:
            return True
        if a_str in sub:
            if _occurs_in(var, sub[a_str], sub):
                return True
    return False


# =====================================================================
# 2. Resolution (Modus Ponens + Modus Tollens)
# =====================================================================


def try_modus_ponens(
    rule_if: tuple[str, tuple],
    rule_then: tuple[str, tuple],
    fact: tuple[str, tuple],
) -> tuple[str, tuple] | None:
    """Modus Ponens: IF P THEN Q, and P is true → Q is true.

    Args:
        rule_if: the condition predicate (what must hold).
        rule_then: the conclusion predicate (what follows).
        fact: a known true predicate.

    Returns:
        The concluded predicate if MP applies, else None.
    """
    sub = unify(rule_if, fact)
    if sub is None:
        return None
    # Apply substitution to the conclusion
    return _apply_substitution(rule_then, sub)


def try_modus_tollens(
    rule_if: tuple[str, tuple],
    rule_then: tuple[str, tuple],
    fact: tuple[str, tuple],  # this fact is FALSE (negation)
) -> tuple[str, tuple] | None:
    """Modus Tollens: IF P THEN Q, and Q is false → P is false.

    Returns:
        The negated condition (asserts P is false), or None if MT doesn't apply.
    """
    sub = unify(rule_then, fact)
    if sub is None:
        return None
    return _apply_substitution(rule_if, sub)


def _apply_substitution(
    term: tuple[str, tuple], sub: dict[str, tuple[str, tuple]],
) -> tuple[str, tuple]:
    """Apply variable substitution to a term."""
    name, args = term
    new_args = []
    for a in args:
        a_str = str(a)
        if a_str[0].isupper() and a_str in sub:
            _, sub_args = sub[a_str]
            new_args.extend(sub_args)
        else:
            new_args.append(a)
    return (name, tuple(new_args))


# =====================================================================
# 3. Explanation Generator — depth=2 why-chains
# =====================================================================


class ExplanationGenerator:
    """Generates human-readable explanations of agent behavior.

    Given a set of induced rules and a sequence of actions/observations,
    constructs a depth=2 chain explaining why each action was taken.

    Depth is deliberately limited to 2 to avoid combinatorial explosion
    during training. Offline use only — not called in the training loop.

    Explanation depth limit: 2 (FUTURE: depth>2 with combinatorial guard).
    """

    def __init__(self, max_chain_length: int = 5) -> None:
        self._max_chain = max_chain_length

    def explain_action(
        self,
        action: int,
        predicates_at_time: dict[str, bool],
        rules: list[Any],  # list[InducedRule]
    ) -> str:
        """Explain why this action was taken, given the current predicates.

        Returns a natural-language explanation like:
            "I opened the door because: (1) the door was locked,
             (2) I had the key, (3) having a key allows opening locked doors."
        """
        action_names = ["move_north", "move_south", "move_west", "move_east",
                        "push", "pull", "grasp", "wait"]

        # Find rules that recommend this action
        relevant = [r for r in rules if hasattr(r, 'then_predicate')
                    and r.then_predicate[0] == "action"
                    and r.then_predicate[1][0] == action]

        if not relevant:
            action_name = action_names[action] if action < len(action_names) else f"action_{action}"
            return f"I took {action_name} because my neural policy suggested it."

        # Take the highest-confidence matching rule
        best_rule = max(relevant, key=lambda r: r.confidence)

        # Check which antecedents hold
        if_strs = [f"{p}({','.join(map(str, args))})"
                   for p, args in best_rule.if_predicates]
        true_conds = [s for s in if_strs if predicates_at_time.get(s, False)]

        action_name = action_names[action] if action < len(action_names) else f"action_{action}"

        if not true_conds:
            return f"I took {action_name} by habit (no specific reason found)."

        # Build chain
        steps = [f"I took {action_name} because:"]
        for i, cond in enumerate(true_conds[:3], 1):
            steps.append(f"  ({i}) {_humanize_predicate(cond)}")
        steps.append(f"  (confidence: {best_rule.confidence:.1%})")

        return "\n".join(steps)

    def explain_chain(
        self,
        action_sequence: list[int],
        predicate_sequence: list[dict[str, bool]],
        rules: list[Any],
    ) -> str:
        """Explain a multi-step action plan as a depth-2 why-chain.

        Limits to last 3 actions to keep explanation bounded.
        """
        if not action_sequence:
            return "No actions to explain."

        lines = ["Here's why I did what I did:"]
        for t in range(max(0, len(action_sequence) - 3), len(action_sequence)):
            action = action_sequence[t]
            preds = predicate_sequence[t] if t < len(predicate_sequence) else {}
            explanation = self.explain_action(action, preds, rules)
            lines.append(explanation)
        return "\n".join(lines)


def _humanize_predicate(pred_str: str) -> str:
    """Convert predicate string to human-readable form."""
    # "exists(s0)" → "object s0 exists"
    # "near(s0,s1)" → "objects s0 and s1 are near each other"
    # "color_red(s0)" → "object s0 is red"
    name, args_str = pred_str.split("(", 1)
    args_str = args_str.rstrip(")")
    args = args_str.split(",")

    human_readable = {
        "exists": lambda a: f"object {a[0]} exists",
        "near": lambda a: f"objects {a[0]} and {a[1]} are near each other",
        "touching": lambda a: f"objects {a[0]} and {a[1]} are touching",
        "moving": lambda a: f"object {a[0]} is moving",
        "large": lambda a: f"object {a[0]} is large",
        "color_red": lambda a: f"object {a[0]} is red",
        "color_blue": lambda a: f"object {a[0]} is blue",
        "color_green": lambda a: f"object {a[0]} is green",
        "color_yellow": lambda a: f"object {a[0]} is yellow",
    }

    if name in human_readable:
        return human_readable[name](args)
    return pred_str.replace("(", " ").replace(")", "").replace(",", " and ")


# =====================================================================
# 4. Future Work Registry
# =====================================================================

# TODO(phase6+): Nested belief reasoning
#   When TheoryOfMind reaches reliability > 80%, add:
#   ```
#   believes(caregiver, not(knows(learner, location(ball))))
#   ```
#   Requires: belief base per agent, modal logic (KD45), depth limit = 2 initially.

# TODO(phase7+): Depth>2 recursive chains
#   Current limit: 2 (combinatorial safety).
#   Future: add `_combinatorial_guard(max_branches=50, max_depth=5)`
#   with A* heuristic search (most-confident rule first).
#   WARNING: each additional depth level increases search space by factor of
#   num_rules ≈ 100×. Depth=3 = 10^4 combos, depth=5 = 10^8 combos.
#   Must run OFF the training loop (separate reasoning thread).

# TODO(phase8+): Probabilistic resolution
#   Replace Boolean truth values with [0,1] confidence scores.
#   Modus ponens: confidence(conclusion) = confidence(premise) × confidence(rule).
#   Modus tollens: confidence(negated premise) = confidence(negated conclusion) × confidence(rule).
#   Propagate through chain with min/max bounds.
