"""Rule Induction Engine — Symbolic Logic Without Cosine Matching.

Replaces the cosine-similarity based "rule matching" in NeuralSymbolicLayer
with a proper mini rule-induction system based on David Poole's probabilistic
logic programming paradigm.

Core insight:
    Instead of "cosine(hidden_state, rule_condition) > 0.7",
    we discretize slot attention output into boolean predicates:
        has_attribute(slot_N, color=red) → True/False
        is_near(slot_N, slot_M) → True/False
    and then apply forward-chaining AND backward-chaining over these.

This is NOT a general Prolog interpreter. It's a bounded, GPU-friendly
rule engine with:
- Discrete predicate extraction from slot states
- Forward chaining (MP: if A then B, and A holds → B holds)
- Backward chaining (goal-driven proof search)
- Inductive rule learning (from positive/negative examples)
- Rule confidence tracking with Bayesian updates

Bounded: max_predicates, max_rules, max_chain_depth all fixed.

规则归纳引擎：把 Slot Attention 输出离散化为布尔谓词，做真正的逻辑推理。
替代余弦匹配。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# Predicate — a boolean property of the world
# =====================================================================


@dataclass
class Predicate:
    name: str
    arity: int
    description: str = ""


# Built-in predicates extractable from Slot Attention
BUILTIN_PREDICATES: list[Predicate] = [
    Predicate("exists", 1, "slot_i has content"),
    Predicate("color_red", 1, "slot_i is red"),
    Predicate("color_blue", 1, "slot_i is blue"),
    Predicate("color_green", 1, "slot_i is green"),
    Predicate("color_yellow", 1, "slot_i is yellow"),
    Predicate("near", 2, "slot_i is near slot_j"),
    Predicate("touching", 2, "slot_i is touching slot_j"),
    Predicate("moving", 1, "slot_i has significant velocity"),
    Predicate("large", 1, "slot_i has above-average norm"),
]


# =====================================================================
# InducedRule — a learned rule
# =====================================================================


@dataclass
class InducedRule:
    if_predicates: list[tuple[str, tuple[int, ...]]]
    then_predicate: tuple[str, tuple[int, ...]]
    confidence: float = 0.5
    positive_examples: int = 0
    negative_examples: int = 0
    derivation_id: int = -1

    def __repr__(self) -> str:
        ifs = " AND ".join(f"{p}({','.join(map(str, args))})" for p, args in self.if_predicates)
        then = f"{self.then_predicate[0]}({','.join(map(str, self.then_predicate[1]))})"
        return f"IF {ifs} THEN {then} [conf={self.confidence:.2f}, pos={self.positive_examples}]"


# =====================================================================
# RuleInductionEngine — the core
# =====================================================================


class RuleInductionEngine:
    """Extracts discrete predicates from continuous slot states and reasons over them.

    Three stages:
    1. Perception: SlotAttention output → boolean predicate values (discretization)
    2. Induction: From (predicates, action, outcome) triples, learn IF-THEN rules
    3. Deduction: Apply learned rules via forward/backward chaining

    Bounded: max_predicates, max_slots, max_rules, max_chain_depth.
    """

    def __init__(
        self,
        num_slots: int = 7,
        max_rules: int = 128,
        max_chain_depth: int = 5,
        min_confidence: float = 0.3,
        induction_min_positive: int = 3,
    ) -> None:
        self._num_slots = num_slots
        self._max_rules = max_rules
        self._max_depth = max_chain_depth
        self._min_confidence = min_confidence
        self._induction_min_pos = induction_min_positive

        self._rules: dict[int, InducedRule] = {}
        self._next_id = 0

        # Accumulated predicate statistics (跨 episode 累积，定期衰减)
        self._positive_combos: dict[str, dict[int, float]] = {}  # pred_sig → {action: float count}
        self._negative_combos: dict[str, dict[int, float]] = {}
        self._accumulation_decay: float = 0.98  # per-episode decay (slow, floats preserve signal)
        self._episodes_accumulated: int = 0
        self._decay_every: int = 5  # only decay every N episodes, not every single one

        # Color centroids (learnt from slot stats)
        self._color_centroids: dict[str, torch.Tensor] = {}

        # Episode buffer for induction
        self._episode_predicates: list[dict[str, bool]] = []
        self._episode_actions: list[int] = []
        self._episode_outcomes: list[float] = []

    # ------------------------------------------------------------------ perception

    def extract_predicates(self, slots: torch.Tensor) -> dict[str, bool]:
        """Convert continuous slot vectors to boolean predicates.

        slots: (B, num_slots, slot_dim) → predicate dict.
        Only uses B=0 (first batch element).
        """
        facts: dict[str, bool] = {}
        s = slots[0] if slots.dim() == 3 else slots  # (num_slots, slot_dim)

        slot_norms = s.norm(dim=-1).cpu().numpy()
        mean_norm = float(np.mean(slot_norms))

        for i in range(min(self._num_slots, s.shape[0])):
            si = f"s{i}"
            # Existence
            facts[f"exists({si})"] = bool(slot_norms[i] > 0.1 * mean_norm)

            # Color detection (simplified: use RGB-like centroids)
            slot_vec = s[i].cpu()
            best_color = "none"
            best_sim = 0.0
            for color, centroid in self._color_centroids.items():
                sim = float(torch.cosine_similarity(
                    slot_vec.unsqueeze(0), centroid.unsqueeze(0), dim=-1
                ).item())
                if sim > best_sim and sim > 0.5:
                    best_sim = sim
                    best_color = color
            if best_color != "none":
                facts[f"color_{best_color}({si})"] = True

            # Motion detection (simplified)
            facts[f"moving({si})"] = bool(slot_norms[i] > 1.5 * mean_norm)
            facts[f"large({si})"] = bool(slot_norms[i] > 1.2 * mean_norm)

            # Proximity
            for j in range(i + 1, min(self._num_slots, s.shape[0])):
                sj = f"s{j}"
                dist = float((s[i] - s[j]).norm().item())
                facts[f"near({si},{sj})"] = bool(dist < 0.5)
                facts[f"touching({si},{sj})"] = bool(dist < 0.2)

        return facts

    # ------------------------------------------------------------------ induction

    def record_episode(
        self,
        predicates_sequence: list[dict[str, bool]],
        actions: list[int],
        outcome: float,
    ) -> None:
        """Store an episode for batch rule induction."""
        self._episode_predicates = predicates_sequence
        self._episode_actions = actions
        self._episode_outcomes = [outcome] * len(actions)

    def induce_rules(self) -> list[InducedRule]:
        """Induce IF-THEN rules from accumulated episode statistics.

        Uses instance-level accumulators that persist across episodes.
        Statistics are decayed each episode so old patterns fade.
        """
        if not self._episode_predicates:
            return []

        new_rules: list[InducedRule] = []

        # Accumulate from this episode
        for t, preds in enumerate(self._episode_predicates):
            if t >= len(self._episode_actions):
                break
            action = self._episode_actions[t]
            outcome = self._episode_outcomes[min(t, len(self._episode_outcomes) - 1)]

            true_preds = tuple(sorted(k for k, v in preds.items() if v))
            if not true_preds:
                continue
            sig = "&".join(true_preds[:6])

            if outcome > 0:
                self._positive_combos.setdefault(sig, {}).setdefault(action, 0.0)
                self._positive_combos[sig][action] += 1.0
            else:
                self._negative_combos.setdefault(sig, {}).setdefault(action, 0.0)
                self._negative_combos[sig][action] += 1.0

        self._episodes_accumulated += 1

        # Check for rules that now meet the float threshold
        for sig, action_counts in self._positive_combos.items():
            for action, pos_count in list(action_counts.items()):
                neg_count = self._negative_combos.get(sig, {}).get(action, 0.0)
                total = pos_count + neg_count
                if total < float(self._induction_min_pos):
                    continue
                confidence = pos_count / total if total > 0 else 0.0
                if confidence < self._min_confidence:
                    continue
                # Deduplicate: check if a rule with this sig+action already exists
                pred_tuples = [
                    _parse_predicate_tuple(p) for p in sig.split("&") if p
                ]
                already_exists = any(
                    set(r.if_predicates) == set(pred_tuples) and
                    r.then_predicate == ("action", (action,))
                    for r in self._rules.values()
                )
                if already_exists:
                    continue
                rule = InducedRule(
                    if_predicates=[pt for pt in pred_tuples if pt is not None],
                    then_predicate=("action", (action,)),
                    confidence=confidence,
                    positive_examples=int(pos_count),
                    negative_examples=int(neg_count),
                    derivation_id=-1,
                )
                self._add_rule(rule)
                new_rules.append(rule)

        # Decay only when enough episodes accumulated, and use float decay
        if self._episodes_accumulated > 0 and self._episodes_accumulated % self._decay_every == 0:
            for combos in [self._positive_combos, self._negative_combos]:
                for sig in list(combos.keys()):
                    for action in list(combos[sig].keys()):
                        combos[sig][action] *= self._accumulation_decay
                        # Only delete if truly negligible (below 0.1, not 1.0)
                        if combos[sig][action] < 0.1:
                            del combos[sig][action]
                    if not combos[sig]:
                        del combos[sig]

        return new_rules

    def _add_rule(self, rule: InducedRule) -> None:
        if len(self._rules) >= self._max_rules:
            self._evict_rule()
        rule.derivation_id = self._next_id
        self._rules[self._next_id] = rule
        self._next_id += 1

    def _evict_rule(self) -> None:
        if not self._rules:
            return
        worst_id = min(
            self._rules,
            key=lambda rid: self._rules[rid].confidence * self._rules[rid].positive_examples,
        )
        del self._rules[worst_id]

    # ------------------------------------------------------------------ deduction

    def forward_chain(self, facts: dict[str, bool]) -> dict[str, bool]:
        """Apply rules to derive new facts. Sound. Bounded.

        Returns updated facts dict with derived facts added.
        """
        derived = dict(facts)
        for _ in range(self._max_depth):
            new_derived = 0
            for rule in self._rules.values():
                if rule.confidence < self._min_confidence:
                    continue
                # Check all antecedents hold
                if_pred_strs = [f"{p}({','.join(map(str,args))})" for p, args in rule.if_predicates]
                if all(derived.get(p, False) for p in if_pred_strs):
                    then_str = f"{rule.then_predicate[0]}({','.join(map(str, rule.then_predicate[1]))})"
                    if not derived.get(then_str, False):
                        derived[then_str] = True
                        new_derived += 1
            if new_derived == 0:
                break
        return derived

    def backward_chain(
        self, goal: tuple[str, tuple[int, ...]], facts: dict[str, bool],
    ) -> list[list[InducedRule]]:
        """Goal-driven backward chaining. Returns chains of rules that prove the goal."""
        chains: list[list[InducedRule]] = []
        goal_str = f"{goal[0]}({','.join(map(str, goal[1]))})"

        if facts.get(goal_str, False):
            return [[]]  # already true

        for rule in self._rules.values():
            then_str = f"{rule.then_predicate[0]}({','.join(map(str, rule.then_predicate[1]))})"
            if then_str == goal_str and rule.confidence >= self._min_confidence:
                chains.append([rule])

        # Extend chains by searching for rules that prove antecedents
        for depth in range(self._max_depth - 1):
            new_chains: list[list[InducedRule]] = []
            for chain in chains:
                if not chain:
                    continue
                first_rule = chain[0]
                for if_pred in first_rule.if_predicates:
                    sub_chains = self.backward_chain(if_pred, facts)
                    for sc in sub_chains:
                        combined = sc + chain
                        if not _chain_exists(combined, chains + new_chains):
                            new_chains.append(combined)
            if not new_chains:
                break
            chains.extend(new_chains)

        return chains

    # ------------------------------------------------------------------ diagnostics

    @property
    def capacity(self) -> int:
        return self._max_rules

    def __len__(self) -> int:
        return len(self._rules)

    def summary(self) -> dict:
        return {
            "num_rules": len(self._rules),
            "capacity": self._max_rules,
            "mean_confidence": (
                sum(r.confidence for r in self._rules.values()) /
                max(1, len(self._rules))
            ),
            "induced_from_scratch": sum(
                1 for r in self._rules.values() if r.derivation_id >= 0
            ),
        }

    def get_rules_text(self) -> list[str]:
        return [str(r) for r in sorted(
            self._rules.values(), key=lambda r: -r.confidence,
        )[:20]]

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_id": self._next_id,
            "episodes_accumulated": self._episodes_accumulated,
            "positive_combos": {sig: dict(acts) for sig, acts in self._positive_combos.items()},
            "negative_combos": {sig: dict(acts) for sig, acts in self._negative_combos.items()},
            "rules": [
                {
                    "if_predicates": [(p, tuple(args)) for p, args in r.if_predicates],
                    "then_predicate": (r.then_predicate[0], tuple(r.then_predicate[1])),
                    "confidence": r.confidence,
                    "positive_examples": r.positive_examples,
                    "negative_examples": r.negative_examples,
                    "derivation_id": r.derivation_id,
                }
                for r in self._rules.values()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._next_id = int(state["next_id"])
        self._episodes_accumulated = int(state.get("episodes_accumulated", 0))
        self._positive_combos = state.get("positive_combos", {})
        self._negative_combos = state.get("negative_combos", {})
        self._rules.clear()
        for r_dict in state["rules"]:
            rule = InducedRule(
                if_predicates=[(p, tuple(args)) for p, args in r_dict["if_predicates"]],
                then_predicate=(r_dict["then_predicate"][0], tuple(r_dict["then_predicate"][1])),
                confidence=r_dict["confidence"],
                positive_examples=r_dict["positive_examples"],
                negative_examples=r_dict["negative_examples"],
                derivation_id=r_dict["derivation_id"],
            )
            self._rules[rule.derivation_id] = rule

    def set_color_centroid(self, color: str, embedding: torch.Tensor) -> None:
        self._color_centroids[color] = embedding.detach().cpu()


# ---------------------------------------------------------------------- helpers


def _parse_predicate_tuple(s: str) -> tuple[str, tuple[int, ...]] | None:
    """Parse 'near(s0,s1)' → ('near', ('s0', 's1'))."""
    try:
        name, args_str = s.split("(", 1)
        args_str = args_str.rstrip(")")
        args = tuple(int(a.strip()[1:]) for a in args_str.split(",") if a.strip())
        return name, args
    except (ValueError, IndexError):
        return None


def _chain_exists(
    chain: list[InducedRule],
    existing: list[list[InducedRule]],
) -> bool:
    ids = tuple(r.derivation_id for r in chain)
    for ex in existing:
        if tuple(r.derivation_id for r in ex) == ids:
            return True
    return False
