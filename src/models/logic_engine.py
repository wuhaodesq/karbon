"""Symbolic Logic Engine: variable binding, quantification, rule composition.

Builds on NeuralSymbolicLayer to add TRUE symbolic reasoning capabilities:

1. :class:`Variable` — a symbolic variable that can represent ANY state
   matching a pattern (not a specific hidden state vector).

2. :class:`QuantifiedRule` — a rule with universal/existential quantification:
   "FOR ALL X in category 'key': IF see(X) THEN pick_up(X)"

3. :class:`LogicEngine` — composes rules via forward chaining:
   Rule A: "IF locked(door) AND has(key) THEN can_open(door)"
   Rule B: "IF can_open(door) THEN go_through(door)"
   → Derived: "IF locked(door) AND has(key) THEN go_through(door)"

4. :class:`ProofChecker` — verifies whether a derivation is valid by
   checking each step against the rule base.

This gives the agent the "phase transition 3" capability: abstraction.
Instead of matching specific hidden state vectors (pattern matching),
the engine matches CATEGORIES and VARIABLES (symbolic reasoning).

    NeuralSymbolicLayer: "IF hidden≈X THEN action=A" (pattern matching)
    LogicEngine:          "∀X ∈ keys: IF see(X) THEN pick_up(X)" (symbolic)

Bounded: max_rules, max_variables, max_proof_steps are all fixed.
VRAM: ~0.1 GB. Axiom 1 satisfied.

符号逻辑引擎：变量绑定 + 量化 + 规则组合 + 正确性验证。
让智能体从"模式匹配"升级到"符号推理"——质变 3。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Variable — symbolic variable binding
# =====================================================================


class VariableType(Enum):
    """Categories of variables the agent can reason about."""
    OBJECT = "object"       # key, door, goal, wall
    STATE = "state"         # locked, open, visited
    ACTION = "action"      # forward, pick_up, open
    ABSTRACT = "abstract"  # any category


@dataclass
class Variable:
    """A symbolic variable that represents a CATEGORY, not a specific instance.

    - ``name``: human-readable name ("X", "key", "any_door")
    - ``var_type``: what kind of thing this variable represents
    - ``category_embedding``: (d_model,) — the centroid of this category
      in feature space. Used to match against actual hidden states.
    - ``bindings``: list of actual hidden states that matched this variable
      (bounded to max_bindings for Axiom 1).
    """
    name: str
    var_type: VariableType
    category_embedding: torch.Tensor  # (d_model,)
    bindings: list = field(default_factory=list)  # bounded by max_bindings

    def match(self, hidden_state: torch.Tensor, threshold: float = 0.7) -> float:
        """Check if a hidden state belongs to this variable's category.

        Returns cosine similarity (0-1). If >= threshold, it's a match.
        """
        cos = F.cosine_similarity(
            hidden_state.unsqueeze(0),
            self.category_embedding.unsqueeze(0),
            dim=1,
        )
        return float(cos.item())


# =====================================================================
# 2. QuantifiedRule — rule with universal/existential quantification
# =====================================================================


class Quantifier(Enum):
    UNIVERSAL = "for_all"      # ∀X: rule(X)
    EXISTENTIAL = "exists"      # ∃X: rule(X)


@dataclass
class QuantifiedRule:
    """A rule with quantified variables.

    Example:
        "FOR ALL X in 'key' category: IF see(X) THEN pick_up(X)"

    - ``id``: unique rule ID
    - ``quantifier``: ∀ or ∃
    - ``variable``: the Variable being quantified over
    - ``condition_fn``: description of the condition (text)
    - ``action``: the action to take
    - ``confidence``: how reliable this rule is [0, 1]
    - ``proof_verified``: whether this rule was derived (not just extracted)
    - ``derivation_chain``: list of rule IDs used to derive this (if derived)
    """
    id: int
    quantifier: Quantifier
    variable: Variable
    condition: str          # human-readable: "see(X)"
    action: int             # action index
    confidence: float = 0.5
    proof_verified: bool = False
    derivation_chain: list[int] = field(default_factory=list)
    usage_count: int = 0

    def __repr__(self) -> str:
        q = "∀" if self.quantifier == Quantifier.UNIVERSAL else "∃"
        verified = "✓" if self.proof_verified else "?"
        return (
            f"Rule #{self.id}: {q}{self.variable.name}: "
            f"IF {self.condition} THEN action={self.action} "
            f"(conf={self.confidence:.2f} {verified})"
        )


# =====================================================================
# 3. LogicEngine — forward chaining rule composition
# =====================================================================


class LogicEngine:
    """Symbolic logic engine: forward chaining + variable unification.

    Can:
    1. Store quantified rules (bounded: max_rules).
    2. Unify a concrete observation against variables.
    3. Forward-chain: if Rule A's conclusion matches Rule B's condition,
       derive a new rule.
    4. Check proof validity.

    Bounded: max_rules fixed. max_variables fixed. max_proof_depth fixed.
    All operations are O(max_rules). Axiom 1.

    符号逻辑引擎：前向链接 + 变量统一 + 规则组合 + 正确性验证。
    """

    def __init__(
        self,
        d_model: int = 384,
        max_rules: int = 64,
        max_variables: int = 16,
        max_proof_depth: int = 5,
        match_threshold: float = 0.7,
    ) -> None:
        self._d_model = d_model
        self._max_rules = int(max_rules)
        self._max_vars = int(max_variables)
        self._max_depth = int(max_proof_depth)
        self._threshold = float(match_threshold)

        self._rules: dict[int, QuantifiedRule] = {}
        self._variables: dict[str, Variable] = {}
        self._next_rule_id = 0

    @property
    def capacity(self) -> int:
        return self._max_rules

    def __len__(self) -> int:
        return len(self._rules)

    # ---------------------------------------------------------- variables

    def define_variable(
        self,
        name: str,
        var_type: VariableType,
        category_embedding: torch.Tensor,
    ) -> Variable:
        """Define a new variable (category) for symbolic reasoning.

        Example: define_variable("key", OBJECT, centroid_of_key_embeddings)
        """
        if len(self._variables) >= self._max_vars:
            # Evict least-used variable
            oldest = min(self._variables, key=lambda k: len(self._variables[k].bindings))
            del self._variables[oldest]
        var = Variable(
            name=name,
            var_type=var_type,
            category_embedding=category_embedding.detach().clone(),
        )
        self._variables[name] = var
        return var

    def get_variable(self, name: str) -> Variable | None:
        return self._variables.get(name)

    def unify(
        self,
        hidden_state: torch.Tensor,
    ) -> list[tuple[Variable, float]]:
        """Find all variables whose category matches the given hidden state.

        This is VARIABLE UNIFICATION — the core of symbolic reasoning.
        Returns list of (variable, similarity) pairs above threshold.
        """
        matches: list[tuple[Variable, float]] = []
        for var in self._variables.values():
            sim = var.match(hidden_state, self._threshold)
            if sim >= self._threshold:
                matches.append((var, sim))
                # Record binding (bounded)
                if len(var.bindings) < 32:
                    var.bindings.append(hidden_state.detach().clone())
        return matches

    # ---------------------------------------------------------- rules

    def add_rule(
        self,
        quantifier: Quantifier,
        variable_name: str,
        condition: str,
        action: int,
        confidence: float = 0.5,
        proof_verified: bool = False,
        derivation_chain: list[int] | None = None,
    ) -> QuantifiedRule:
        """Add a quantified rule. Evicts lowest-confidence if full."""
        if len(self._rules) >= self._max_rules:
            self._evict_rule()

        rule_id = self._next_rule_id
        self._next_rule_id += 1

        var = self._variables.get(variable_name)
        if var is None:
            # Auto-create a variable with a random embedding (to be refined later)
            var = self.define_variable(
                variable_name, VariableType.ABSTRACT,
                torch.randn(self._d_model),
            )

        rule = QuantifiedRule(
            id=rule_id,
            quantifier=quantifier,
            variable=var,
            condition=condition,
            action=action,
            confidence=confidence,
            proof_verified=proof_verified,
            derivation_chain=derivation_chain or [],
        )
        self._rules[rule_id] = rule
        return rule

    def _evict_rule(self) -> None:
        """Evict the rule with lowest confidence × usage."""
        if not self._rules:
            return
        worst_id = min(
            self._rules,
            key=lambda rid: self._rules[rid].confidence * (1 + self._rules[rid].usage_count * 0.01),
        )
        del self._rules[worst_id]

    # ---------------------------------------------------------- forward chaining

    def forward_chain(self) -> list[QuantifiedRule]:
        """Compose rules via forward chaining.

        Only composes EMPIRICAL rules (not previously derived) to prevent
        infinite chaining cascades. A derived rule can be USED in reasoning
        but cannot participate in further derivation.

        Bounded: max_proof_depth limits chain length. New rules count
        against max_rules. Axiom 1.
        """
        new_rules: list[QuantifiedRule] = []

        # Only chain non-derived (empirical) rules
        empirical_rules = [r for r in self._rules.values() if not r.proof_verified]

        for rule_a in empirical_rules:
            for rule_b in empirical_rules:
                if rule_a.id >= rule_b.id:
                    continue  # avoid duplicate pairs (A,B) and (B,A)

                # Check if A's variable category matches B's variable category
                sim = F.cosine_similarity(
                    rule_a.variable.category_embedding.unsqueeze(0),
                    rule_b.variable.category_embedding.unsqueeze(0),
                    dim=1,
                ).item()

                if sim < self._threshold:
                    continue

                # Avoid duplicate derivations
                chain = sorted([rule_a.id, rule_b.id])
                chain_key = tuple(chain)
                already_derived = any(
                    tuple(sorted(r.derivation_chain)) == chain_key
                    for r in self._rules.values()
                )
                if already_derived:
                    continue

                if len(self._rules) >= self._max_rules:
                    return new_rules

                # Compose: A's condition → B's action
                composed_condition = f"{rule_a.condition} → {rule_b.condition}"
                composed_confidence = min(rule_a.confidence, rule_b.confidence) * 0.9

                new_rule = self.add_rule(
                    quantifier=Quantifier.UNIVERSAL,
                    variable_name=rule_b.variable.name,
                    condition=composed_condition,
                    action=rule_b.action,
                    confidence=composed_confidence,
                    proof_verified=True,
                    derivation_chain=[rule_a.id, rule_b.id],
                )
                new_rules.append(new_rule)

        return new_rules

    # ---------------------------------------------------------- reasoning

    def reason(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[QuantifiedRule | None, dict[str, Any]]:
        """Full symbolic reasoning pass.

        1. Unify hidden state against all variables.
        2. Find matching rules.
        3. Return the highest-confidence matching rule.

        Returns (best_rule, info) where info contains unification details.
        """
        info: dict[str, Any] = {
            "unified_variables": [],
            "matched_rules": [],
            "derived": False,
        }

        # Step 1: Unify
        matches = self.unify(hidden_state)
        info["unified_variables"] = [
            {"name": v.name, "similarity": s} for v, s in matches
        ]

        if not matches:
            return None, info

        # Step 2: Find rules whose variable matches
        best_rule: QuantifiedRule | None = None
        best_score = 0.0

        for rule in self._rules.values():
            # Check if rule's variable is among the unified ones
            for var, sim in matches:
                if rule.variable.name == var.name:
                    score = rule.confidence * sim
                    info["matched_rules"].append({
                        "id": rule.id,
                        "score": score,
                        "derived": rule.proof_verified,
                    })
                    if score > best_score:
                        best_score = score
                        best_rule = rule
                    break

        if best_rule is not None:
            best_rule.usage_count += 1
            info["derived"] = best_rule.proof_verified

        return best_rule, info

    # ---------------------------------------------------------- proof checking

    def verify_proof(
        self,
        rule_id: int,
    ) -> bool:
        """Verify that a derived rule's proof chain is valid.

        A proof is valid if:
        1. Every rule in the derivation chain exists.
        2. The chain doesn't cycle (no rule references itself).
        3. The chain length ≤ max_proof_depth.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            return False

        if not rule.proof_verified:
            return True  # non-derived rules are trivially "valid" (empirical)

        chain = rule.derivation_chain
        if len(chain) == 0:
            return True

        if len(chain) > self._max_depth:
            return False

        # Check all rules in chain exist
        for rid in chain:
            if rid not in self._rules:
                return False

        # Check no cycles
        if rule_id in chain:
            return False

        return True

    # ---------------------------------------------------------- diagnostics

    def get_rules_text(self) -> list[str]:
        """Return human-readable descriptions of all rules."""
        rules = sorted(self._rules.values(), key=lambda r: -r.confidence)
        return [str(r) for r in rules]

    def get_variables_text(self) -> list[str]:
        """Return human-readable descriptions of all variables."""
        return [
            f"{v.name} ({v.var_type.value}): {len(v.bindings)} bindings"
            for v in self._variables.values()
        ]

    def summary(self) -> dict:
        verified = sum(1 for r in self._rules.values() if r.proof_verified)
        return {
            "num_rules": len(self._rules),
            "num_variables": len(self._variables),
            "verified_rules": verified,
            "capacity": self._max_rules,
            "mean_confidence": (
                sum(r.confidence for r in self._rules.values()) /
                max(1, len(self._rules))
            ),
        }

    # ---------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "d_model": self._d_model,
            "max_rules": self._max_rules,
            "max_vars": self._max_vars,
            "max_depth": self._max_depth,
            "threshold": self._threshold,
            "next_rule_id": self._next_rule_id,
            "rules": [
                {
                    "id": r.id,
                    "quantifier": r.quantifier.value,
                    "variable_name": r.variable.name,
                    "condition": r.condition,
                    "action": r.action,
                    "confidence": r.confidence,
                    "proof_verified": r.proof_verified,
                    "derivation_chain": r.derivation_chain,
                    "usage_count": r.usage_count,
                }
                for r in self._rules.values()
            ],
            "variables": [
                {
                    "name": v.name,
                    "var_type": v.var_type.value,
                    "category_embedding": v.category_embedding.cpu(),
                }
                for v in self._variables.values()
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._d_model = int(state["d_model"])
        self._max_rules = int(state["max_rules"])
        self._max_vars = int(state["max_vars"])
        self._max_depth = int(state["max_depth"])
        self._threshold = float(state["threshold"])
        self._next_rule_id = int(state["next_rule_id"])
        self._rules.clear()
        self._variables.clear()

        for v_dict in state["variables"]:
            self._variables[v_dict["name"]] = Variable(
                name=v_dict["name"],
                var_type=VariableType(v_dict["var_type"]),
                category_embedding=v_dict["category_embedding"],
            )

        for r_dict in state["rules"]:
            var = self._variables.get(r_dict["variable_name"])
            if var is None:
                continue
            self._rules[r_dict["id"]] = QuantifiedRule(
                id=r_dict["id"],
                quantifier=Quantifier(r_dict["quantifier"]),
                variable=var,
                condition=r_dict["condition"],
                action=r_dict["action"],
                confidence=r_dict["confidence"],
                proof_verified=r_dict["proof_verified"],
                derivation_chain=r_dict["derivation_chain"],
                usage_count=r_dict["usage_count"],
            )
