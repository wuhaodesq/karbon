"""Kanren-backed symbol engine for Stage 16 (Y1 path).

Replaces cosine-similarity matching with real logical unification.
Implements the ROADMAP Y1 design:
  - neural predicates -> kanren facts/rules
  - kanren inference -> structured results
  - results feed back via REINFORCE (gradient does NOT flow through engine)

Bounded: max_facts=512, max_rules=128, max_resolution_steps=200.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# kanren is optional; gracefully degrade to symbolic_reasoning fallback
_kanren_available = False
try:
    from kanren import facts, Relation, var, run, conde, lany
    _kanren_available = True
except ImportError:
    pass

# Fallback: use symbolic_reasoning's unification
try:
    from src.models.symbolic_reasoning import unify, try_modus_ponens
    _fallback_unify = True
except ImportError:
    _fallback_unify = False


@dataclass
class SymbolResult:
    """Result of a symbolic inference query."""
    query: str
    answers: list[Any]          # list of bindings/answers
    confidence: float           # 0..1, based on rule support
    rule_chain: list[str]       # human-readable derivation
    correct: bool | None = None # set by environment feedback (learning-back)


class SymbolBackend:
    """Real logical reasoning backend using kanren (or fallback).

    Converts neural predicate extractions into kanren facts/rules,
    runs inference, and returns structured results for learning-back.

    Bounded guarantees:
        - max_facts: 512
        - max_rules: 128
        - max_resolution_steps: 200
    """

    def __init__(
        self,
        max_facts: int = 512,
        max_rules: int = 128,
        max_resolution_steps: int = 200,
    ) -> None:
        self.max_facts = max_facts
        self.max_rules = max_rules
        self.max_resolution_steps = max_resolution_steps

        # kanren relations (created lazily)
        self._relations: dict[str, Any] = {}
        self._fact_count: int = 0
        self._rule_count: int = 0

        # Fallback storage (when kanren unavailable): use simple dict
        self._facts_db: dict[str, list[tuple]] = {}
        self._rules_db: list[dict] = []

        # Learning-back buffer: stores recent inference results
        # for REINFORCE reward computation
        self._inference_buffer: list[SymbolResult] = []
        self._buffer_capacity = 256  # BOUNDS-OK: fixed cap

        # Statistics
        self._total_queries = 0
        self._correct_predictions = 0

    @property
    def available(self) -> bool:
        return _kanren_available or _fallback_unify

    @property
    def capacity(self) -> int:
        return self._buffer_capacity

    def __len__(self) -> int:
        return len(self._inference_buffer)

    # -------------------------------------------------------- fact/rule ingestion

    def add_causal_edges(self, edges: list[dict]) -> None:
        """Convert causal discovery edges to facts.

        Each edge: {src, tgt, strength} -> causes(src, tgt)
        """
        for edge in edges:
            if self._fact_count >= self.max_facts:
                break
            src = self._clean_name(str(edge.get("src", "")))
            tgt = self._clean_name(str(edge.get("tgt", "")))
            strength = float(edge.get("strength", 0.0))
            if strength < 0.005 or not src or not tgt:
                continue
            fact_key = "causes"
            if _kanren_available and fact_key not in self._relations:
                self._relations[fact_key] = Relation()
            self._facts_db.setdefault(fact_key, []).append((src, tgt))
            if _kanren_available:
                facts(self._relations[fact_key], (src, tgt))
            self._fact_count += 1

    def add_induced_rules(self, rules: list) -> None:
        """Convert RuleInductionEngine rules to logical rules.

        Each InducedRule: if_predicates + then_predicate -> Horn clause
        """
        for rule in rules:
            if self._rule_count >= self.max_rules:
                break
            if_p = rule.if_predicates if hasattr(rule, "if_predicates") else rule.get("if_predicates", [])
            then_p = rule.then_predicate if hasattr(rule, "then_predicate") else rule.get("then_predicate", None)
            conf = rule.confidence if hasattr(rule, "confidence") else rule.get("confidence", 0.5)
            if not then_p:
                continue
            self._rules_db.append({
                "if": if_p,
                "then": then_p,
                "confidence": conf,
            })
            self._rule_count += 1

    def add_symbolic_rules(self, rules: dict) -> None:
        """Add NeuralSymbolicLayer RuleMemory rules (embedding-based).

        These are stored as action-condition mappings.
        """
        for rid, rule_data in rules.items():
            if self._rule_count >= self.max_rules:
                break
            action = rule_data.get("action", -1)
            conf = rule_data.get("confidence", 0.5)
            desc = rule_data.get("description", "")
            if action < 0 or not desc:
                continue
            self._rules_db.append({
                "if": [("condition", (rid,))],
                "then": ("action", (action,)),
                "confidence": conf,
                "description": desc,
            })
            self._rule_count += 1

    def add_fact(self, predicate: str, args: tuple) -> bool:
        """Add a ground fact (bounded by ``max_facts``).

        Public premise-injection entry point used by the formal-reasoning
        probe and external knowledge injection. 显式注入事实。
        """
        if self._fact_count >= self.max_facts:
            return False
        key = self._clean_name(str(predicate))
        if not key:
            return False
        if _kanren_available and key not in self._relations:
            self._relations[key] = Relation()
        self._facts_db.setdefault(key, []).append(tuple(args))
        if _kanren_available:
            facts(self._relations[key], tuple(args))
        self._fact_count += 1
        return True

    def add_rule(
        self,
        if_predicates: list,
        then_predicate: tuple,
        confidence: float = 1.0,
    ) -> bool:
        """Add a Horn clause (bounded by ``max_rules``).

        ``if_predicates``: list of (pred, args) with ``"_"`` wildcards;
        ``then_predicate``: (pred, args). 显式注入规则。
        """
        if self._rule_count >= self.max_rules:
            return False
        if not then_predicate:
            return False
        self._rules_db.append({
            "if": [tuple(p) for p in if_predicates],
            "then": tuple(then_predicate),
            "confidence": float(confidence),
        })
        self._rule_count += 1
        return True

    # -------------------------------------------------------- inference

    def query(self, predicate: str, args: tuple) -> SymbolResult:
        """Run a logical inference query.

        Args:
            predicate: e.g. "causes", "near", "action"
            args: tuple of arguments (constants or variable specs)

        Returns:
            SymbolResult with answers and derivation chain
        """
        self._total_queries += 1
        answers = []
        chain = []

        # 2026-09-23: query() serves the fact+rule semantics directly
        # (_match_args wildcards + single-level Horn chaining). The kanren
        # relations mirror facts only; their enumeration shape proved
        # unreliable in probes (populated relation returned 0 answers) and
        # rules were never compiled into relations, so the kanren branch was
        # removed from query(). relations remain maintained for future
        # proper resolution work.
        # 事实查询 + 单层规则前向链; kanren 关系仅镜像事实, 枚举路径不可靠已弃用。
        if predicate in self._facts_db:
            for fact in self._facts_db[predicate]:
                if self._match_args(fact, args):
                    answers.append(fact)
            chain.append(f"fact_lookup({predicate}{args}) -> {len(answers)} matches")

        # Multi-hop derivation layer (2026-09-25): bounded forward chaining
        # so rules can consume other rules' consequents (G3 composition).
        derived = self._derive_facts()
        if derived:
            chain.append(
                f"derived({sum(len(v) for v in derived.values())} facts)")

        # Forward chain through rules
        for rule in self._rules_db:
            if_preds = rule["if"]
            then_pred = rule["then"]
            if then_pred[0] == predicate:
                all_match = True
                for cond_pred, cond_args in if_preds:
                    if not (self._check_predicate(cond_pred, cond_args)
                            or self._check_derived(derived, cond_pred, cond_args)):
                        all_match = False
                        break
                if all_match:
                    # answers are ARG TUPLES (same shape as fact lookups and
                    # the derived layer); a rule conclusion contributes its
                    # args only, deduped against existing answers.
                    if then_pred[1] not in answers:
                        answers.append(then_pred[1])
                    chain.append(f"rule({if_preds} -> {then_pred}, conf={rule['confidence']:.2f})")

        # Query matches against the derived layer as well
        for fact in derived.get(predicate, []):
            if self._match_args(fact, args) and fact not in answers:
                answers.append(fact)
                chain.append(f"derived_match({predicate}{fact})")

        conf = min(1.0, len(answers) / max(1, len(self._rules_db))) if answers else 0.0

        result = SymbolResult(
            query=f"{predicate}{args}",
            answers=answers,
            confidence=conf,
            rule_chain=chain,
        )

        # Store in buffer for learning-back
        if len(self._inference_buffer) >= self._buffer_capacity:
            self._inference_buffer.pop(0)  # BOUNDS-OK: bounded
        self._inference_buffer.append(result)

        return result

    def predict_action(self, predicates: list[tuple[str, tuple]]) -> SymbolResult:
        """Predict best action given current predicates.

        Args:
            predicates: list of (predicate_name, args) that are currently true

        Returns:
            SymbolResult with predicted action in answers
        """
        self._total_queries += 1
        best_action = -1
        best_conf = 0.0
        chain = []

        for rule in self._rules_db:
            then_pred = rule["then"]
            if then_pred[0] != "action":
                continue
            if_preds = rule["if"]
            matched = 0
            for cond_pred, cond_args in if_preds:
                for pred_name, pred_args in predicates:
                    if cond_pred == pred_name and self._match_args(cond_args, pred_args):
                        matched += 1
                        break
            if matched == len(if_preds) and rule["confidence"] > best_conf:
                best_conf = rule["confidence"]
                best_action = then_pred[1][0] if len(then_pred[1]) > 0 else -1
                chain.append(f"rule_match({if_preds} -> action={best_action}, conf={best_conf:.2f})")

        result = SymbolResult(
            query=f"predict_action({predicates})",
            answers=[("action", best_action)] if best_action >= 0 else [],
            confidence=best_conf,
            rule_chain=chain,
        )

        if len(self._inference_buffer) >= self._buffer_capacity:
            self._inference_buffer.pop(0)
        self._inference_buffer.append(result)

        return result

    # -------------------------------------------------------- learning-back

    def feedback(self, query_idx: int, correct: bool) -> float:
        """Provide environment feedback for a past inference.

        This is the learning-back mechanism: the environment tells us
        whether a prediction was correct, and we compute a REINFORCE
        reward for the neural predicate extractor.

        Returns:
            reward: +1 for correct, -1 for incorrect, 0 for no data
        """
        if 0 <= query_idx < len(self._inference_buffer):
            result = self._inference_buffer[query_idx]
            result.correct = correct
            if correct:
                self._correct_predictions += 1
                return 1.0
            else:
                return -1.0
        return 0.0

    def get_reinforce_rewards(self) -> list[float]:
        """Get all REINFORCE rewards from the inference buffer.

        Rewards: +1 for correct predictions, -1 for incorrect, 0 for unverified.
        Call this at the end of each PPO update cycle.
        """
        rewards = []
        for result in self._inference_buffer:
            if result.correct is True:
                rewards.append(1.0)
            elif result.correct is False:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        return rewards

    def clear_buffer(self) -> None:
        """Clear the inference buffer after REINFORCE rewards are consumed."""
        self._inference_buffer.clear()

    # -------------------------------------------------------- state

    def state_dict(self) -> dict:
        return {
            "facts_db": self._facts_db,
            "rules_db": self._rules_db,
            "fact_count": self._fact_count,
            "rule_count": self._rule_count,
            "total_queries": self._total_queries,
            "correct_predictions": self._correct_predictions,
        }

    def load_state_dict(self, state: dict) -> None:
        self._facts_db = state.get("facts_db", {})
        self._rules_db = state.get("rules_db", [])
        self._fact_count = state.get("fact_count", 0)
        self._rule_count = state.get("rule_count", 0)
        self._total_queries = state.get("total_queries", 0)
        self._correct_predictions = state.get("correct_predictions", 0)
        # Rebuild kanren relations if available
        if _kanren_available:
            self._relations.clear()
            for pred, facts_list in self._facts_db.items():
                if pred not in self._relations:
                    self._relations[pred] = Relation()
                for f in facts_list:
                    facts(self._relations[pred], f)

    def summary(self) -> dict:
        return {
            "facts": self._fact_count,
            "rules": self._rule_count,
            "queries": self._total_queries,
            "correct": self._correct_predictions,
            "accuracy": self._correct_predictions / max(1, self._total_queries),
            "backend": "kanren" if _kanren_available else "fallback",
        }

    # -------------------------------------------------------- helpers

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean a name for use as a logical constant."""
        return name.replace(" ", "_").replace("-", "_").replace(".", "_").lower()

    @staticmethod
    def _match_args(fact_args: tuple, query_args: tuple) -> bool:
        """Check if fact args match query args (wildcard _ matches anything)."""
        if len(fact_args) != len(query_args):
            return False
        for f, q in zip(fact_args, query_args):
            if q == "_" or q is None:
                continue
            if f != q:
                return False
        return True

    def _check_predicate(self, pred_name: str, pred_args: tuple) -> bool:
        """Check if a predicate is true in the fact database."""
        if pred_name in self._facts_db:
            for fact in self._facts_db[pred_name]:
                if self._match_args(fact, pred_args):
                    return True
        return False

    @staticmethod
    def _check_derived(derived: dict, pred_name: str, pred_args: tuple) -> bool:
        """Check a predicate against the derived-fact layer (multi-hop)."""
        for fact in derived.get(pred_name, []):
            if SymbolBackend._match_args(fact, pred_args):
                return True
        return False

    def _derive_facts(self, max_rounds: int = 2, max_derived: int = 64) -> dict:
        """Bounded forward chaining over ground Horn rules (multi-hop).

        2026-09-25: closes the G3 composition gap — a rule whose antecedent
        is only derivable from ANOTHER rule (e.g. on(a,b),on(b,c) -> on(a,c),
        then on(a,c) -> finish(a)) never fired before because antecedents
        were checked against direct facts only. Each round applies every
        rule once against facts plus the PREVIOUS round's derived layer, so
        the round cap equals the derivation-depth cap and the result does
        not depend on rule order. Rules with wildcard consequents are
        skipped (their instantiation needs real unification; appending
        pre-image wildcards would create false positives).
        """
        derived: dict[str, list[tuple]] = {}
        count = 0
        for _ in range(max(0, int(max_rounds))):
            new_layer: dict[str, list[tuple]] = {}
            for rule in self._rules_db:
                if count + sum(len(v) for v in new_layer.values()) >= max_derived:
                    break
                then_pred, then_args = rule["then"][0], tuple(rule["then"][1])
                if any(str(a) == "_" or a is None for a in then_args):
                    continue
                ok = True
                for cond_pred, cond_args in rule["if"]:
                    if not (self._check_predicate(cond_pred, cond_args)
                            or self._check_derived(derived, cond_pred, cond_args)):
                        ok = False
                        break
                if not ok:
                    continue
                if then_args in self._facts_db.get(then_pred, []):
                    continue
                if (then_args in new_layer.get(then_pred, [])
                        or then_args in derived.get(then_pred, [])):
                    continue
                new_layer.setdefault(then_pred, []).append(then_args)
            if not new_layer:
                break
            for _pred, _lst in new_layer.items():
                derived.setdefault(_pred, []).extend(_lst)
                count += len(_lst)
        return derived
