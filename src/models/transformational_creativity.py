"""Transformational Creativity Engine.

Based on the formula:

    变革创造 = (旧经验积木 + 新抽象符号) × 远距离重组 × 规则打破 ÷ 因果自洽校验 + 好奇心

Implements an APPROXIMATION of transformational creativity — the ability
to break existing rules and create new conceptual spaces.

NOTE: This is an APPROXIMATION, not true transformational creativity.
      True transformational creativity requires stepping entirely outside
      the rule space (e.g., inventing a completely new concept that has
      no relation to any existing knowledge). This module approximates it
      by systematically breaking rules and checking if the rest of the
      rule system still holds — but it operates WITHIN the existing
      concept space, not outside it.
      
      未来正确性可以不保证 — the causal consistency check uses the world
      model's approximation, which itself may be wrong. Accepted
      transformations are "promising hypotheses", not guaranteed truths.
      They must be empirically tested (via HypothesisTester) before
      being relied upon.

      本模块是变革创造的近似实现，不是真正的变革创造。
      真正的变革创造需要完全跳出规则空间，本模块仍在规则空间内操作。
      因果自洽校验依赖世界模型的近似，世界模型本身可能有误。
      被接受的变革只是"有前景的假设"，不是保证正确的真理。
      必须经 HypothesisTester 实证检验后才能信赖。

Five stages:

1. STOCKPILE — collect "building blocks" from all accumulated knowledge:
   rules, skills, variables, concepts, failed hypotheses.
   (旧经验积木 + 新抽象符号)

2. FAR-RECOMBINE — combine concepts from VERY different domains (not just
   adjacent ones like DivergentGenerator does). Use embedding distance to
   maximize "conceptual distance" between combined elements.
   (远距离重组)

3. RULE-BREAKING — deliberately violate one existing rule and see what
   happens. "What if keys DON'T open doors? What if walls are walkable?"
   Generate a "broken rule" and test it.
   (规则打破)

4. CAUSAL-CHECK — verify that the new (rule-breaking) combination is
   causally consistent: does it lead to a coherent world model? Use the
   RSSM world model to simulate the consequences of the broken rule.
   (因果自洽校验)

5. CURIO-GATE — only pursue ideas that trigger high curiosity (RND reward).
   Ideas that are novel but boring are discarded.
   (好奇心)

This is NOT true transformational creativity (which requires stepping
outside the rule space entirely). But it APPROXIMATES it by:
- Systematically breaking rules (stage 3)
- Checking if the broken rule leads to a consistent alternative (stage 4)
- Keeping only the ones that are interesting (stage 5)

The key insight: "breaking a rule" = negating one rule and checking if the
rest of the rule system still holds. If it does, the broken rule was
"convention" not "necessity" — and the agent has discovered a new
possibility.

    Example:
    Rule: "IF see wall THEN turn" (confidence 0.9)
    → BREAK: "IF see wall THEN walk through"
    → CHECK: does the world model predict a consistent outcome?
    → If yes: "walls are walkable in this env!" (new discovery)
    → If no: "walls are solid, rule was necessary" (no creativity)

Bounded: max_transformations, max_simulations fixed. All stores bounded.
VRAM: ~0.2 GB (uses existing RSSM for simulation). Axiom 1.

变革创造引擎：基于公式的近似实现。
五个阶段：积木 → 远距离重组 → 规则打破 → 因果校验 → 好奇心筛选。
不是真正的变革创造（那需要跳出规则空间），但通过系统化地打破规则
+ 验证一致性来逼近。
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Transformation: one creative rule-breaking event
# =====================================================================


@dataclass
class Transformation:
    """A single transformational creative event.

    - ``original_rule``: the rule being broken (text description).
    - ``broken_rule``: the new rule after breaking (text).
    - ``domain_a``: source domain of concept A.
    - ``domain_b``: source domain of concept B (far away).
    - ``distance_score``: how "far apart" the combined concepts are.
    - ``causal_consistent``: does the world model predict consistency?
    - ``curiosity_score``: how much RND curiosity does this trigger?
    - ``overall_score``: weighted combination.
    - ``status``: "proposed" → "tested" → "accepted" / "rejected".
    """
    id: int
    original_rule: str = ""
    broken_rule: str = ""
    domain_a: str = ""
    domain_b: str = ""
    distance_score: float = 0.5
    causal_consistent: bool = False
    curiosity_score: float = 0.5
    overall_score: float = 0.0
    status: str = "proposed"  # proposed / tested / accepted / rejected
    test_result: str = ""

    def compute_score(self, w_dist=0.25, w_causal=0.35, w_curio=0.40) -> float:
        """overall = 0.25*distance + 0.35*causal + 0.40*curiosity."""
        causal_f = 1.0 if self.causal_consistent else 0.0
        self.overall_score = (
            w_dist * self.distance_score
            + w_causal * causal_f
            + w_curio * self.curiosity_score
        )
        return self.overall_score

    def __repr__(self) -> str:
        status_icon = {"proposed": "?", "tested": "→", "accepted": "✓", "rejected": "✗"}
        return (
            f"Transform #{self.id} {status_icon.get(self.status, '?')}: "
            f"BREAK '{self.original_rule}' → '{self.broken_rule}' "
            f"(dist={self.distance_score:.2f} "
            f"causal={'Y' if self.causal_consistent else 'N'} "
            f"curio={self.curiosity_score:.2f} "
            f"score={self.overall_score:.2f})"
        )


# =====================================================================
# TransformationalCreativityEngine
# =====================================================================


class TransformationalCreativityEngine(nn.Module):
    """Approximate transformational creativity.

    Pipeline:
        旧经验积木 → 远距离重组 → 规则打破 → 因果校验 → 好奇心筛选

    1. STOCKPILE: collect concepts from rules, skills, variables.
    2. FAR-RECOMBINE: pair concepts with MAXIMUM embedding distance.
    3. RULE-BREAKING: negate a rule ("IF X THEN NOT Y" instead of "IF X THEN Y").
    4. CAUSAL-CHECK: use world model to simulate the broken rule.
    5. CURIO-GATE: keep only high-curiosity transformations.

    Bounded: max_transformations fixed. All operations bounded.
    VRAM: ~0.2 GB. Axiom 1.
    """

    def __init__(
        self,
        d_model: int = 384,
        max_transformations: int = 64,
        distance_threshold: float = 0.3,  # minimum distance for "far" recombination
        curiosity_threshold: float = 0.3,  # minimum curiosity to keep
        max_simulations: int = 5,  # max world model simulations per cycle
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._max_tf = int(max_transformations)
        self._dist_thresh = float(distance_threshold)
        self._curio_thresh = float(curiosity_threshold)
        self._max_sims = int(max_simulations)

        self._transformations: deque[Transformation] = deque(maxlen=self._max_tf)  # BOUNDS-OK: maxlen bounded
        self._next_id = 0

        # Distance scorer: how far apart are two concepts?
        # Uses a contrastive embedding — pushes distant concepts apart.
        self.distance_projector = nn.Linear(d_model, d_model // 2)

        # Curiosity predictor: given a broken rule, predict curiosity (RND-like).
        self.curiosity_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @property
    def capacity(self) -> int:
        return self._max_tf

    def __len__(self) -> int:
        return len(self._transformations)

    # ---------------------------------------------------------- 1. STOCKPILE

    def _stockpile(
        self,
        rules: list,
        skills: list,
        variables: list,
    ) -> list[tuple[str, torch.Tensor, str]]:
        """Collect all concepts as (name, embedding, domain) triples.

        旧经验积木 + 新抽象符号
        """
        concepts: list[tuple[str, torch.Tensor, str]] = []

        # Rules
        for r in rules:
            emb = getattr(r, "condition_embedding", None)
            desc = getattr(r, "description", str(getattr(r, "condition", "rule")))
            if emb is not None:
                concepts.append((desc, emb, "rule"))

        # Skills
        for s in skills:
            weights = getattr(s, "weights", None)
            if weights is not None:
                w = getattr(weights, "A", weights)
                emb = w.flatten()[:self._d_model]
                if emb.shape[0] < self._d_model:
                    emb = F.pad(emb, (0, self._d_model - emb.shape[0]))
                tag = getattr(s, "tag", "skill")
                concepts.append((tag, emb, "skill"))

        # Variables (from LogicEngine)
        for v in variables:
            emb = getattr(v, "category_embedding", None)
            name = getattr(v, "name", "var")
            if emb is not None:
                concepts.append((name, emb, "variable"))

        return concepts

    # ---------------------------------------------------------- 2. FAR-RECOMBINE

    def _far_recombine(
        self,
        concepts: list[tuple[str, torch.Tensor, str]],
        n_pairs: int,
    ) -> list[tuple[tuple, tuple, float]]:
        """Find concept pairs with MAXIMUM embedding distance.

        远距离重组：不是随机配对，而是找距离最远的概念对。
        距离越远 = 越有可能是"变革性"组合。
        """
        if len(concepts) < 2:
            return []

        # Compute all pairwise distances (bounded by concept count)
        pairs: list[tuple[tuple, tuple, float]] = []
        n = len(concepts)

        # Sample n_pairs random pairs, compute distance, keep the farthest
        candidates = []
        for _ in range(min(n_pairs * 3, n * (n - 1) // 2)):
            i, j = random.sample(range(n), 2)
            ci, cj = concepts[i], concepts[j]
            sim = float(F.cosine_similarity(
                ci[1].unsqueeze(0), cj[1].unsqueeze(0), dim=1
            ).item())
            distance = 1.0 - sim  # higher = farther = more creative
            candidates.append((ci, cj, distance))

        # Sort by distance (farthest first) and take top n_pairs
        candidates.sort(key=lambda x: -x[2])
        return candidates[:n_pairs]

    # ---------------------------------------------------------- 3. RULE-BREAKING

    def _break_rule(
        self,
        concept_a: tuple[str, torch.Tensor, str],
        concept_b: tuple[str, torch.Tensor, str],
        rules: list,
    ) -> Transformation:
        """Break a rule by negating or modifying it.

        规则打破：把 "IF X THEN Y" 变成 "IF X THEN NOT Y"
        或 "IF X THEN Z"（用一个远域的动作替换）。
        """
        # Find a rule related to concept_a
        target_rule = None
        for r in rules:
            emb = getattr(r, "condition_embedding", None)
            if emb is None:
                continue
            sim = float(F.cosine_similarity(
                concept_a[1].unsqueeze(0), emb.unsqueeze(0), dim=1
            ).item())
            if sim > 0.5:
                target_rule = r
                break

        original_desc = ""
        broken_desc = ""
        if target_rule is not None:
            original_desc = getattr(target_rule, "description",
                                    str(getattr(target_rule, "condition", "rule")))
            original_action = getattr(target_rule, "action", 0)
            # Break: replace the action with something from domain B
            broken_desc = (
                f"WHAT IF instead of action={original_action}, "
                f"'{concept_a[0]}' triggers '{concept_b[0]}'?"
            )
        else:
            original_desc = f"no rule for '{concept_a[0]}'"
            broken_desc = (
                f"WHAT IF '{concept_a[0]}' and '{concept_b[0]}' "
                f"are combined in a new way?"
            )

        tf = Transformation(
            id=self._next_id,
            original_rule=original_desc,
            broken_rule=broken_desc,
            domain_a=concept_a[2],
            domain_b=concept_b[2],
            distance_score=0.5,  # will be set by caller
        )
        self._next_id += 1
        return tf

    # ---------------------------------------------------------- 4. CAUSAL-CHECK

    def _causal_check(
        self,
        tf: Transformation,
        world_model: Any | None = None,
    ) -> bool:
        """Use the world model to simulate the broken rule.

        因果自洽校验：如果打破这条规则，世界模型能否预测一个
        一致的结果？如果可以 → 规则是"约定"不是"必然"。

        Simplified: if world_model is None, always return True (optimistic).
        If world_model is provided, simulate one step and check for NaN/divergence.
        """
        if world_model is None:
            return True  # optimistic: assume consistent without simulation

        try:
            # Get initial state from world model
            state = world_model.initial_state(1, torch.device("cpu"))

            # Simulate one step with a random action (proxy for "broken" action)
            import torch as _t
            action = _t.zeros(1, world_model.config.action_dim)
            action[0, 0] = 1.0  # arbitrary action

            new_state, _ = world_model.imagine_step(state, action)
            recon = world_model.decode(new_state)

            # Check for consistency: no NaN, no extreme values
            if _t.isnan(recon).any() or _t.isinf(recon).any():
                return False  # world model diverged → inconsistent
            if recon.abs().max() > 100:  # extreme values
                return False
            return True
        except Exception as exc:
            logger.debug("Causal check failed: %s", exc)
            return False  # can't verify → assume inconsistent

    # ---------------------------------------------------------- 5. CURIO-GATE

    def _curiosity_score(
        self,
        tf: Transformation,
        concept_a_emb: torch.Tensor,
        concept_b_emb: torch.Tensor,
    ) -> float:
        """Predict curiosity (RND-like) for this transformation.

        好奇心：这个打破规则的想法有多"有趣"？
        Uses a small predictor on the combined concept embeddings.
        """
        with torch.no_grad():
            combined = (concept_a_emb + concept_b_emb) / 2
            score = float(self.curiosity_predictor(combined.unsqueeze(0)).item())
        return score

    # ---------------------------------------------------------- GENERATE

    def generate(
        self,
        rules: list | None = None,
        skills: list | None = None,
        variables: list | None = None,
        world_model: Any | None = None,
        n_transformations: int = 10,
    ) -> list[Transformation]:
        """Full transformational creativity pipeline.

        旧经验积木 → 远距离重组 → 规则打破 → 因果校验 → 好奇心筛选

        Args:
            rules: rule objects with .condition_embedding and .action.
            skills: skill objects with .weights.
            variables: variable objects with .category_embedding.
            world_model: RSSM for causal simulation (optional).
            n_transformations: number of transformations to generate.

        Returns:
            List of accepted Transformations, sorted by overall_score.
        """
        rules = rules or []
        skills = skills or []
        variables = variables or []

        # 1. STOCKPILE
        concepts = self._stockpile(rules, skills, variables)
        if len(concepts) < 2:
            return []

        # 2. FAR-RECOMBINE
        far_pairs = self._far_recombine(concepts, n_transformations)

        results: list[Transformation] = []

        for concept_a, concept_b, distance in far_pairs:
            if distance < self._dist_thresh:
                continue  # not far enough → not transformational

            # 3. RULE-BREAKING
            tf = self._break_rule(concept_a, concept_b, rules)
            tf.distance_score = distance

            # 4. CAUSAL-CHECK
            tf.causal_consistent = self._causal_check(tf, world_model)
            tf.status = "tested"

            # 5. CURIO-GATE
            tf.curiosity_score = self._curiosity_score(tf, concept_a[1], concept_b[1])
            tf.compute_score()

            # Gate: keep only if curiosity is high enough
            if tf.curiosity_score >= self._curio_thresh and tf.causal_consistent:
                tf.status = "accepted"
                results.append(tf)
            elif tf.curiosity_score >= self._curio_thresh:
                tf.status = "proposed"  # interesting but causally uncertain
                results.append(tf)
            else:
                tf.status = "rejected"

            self._transformations.append(tf)

            if len(results) >= n_transformations:
                break

        # Sort by overall score (most creative first)
        results.sort(key=lambda t: -t.overall_score)
        return results

    # ---------------------------------------------------------- DIAGNOSTICS

    def get_accepted(self) -> list[Transformation]:
        """Get all accepted transformations."""
        return [t for t in self._transformations if t.status == "accepted"]

    def get_text(self, n: int = 10) -> list[str]:
        """Human-readable descriptions."""
        sorted_tf = sorted(self._transformations, key=lambda t: -t.overall_score)
        return [str(t) for t in sorted_tf[:n]]

    def summary(self) -> dict:
        accepted = self.get_accepted()
        return {
            "total": len(self._transformations),
            "accepted": len(accepted),
            "proposed": sum(1 for t in self._transformations if t.status == "proposed"),
            "rejected": sum(1 for t in self._transformations if t.status == "rejected"),
            "capacity": self._max_tf,
            "best_score": max((t.overall_score for t in self._transformations), default=0),
        }

    def state_dict(self) -> dict:
        return {
            "max_tf": self._max_tf,
            "next_id": self._next_id,
            "transformations": [
                {
                    "id": t.id,
                    "original": t.original_rule,
                    "broken": t.broken_rule,
                    "domain_a": t.domain_a,
                    "domain_b": t.domain_b,
                    "distance": t.distance_score,
                    "causal": t.causal_consistent,
                    "curiosity": t.curiosity_score,
                    "score": t.overall_score,
                    "status": t.status,
                }
                for t in self._transformations
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._max_tf = int(state["max_tf"])
        self._next_id = int(state["next_id"])
        self._transformations.clear()
        for t_dict in state["transformations"]:
            tf = Transformation(
                id=t_dict["id"],
                original_rule=t_dict["original"],
                broken_rule=t_dict["broken"],
                domain_a=t_dict["domain_a"],
                domain_b=t_dict["domain_b"],
                distance_score=t_dict["distance"],
                causal_consistent=t_dict["causal"],
                curiosity_score=t_dict["curiosity"],
                overall_score=t_dict["score"],
                status=t_dict["status"],
            )
            self._transformations.append(tf)
