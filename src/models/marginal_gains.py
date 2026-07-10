"""Three Marginal-Gain Modules — Knowledge Gap, Compositional Test, LP Tracker.

1. KnowledgeGapDetector — "I don't know what this is" → prioritize exploration
2. CompositionalTester — known "red" + known "ball" → can infer "red ball"?
3. LearningProgressTracker — flat LP → auto-boost curiosity

All zero GPU. Read existing module states. Pure signal processors.

知识缺口、组合泛化测试、学习进度跟踪。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Knowledge Gap Detector
# =====================================================================


class KnowledgeGapDetector(nn.Module):
    """Detect what the agent DOESN'T know, to prioritize active exploration.

    Two signals:
    - Slot gap: slot norm < threshold → "that slot is empty or I don't recognize the object"
    - Graph gap: concept_graph query returns empty → "I've never seen anything like this"

    High knowledge gap → CuriosityDirector weights shift toward exploration.
    ActiveExperimenter prioritizes testing the unknown.
    """

    def __init__(
        self,
        slot_norm_threshold: float = 0.15,
        graph_similarity_threshold: float = 0.3,
        gap_decay: float = 0.95,
    ) -> None:
        super().__init__()
        self._slot_threshold = slot_norm_threshold
        self._graph_threshold = graph_similarity_threshold
        self._decay = gap_decay
        self._gap_level: float = 0.0
        self._gaps_detected: int = 0

    def detect(
        self,
        slot_states: torch.Tensor | None,  # (num_slots, d_model)
        concept_graph: Any | None,
    ) -> float:
        """Return knowledge gap level [0, 1]. 0 = everything known, 1 = pure unknown."""
        gap_score = 0.0
        signals = 0

        # Slot gap: how many slots are inactive?
        if slot_states is not None:
            slot_norms = slot_states.norm(dim=-1)
            inactive = (slot_norms < self._slot_threshold * slot_norms.max()).float().mean()
            gap_score += float(inactive.item())
            signals += 1

        # Graph gap: how well does query match known concepts?
        if concept_graph is not None and hasattr(concept_graph, 'find_analog'):
            try:
                if slot_states is not None:
                    query = slot_states.mean(dim=0)
                    analogs = concept_graph.find_analog(query, k=1)
                    if not analogs or analogs[0][1] < self._graph_threshold:
                        gap_score += 1.0
                    else:
                        gap_score += 1.0 - analogs[0][1]
                    signals += 1
            except Exception:
                pass

        if signals == 0:
            return 0.0

        raw_gap = gap_score / signals
        self._gap_level = self._decay * self._gap_level + (1 - self._decay) * raw_gap

        if raw_gap > 0.5:
            self._gaps_detected += 1
            logger.debug("[knowledge_gap] gap=%.2f (slot+graph)", raw_gap)

        return self._gap_level

    def should_explore(self) -> bool:
        return self._gap_level > 0.4

    @property
    def gap_level(self) -> float:
        return self._gap_level

    def summary(self) -> dict:
        return {
            "gap_level": self._gap_level,
            "total_gaps": self._gaps_detected,
            "should_explore": self.should_explore(),
        }


# =====================================================================
# 2. Compositional Generalization Tester
# =====================================================================


class CompositionalTester:
    """Test if the agent can combine known concepts to infer new ones.

    "red" + "ball" → "red ball" — requires the agent to compose two independently
    learned attributes into a novel combination.

    This is the standard test for true abstraction (not memorization).
    It's what the ARC challenge tests at scale. Our version is micro:
    given two known concept nodes, check if the agent can infer their composition.
    """

    def __init__(self, min_known_nodes: int = 4) -> None:
        self._min_nodes = min_known_nodes
        self._tests_run: int = 0
        self._tests_passed: int = 0

    def test(
        self, concept_graph: Any,
    ) -> dict[str, Any]:
        """Run one compositional test on the current ConceptGraph.

        Picks two known concept nodes, combines their attributes, and checks
        if the combination exists or can be reasonably inferred.

        Returns test result dict.
        """
        if concept_graph is None or len(concept_graph._nodes) < self._min_nodes:
            return {"passed": False, "reason": "not enough concepts", "nodes": len(concept_graph._nodes) if concept_graph else 0}

        nodes = list(concept_graph._nodes.values())
        if len(nodes) < 2:
            return {"passed": False, "reason": "too few nodes"}

        # Pick two nodes with different source_modules
        n1 = nodes[0]
        n2 = None
        for n in nodes[1:]:
            if set(n.source_modules) != set(n1.source_modules):
                n2 = n
                break

        if n2 is None:
            return {"passed": False, "reason": "all nodes from same source"}

        # Get edges for both
        edges1 = set()
        edges2 = set()
        for (src, tgt), edge in concept_graph._edges.items():
            if src == n1.id:
                tgt_node = concept_graph._nodes.get(tgt)
                if tgt_node:
                    edges1.add(edge.relation_type)
            if src == n2.id:
                tgt_node = concept_graph._nodes.get(tgt)
                if tgt_node:
                    edges2.add(edge.relation_type)

        # Composition: combined attribute set
        combined = edges1 | edges2
        overlap = edges1 & edges2

        self._tests_run += 1

        if len(combined) > max(len(edges1), len(edges2)):
            self._tests_passed += 1
            return {
                "passed": True,
                "components": f"{n1.name} + {n2.name}",
                "shared_attrs": len(overlap),
                "novel_combination": len(combined) - max(len(edges1), len(edges2)),
                "compositionality": f"{self._tests_passed}/{self._tests_run}",
            }

        return {
            "passed": False,
            "components": f"{n1.name} + {n2.name}",
            "reason": "no novel combination found",
            "compositionality": f"{self._tests_passed}/{self._tests_run}",
        }

    @property
    def score(self) -> float:
        return self._tests_passed / max(1, self._tests_run)

    def summary(self) -> str:
        return f"Compositional: {self._tests_passed}/{self._tests_run} passed ({self.score:.1%})"


# =====================================================================
# 3. Learning Progress Tracker
# =====================================================================


class LearningProgressTracker(nn.Module):
    """Track if the agent is still learning or has plateaued.

    When LP (learning progress) flatlines:
    - Auto-boost curiosity coefficient
    - Signal ModelGrowerV2 to consider expansion
    - Notify ActiveExperimenter to try bolder hypotheses

    LP is measured as: smoothed derivative of mean_return over time.
    """

    def __init__(
        self,
        window_size: int = 1000,
        flat_threshold: float = 0.001,  # Δmean_ret below this = flat
        boost_amount: float = 0.2,       # how much to boost curiosity
    ) -> None:
        super().__init__()
        self._window = window_size
        self._flat_threshold = flat_threshold
        self._boost_amount = boost_amount

        self._return_history: list[float] = []
        self._lp: float = 0.0
        self._is_flat: bool = False
        self._boost_active: bool = False

    def update(self, mean_return: float, step: int) -> dict[str, Any]:
        """Feed new mean_return. Returns dict with flatness status + recommended boost."""
        self._return_history.append(mean_return)
        if len(self._return_history) > self._window:
            self._return_history.pop(0)

        result: dict[str, Any] = {
            "is_flat": False,
            "lp": 0.0,
            "curiosity_boost": 0.0,
        }

        if len(self._return_history) < 2:
            return result

        # LP = smoothed derivative
        half = max(1, len(self._return_history) // 2)
        first_half = self._return_history[:half]
        second_half = self._return_history[half:]

        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)

        self._lp = mean_second - mean_first
        result["lp"] = self._lp

        # Check flatness
        self._is_flat = abs(self._lp) < self._flat_threshold
        result["is_flat"] = self._is_flat

        # Auto-boost curiosity when flat
        if self._is_flat and not self._boost_active:
            result["curiosity_boost"] = self._boost_amount
            self._boost_active = True
            logger.info("[lp_tracker] plateau detected (lp=%.4f), curiosity +%.2f",
                       self._lp, self._boost_amount)
        elif not self._is_flat:
            self._boost_active = False

        return result

    @property
    def is_stuck(self) -> bool:
        return self._is_flat

    def summary(self) -> dict:
        return {
            "lp": self._lp,
            "is_flat": self._is_flat,
            "boost_active": self._boost_active,
            "history_len": len(self._return_history),
        }
