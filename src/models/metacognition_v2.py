"""Meta-Cognitive Abilities — Self-Reflection + Unsupervised Concept Discovery.

Two lightweight upgrades that activate when enough data exists (Phase 3+):

1. SelfReflectionValidator — checks if planned action achieved expected outcome.
   If wrong → records "I was wrong about X" → feeds back to RuleInductionEngine.
   This closes the loop: plan → execute → verify → learn.

2. ConceptClusterer — discovers emergent categories from ConceptGraph nodes.
   When ≥3 nodes share ≥3 same edges → creates a parent concept node.
   This is "fruit" emerging from "apple, orange, banana" sharing common edges.

Both are dormant until data accumulates (Phase 0-2: no effect. Phase 3+: kicks in).

自省验证器 + 无监督概念聚类。数据够了才激活。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. SelfReflectionValidator
# =====================================================================


@dataclass
class ReflectionRecord:
    expected_outcome: str
    actual_outcome: str
    was_correct: bool
    step: int
    lesson: str


class SelfReflectionValidator(nn.Module):
    """Checks if planned actions achieved expected outcomes.

    Hooked into the training loop at episode end:
        planned_action → execute → observe actual outcome
        → compare with expected (from LongRangePlanner / RuleInductionEngine)
        → record discrepancy → feed back to RuleInductionEngine

    This gives the agent the ability to say:
        "I thought pushing it would make it roll, but it didn't move.
         Maybe it's too heavy."
    """

    def __init__(
        self,
        max_records: int = 500,
        error_threshold: float = 0.3,     # min discrepancy to record as "wrong"
    ) -> None:
        super().__init__()
        self._max = int(max_records)
        self._threshold = float(error_threshold)
        self._records: list[ReflectionRecord] = []
        self._correct_count: int = 0
        self._wrong_count: int = 0

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._records)

    def reflect(
        self,
        expected: str,
        actual_reward: float,
        predicted_reward: float,
        step: int,
        context: str = "",
    ) -> str | None:
        """Compare expected vs actual outcome. Return lesson if wrong.

        Args:
            expected: description of what the agent thought would happen.
            actual_reward: the reward actually received.
            predicted_reward: what the agent's model predicted.
            step: global step.
            context: additional context (e.g., object involved, action taken).

        Returns:
            Lesson string if discrepancy detected, None if prediction was correct.
        """
        error = abs(actual_reward - predicted_reward)
        was_correct = error < self._threshold

        if was_correct:
            self._correct_count += 1
            return None

        self._wrong_count += 1

        # Generate lesson
        if actual_reward > predicted_reward:
            lesson = f"I underestimated the outcome: expected {predicted_reward:.2f} but got {actual_reward:.2f}. {context}"
        else:
            lesson = f"I overestimated the outcome: expected {predicted_reward:.2f} but got {actual_reward:.2f}. {context}"

        record = ReflectionRecord(
            expected_outcome=expected,
            actual_outcome=f"reward={actual_reward:.2f}",
            was_correct=False,
            step=step,
            lesson=lesson,
        )

        if len(self._records) >= self._max:
            self._records.pop(0)
        self._records.append(record)

        return lesson

    def query_past_errors(
        self, situation_embedding: torch.Tensor, k: int = 5,
    ) -> list[str]:
        """Return past lessons for similar situations (simplified: recent first)."""
        recent = self._records[-k:] if self._records else []
        return [r.lesson for r in reversed(recent) if not r.was_correct]

    @property
    def accuracy(self) -> float:
        total = self._correct_count + self._wrong_count
        return self._correct_count / max(1, total)

    def summary(self) -> dict:
        return {
            "records": len(self._records),
            "correct": self._correct_count,
            "wrong": self._wrong_count,
            "accuracy": f"{self.accuracy:.1%}",
            "last_lesson": self._records[-1].lesson if self._records else "none",
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "correct_count": self._correct_count,
            "wrong_count": self._wrong_count,
            "records": [
                {"expected": r.expected_outcome, "actual": r.actual_outcome,
                 "correct": r.was_correct, "step": r.step, "lesson": r.lesson}
                for r in self._records[-50:]  # bounded
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._correct_count = int(state.get("correct_count", 0))
        self._wrong_count = int(state.get("wrong_count", 0))
        self._records.clear()
        for r in state.get("records", []):
            self._records.append(ReflectionRecord(
                expected_outcome=r["expected"],
                actual_outcome=r["actual"],
                was_correct=r["correct"],
                step=r["step"],
                lesson=r["lesson"],
            ))


# =====================================================================
# 2. ConceptClusterer — unsupervised category discovery
# =====================================================================


class ConceptClusterer(nn.Module):
    """Discovers emergent categories from ConceptGraph nodes.

    Logic:
        1. Scan all nodes in ConceptGraph
        2. Find groups of ≥min_cluster_size nodes sharing ≥min_shared_edges edges
        3. Create a parent concept node connecting them
        4. The parent represents an emergent category ("fruit", "tool", "container")

    This requires the ConceptGraph to already have nodes + edges (Phase 3+).
    Before that, it's dormant.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_shared_edges: int = 3,
        max_categories: int = 50,
        cluster_every_steps: int = 5000,
        merge_similarity: float = 0.7,
    ) -> None:
        super().__init__()
        self._min_size = int(min_cluster_size)
        self._min_edges = int(min_shared_edges)
        self._max_cats = int(max_categories)
        self._cluster_every = int(cluster_every_steps)
        self._merge_sim = float(merge_similarity)
        self._last_cluster_step = -cluster_every_steps
        self._categories: dict[str, list[int]] = {}  # category_name → [node_ids]
        self._next_cat_id = 0

    @property
    def capacity(self) -> int:
        return self._max_cats

    def __len__(self) -> int:
        return len(self._categories)

    def should_cluster(self, step: int) -> bool:
        return (step - self._last_cluster_step) >= self._cluster_every

    def cluster(
        self, concept_graph: Any, step: int,
    ) -> list[dict[str, Any]]:
        """Run one clustering pass on the ConceptGraph.

        Returns list of newly discovered categories.
        """
        self._last_cluster_step = step

        if len(concept_graph._nodes) < self._min_size:
            return []

        node_ids = list(concept_graph._nodes.keys())
        if len(node_ids) < self._min_size:
            return []

        # Build adjacency: node_id → set of (neighbor_id, relation_type)
        adjacency: dict[int, set[tuple[int, str]]] = {
            nid: set() for nid in node_ids
        }
        for (src, tgt), edge in concept_graph._edges.items():
            adjacency.setdefault(src, set()).add((tgt, edge.relation_type))
            adjacency.setdefault(tgt, set()).add((src, edge.relation_type))

        # Find groups sharing ≥min_shared_edges
        new_categories: list[dict[str, Any]] = []
        visited: set[int] = set()

        for nid in node_ids:
            if nid in visited or len(self._categories) >= self._max_cats:
                continue
            group = self._find_cluster(nid, adjacency, visited)
            if len(group) >= self._min_size:
                cat_name = f"category_{self._next_cat_id}"
                self._next_cat_id += 1
                self._categories[cat_name] = list(group)

                # Compute centroid embedding
                centroids = [concept_graph._nodes[gid].embedding for gid in group]
                centroid = torch.stack(centroids).mean(dim=0)

                # Add parent node to ConceptGraph
                parent_id = concept_graph.add_concept(
                    embedding=centroid,
                    name=cat_name,
                    source="concept_clusterer",
                    step=step,
                )
                # Add edges from parent to children
                for child_id in group:
                    concept_graph.add_edge(
                        parent_id, child_id, "is_category_of",
                        confidence=0.7, source_module="concept_clusterer", step=step,
                    )

                names = [concept_graph._nodes[gid].name for gid in group]
                new_categories.append({
                    "name": cat_name,
                    "members": names,
                    "size": len(group),
                })
                logger.info("[concept] new category '%s': %s", cat_name, names[:5])

        return new_categories

    def _find_cluster(
        self,
        seed: int,
        adjacency: dict[int, set[tuple[int, str]]],
        visited: set[int],
    ) -> set[int]:
        """Greedy cluster expansion: find all nodes sharing ≥min_edges with seed's neighbors."""
        seed_neighbors = {tgt for tgt, _ in adjacency.get(seed, set())}

        group: set[int] = {seed}
        for candidate in adjacency:
            if candidate == seed or candidate in visited:
                continue
            cand_neighbors = {tgt for tgt, _ in adjacency.get(candidate, set())}
            shared = seed_neighbors & cand_neighbors
            if len(shared) >= self._min_edges:
                group.add(candidate)

        visited.update(group)
        return group

    def query_category(self, node_id: int, concept_graph: Any) -> str | None:
        """Return the category name for a node, if any."""
        for cat_name, members in self._categories.items():
            if node_id in members:
                return cat_name
        return None

    def summary(self) -> dict:
        return {
            "categories": len(self._categories),
            "max_categories": self._max_cats,
            "members": {k: len(v) for k, v in self._categories.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "categories": dict(self._categories),
            "next_cat_id": self._next_cat_id,
            "last_cluster_step": self._last_cluster_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._categories = state.get("categories", {})
        self._next_cat_id = int(state.get("next_cat_id", 0))
        self._last_cluster_step = int(
            state.get("last_cluster_step", -self._cluster_every)
        )
