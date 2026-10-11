"""Stage 22 M1 — bounded concept hierarchy (instance → category → super).

Builds a depth-≤3 concept hierarchy from example embeddings (online greedy
prototype clustering, two levels) and classifies NOVEL instances by
routing through the hierarchy (super first, then category within it) —
with a flat (single-level nearest) mode for honest A/B comparison.

M1 verification (docs/stage22_design.md): novel-instance classification
via the hierarchy beats the flat routing of the SAME level-0 prototypes
by > 0.05 on held-out instances (synthetic structured data in tests;
`scripts/eval/concept_hierarchy_eval.py` produces the record).

Bounded by construction: node budget (<= max_nodes), depth <= max_depth,
per-node running prototypes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ConceptNode:
    """One concept (category at depth 0, super-category at depth 1)."""

    id: int
    depth: int
    proto: torch.Tensor  # running-mean prototype (unit-normalized)
    parent_id: int = -1  # for depth-0 nodes: owning super (-1 = none yet)
    members: int = 0
    label_counts: dict = field(default_factory=dict)  # training-label votes


class ConceptHierarchy:
    """Bounded two-level concept hierarchy over embeddings.

    2026-10-11: switched from cosine to EUCLIDEAN distance — high-dim
    Gaussian embeddings concentrate in angle (all cosine ≈ 1), which
    collapsed every category into one node; distance thresholds recover
    the generative structure honestly.
    """

    def __init__(
        self,
        dim: int = 16,
        max_nodes: int = 256,
        cat_dist: float = 1.8,
        super_dist: float = 6.0,
    ) -> None:
        self.dim = int(dim)
        self.max_nodes = int(max_nodes)
        self.cat_dist = float(cat_dist)
        self.super_dist = float(super_dist)
        self._nodes: dict[int, ConceptNode] = {}
        self._next_id = 0
        self._cats: list[int] = []  # depth-0 ids
        self._supers: list[int] = []  # depth-1 ids

    # ------------------------------------------------------------- building

    def _nearest(self, emb: torch.Tensor, ids: list[int]) -> tuple[int, float]:
        """Nearest prototype by Euclidean distance; returns (id, dist)."""
        best_id, best_d = -1, float("inf")
        for nid in ids:
            d = float(torch.norm(self._nodes[nid].proto - emb).item())
            if d < best_d:
                best_id, best_d = nid, d
        return best_id, best_d

    def _observe(self, emb: torch.Tensor, ids: list[int], depth: int,
                 threshold: float, label: str | None,
                 level_cap: int = 10 ** 9) -> int:
        """Greedy online prototype clustering; returns the matched/new id.

        ``level_cap`` bounds this level's node count independently so the
        super level always has budget (the previous global-budget check
        starved supers whenever level-0 saturated — hierarchy == flat).
        """
        emb = emb.float().reshape(-1)
        if ids:
            nid, dist = self._nearest(emb, ids)
            if dist <= threshold:
                node = self._nodes[nid]
                node.proto = node.proto + (emb - node.proto) / (node.members + 1)
                node.members += 1
                if label is not None:
                    node.label_counts[label] = node.label_counts.get(label, 0) + 1
                return nid
        if len(ids) >= level_cap:
            # this level's budget exhausted: attach to the best match
            nid, _ = self._nearest(emb, ids)
            if nid >= 0:
                node = self._nodes[nid]
                node.proto = node.proto + (emb - node.proto) / (node.members + 1)
                node.members += 1
                if label is not None:
                    node.label_counts[label] = node.label_counts.get(label, 0) + 1
                return nid
            return -1
        node = ConceptNode(id=self._next_id, depth=depth,
                           proto=emb.clone(), members=1)
        if label is not None:
            node.label_counts[label] = 1
        self._nodes[self._next_id] = node
        ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def build(self, embeddings: torch.Tensor,
              labels: list[str] | None = None) -> None:
        """Build level-0 categories then level-1 supers over their prototypes.

        Deterministic given the input order (fixed seed in evaluators).
        Budget split: 3/4 for categories, 1/4 reserved for supers.
        """
        cat_cap = max(1, (self.max_nodes * 3) // 4)
        super_cap = max(1, self.max_nodes - cat_cap)
        embeddings = embeddings.float()
        for i in range(embeddings.shape[0]):
            lab = labels[i] if labels is not None else None
            self._observe(embeddings[i], self._cats, 0, self.cat_dist, lab,
                          level_cap=cat_cap)
        # supers over category prototypes (order by id -> deterministic)
        for cid in list(self._cats):
            proto = self._nodes[cid].proto
            sid = self._observe(proto, self._supers, 1,
                                self.super_dist, None, level_cap=super_cap)
            self._nodes[cid].parent_id = sid
        # orphan categories (budget edge cases) get their own super
        for cid in list(self._cats):
            if self._nodes[cid].parent_id < 0:
                sid = self._observe(self._nodes[cid].proto, self._supers, 1,
                                    -1.0, None, level_cap=super_cap)
                self._nodes[cid].parent_id = sid

    # ---------------------------------------------------------- inference

    def _majority_label(self, nid: int) -> str | None:
        counts = self._nodes[nid].label_counts
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def predict(self, emb: torch.Tensor, hierarchical: bool = True) -> str | None:
        """Classify a novel instance by majority label of the matched category.

        ``hierarchical=True``: route via the nearest super, then the nearest
        category nested under it. ``False``: flat nearest category.
        """
        emb = emb.float().reshape(-1)
        if not self._cats:
            return None
        if hierarchical and self._supers:
            sid, _ = self._nearest(emb, self._supers)
            nested = [cid for cid in self._cats
                      if self._nodes[cid].parent_id == sid]
            pool = nested if nested else self._cats
            cid, _ = self._nearest(emb, pool)
        else:
            cid, _ = self._nearest(emb, self._cats)
        return self._majority_label(cid)

    # ------------------------------------------------------------- state

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_categories(self) -> int:
        return len(self._cats)

    @property
    def n_supers(self) -> int:
        return len(self._supers)

    def state_dict(self) -> dict:
        return {
            "nodes": {nid: {"depth": n.depth,
                            "proto": n.proto.detach().cpu(),
                            "parent_id": n.parent_id,
                            "members": n.members,
                            "label_counts": dict(n.label_counts)}
                      for nid, n in self._nodes.items()},
            "next_id": self._next_id,
            "cats": list(self._cats),
            "supers": list(self._supers),
        }

    def load_state_dict(self, state: dict) -> None:
        self._nodes = {}
        for nid, nd in state.get("nodes", {}).items():
            self._nodes[int(nid)] = ConceptNode(
                id=int(nid), depth=int(nd["depth"]),
                proto=nd["proto"], parent_id=int(nd["parent_id"]),
                members=int(nd["members"]),
                label_counts=dict(nd["label_counts"]))
        self._next_id = int(state.get("next_id", 0))
        self._cats = list(state.get("cats", []))
        self._supers = list(state.get("supers", []))
