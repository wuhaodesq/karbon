"""Stage 22 M4 — ConceptBridge: concept stack wired into the training loop.

Per-episode integration (passive, bounded — never touches the policy):

  1. Every object instance observed in the world is fed (with its
     attributes) into the ConceptHierarchy — categories and supers form
     from REAL experience.
  2. Physics interactions (a pushed/moved object) emit property facts:
     ``prop(instance, movable)`` and, once a category has enough movable
     members, ``prop(category, movable)``.
  3. The ConceptProver then answers concept-level queries: a NOVEL
     instance inherits "movable" through its category/super — the M2
     deduction running on live training data.
  4. ``summary()`` provides the ``[concept]`` telemetry line.

Bounded by construction: instance store (max_instances) + hierarchy and
prover capacities; attribute embeddings are a fixed-size one-hot frame
(kind/color/size) — honest and cheap, no policy dependence.
"""

from __future__ import annotations

from collections import deque

import torch

from src.models.concept_hierarchy import ConceptHierarchy
from src.models.concept_rules import ConceptProver

_KINDS = ("sphere", "box", "cylinder", "capsule")
_COLORS = 8
_SIZES = 3


def attribute_embedding(kind: str, color_idx: int, size_idx: int,
                        dim: int = 16) -> torch.Tensor:
    """Fixed attribute frame (kind-dominant one-hots, padded to dim).

    Weights: kind 2.0, color 0.5, size 0.5 — two objects of the same KIND
    with different colors sit ~0.71 apart (same category), while different
    kinds sit ~2.8 apart (different categories). Equal weights made every
    color its own category and broke category-level generalization.
    """
    v = torch.zeros(dim)
    k = _KINDS.index(kind) if kind in _KINDS else 0
    v[k] = 2.0
    v[4 + (int(color_idx) % _COLORS)] = 0.5
    v[12 + (int(size_idx) % _SIZES)] = 0.5
    return v


class ConceptBridge:
    """Bounded concept-formation + deduction bridge over live experience."""

    def __init__(self, max_instances: int = 256, min_cat_movable: int = 2,
                 dim: int = 16) -> None:
        self.max_instances = int(max_instances)
        self.min_cat_movable = int(min_cat_movable)
        self.dim = int(dim)
        self.hierarchy = ConceptHierarchy(dim=dim, max_nodes=128,
                                          cat_dist=0.9, super_dist=2.5)
        self.prover = ConceptProver(max_facts=2048)
        self.prover.add_concept_rules()
        self._seen: deque = deque(maxlen=self.max_instances)  # BOUNDS-OK
        self._movable_by_cat: dict[str, set] = {}
        self.n_movable_facts = 0
        self.n_inherit_queries = 0
        self.n_inherit_hits = 0

    # ------------------------------------------------------------ ingestion

    @property
    def capacity(self) -> int:
        """Bounded-storage contract (HealthChecker / AGENTS §4)."""
        return self.max_instances

    def __len__(self) -> int:
        return len(self._seen)

    def feed_episode(self, objs, info: dict) -> None:
        """Absorb one episode's world objects + real grasp experience.

        ``objs``: iterable of ObjectDef-like items (name/kind/color/size).
        ``info``: env step info; ``grasp_carry_events`` entries with
        ``obj_id`` + ``carry_distance`` mark objects that were actually
        carried (> 0.05) — the honest movable-property source. Extraction
        lives here (not in train.py) so unit tests cover the exact
        semantics the training loop runs.
        """
        objs = list(objs) if objs else []
        for ci, od in enumerate(objs[:64]):
            color = getattr(od, "color", (0.5, 0.5, 0.5, 1.0))
            size = getattr(od, "size", (0.05, 0.05, 0.05))
            self.observe_object(
                inst_id=str(getattr(od, "name", "obj%d" % ci)),
                kind=str(getattr(od, "kind", "box")),
                color_idx=int(max(0.0, min(1.0, float(color[0]))) * 7.99),
                size_idx=min(2, int(max(0.0, float(size[0])) / 0.06)),
            )
        events = (info.get("grasp_carry_events") or [])[-16:]
        for ev in events:
            if float(ev.get("carry_distance", 0.0)) > 0.05:
                oi = int(ev.get("obj_id", -1))
                if 0 <= oi < len(objs):
                    self.record_movable(str(getattr(objs[oi], "name", "obj%d" % oi)))

    def observe_object(self, inst_id: str, kind: str, color_idx: int,
                       size_idx: int) -> None:
        """Feed one object instance (idempotent per id)."""
        if inst_id in self._seen:
            return
        emb = attribute_embedding(kind, color_idx, size_idx, self.dim)
        self._seen.append(inst_id)
        self.hierarchy.build(emb.unsqueeze(0), [inst_id])
        # update prover membership from the assignment
        cats = self.hierarchy._cats
        if cats:
            cid, _ = self.hierarchy._nearest(emb, cats)
            self.prover.add_fact("member", (inst_id, f"cat{cid}"))
            node = self.hierarchy._nodes[cid]
            if node.parent_id >= 0:
                self.prover.add_fact("member", (inst_id, f"sup{node.parent_id}"))

    def record_movable(self, inst_id: str) -> None:
        """Physics fact: this instance was actually moved by contact."""
        self.prover.add_fact("prop", (inst_id, "movable"))
        self.n_movable_facts += 1
        # category-level generalization once enough members agree
        for fact in self.prover._facts.get("member", ()):  # BOUNDS-OK
            if fact[0] == inst_id:
                cat = fact[1]
                bucket = self._movable_by_cat.setdefault(cat, set())
                if len(bucket) < self.max_instances:  # BOUNDS-OK: per-cat cap
                    bucket.add(inst_id)
                if len(bucket) >= self.min_cat_movable:
                    self.prover.add_fact("prop", (cat, "movable"))

    # ------------------------------------------------------------ deduction

    def novel_movable(self, inst_id: str) -> bool:
        """Concept-level query: does the bridge deduce this instance movable?"""
        self.n_inherit_queries += 1
        ans = self.prover.query("prop", (inst_id, "movable"))
        hit = bool(ans)
        if hit:
            self.n_inherit_hits += 1
        return hit

    # ------------------------------------------------------------- telemetry

    def summary(self) -> dict:
        return {
            "instances": len(self._seen),
            "categories": self.hierarchy.n_categories,
            "supers": self.hierarchy.n_supers,
            "movable_facts": self.n_movable_facts,
            "inherit_queries": self.n_inherit_queries,
            "inherit_hits": self.n_inherit_hits,
            "facts": self.prover.n_facts,
        }

    def state_dict(self) -> dict:
        facts = []
        for pred, bucket in self.prover._facts.items():  # BOUNDS-OK bounded store
            for args in bucket:
                facts.append((pred, list(args)))
        return {
            "hierarchy": self.hierarchy.state_dict(),
            "seen": list(self._seen),
            "movable_by_cat": {k: sorted(v) for k, v in self._movable_by_cat.items()},
            "counts": {"movable": self.n_movable_facts,
                       "q": self.n_inherit_queries, "h": self.n_inherit_hits},
            "facts": facts,
        }

    def load_state_dict(self, state: dict) -> None:
        self.hierarchy.load_state_dict(state.get("hierarchy", {}))
        self._seen = deque(state.get("seen", []), maxlen=self.max_instances)  # BOUNDS-OK
        self._movable_by_cat = {k: set(v) for k, v in
                                state.get("movable_by_cat", {}).items()}
        # rebuild the prover (fresh concept rules + persisted facts)
        self.prover = ConceptProver(max_facts=2048)
        self.prover.add_concept_rules()
        for pred, args in state.get("facts", []):
            self.prover.add_fact(str(pred), tuple(args))
        cts = state.get("counts", {})
        self.n_movable_facts = int(cts.get("movable", 0))
        self.n_inherit_queries = int(cts.get("q", 0))
        self.n_inherit_hits = int(cts.get("h", 0))
