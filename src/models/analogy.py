"""Stage 22 M3 — relational analogy engine (bounded).

Classic relation transfer: given a demonstrated pair (a, b) instantiating
a relation and a new source c, the answer is the entity whose embedding is
closest to ``emb(c) + (emb(b) - emb(a))``. Relations need no labels at
query time — the offset from the example pair carries them.

M3 verification (docs/stage22_design.md): held-out analogy triples
(novel pair + novel source) top-1 accuracy > random + 0.1
(`scripts/eval/analogy_eval.py` produces the record).

Bounded by construction: entity store and mapping store capped; retrieval
is a bounded argmin over the store.
"""

from __future__ import annotations

from collections import deque

import torch


class AnalogyEngine:
    """Bounded embedding-space analogy engine (A:B :: C:?)."""

    def __init__(self, dim: int = 16, max_entities: int = 512,
                 max_mappings: int = 1024) -> None:
        self.dim = int(dim)
        self.max_entities = int(max_entities)
        self.max_mappings = int(max_mappings)
        self._entities: dict[str, torch.Tensor] = {}
        # a few recent demonstration pairs per relation-style (bounded)
        self._mappings: deque = deque(maxlen=self.max_mappings)  # BOUNDS-OK

    # ------------------------------------------------------------- ingestion

    def add_entity(self, name: str, emb: torch.Tensor) -> bool:
        if name not in self._entities and len(self._entities) >= self.max_entities:
            return False
        self._entities[name] = emb.detach().float().reshape(-1)[: self.dim]
        return True

    def add_mapping(self, a: str, b: str) -> bool:
        if a not in self._entities or b not in self._entities:
            return False
        self._mappings.append((a, b))
        return True

    # ------------------------------------------------------------- analogy

    def _offset(self, a: str, b: str) -> torch.Tensor:
        """Relation offset emb(b) - emb(a) (the demonstrated relation)."""
        return self._entities[b] - self._entities[a]

    def analogize(self, a: str, b: str, c: str,
                  top_k: int = 1) -> list[tuple[str, float]]:
        """Answer A:B :: C:? — nearest store entities to c + (b - a).

        Returns up to ``top_k`` (name, distance) pairs, excluding a/b/c.
        """
        if a not in self._entities or b not in self._entities \
                or c not in self._entities:
            return []
        target = self._entities[c] + self._offset(a, b)
        cands = []
        for name, emb in self._entities.items():
            if name in (a, b, c):
                continue
            d = float(torch.norm(emb - target).item())
            cands.append((name, d))
        cands.sort(key=lambda kv: kv[1])
        return cands[: max(1, int(top_k))]

    # ---------------------------------------------------------------- state

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    def state_dict(self) -> dict:
        return {
            "entities": {k: v.detach().cpu() for k, v in self._entities.items()},
            "mappings": list(self._mappings),
        }

    def load_state_dict(self, state: dict) -> None:
        self._entities = {k: v.float() for k, v in state.get("entities", {}).items()}
        self._mappings = deque(state.get("mappings", []),
                               maxlen=self.max_mappings)  # BOUNDS-OK
