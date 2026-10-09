"""Stage 21 M3 — contradiction detection & self-correction for the rule base.

Operational definition: two rules whose CONDITION embeddings are nearly
identical (cosine similarity >= threshold) but whose PREDICTED ACTIONS
differ are a contradiction pair — the agent predicts conflicting outcomes
for the same situation. The self-correction loop resolves a pair by
lowering the confidence of the weaker rule (lower success rate / fewer
uses), keeping the better one.

M3 verification (docs/stage21_design.md): injected-contradiction detection
rate > 0.8 with false-positive rate < 0.1 on non-conflicting rules —
covered by tests/test_contradiction.py on synthetic rule memories.

Bounded by construction: pairwise scan over the bounded RuleMemory
(<= max_rules, typically 64 -> <= 2016 pairs).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ContradictionPair:
    """Two conflicting rules (ids) + their similarity."""

    a_id: int
    b_id: int
    similarity: float


class ContradictionDetector:
    """Detect + resolve conflicting rules in a RuleMemory."""

    def __init__(self, sim_threshold: float = 0.9,
                 resolve_factor: float = 0.7, resolved_memory: int = 256) -> None:
        self.sim_threshold = float(sim_threshold)
        self.resolve_factor = float(resolve_factor)
        # 2026-10-09: remember already-resolved pairs so the periodic loop does
        # not re-punish the same conflict every cycle (0.7^k confidence decay).
        self._resolved: deque = deque(maxlen=int(resolved_memory))  # BOUNDS-OK

    # ---------------------------------------------------------------- detect

    def detect(self, rules: dict) -> list[ContradictionPair]:
        """Return near-identical-condition / different-action rule pairs."""
        ids = list(rules.keys())
        if len(ids) < 2:
            return []
        emb = []
        acts = []
        for rid in ids:
            r = rules[rid]
            e = getattr(r, "condition_embedding", None)
            if e is None or not torch.is_tensor(e):
                return []  # cannot judge without embeddings (honest no-op)
            # 2026-10-09: rule embeddings can live on mixed devices (rules
            # created at different times / merged); normalize and move to CPU
            # before stacking — the pairwise scan is tiny and CPU is safe.
            emb.append(F.normalize(e.detach().float().reshape(-1), dim=0).cpu())
            acts.append(int(getattr(r, "action", -1)))
        mat = torch.stack(emb)  # (R, D)
        sim = mat @ mat.t()  # (R, R)
        pairs: list[ContradictionPair] = []
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j].item())
                if s >= self.sim_threshold and acts[i] != acts[j]:
                    pairs.append(ContradictionPair(ids[i], ids[j], s))
        return pairs

    # --------------------------------------------------------------- resolve

    def resolve(self, rules: dict, pairs: list[ContradictionPair]) -> int:
        """Lower the weaker rule's confidence in every pair (once per pair)."""
        adjusted = 0
        for p in pairs:
            key = frozenset((p.a_id, p.b_id))
            if key in self._resolved:
                continue
            ra, rb = rules.get(p.a_id), rules.get(p.b_id)
            if ra is None or rb is None:
                continue
            sa, sb = self._score(ra), self._score(rb)
            loser = ra if sa < sb else rb
            loser.confidence = max(0.05, float(loser.confidence) * self.resolve_factor)
            self._resolved.append(key)
            adjusted += 1
        return adjusted

    @staticmethod
    def _score(rule) -> float:
        sr = float(getattr(rule, "success_rate", 0.0))
        uses = float(getattr(rule, "usage_count", 0))
        return sr * 100.0 + uses  # prefer success rate, then usage

    # ---------------------------------------------------------------- cycle

    def step(self, rules: dict) -> dict:
        """Detect + resolve; returns a bounded summary for logging."""
        pairs = self.detect(rules)
        resolved = self.resolve(rules, pairs) if pairs else 0
        return {"pairs": len(pairs), "resolved": resolved}
