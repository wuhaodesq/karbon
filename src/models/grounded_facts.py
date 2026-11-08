"""grounded_facts — Stage 26 R2: verified assertions -> bounded semantic store.

Pipeline:
  AssertionGenerator (programmatic text, no external corpus)
      -> assertion_verifier.verify(text, scene_facts)
      -> only status=="true" assertions are ingested
      -> GroundedFactStore (bounded dict, dedupe/merge, prune, save/restore)
      -> rejected texts kept in a bounded audit log (maxlen 64)

Stored fact keys are STABLE category facts (some kind/colour exists, some
kind exists, occlusion is possible); scene-specific counts are verified
but not persisted (they are episode-local, not knowledge). This is the
"semantic layer" of the three-tier memory design; consolidation into
weights is R3.

语义入库: 程序化断言 → 验证 → 有界事实库 (去重/合并/修剪) + 拒绝审计。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from .assertion_verifier import SceneFacts, SceneObject, verify
from .instruction_channel import KINDS, COLORS

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

# out-of-scope probes (must be rejected as unverifiable; they train the gate)
UNVERIFIABLE_PROBES = (
    "the weather is nice today",
    "the box is left of the sphere",
    "the red box costs five coins",
)


@dataclass
class FactRecord:
    key: str
    hits: int = 1
    first_step: int = 0
    last_step: int = 0


class GroundedFactStore:
    """Bounded semantic store with dedupe/merge and value-based pruning."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = max(16, int(capacity))
        self._facts: dict[str, FactRecord] = {}
        self._rejected: deque[str] = deque(maxlen=64)
        self._accepted_total = 0
        self._rejected_total = 0
        self._unverifiable_total = 0

    # ------------------------------------------------------------- ingest
    def ingest(self, text: str, status: str, step: int) -> str:
        """Record one verdict: 'true' accepted, others rejected (audited)."""
        if status == "true":
            key = self._key_from_text(text)
            if key is None:
                self._unverifiable_total += 1
                return "no-key"
            rec = self._facts.get(key)
            if rec is None:
                self._facts[key] = FactRecord(key=key, hits=1,
                                              first_step=int(step),
                                              last_step=int(step))
            else:
                rec.hits += 1
                rec.last_step = int(step)
            self._accepted_total += 1
            if len(self._facts) > self._capacity:
                self._prune()
            return "accepted"
        if status == "false":
            self._rejected_total += 1
            self._rejected.append(str(text)[:120])
            return "rejected"
        self._unverifiable_total += 1
        return "unverifiable"

    @staticmethod
    def _key_from_text(text: str) -> str | None:
        low = text.lower()
        kinds = [k for k in KINDS if k in low]
        colors = [c for c in COLORS if c in low]
        occl = any(w in low for w in ("hidden", "occluded", "out of sight",
                                      "behind the wall", "behind a wall"))
        if occl and kinds:
            return f"occludable:{kinds[0]}"
        if kinds and colors:
            return f"exists:{kinds[0]}:{colors[0]}"
        if kinds:
            return f"exists:{kinds[0]}"
        if occl:
            return "occludable:any"
        return None

    def _prune(self) -> None:
        keep = sorted(self._facts.items(),
                      key=lambda kv: (kv[1].hits, kv[1].last_step),
                      reverse=True)[: self._capacity]
        self._facts = dict(keep)

    # --------------------------------------------------------------- stats
    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._facts)

    def telemetry(self) -> dict[str, Any]:
        return {
            "facts": len(self._facts),
            "accepted_total": self._accepted_total,
            "rejected_total": self._rejected_total,
            "unverifiable_total": self._unverifiable_total,
            "recent_rejected": list(self._rejected)[-3:],
        }

    # --------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        return {
            "facts": {k: [v.hits, v.first_step, v.last_step]
                      for k, v in self._facts.items()},
            "rejected": list(self._rejected),
            "accepted_total": self._accepted_total,
            "rejected_total": self._rejected_total,
            "unverifiable_total": self._unverifiable_total,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._facts = {}
        for k, v in state.get("facts", {}).items():
            if len(self._facts) >= self._capacity:
                break
            self._facts[str(k)] = FactRecord(
                key=str(k), hits=int(v[0]), first_step=int(v[1]),
                last_step=int(v[2]))
        self._rejected.clear()
        self._rejected.extend(str(x) for x in state.get("rejected", [])[-64:])
        self._accepted_total = int(state.get("accepted_total", 0))
        self._rejected_total = int(state.get("rejected_total", 0))
        self._unverifiable_total = int(state.get("unverifiable_total", 0))


class AssertionGenerator:
    """Programmatic assertions from a scene (true / false / unverifiable)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def _plural(self, kind: str) -> str:
        return {"box": "boxes", "sphere": "spheres",
                "cylinder": "cylinders", "capsule": "capsules"}[kind]

    def generate(self, facts: SceneFacts, n: int = 24) -> list[str]:
        objs = facts.objects
        if not objs:
            return []
        out: list[str] = []
        kinds_present = sorted({o.kind for o in objs})
        colours_present = sorted({o.color for o in objs if o.color in COLORS})
        occluded = [o for o in objs if o.id in facts.occluded_ids]
        for _ in range(n):
            r = self._rng.random()
            if r < 0.30 and kinds_present:
                k = self._rng.choice(kinds_present)
                c = sum(1 for o in objs if o.kind == k)
                roll = self._rng.random()
                # 50% TRUE (claim == c), else an off-by-one FALSE claim.
                # (The first version always claimed c±1 -> every count
                # assertion was false; the accepted rate collapsed to 26%.)
                if roll < 0.5:
                    claim = c
                else:
                    claim = max(0, c + (1 if roll < 0.75 else -1))
                word = NUM_WORDS.get(claim, str(claim))
                noun = self._plural(k) if claim != 1 else k
                out.append(f"there are {word} {noun}")
            elif r < 0.55 and kinds_present and colours_present:
                k = self._rng.choice(kinds_present)
                c = self._rng.choice(colours_present)
                out.append(f"the {c} {k}")
            elif r < 0.70 and kinds_present:
                k = self._rng.choice(kinds_present)
                if self._rng.random() < 0.5 and occluded:
                    o = self._rng.choice(occluded)
                    out.append(f"the {o.color} {o.kind} is hidden")
                else:
                    o = self._rng.choice(objs)
                    out.append(f"the {o.color} {o.kind} is hidden")
            elif r < 0.85 and colours_present and kinds_present:
                # absent combination (false existence)
                k = self._rng.choice(KINDS)
                c = self._rng.choice(COLORS)
                out.append(f"the {c} {k}")
            else:
                out.append(self._rng.choice(UNVERIFIABLE_PROBES))
        return out


def count_within_radius(positions, agent_xy, radius: float) -> int:
    """R3: cardinality within the perceptual horizon (shared caliber).

    positions: iterable of (x, y); agent_xy: (x, y). Pure logic so the
    training target and the eval target can never drift apart again.
    """
    ax, ay = float(agent_xy[0]), float(agent_xy[1])
    r2 = float(radius) ** 2
    n = 0
    for px, py in positions:
        dx = float(px) - ax
        dy = float(py) - ay
        if dx * dx + dy * dy <= r2:
            n += 1
    return n


def ingest_scene(store: GroundedFactStore, facts: SceneFacts,
                 generator: AssertionGenerator, step: int,
                 n: int = 24) -> dict[str, int]:
    """Generate -> verify -> ingest; returns per-verdict counters."""
    stats = {"true": 0, "false": 0, "unverifiable": 0}
    for text in generator.generate(facts, n):
        verdict = verify(text, facts)["status"]
        stats[verdict] = stats.get(verdict, 0) + 1
        store.ingest(text, verdict, step)
    return stats
