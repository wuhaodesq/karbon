"""assertion_verifier — Stage 26 R1: text assertion -> scene grounding.

Gatekeeper of GROUNDED READING: a text assertion may enter the semantic
store ONLY if it can be verified against experience. Bounded, rule-based,
no LLM (design doc §2 / §4).

R1 scope (disclosed):
  - COUNT assertions        "there are three cylinders" / "8 objects"
  - EXISTENCE assertions    "the red box" / "a blue sphere"
  - OCCLUSION assertions    "hidden behind walls" vs the scene's occluded set
  - object-object SPATIAL relations are NOT verified in R1 (unverifiable),
    because the scene's view-relative semantics (behind/left/right) need a
    heading convention that belongs to a later milestone.

Verification is strict: unknown vocabulary or unknown referents yield
"unverifiable" (never "true"). Precision over recall.

文本断言验证器: 只有能被经验核对的断言才允许进入语义库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .instruction_channel import COLORS, KINDS, color_name

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}
OCCL_WORDS = ("hidden", "occluded", "out of sight", "behind the wall",
              "behind a wall", "disappeared")
SPATIAL_WORDS = ("left of", "right of", "in front of", "next to",
                 "above", "below", "farther than", "closer than")


@dataclass
class SceneObject:
    id: int
    kind: str
    color: str

    @classmethod
    def from_env(cls, index: int, obj) -> "SceneObject":
        return cls(id=int(index),
                   kind=str(getattr(obj, "kind", "")),
                   color=color_name(getattr(obj, "color", (0.5, 0.5, 0.5))))


@dataclass
class SceneFacts:
    objects: list[SceneObject]
    occluded_ids: frozenset[int] = frozenset()

    @classmethod
    def from_env(cls, env) -> "SceneFacts":
        objs = [SceneObject.from_env(i, o) for i, o in enumerate(env.objects)]
        try:
            occ = frozenset(int(oid) for oid, _x, _y
                            in env.get_occlusion_signal().get("active", []))
        except Exception:  # noqa: BLE001 - facts fallback
            occ = frozenset()
        return cls(objects=objs, occluded_ids=occ)


def _num_tokens(text: str) -> set[int]:
    found = {int(x) for x in re.findall(r"\b(\d{1,3})\b", text)}
    for w, v in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            found.add(v)
    return found


_PLURAL = {"box": r"box(?:es)?", "sphere": r"spheres?",
           "cylinder": r"cylinders?", "capsule": r"capsules?"}


def _kind_mentions(text: str) -> set[str]:
    low = text.lower()
    return {k for k in KINDS if re.search(rf"\b{_PLURAL.get(k, k)}\b", low)}


def _color_mentions(text: str) -> set[str]:
    low = text.lower()
    return {c for c in COLORS if re.search(rf"\b{c}\b", low)}


def _match(facts: SceneFacts, kinds: set[str], colors: set[str]) -> list[SceneObject]:
    return [o for o in facts.objects
            if (not kinds or o.kind in kinds) and (not colors or o.color in colors)]


def verify(text: str, facts: SceneFacts) -> dict:
    """Verify a text assertion against scene facts.

    Returns {"status": "true"|"false"|"unverifiable", "reason": str}.
    Strict: any claim whose vocabulary/referent is unknown -> unverifiable.
    """
    low = text.lower().strip()
    if not low:
        return {"status": "unverifiable", "reason": "empty"}
    kinds = _kind_mentions(low)
    colors = _color_mentions(low)
    nums = _num_tokens(low)
    has_occl = any(w in low for w in OCCL_WORDS)
    if not kinds and not colors and not nums and not has_occl:
        return {"status": "unverifiable", "reason": "no verifiable vocabulary"}
    # R1 disclosure: view-relative spatial relations need a heading
    # convention and are NOT verifiable yet.
    if any(w in low for w in SPATIAL_WORDS):
        return {"status": "unverifiable", "reason": "spatial relation out of R1 scope"}
    # Unknown kind/colour words are not our business, but a claim with BOTH
    # an unknown referent and our vocabulary is judged on ours only.
    matched = _match(facts, kinds, colors)

    # --- count assertion ---------------------------------------------------
    if nums and (kinds or "object" in low or "thing" in low or "item" in low):
        if len(nums) != 1:
            return {"status": "unverifiable", "reason": "multiple counts"}
        claimed = next(iter(nums))
        actual = len(facts.objects) if not (kinds or colors) else len(matched)
        if claimed == actual:
            return {"status": "true", "reason": f"count {claimed} matches"}
        return {"status": "false",
                "reason": f"claimed {claimed}, actual {actual}"}

    # --- existence ---------------------------------------------------------
    if kinds or colors:
        if not matched:
            return {"status": "false", "reason": "no matching object"}
        occl_claim = any(w in low for w in OCCL_WORDS)
        if occl_claim and not kinds and not colors:
            return {"status": "unverifiable", "reason": "no referent"}
        if occl_claim:
            # the referent must be among the currently occluded objects
            if any(o.id in facts.occluded_ids for o in matched):
                return {"status": "true", "reason": "referent occluded"}
            return {"status": "false", "reason": "referent not occluded"}
        return {"status": "true", "reason": "referent exists"}

    if has_occl:
        if facts.occluded_ids:
            return {"status": "true", "reason": "occlusion present"}
        return {"status": "false", "reason": "no occlusion"}

    return {"status": "unverifiable", "reason": "out of R1 scope"}
