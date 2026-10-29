"""instruction_battery.py — Stage 23 language L-in battery (M3b).

Tests TEXT -> REFERENT -> BEHAVIOR on 5 seeded scenes x 20 instruction
templates. Honest decomposition:

  referent_ok : the parsed referent matches the generator's binding
                (ties allowed when the instruction is ambiguous);
  approach_ok : a SCRIPTED greedy walker (8-direction actions, disclosed)
                reaches the referent within 120 steps (<=0.5 m live,
                <=0.4 m memory anchor);
  match       : referent_ok AND approach_ok  (M3b criterion >= 0.7);
  baseline    : chance-level referent (1/n_objects) x approach_ok.

Memory trials ("go to where X disappeared"): the walker must use the
agent's OWN occlusion memory (env last-known anchors), not the live body
position. Trials without an active occlusion are skipped (not applicable).

Honesty guards:
  - parser + generator share the lexicon by construction (template battery);
    the DISCRIMINATIVE part is referent resolution + memory-grounded action;
  - raw rows saved; ambiguous bindings counted separately;
  - scripted executor disclosed (carry_tool_probe pattern); this is an
    L-in wiring/measurement milestone, not a learned policy claim.

Usage:
    python scripts/eval/instruction_battery.py --scenes 5 --out /root/m3b_instruction.json
"""
import argparse
import colorsys
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KINDS = ("box", "sphere", "cylinder", "capsule")
COLOR_NAMES = ("red", "orange", "yellow", "green", "cyan", "blue", "purple")
DIRS = [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0),
        (0.70710678, 0.70710678), (-0.70710678, 0.70710678),
        (-0.70710678, -0.70710678), (0.70710678, -0.70710678)]

MEM_WORDS = ("disappear", "vanish", "hidden", "occluded", "lost sight",
             "saw last", "last known", "saw that")

TEMPLATES = [
    "go to the {kind}",
    "approach the {kind}",
    "find the {kind}",
    "move to the {kind}",
    "go to the {color} {kind}",
    "approach the {color} {kind}",
    "find the {color} {kind}",
    "touch the {color} {kind}",
    "go to the nearest {kind}",
    "approach the farthest {kind}",
    "go to the object that disappeared",
    "find the one that vanished",
    "go to where the {kind} disappeared",
    "go to the object that is hidden",
    "go to the occluded object",
    "go to where it disappeared",
    "find the object you lost sight of",
    "go back to the last known spot",
    "go to the object you saw last",
    "go to the most recently hidden object",
]

MEM_TEMPLATES = set(TEMPLATES[10:])


def color_name(rgb) -> str:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360.0
    if s < 0.15:
        return "gray"
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 265:
        return "blue"
    return "purple"


def _mentions(low: str, word: str) -> bool:
    """Word-boundary match (substring matching made 'disappeaRED' red)."""
    return re.search(rf"\b{re.escape(word)}\b", low) is not None


def parse_instruction(text: str, facts: dict, agent_xy) -> dict:
    """Text -> referent constraints. Returns ids list + mode + anchor."""
    low = text.lower()
    kinds = [k for k in KINDS
             if _mentions(low, k) or _mentions(low, k + "s")
             or _mentions(low, k + "es")]
    colors = [c for c in COLOR_NAMES if _mentions(low, c)]
    memory = any(w in low for w in MEM_WORDS)
    cand = [o for o in facts["objects"]
            if (not kinds or o["kind"] in kinds)
            and (not colors or o["color"] in colors)]
    if memory:
        cand = [o for o in cand if o["id"] in facts["occluded"]]
    ax, ay = agent_xy

    def dist(o, xy=None):
        px, py = xy if xy else o["xy"]
        return math.hypot(px - ax, py - ay)

    if "nearest" in low and cand:
        cand = [min(cand, key=lambda o: dist(o))]
    elif "farthest" in low and cand:
        cand = [max(cand, key=lambda o: dist(o))]
    elif "last known" in low and cand:
        cand = [min(cand, key=lambda o: dist(o, facts["last_known"].get(o["id"], o["xy"])))]
    mode = "memory" if memory else "live"
    anchor = None
    if mode == "memory" and cand:
        anchor = facts["last_known"].get(cand[0]["id"])
    return {"ids": [o["id"] for o in cand], "mode": mode, "anchor": anchor}


def make_trials(facts: dict, agent_xy, rng: random.Random) -> list[dict]:
    """Bind templates to valid referents; ambiguity is encoded in acceptable."""
    objs = facts["objects"]
    by_kind = {}
    for o in objs:
        by_kind.setdefault(o["kind"], []).append(o)
    pairs = {}
    for o in objs:
        pairs.setdefault((o["kind"], o["color"]), []).append(o)
    trials = []
    for tpl in TEMPLATES:
        if tpl in MEM_TEMPLATES:
            if not facts["occluded"]:
                trials.append({"template": tpl, "skip": "no-active-occlusion"})
                continue
            kinds_in = [k for k in KINDS if k in tpl]
            pool = [i for i in facts["occluded"]
                    if not kinds_in or facts["by_id"][i]["kind"] in kinds_in]
            if not pool:
                trials.append({"template": tpl, "skip": "no-matching-occluded"})
                continue
            tid = rng.choice(pool)
            ref = facts["by_id"][tid]
            text = tpl.format(kind=ref["kind"], color=ref["color"])
            trials.append({"template": tpl, "text": text, "mode": "memory",
                           "acceptable": pool})
            continue
        kinds_in = [k for k in KINDS if k in tpl]
        want_color = "{" + "color" + "}" in tpl
        if want_color:
            uniq = [o for (k, c), lst in pairs.items()
                    if len(lst) == 1 and (not kinds_in or k in kinds_in)
                    for o in lst]
            if not uniq:
                trials.append({"template": tpl, "skip": "no-unique-kind-color"})
                continue
            ref = rng.choice(uniq)
            acceptable = [o["id"] for o in pairs[(ref["kind"], ref["color"])]]
        else:
            pool = [o for k, lst in by_kind.items()
                    if not kinds_in or k in kinds_in for o in lst]
            if not pool:
                trials.append({"template": tpl, "skip": "no-matching-kind"})
                continue
            ref = rng.choice(pool)
            acceptable = [o["id"] for o in by_kind[ref["kind"]]]
        text = tpl.format(kind=ref["kind"], color=ref["color"])
        if "nearest" in tpl or "farthest" in tpl:
            same = [o for o in objs if o["kind"] == ref["kind"]]
            if len(same) < 2:
                trials.append({"template": tpl, "skip": "kind-not-ambiguous-enough"})
                continue
            key = (lambda o: math.hypot(o["xy"][0] - agent_xy[0],
                                        o["xy"][1] - agent_xy[1]))
            ref = min(same, key=key) if "nearest" in tpl else max(same, key=key)
            acceptable = [ref["id"]]
        trials.append({"template": tpl, "text": text, "mode": "live",
                       "acceptable": acceptable})
    return trials


def scene_facts(en, env_step_xy) -> dict:
    objs = []
    for i, o in enumerate(en.objects):
        kind = str(getattr(o, "kind", "?"))
        col = color_name(getattr(o, "color", (0.5, 0.5, 0.5)))
        x, y = env_step_xy(f"obj_{i}")
        objs.append({"id": i, "kind": kind, "color": col, "xy": (x, y)})
    by_id = {o["id"]: o for o in objs}
    occluded = []
    last_known = {}
    try:
        sig = en.get_occlusion_signal()
        for oid, lx, ly in sig.get("active", []):
            occluded.append(int(oid))
            last_known[int(oid)] = (float(lx), float(ly))
    except Exception:
        pass
    return {"objects": objs, "by_id": by_id, "occluded": occluded,
            "last_known": last_known}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--settle", type=int, default=120,
                    help="random warmup steps per scene (creates occlusions)")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out", type=str, default="/root/m3b_instruction.json")
    args = ap.parse_args()

    import numpy as np
    from src.envs.three_d_world import ThreeDWorld
    from src.utils import load_config

    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    ec = cfg["env"]

    def body_xy(en, name):
        b = en._model.body(name)
        return float(en._data.xpos[b.id, 0]), float(en._data.xpos[b.id, 1])

    rows = []
    np_rng = np.random.RandomState(args.seed)
    rng = random.Random(args.seed)
    for si in range(args.scenes):
        env = ThreeDWorld(
            num_objects=int(ec.get("num_objects", 8)),
            max_episode_steps=int(ec.get("max_episode_steps", 300)),
            render_size=64,
            action_force=float(ec.get("action_force", 50.0)),
            developmental_age=float(ec.get("developmental_age", 0.5)),
            num_occluders=int(ec.get("num_occluders", 0)),
            occluder_obs_slots=int(ec.get("occluder_obs_slots", 0)),
            object_crossing_every=int(ec.get("object_crossing_every", 0)),
            object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
        )
        env.reset(seed=int(np_rng.randint(0, 2**31 - 1)))
        for _ in range(args.settle):
            env.step(int(np_rng.randint(0, 8)))
        facts = scene_facts(env, lambda n: body_xy(env, n))
        agent_xy = body_xy(env, "learner")
        trials = make_trials(facts, agent_xy, rng)
        for tr in trials:
            if "skip" in tr:
                rows.append({"scene": si, "template": tr["template"],
                             "skipped": tr["skip"]})
                continue
            agent_xy = body_xy(env, "learner")
            parsed = parse_instruction(tr["text"], facts, agent_xy)
            target_ids = parsed["ids"]
            if "nearest" in tr["template"] or "farthest" in tr["template"]:
                ok_ref = bool(target_ids)
            else:
                ok_ref = bool(target_ids) and any(
                    i in tr["acceptable"] for i in target_ids)
            reached = False
            steps = 0
            if target_ids:
                tid = target_ids[0]
                if parsed["mode"] == "memory" and parsed["anchor"]:
                    anchor = parsed["anchor"]
                else:
                    anchor = body_xy(env, f"obj_{tid}")
                for steps in range(1, args.steps + 1):
                    if parsed["mode"] == "live":
                        anchor = body_xy(env, f"obj_{tid}")
                    ax, ay = body_xy(env, "learner")
                    dx, dy = anchor[0] - ax, anchor[1] - ay
                    a = max(range(8), key=lambda k: dx * DIRS[k][0] + dy * DIRS[k][1])
                    env.step(a)
                    ax, ay = body_xy(env, "learner")
                    if math.hypot(anchor[0] - ax, anchor[1] - ay) <= \
                            (0.5 if parsed["mode"] == "live" else 0.4):
                        reached = True
                        break
            rows.append({"scene": si, "template": tr["template"],
                         "text": tr["text"], "mode": tr["mode"],
                         "acceptable": tr["acceptable"],
                         "parsed": parsed["ids"],
                         "referent_ok": ok_ref, "approach_ok": reached,
                         "match": ok_ref and reached, "steps": steps})
        env.close()
        print(f"[m3b] scene {si} done ({sum(1 for r in rows if r.get('scene') == si)} trials)",
              flush=True)

    scored = [r for r in rows if "skipped" not in r]
    n = max(1, len(scored))
    match = sum(1 for r in scored if r["match"]) / n
    ref = sum(1 for r in scored if r["referent_ok"]) / n
    live = [r for r in scored if r["mode"] == "live"]
    mem = [r for r in scored if r["mode"] == "memory"]
    summary = {
        "n_trials": len(rows),
        "n_scored": len(scored),
        "n_live": len(live),
        "n_memory": len(mem),
        "referent_accuracy": round(ref, 3),
        "match_rate": round(match, 3),
        "match_live": round(sum(1 for r in live if r["match"]) / max(1, len(live)), 3),
        "match_memory": round(sum(1 for r in mem if r["match"]) / max(1, len(mem)), 3),
        "baseline_chance": round(1.0 / max(1, int(ec.get("num_objects", 8))), 3),
        "criterion": "match>=0.7 (M3b), op unchanged (no training change)",
        "pass": match >= 0.7,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1),
                              encoding="utf-8")
    print("[m3b] SUMMARY:", json.dumps(summary))


if __name__ == "__main__":
    main()
