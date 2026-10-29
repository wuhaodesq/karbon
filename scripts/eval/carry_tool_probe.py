"""carry_tool_probe.py — M3.5 momentum-semantics safety probe.

Scripted (no policy) verification of the carry/release/tool mechanics:
  1. navigate to the nearest graspable object, grasp it
  2. carry (move) — the held object must follow the agent
  3. release WHILE moving — the object detaches; the agent keeps moving
     (momentum semantics must not break carrying/placing)
  4. tool action 10 executes and does not freeze the body
"""
import json
import math
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.three_d_world import ThreeDWorld

en = ThreeDWorld(num_objects=8, seed=7, max_episode_steps=600, render_size=64,
                 action_force=50.0, developmental_age=0.5)
obs = en.reset(seed=7)
DIRS = [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0),
        (0.707, 0.707), (-0.707, 0.707), (-0.707, -0.707), (0.707, -0.707)]


def axy():
    b = en._model.body("learner")
    return float(en._data.xpos[b.id, 0]), float(en._data.xpos[b.id, 1])


def oxy(i):
    b = en._model.body(f"obj_{i}")
    return float(en._data.xpos[b.id, 0]), float(en._data.xpos[b.id, 1])


def act_toward(tx, ty):
    a = axy()
    dx, dy = tx - a[0], ty - a[1]
    return max(range(8), key=lambda k: dx * DIRS[k][0] + dy * DIRS[k][1])


def dist2o(i):
    a, o = axy(), oxy(i)
    return math.hypot(a[0] - o[0], a[1] - o[1])


def nav_to(i, max_steps=300):
    for _ in range(max_steps):
        if dist2o(i) < 0.32:
            return True
        o = oxy(i)
        en.step(act_toward(o[0], o[1]))
    return dist2o(i) < 0.45


res = {"pass": True}
objs = [i for i, o in enumerate(en.objects) if getattr(o, "graspable", True)]
target = min(objs, key=dist2o)

res["reached"] = nav_to(target)
en.step(8)  # grasp
if en._held_obj_id is not None:
    target = en._held_obj_id  # may grasp a nearer object; that is fine
res["grasp_ok"] = en._held_obj_id is not None

if res["grasp_ok"]:
    a0, o0 = axy(), oxy(target)
    for _ in range(12):
        en.step(3)  # East
    a1, o1 = axy(), oxy(target)
    res["carry_agent_move"] = round(math.hypot(a1[0] - a0[0], a1[1] - a0[1]), 3)
    res["carry_obj_move"] = round(math.hypot(o1[0] - o0[0], o1[1] - o0[1]), 3)
    res["carry_follow_gap"] = round(math.hypot(a1[0] - o1[0], a1[1] - o1[1]), 3)

    # release WHILE moving: momentum must keep the agent moving
    a2 = axy()
    n_rel = len(en._release_events)
    for _ in range(5):
        en.step(9)
    a3 = axy()
    res["release_events_added"] = len(en._release_events) - n_rel
    res["released"] = en._held_obj_id is None
    res["release_keeps_moving"] = round(
        math.hypot(a3[0] - a2[0], a3[1] - a2[1]), 3)

    # tool action executes without freezing
    a4 = axy()
    for _ in range(4):
        en.step(10)
    a5 = axy()
    res["tool_keeps_moving"] = round(math.hypot(a5[0] - a4[0], a5[1] - a4[1]), 3)
    res["tool_events"] = len(en._tool_use_events)

checks = {
    "reached": res.get("reached", False),
    "grasp_ok": res.get("grasp_ok", False),
    "carry": res.get("carry_agent_move", 0) > 0.15
             and abs(res.get("carry_obj_move", 0) - res.get("carry_agent_move", 0)) < 0.4
             and res.get("carry_follow_gap", 9) < 0.6,
    "release": res.get("release_events_added", 0) >= 1
               and res.get("released", False)
               and res.get("release_keeps_moving", 0) > 0.03,
    "tool": res.get("tool_keeps_moving", 0) > 0.03,
}
res["checks"] = checks
res["pass"] = all(checks.values())
print(json.dumps(res, indent=1))
en.close()
