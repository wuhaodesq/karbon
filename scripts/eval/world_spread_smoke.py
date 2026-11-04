"""world_spread_smoke.py — M2-p3 smoke: object arena spread + shape contract.

Builds the training env with the config's object_spawn_half, checks the
obs/proprio/action contract and reports pairwise object distances at reset.
Must run in a GPU-idle window (creates a MuJoCo renderer).

Usage:
    python scripts/eval/world_spread_smoke.py --spawn-half 3.0
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.extended_3d_world import ExtendedThreeDWorld
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn-half", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="/root/world_spread_smoke.json")
    args = ap.parse_args()
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    ec = cfg["env"]
    env = ExtendedThreeDWorld(
        num_objects=int(ec.get("num_objects", 8)),
        num_siblings=int(ec.get("num_siblings", 2)),
        season_cycle_steps=int(ec.get("season_cycle_steps", 500000)),
        max_episode_steps=int(ec.get("max_episode_steps", 600)),
        render_size=int(ec.get("render_size", 128)),
        action_force=float(ec.get("action_force", 50.0)),
        developmental_age=float(ec.get("developmental_age", 0.5)),
        num_occluders=int(ec.get("num_occluders", 0)),
        occluder_obs_slots=int(ec.get("occluder_obs_slots", 0)),
        object_crossing_every=int(ec.get("object_crossing_every", 0)),
        object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
        object_spawn_half=float(args.spawn_half),
    )
    obs = env.reset(seed=7)
    pos = []
    for i in range(env._num_objects):
        b = env._model.body(f"obj_{i}")
        pos.append((float(env._data.xpos[b.id, 0]),
                    float(env._data.xpos[b.id, 1])))
    d = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
         for a, b in itertools.combinations(pos, 2)]
    # M2-p3 fix check: random-walk 300 steps and verify the learner stays
    # inside the arena (no boundary drift; observed 2026-11-04).
    lim = float(args.spawn_half) + 0.5
    max_abs = 0.0
    rng = np.random.RandomState(11)
    lb = env._model.body("learner")
    for _ in range(300):
        env.step(int(rng.randint(0, 8)))
        max_abs = max(max_abs, abs(float(env._data.xpos[lb.id, 0])),
                      abs(float(env._data.xpos[lb.id, 1])))
    # directed drift test: hold +x (action 3) for 400 steps; the clamp must
    # stop the learner at the arena bound (the real failure mode observed).
    env.reset(seed=13)
    lb = env._model.body("learner")
    dir_max = 0.0
    for _ in range(400):
        env.step(3)
        dir_max = max(dir_max, abs(float(env._data.xpos[lb.id, 0])))
    res = {
        "spawn_half": float(args.spawn_half),
        "learner_max_abs_xy": round(max_abs, 3),
        "learner_bound_ok": bool(max_abs <= lim + 0.2),
        "directed_drift_max": round(dir_max, 3),
        "directed_bound_ok": bool(dir_max <= lim + 0.2),
        "obs_shape": list(env.observation_shape),
        "proprio_dim": int(env.proprio_dim),
        "num_actions": int(env.action_space_n),
        "max_abs_xy": round(float(np.max(np.abs(np.asarray(pos)))), 3),
        "pairwise_min": round(min(d), 3) if d else None,
        "pairwise_median": round(float(np.median(d)), 3) if d else None,
        "pairwise_max": round(max(d), 3) if d else None,
    }
    env.close()
    print("[spread]", json.dumps(res))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
