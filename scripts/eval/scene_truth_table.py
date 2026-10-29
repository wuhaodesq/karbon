"""scene_truth_table.py — Stage 23 language L-out: scene facts for describe().

Builds a MIXED truth table:
  - 16 procedural scenes: object count 3-12, varied kind subsets, varied
    hidden counts 0..total (tests counting/kind/hidden fidelity);
  - 8 real env episodes (seeded rolls, no policy): authentic occlusion /
    task activity from ThreeDWorld (env composition is fixed: 8 objects).

Output feeds language_grounding_eval.py (strict scoring).

Usage:
    python scripts/eval/scene_truth_table.py --episodes 8 --steps 150 \
        --out /root/lang_scenes.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KINDS = ["box", "sphere", "cylinder", "capsule"]
TASKS = ["free_play", "obstacle", "key_door", "collect"]


def _procedural(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    scenes = []
    for i in range(n):
        total = rng.randint(3, 12)
        kinds = sorted(rng.sample(KINDS, rng.randint(1, 3)))
        scenes.append({
            "id": i,
            "source": "proc",
            "num_objects": total,
            "kinds": kinds,
            "n_colors": rng.randint(1, 6),
            "task_type": rng.choice(TASKS),
            "occluded_now": rng.randint(0, total),
        })
    return scenes


def _real_episodes(cfg: dict, episodes: int, steps: int,
                   seed: int, start_id: int) -> list[dict]:
    from src.envs.three_d_world import ThreeDWorld
    ec = cfg["env"]
    env = ThreeDWorld(
        num_objects=int(ec.get("num_objects", 8)),
        max_episode_steps=int(ec.get("max_episode_steps", 300)),
        render_size=int(ec.get("render_size", 128)),
        action_force=float(ec.get("action_force", 50.0)),
        developmental_age=float(ec.get("developmental_age", 0.5)),
        num_occluders=int(ec.get("num_occluders", 0)),
        occluder_obs_slots=int(ec.get("occluder_obs_slots", 0)),
        occluder_arrival_reveal=bool(ec.get("occluder_arrival_reveal", False)),
        object_crossing_every=int(ec.get("object_crossing_every", 0)),
        object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
    )
    env._auto_reset = False
    rng = np.random.RandomState(seed)
    scenes = []
    for i in range(episodes):
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        try:
            env._occlusion_events.clear()
        except Exception:
            pass
        for _ in range(steps):
            env.step(int(rng.randint(0, env.action_space_n)))
        objs = env.objects
        kinds = sorted({str(getattr(o, "kind", "?")) for o in objs})
        scenes.append({
            "id": start_id + i,
            "source": "env",
            "num_objects": len(objs),
            "kinds": kinds,
            "n_colors": len({round(float(getattr(o, "color", (0.5,))[0]), 1)
                             for o in objs}),
            "task_type": str(getattr(env, "_task_type", "free_play")),
            "occluded_now": len(getattr(env, "_active_occlusions_3d", {}) or {}),
        })
        print(f"[scene] env {start_id + i}: objects={len(objs)} kinds={kinds} "
              f"occ={scenes[-1]['occluded_now']}", flush=True)
    env.close()
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="stage20_hypothesis_v2.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default="/root/lang_scenes.json")
    args = ap.parse_args()

    from src.utils import load_config
    cfg = load_config(args.config, args.preset)
    scenes = _procedural(args.seed, args.proc)
    scenes += _real_episodes(cfg, args.episodes, args.steps,
                             args.seed + 1, start_id=len(scenes))
    for s in scenes[:4]:
        print("[scene] proc", s, flush=True)
    Path(args.out).write_text(json.dumps(scenes, indent=1), encoding="utf-8")
    print("[scene] saved", args.out, f"({len(scenes)} scenes)")


if __name__ == "__main__":
    main()
