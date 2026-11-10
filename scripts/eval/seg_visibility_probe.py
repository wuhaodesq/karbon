"""seg_visibility_probe.py — precise visible-object count via segmentation.

Renders with the SAME camera as the observation (ego) using MuJoCo
segmentation rendering and counts the distinct object body IDs visible.
Decisive check: can the agent even SEE the objects its count trial asks
about (count within the 4m perception radius)?

Usage:
    MUJOCO_GL=osmesa python scripts/eval/seg_visibility_probe.py --samples 200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.extended_3d_world import ExtendedThreeDWorld
from src.models.grounded_facts import count_within_radius
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--out", type=str, default="/root/seg_visibility_probe.json")
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
        object_spawn_half=float(ec.get("object_spawn_half", 4.0)),
        num_sense_radius=float(ec.get("num_sense_radius", 4.0)),
    )
    ok = True
    obj_ids = {}
    for oi in range(int(getattr(env, "_num_objects", 0))):
        try:
            obj_ids[env._model.body(f"obj_{oi}").id] = oi
        except Exception:
            pass
    rng = np.random.RandomState(7)
    vis, lbl = [], []
    while len(vis) < args.samples and ok:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))  # rebuilds the scene+renderer
        seg = env._renderer
        try:
            seg.enable_segmentation_rendering()
        except Exception as ex:  # noqa: BLE001 - version fallback
            print("[seg] segmentation unavailable:", ex)
            ok = False
            break
        for _ in range(20):
            try:
                seg.update_scene(env._data, camera="ego")
                im = seg.render()
            except Exception as ex:  # noqa: BLE001
                print("[seg] render failed:", ex)
                break
            ids = np.unique(im[:, :, 0])
            seen = sum(1 for i in ids if int(i) in obj_ids)
            vis.append(seen)
            try:
                lb = env._model.body("learner")
                lp = (float(env._data.xpos[lb.id, 0]),
                      float(env._data.xpos[lb.id, 1]))
                pos = []
                for oi in range(int(getattr(env, "_num_objects", 0))):
                    ob = env._model.body(f"obj_{oi}")
                    pos.append((float(env._data.xpos[ob.id, 0]),
                                float(env._data.xpos[ob.id, 1])))
                lbl.append(count_within_radius(pos, lp,
                                               float(ec.get("num_sense_radius", 4.0))))
            except Exception:
                vis.pop()
                continue
            env.step(int(rng.randint(0, env.action_space_n)))
            if len(vis) >= args.samples:
                break
    env.close()
    res = {"segmentation_ok": ok, "samples": len(vis)}
    if vis:
        res.update({
            "visible_mean": round(float(np.mean(vis)), 2),
            "visible_max": int(np.max(vis)),
            "label_mean": round(float(np.mean(lbl)), 2),
            "corr_visible_label": round(float(np.corrcoef(vis, lbl)[0, 1]), 3)
            if np.std(vis) > 0 else None,
        })
    print("[seg]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
