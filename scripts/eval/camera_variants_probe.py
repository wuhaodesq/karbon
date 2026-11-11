"""camera_variants_probe.py — choose a wider observation camera offline.

Tests candidate (camera_pos, fovy) settings against the segmentation
visibility metric (how many objects within the 4m horizon are visible).
Runs under MUJOCO_GL=osmesa; the training env is untouched.

Usage:
    MUJOCO_GL=osmesa python scripts/eval/camera_variants_probe.py
"""
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

VARIANTS = [
    {"name": "default", "pos": (0.0, -1.0, 0.8), "fovy": 60.0},
    {"name": "wide90", "pos": (0.0, -1.5, 1.2), "fovy": 90.0},
    {"name": "overhead", "pos": (0.0, -2.5, 2.8), "fovy": 95.0},
    {"name": "overhead_wide", "pos": (0.0, -3.0, 3.2), "fovy": 110.0},
]


def run_variant(cfg, device, variant, samples=120, seed=7):
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
        camera_pos=tuple(variant["pos"]),
        camera_fovy=float(variant["fovy"]),
    )
    obj_ids = {}
    for oi in range(int(getattr(env, "_num_objects", 0))):
        try:
            obj_ids[env._model.body(f"obj_{oi}").id] = oi
        except Exception:
            pass
    rng = np.random.RandomState(seed)
    vis, lbl = [], []
    while len(vis) < samples:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        seg = env._renderer
        try:
            seg.enable_segmentation_rendering()
        except Exception:
            break
        for _ in range(15):
            try:
                seg.update_scene(env._data, camera="ego")
                im = seg.render()
            except Exception:
                break
            ids = np.unique(im[:, :, 0])
            seen = sum(1 for i in ids if int(i) in obj_ids)
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
                continue
            vis.append(seen)
            env.step(int(rng.randint(0, env.action_space_n)))
            if len(vis) >= samples:
                break
    env.close()
    return {
        "name": variant["name"], "pos": variant["pos"],
        "fovy": variant["fovy"],
        "visible_mean": round(float(np.mean(vis)), 2) if vis else None,
        "visible_max": int(np.max(vis)) if vis else None,
        "label_mean": round(float(np.mean(lbl)), 2) if lbl else None,
        "samples": len(vis),
    }


def main() -> None:
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    device = None
    out = []
    for v in VARIANTS:
        try:
            r = run_variant(cfg, device, v)
            print("[cam]", json.dumps(r), flush=True)
            out.append(r)
        except Exception as ex:  # noqa: BLE001
            print("[cam]", v["name"], "FAILED", ex, flush=True)
    Path("/root/camera_variants.json").write_text(json.dumps(out, indent=1),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
