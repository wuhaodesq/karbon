"""render_visibility_probe.py — does the render show the objects we ask to count?

For each sample: render the scene, count distinct saturated colour blobs
(objects are randomly coloured; floor/walls are grey), and compare with the
count of objects within the perception radius. If the render shows ~1-3
while the label is ~6, the observation interface (camera/framing) is the
bottleneck for number sense.

Usage:
    MUJOCO_GL=osmesa python scripts/eval/render_visibility_probe.py \
        --samples 200 --out /root/visibility_probe.json
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


def _hsv_sat(pix: np.ndarray) -> np.ndarray:
    mx = pix.max(axis=2)
    mn = pix.min(axis=2)
    return np.where(mx > 0.05, (mx - mn) / (mx + 1e-6), 0.0)


def _visible_colors(pix: np.ndarray) -> int:
    """Distinct quantised hues among saturated, non-tiny blobs."""
    sat = _hsv_sat(pix)
    mask = sat > 0.45
    pix8 = (np.clip(pix, 0, 1) * 255).astype(np.int32)
    hsv = np.zeros_like(pix)
    # cheap hue: dominant channel ordering
    r, g, b = pix8[:, :, 0], pix8[:, :, 1], pix8[:, :, 2]
    mx = np.max(pix8, axis=2).astype(np.float64) + 1e-6
    keys = ((r * 131 + g * 197 + b * 263) % 4096) // 64  # 64 coarse bins
    keys = keys[mask]
    if keys.size < 5:  # a blob needs a few pixels
        return 0
    vals, counts = np.unique(keys, return_counts=True)
    # a real object blob has >= 4 pixels in its colour bin
    return int((counts >= 4).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--out", type=str, default="/root/visibility_probe.json")
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
    rng = np.random.RandomState(7)
    vis, lbl = [], []
    while len(vis) < args.samples:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        for _ in range(20):
            obs = env._render()
            pix = np.asarray(obs, dtype=np.float32)
            if pix.max() > 1.5:
                pix /= 255.0
            vis.append(_visible_colors(pix))
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
    res = {
        "samples": len(vis),
        "visible_colors_mean": round(float(np.mean(vis)), 2),
        "visible_colors_max": int(np.max(vis)),
        "label_mean": round(float(np.mean(lbl)), 2),
        "label_max": int(np.max(lbl)),
        "corr_visible_label": round(float(np.corrcoef(vis, lbl)[0, 1]), 3)
        if np.std(vis) > 0 else None,
    }
    print("[vis]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
