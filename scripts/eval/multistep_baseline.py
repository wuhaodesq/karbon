"""multistep_baseline.py — M-P1: honest multi-step capability baseline.

Forces the env's key_door chain task (grasp the key obj_1 -> bring it near
the door obj_0 -> reach the door) and measures the CURRENT policy:
grasp_rate, door_rate, completion_rate over N episodes.

Usage:
    MUJOCO_GL=osmesa python scripts/eval/multistep_baseline.py \
        --ckpt <ckpt> --episodes 20 --out /root/multistep_baseline.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.eval.run_stage18_full_eval as s18e
from src.envs.extended_3d_world import ExtendedThreeDWorld
from src.platform import get_device
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/multistep_baseline.json")
    args = ap.parse_args()
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    ec = cfg["env"]
    device = get_device()
    env = ExtendedThreeDWorld(
        num_objects=int(ec.get("num_objects", 8)),
        num_siblings=int(ec.get("num_siblings", 2)),
        season_cycle_steps=int(ec.get("season_cycle_steps", 500000)),
        max_episode_steps=args.max_steps,
        render_size=int(ec.get("render_size", 128)),
        action_force=float(ec.get("action_force", 50.0)),
        developmental_age=0.5,
        num_occluders=int(ec.get("num_occluders", 3)),
        occluder_obs_slots=int(ec.get("occluder_obs_slots", 3)),
        object_crossing_every=0,
        object_crossing_hold_steps=0,
        object_spawn_half=float(ec.get("object_spawn_half", 4.0)),
        num_sense_radius=float(ec.get("num_sense_radius", 4.0)),
        camera_pos=tuple(ec.get("camera_pos", (0.0, -1.5, 1.2))),
        camera_fovy=float(ec.get("camera_fovy", 90.0)),
    )
    env._auto_reset = False
    n_layers = int(s18e._ckpt_layer_count(args.ckpt) or
                   int(cfg["model"].get("hybrid_n_layers", 7)))
    model = s18e.build_model(env.observation_shape, env.action_space_n,
                             cfg["model"], device, n_layers,
                             proprio_dim=int(env.proprio_dim))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=False)
    model.eval()
    rng = np.random.RandomState(args.seed)

    grasp_n = door_n = done_n = 0
    rows = []
    for ep in range(args.episodes):
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        # Force the key_door chain task after the scene is built.
        env._task_type = "key_door"
        env._task_target_body = "obj_0"
        env._task_key_body = "obj_1"
        env._task_door_open = False
        env._task_reward_collected = False
        env._task_progress = 0.0
        gripped = False
        step_i = 0
        while step_i < args.max_steps:
            obs_t = s18e._obs_to_tensor(obs, device)
            prop_t = s18e._prop_to_tensor(env, device)
            with torch.no_grad():
                out = model(obs_t, proprio=prop_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if rng.random() < args.epsilon:
                a = int(rng.randint(0, env.action_space_n))
            else:
                a = int(torch.argmax(logits, dim=-1).item())
            st = env.step(a)
            obs = st.obs
            if env._held_obj_id is not None:
                gripped = True
            step_i += 1
            if st.terminated or st.truncated:
                break
        row = {"ep": ep, "grasp": bool(gripped),
               "door": bool(getattr(env, "_task_door_open", False)),
               "done": bool(getattr(env, "_task_reward_collected", False))}
        grasp_n += int(row["grasp"])
        door_n += int(row["door"])
        done_n += int(row["done"])
        rows.append(row)
        print("[mstep]", json.dumps(row), flush=True)
    env.close()
    n = max(1, args.episodes)
    res = {
        "ckpt": args.ckpt, "episodes": args.episodes,
        "grasp_rate": round(grasp_n / n, 3),
        "door_rate": round(door_n / n, 3),
        "completion_rate": round(done_n / n, 3),
        "criterion": "M-P1 baseline (reference for the planner gain +0.2)",
    }
    print("[mstep]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps({"summary": res, "rows": rows}, indent=1),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
