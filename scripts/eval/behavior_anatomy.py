"""Behavior anatomy (plateau analysis, step A).

Compares two checkpoints on identical seeded rollouts and answers:
  * What does each policy DO (argmax action histogram)?
  * Is the current policy stationary more often?
  * Where does its reward come from — moving, rotating in place, or spikes?
  * How much reward flows while stationary (the "passive steady income")?

Usage:
    python scripts/eval/behavior_anatomy.py \
        --ckpts /root/autodl-tmp/karbon_ckpts/backup_stage20_012051552.pt,\
/root/autodl-tmp/karbon_ckpts/backup_stage20_18350000.pt \
        --out /root/behavior_anatomy.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.platform import get_device
from src.utils import load_config

_ev = Path(__file__).resolve().parent / "run_stage18_full_eval.py"
_spec = importlib.util.spec_from_file_location("s18e", str(_ev))
_s18e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s18e)

_dp = Path(__file__).resolve().parent / "op_failure_diag.py"
_dspec = importlib.util.spec_from_file_location("opdiag", str(_dp))
_opdiag = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_opdiag)

ACTION_NAMES = [f"move_{d}" for d in
                ("N", "NE", "E", "SE", "S", "SW", "W", "NW")] + \
               ["grasp", "release", "tool", "rotate"]
STAT_VEL = 0.05      # m/s below which the agent is considered stationary
SPIKE_R = 10.0       # per-step reward magnitude counted as a spike


def run_ckpt(ckpt_path: str, cfg: dict, episodes: int, seed: int,
             eps: float, max_steps: int, env_type: str = "threed") -> dict:
    device = get_device()
    tasks = (cfg.get("curriculum") or {}).get("tasks", [])
    n_layers = int(_s18e._ckpt_layer_count(ckpt_path)
                   or int(cfg["model"].get("hybrid_n_layers", 7)))
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0

    model = None
    act_arg = np.zeros(12)
    act_eps = np.zeros(12)
    rew_stat, rew_move = [], []
    rew_by_action = {i: [] for i in range(12)}
    spikes, small, total_steps = 0, 0, 0
    n_events, n_pass, n_no_move = 0, 0, 0
    vel_all = []

    for spec in tasks:
        env = _opdiag._make_env(cfg, int(spec["num_objects"]),
                                float(spec.get("action_force", 50.0)), max_steps,
                                env_type=env_type)
        if model is None:
            model = _s18e.build_model(
                env.observation_shape, env.action_space_n, cfg["model"], device,
                n_layers, proprio_dim=int(getattr(env, "proprio_dim", 0)))
            model.load_state_dict(sd, strict=False)
            model.eval()
        rng = np.random.RandomState(seed + int(spec["id"]))
        for _ep in range(episodes):
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            try:
                env._occlusion_events.clear()
            except Exception:
                pass
            done, si = False, 0
            while not done and si < max_steps:
                obs_t = _s18e._obs_to_tensor(obs, device)
                prop_t = _s18e._prop_to_tensor(env, device)
                with torch.no_grad():
                    out = model(obs_t, proprio=prop_t)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                a_arg = int(torch.argmax(logits, dim=-1).item())
                if rng.random() < eps:
                    a = int(rng.randint(0, env.action_space_n))
                else:
                    a = a_arg
                so = env.step(a)
                obs = so.obs
                r = float(so.reward)
                prop = np.asarray(so.proprio, dtype=float)
                vel = float(np.hypot(prop[3], prop[4])) if prop.size >= 6 else 0.0
                vel_all.append(vel)
                act_arg[a_arg] += 1
                act_eps[a] += 1
                rew_by_action[a].append(r)
                (rew_stat if vel < STAT_VEL else rew_move).append(r)
                if abs(r) >= SPIKE_R:
                    spikes += 1
                elif abs(r) > 1e-6:
                    small += 1
                total_steps += 1
                done = bool(so.terminated) or bool(so.truncated)
                si += 1
            for ev in list(env._occlusion_events):
                e = _opdiag.analyze_event(ev)
                if e:
                    n_events += 1
                    n_pass += int(e["pass"])
                    n_no_move += int(e["bucket"] == "no_move")
        env.close()

    def dist(xs):
        if not xs:
            return {"n": 0}
        xs = np.asarray(xs)
        return {"n": int(xs.size), "mean": round(float(xs.mean()), 4),
                "median": round(float(np.median(xs)), 4),
                "p90": round(float(np.percentile(xs, 90)), 4),
                "pos_frac": round(float((xs > 1e-6).mean()), 4)}

    out = {
        "ckpt": ckpt_path, "step": step, "steps": total_steps,
        "action_hist_argmax": {ACTION_NAMES[i]: int(act_arg[i]) for i in range(12)},
        "action_hist_exec": {ACTION_NAMES[i]: int(act_eps[i]) for i in range(12)},
        "stationary_frac": round(float(np.mean(np.asarray(vel_all) < STAT_VEL)), 4),
        "vel_median": round(float(np.median(vel_all)), 4),
        "reward_when_stationary": dist(rew_stat),
        "reward_when_moving": dist(rew_move),
        "rew_by_action_mean": {ACTION_NAMES[i]: (round(float(np.mean(v)), 4)
                                                 if v else None)
                               for i, v in rew_by_action.items()},
        "spike_frac": round(spikes / max(total_steps, 1), 4),
        "small_rew_frac": round(small / max(total_steps, 1), 4),
        "events": n_events, "pass_rate": round(n_pass / max(n_events, 1), 3),
        "no_move_frac": round(n_no_move / max(n_events, 1), 3),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage20_hypothesis_v2.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/behavior_anatomy.json")
    ap.add_argument("--env-type", type=str, default="threed",
                    choices=["threed", "extended"],
                    help="eval world: threed (historical) or extended (M2-p2)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.preset)
    results = []
    for ckpt in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        r = run_ckpt(ckpt, cfg, args.episodes, args.seed, args.epsilon,
                     args.max_steps, env_type=getattr(args, "env_type", "threed"))
        results.append(r)
        print(f"[anatomy] step={r['step']} stationary={r['stationary_frac']} "
              f"pass={r['pass_rate']} events={r['events']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("[anatomy] saved", args.out)


if __name__ == "__main__":
    main()
