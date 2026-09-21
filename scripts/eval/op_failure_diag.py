"""Object-permanence failure-mode diagnosis (Stage 20 hard-gate work).

Runs the far-probe env per curriculum task, collects occlusion events and
decomposes each event outcome into buckets:

  pass    : best_d < min(0.7*start_d, 0.8)  (milestone criterion)
  no_move : best_d > 0.95*start_d           (agent never approached)
  weak    : 0.7*start_d < best_d <= 0.95*start_d (approached but insufficient)
  close   : ratio <= 0.7 but floor 0.8 not met (tiny start_d; rare)

Also counts episodes with ZERO occlusion events (unscorable regardless of
behaviour) — the probe's coverage, not the policy's competence.

Usage:
    python scripts/eval/op_failure_diag.py \
        --ckpt /root/autodl-tmp/karbon_ckpts/backup_stage20_12000000.pt \
        --config stage20_hypothesis_v2.yaml --preset cloud_24g \
        --episodes 12 --out /root/op_fail_diag_12M.json
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

from src.envs.three_d_world import ThreeDWorld
from src.platform import get_device
from src.utils import load_config

_spec = importlib.util.spec_from_file_location(
    "s18e", str(Path(__file__).resolve().parent / "run_stage18_full_eval.py"))
_s18e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s18e)


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def analyze_event(ev: dict) -> dict | None:
    lk = ev["last_known"]
    traj = ev.get("agent_traj_during_occ") or []
    if not traj:
        return None
    start_d = _dist(traj[0], lk)
    best_d = min(_dist(p, lk) for p in traj)
    end_d = _dist(traj[-1], lk)
    thr = min(start_d * 0.7, 0.8)
    passed = best_d < thr
    if passed:
        bucket = "pass"
    elif best_d > 0.95 * start_d:
        bucket = "no_move"
    elif best_d <= 0.7 * start_d:
        bucket = "close"
    else:
        bucket = "weak"
    return {
        "start_d": round(start_d, 3),
        "best_d": round(best_d, 3),
        "end_d": round(end_d, 3),
        "ratio": round(best_d / max(start_d, 1e-6), 3),
        "pass": bool(passed),
        "bucket": bucket,
        "steps": len(traj),
        "truly": bool(ev.get("truly_occluded", True)),
    }


def _make_env(cfg: dict, num_objects: int, action_force: float, max_steps: int) -> ThreeDWorld:
    env = ThreeDWorld(
        num_objects=int(num_objects),
        max_episode_steps=int(max_steps),
        render_size=int(cfg["env"].get("render_size", 128)),
        action_force=float(action_force),
        developmental_age=float(cfg["env"].get("developmental_age", 0.5)),
        num_occluders=int(cfg["env"].get("num_occluders", 0)),
        occluder_obs_slots=int(cfg["env"].get("occluder_obs_slots", 0)),
        occluder_arrival_reveal=bool(cfg["env"].get("occluder_arrival_reveal", False)),
    )
    env._auto_reset = False
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage20_hypothesis_v2.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/op_fail_diag.json")
    args = ap.parse_args()

    device = get_device()
    cfg = load_config(args.config, args.preset)
    tasks_cfg = (cfg.get("curriculum") or {}).get("tasks", [])
    n_layers = int(_s18e._ckpt_layer_count(args.ckpt) or int(cfg["model"].get("hybrid_n_layers", 7)))

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0

    results: dict[str, dict] = {}
    model = None
    for spec in tasks_cfg:
        tid = int(spec["id"])
        tag = str(spec.get("tag", tid))
        env = _make_env(cfg, int(spec["num_objects"]),
                        float(spec.get("action_force", 50.0)), args.max_steps)
        if model is None:
            model = _s18e.build_model(
                env.observation_shape, env.action_space_n, cfg["model"], device,
                n_layers, proprio_dim=int(getattr(env, "proprio_dim", 0)))
            model.load_state_dict(sd, strict=False)
            model.eval()
        rng = np.random.RandomState(args.seed + tid)
        evs_all: list[dict] = []
        zero_eps = 0
        for _ep in range(args.episodes):
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            try:
                env._occlusion_events.clear()
            except Exception:
                pass
            done = False
            step_i = 0
            while not done and step_i < args.max_steps:
                obs_t = _s18e._obs_to_tensor(obs, device)
                prop_t = _s18e._prop_to_tensor(env, device)
                with torch.no_grad():
                    out = model(obs_t, proprio=prop_t)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                if rng.random() < args.epsilon:
                    action = int(rng.randint(0, env.action_space_n))
                else:
                    action = int(torch.argmax(logits, dim=-1).item())
                step_out = env.step(action)
                obs = step_out.obs
                done = bool(step_out.terminated) or bool(step_out.truncated)
                step_i += 1
            eps_events = [e for e in (analyze_event(ev) for ev in list(env._occlusion_events)) if e]
            if not eps_events:
                zero_eps += 1
            evs_all.extend(eps_events)
        n = len(evs_all)
        pass_rate = sum(1 for e in evs_all if e["pass"]) / max(1, n)
        results[str(tid)] = {
            "tag": tag,
            "num_objects": int(spec["num_objects"]),
            "episodes": args.episodes,
            "zero_event_episodes": zero_eps,
            "n_events": n,
            "pass_rate": round(pass_rate, 3),
            "mean_ratio": round(float(np.mean([e["ratio"] for e in evs_all])) if n else 0.0, 3),
            "buckets": {b: sum(1 for e in evs_all if e["bucket"] == b)
                        for b in ("pass", "no_move", "weak", "close")},
            "events": evs_all,
        }
        print(f"[diag] task {tid} ({tag}): events={n} zero_eps={zero_eps} "
              f"pass={pass_rate:.3f} buckets={results[str(tid)]['buckets']}", flush=True)
        env.close()

    overall = [e for r in results.values() for e in r["events"]]
    n = len(overall)
    summary = {
        "n_events": n,
        "pass_rate": round(sum(1 for e in overall if e["pass"]) / max(1, n), 3),
        "buckets": {b: sum(1 for e in overall if e["bucket"] == b)
                    for b in ("pass", "no_move", "weak", "close")},
        "zero_event_episodes": sum(r["zero_event_episodes"] for r in results.values()),
    }
    print("[diag] OVERALL:", json.dumps(summary))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "step": step, "summary": summary,
                   "per_task": results}, f, indent=1)
    print("[diag] saved", args.out)


if __name__ == "__main__":
    main()
