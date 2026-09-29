"""Systematic-reasoning failure-mode diagnostic.

`_eval_systematic_reasoning` combines three weighted components with a
strict multi-signal gate:

    score = 0.35*entropy_score + 0.30*fm_consistency + 0.35*rule_score
    x0.3 if <2 components active, x0.6 if <3

This script runs the far-probe rollout and prints the three components per
task so the bottleneck is visible (the aggregate 0.28 alone cannot say
whether entropy, force-motion consistency or rule count is failing).

Usage:
    python scripts/eval/systematic_diag.py --ckpt <ckpt> --config ... --preset ... \
        --episodes 8 --out /root/sysdiag.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
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


def components(actions: list[int], pairs: list[dict], rules: int, num_actions: int) -> dict:
    entropy_score = 0.0
    if len(actions) > 10:
        counts = Counter(actions)
        total = len(actions)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in probs)
        max_entropy = math.log(max(num_actions, 2))
        entropy_score = max(0.0, 1.0 - entropy / max_entropy)

    fm_consistency = 0.0
    if len(pairs) > 3:
        force_dirs = []
        for p in pairs:
            f = p.get("force", (0, 0))
            angle = math.atan2(f[1], f[0])
            bucket = int((angle + math.pi) / (math.pi / 4)) % 8
            force_dirs.append(bucket)
        dir_counts = Counter(force_dirs)
        if dir_counts:
            mcr = dir_counts.most_common(1)[0][1] / len(force_dirs)
            fm_consistency = min(1.0, mcr * 3.0)

    rule_score = min(1.0, rules / 20.0)
    score = entropy_score * 0.35 + fm_consistency * 0.3 + rule_score * 0.35
    signal_count = sum(1 for x in [entropy_score, fm_consistency, rule_score] if x > 0.1)
    if signal_count < 2:
        score *= 0.3
    elif signal_count < 3:
        score *= 0.6
    return {
        "entropy_score": round(entropy_score, 4),
        "fm_consistency": round(fm_consistency, 4),
        "rule_score": round(rule_score, 4),
        "signal_count": signal_count,
        "score": round(min(1.0, score), 4),
        "n_actions_taken": len(actions),
        "n_pairs": len(pairs),
        "n_distinct_actions": len(set(actions)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage20_hypothesis_v2.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/sysdiag.json")
    args = ap.parse_args()

    device = get_device()
    cfg = load_config(args.config, args.preset)
    tasks_cfg = (cfg.get("curriculum") or {}).get("tasks", [])
    n_layers = int(_s18e._ckpt_layer_count(args.ckpt) or int(cfg["model"].get("hybrid_n_layers", 7)))
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    rule_count = 0
    sym = (ck.get("extra") or {}).get("symbolic_state") if isinstance(ck, dict) else None
    if isinstance(sym, dict):
        rule_count = int(sym.get("next_id", 0))

    model = None
    results = {}
    for spec in tasks_cfg:
        tid = int(spec["id"])
        env = ThreeDWorld(
            num_objects=int(spec["num_objects"]),
            max_episode_steps=args.max_steps,
            render_size=int(cfg["env"].get("render_size", 128)),
            action_force=float(spec.get("action_force", 50.0)),
            developmental_age=float(cfg["env"].get("developmental_age", 0.5)),
            num_occluders=int(cfg["env"].get("num_occluders", 0)),
            occluder_obs_slots=int(cfg["env"].get("occluder_obs_slots", 0)),
            occluder_arrival_reveal=bool(cfg["env"].get("occluder_arrival_reveal", False)),
        )
        env._auto_reset = False
        if model is None:
            model = _s18e.build_model(env.observation_shape, env.action_space_n, cfg["model"],
                                      device, n_layers, proprio_dim=int(getattr(env, "proprio_dim", 0)))
            model.load_state_dict(sd, strict=False)
            model.eval()
        rng = np.random.RandomState(args.seed + tid)
        eps_actions, eps_pairs = [], []
        for _ep in range(args.episodes):
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            try:
                env._actions.clear()
                env._force_motion_pairs.clear()
            except Exception:
                pass
            done = False
            step = 0
            while not done and step < args.max_steps:
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
                step += 1
            eps_actions.extend(list(env._actions))
            eps_pairs.extend(list(env._force_motion_pairs))
        comp = components(eps_actions, eps_pairs, rule_count, env.action_space_n)
        results[str(tid)] = {"tag": spec.get("tag"), **comp}
        print(f"[sysdiag] task {tid} ({spec.get('tag')}): {json.dumps(comp)}", flush=True)
        env.close()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "rule_count": rule_count, "per_task": results}, f, indent=1)
    print("[sysdiag] saved", args.out)


if __name__ == "__main__":
    main()
