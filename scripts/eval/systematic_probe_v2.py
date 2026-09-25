"""Systematic-reasoning probe v2 — honest replacement for the fm-preimage proxy.

The old milestone component (fm_consistency = most-common push-direction
share * 3) is a pre-image concentration proxy: a policy that pushes in one
repetitive direction scores 1.0 without any force->motion understanding.
v2 measures the agent instead of the distribution:

  1. probe_prediction (held-out) — at every occlusion-event start, ask the
     agent's LEARNED probe gate p(arrival) = probe_net(hidden) (trained by
     real tracking outcomes, not scripted) and compare with the strict event
     outcome (best_d < min(0.7*d0, 0.8) with >=3 trajectory points).
     Accuracy / Brier / n are reported per checkpoint.
  2. rule_quality — usage-weighted success rate of the agent's symbolic
     rules with usage >= 3 (behavioral outcome quality, not a count).
  3. action_entropy — 1 - H(actions)/ln(|A|) (weak, kept for continuity).
  4. fm_alignment — cos(force, displacement) on recorded contact events:
     physics-substrate sanity only, explicitly NOT a skill claim.

Deferred (documented, needs a goal-conditioned pushing task + training):
  target-directed intervention success.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.three_d_world import ThreeDWorld  # noqa: E402
from src.models.advanced_cognition import HypothesisTester  # noqa: E402
from src.platform import get_device  # noqa: E402
from src.utils import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s18e", str(Path(__file__).resolve().parent / "run_stage18_full_eval.py"))
_s18e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s18e)


def _make_env(cfg: dict) -> ThreeDWorld:
    ec = cfg["env"]
    env = ThreeDWorld(
        num_objects=int(ec.get("num_objects", 8)),
        max_episode_steps=int(ec.get("max_episode_steps", 600)),
        render_size=int(ec.get("render_size", 128)),
        action_force=float(ec.get("action_force", 50.0)),
        developmental_age=float(ec.get("developmental_age", 0.5)),
        num_occluders=int(ec.get("num_occluders", 4)),
        occluder_obs_slots=int(ec.get("occluder_obs_slots", 3)),
        occluder_arrival_reveal=bool(ec.get("occluder_arrival_reveal", True)),
        object_crossing_every=int(ec.get("object_crossing_every", 0)),
        object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
    )
    env._auto_reset = False
    return env


def _rule_quality(ck: dict) -> dict:
    rm = (ck.get("extra") or {}).get("symbolic_state") or {}
    rules = rm.get("rules", []) if isinstance(rm, dict) else []
    used = [r for r in rules if int(r.get("usage_count", 0)) >= 3]
    tot_u = sum(int(r.get("usage_count", 0)) for r in used)
    tot_s = sum(int(r.get("success_count", 0)) for r in used)
    return {
        "rules_total": len(rules),
        "rules_used>=3": len(used),
        "usage_weighted_success": round(tot_s / tot_u, 3) if tot_u else 0.0,
        "note": "in-sample behavioral outcomes (not held-out)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage20_hypothesis_v2.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="/root/sysprobe_v2.json")
    args = ap.parse_args()

    device = get_device()
    cfg = load_config(args.config, args.preset)
    env = _make_env(cfg)

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0
    n_layers = int(_s18e._ckpt_layer_count(args.ckpt)
                   or int(cfg["model"].get("hybrid_n_layers", 7)))
    model = _s18e.build_model(env.observation_shape, env.action_space_n,
                              cfg["model"], device, n_layers,
                              proprio_dim=int(getattr(env, "proprio_dim", 0)))
    model.load_state_dict(sd, strict=False)
    model.eval()

    adv = cfg.get("advanced") or {}
    tester = HypothesisTester(
        d_model=int(cfg["model"].get("hidden_size", 384)),
        num_actions=env.action_space_n,
        max_hypotheses=int(adv.get("hypothesis_max", 32)),
        probe_epsilon=float(adv.get("hypothesis_probe_epsilon", 0.1)),
    ).to(device)
    tst = (ck.get("extra") or {}).get("hypothesis_tester_state")
    if isinstance(tst, dict):
        try:
            tester.load_state_dict(tst)
            print("[sysprobe] hypothesis_tester state loaded", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[sysprobe] tester load failed: {ex}", flush=True)
    tester.eval()

    rng = np.random.RandomState(args.seed)
    preds: list[tuple[float, int]] = []
    actions: list[int] = []
    n_eps = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        try:
            env._occlusion_events.clear()
            env._force_motion_pairs.clear()
        except Exception:
            pass
        pending: dict = {}
        done = False
        n_steps = 0
        while not done and n_steps < int(cfg["env"].get("max_episode_steps", 600)):
            obs_t = _s18e._obs_to_tensor(obs, device)
            prop_t = _s18e._prop_to_tensor(env, device)
            with torch.no_grad():
                out = model(obs_t, proprio=prop_t, return_hidden=True)
            logits = out[0]
            hidden = out[2] if len(out) > 2 else None
            if rng.rand() < args.epsilon:
                a = int(rng.randint(0, env.action_space_n))
            else:
                a = int(torch.argmax(logits, dim=-1).item())
            actions.append(a)
            step_out = env.step(a)
            obs = step_out.obs
            done = bool(step_out.terminated) or bool(step_out.truncated)
            n_steps += 1
            ax = float(env._data.body("learner").xpos[0])
            ay = float(env._data.body("learner").xpos[1])
            active = env._active_occlusions_3d
            # new events -> ask the learned probe gate
            for key, occ in list(active.items()):
                lk = occ.get("last_known")
                if lk is None:
                    continue
                d = math.hypot(ax - lk[0], ay - lk[1])
                if key not in pending:
                    p = 0.5
                    if hidden is not None:
                        try:
                            with torch.no_grad():
                                p = float(tester.probe_net(hidden).reshape(-1)[0].item())
                        except Exception:
                            p = 0.5
                    pending[key] = {"p": p, "d0": d, "best": d,
                                    "points": len(occ.get("agent_traj_during_occ") or [])}
                else:
                    st = pending[key]
                    st["best"] = min(st["best"], d)
                    st["points"] = max(
                        st["points"], len(occ.get("agent_traj_during_occ") or []))
            # finalized events -> outcome
            for key in [k for k in pending if k not in active]:
                st = pending.pop(key)
                thr = min(st["d0"] * 0.7, 0.8)
                outcome = 1 if (st["points"] >= 3 and st["best"] < thr) else 0
                preds.append((st["p"], outcome))
        for key in list(pending):
            st = pending.pop(key)
            thr = min(st["d0"] * 0.7, 0.8)
            outcome = 1 if (st["points"] >= 3 and st["best"] < thr) else 0
            preds.append((st["p"], outcome))
        n_eps += 1

    # --- aggregate ---
    n = len(preds)
    acc = (sum(1 for p, y in preds if (p > 0.5) == bool(y)) / n) if n else 0.0
    brier = (sum((p - y) ** 2 for p, y in preds) / n) if n else 0.0
    base_rate = (sum(y for _, y in preds) / n) if n else 0.0
    cnt = Counter(actions)
    tot = max(1, len(actions))
    ent = -sum((c / tot) * math.log(max(c / tot, 1e-9)) for c in cnt.values())
    ent_score = max(0.0, 1.0 - ent / math.log(max(env.action_space_n, 2)))

    pairs = list(env._force_motion_pairs)
    aligns = []
    for pr in pairs:
        fx, fy = pr.get("force", (0.0, 0.0))
        vx, vy = pr.get("velocity_after", (0.0, 0.0))
        mf = math.hypot(fx, fy)
        mv = math.hypot(vx, vy)
        if mf > 1e-6 and mv > 1e-6:
            aligns.append((fx * vx + fy * vy) / (mf * mv))

    report = {
        "ckpt": args.ckpt,
        "step": step,
        "episodes": n_eps,
        "probe_prediction": {
            "n": n,
            "accuracy": round(acc, 3),
            "brier": round(brier, 3),
            "positive_rate": round(base_rate, 3),
            "note": "held-out learned gate predicting strict arrivals",
        },
        "rule_quality": _rule_quality(ck),
        "action_entropy_score": round(ent_score, 3),
        "fm_alignment": {
            "n_pairs": len(pairs),
            "mean_cos": round(float(np.mean(aligns)), 3) if aligns else 0.0,
            "frac_cos_positive": round(
                sum(1 for a in aligns if a > 0) / max(1, len(aligns)), 3),
            "note": "physics-substrate sanity, NOT a skill score",
        },
    }
    print("[sysprobe] " + json.dumps(report, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("[sysprobe] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
