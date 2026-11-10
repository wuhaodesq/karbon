"""count_probe.py — R3-b diagnostic: do the encoder features carry counts?

Collects (render -> model features) samples from the extended world, labels
each with the number of objects within the perceptual horizon, and fits a
RIDGE LINEAR readout on (a) the backbone hidden vector h and (b) the
mean-pooled slot vector. If the linear probe can read counts, the
representation is fine and only the head interface needs fixing; if not,
the representation itself must learn numbering (object-centric work).

Runs under MUJOCO_GL=osmesa (safe concurrently with training).

Usage:
    MUJOCO_GL=osmesa python scripts/eval/count_probe.py --ckpt <ckpt> \
        --samples 3000 --out /root/count_probe.json
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
from src.models.grounded_facts import count_within_radius
from src.platform import get_device
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--steps-per-episode", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/count_probe.json")
    args = ap.parse_args()

    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    ec = cfg["env"]
    device = get_device()
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
    n_layers = int(s18e._ckpt_layer_count(args.ckpt) or
                   int(cfg["model"].get("hybrid_n_layers", 7)))
    model = s18e.build_model(env.observation_shape, env.action_space_n,
                             cfg["model"], device, n_layers,
                             proprio_dim=int(getattr(env, "proprio_dim", 0)))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=False)
    model.eval()

    rng = np.random.RandomState(args.seed)
    feats_h, feats_s, labels = [], [], []
    got = 0
    while got < args.samples:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        for _ in range(args.steps_per_episode):
            obs = env._render()
            obs_t = s18e._obs_to_tensor(obs, device)
            prop_t = s18e._prop_to_tensor(env, device)
            with torch.no_grad():
                out = model(obs_t, proprio=prop_t)
                h = model._last_hidden
                slots = model._last_slots
            try:
                lb = env._model.body("learner")
                lp = (float(env._data.xpos[lb.id, 0]),
                      float(env._data.xpos[lb.id, 1]))
                pos = []
                for oi in range(int(getattr(env, "_num_objects", 0))):
                    ob = env._model.body(f"obj_{oi}")
                    pos.append((float(env._data.xpos[ob.id, 0]),
                                float(env._data.xpos[ob.id, 1])))
                label = count_within_radius(pos, lp, float(ec.get("num_sense_radius", 4.0)))
            except Exception:
                continue
            hh = h.detach().cpu().numpy().reshape(-1) if h is not None else None
            if hh is None:
                continue
            if slots is not None and slots.dim() == 3:
                ss = slots.detach().mean(dim=1).cpu().numpy().reshape(-1)
            else:
                ss = hh
            feats_h.append(hh)
            feats_s.append(ss)
            labels.append(label)
            got += 1
            a = int(rng.randint(0, env.action_space_n))
            env.step(a)
            if got >= args.samples:
                break
    env.close()

    def ridge_probe(X, y, alpha=1.0):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        idx = np.random.RandomState(0).permutation(n)
        tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xn = (X - mu) / sd
        Xtr = np.hstack([Xn[tr], np.ones((len(tr), 1))])
        Xte = np.hstack([Xn[te], np.ones((len(te), 1))])
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ y[tr])
        pred = Xte @ w
        r2 = 1.0 - float(((y[te] - pred) ** 2).sum() /
                         (((y[te] - y[te].mean()) ** 2).sum() + 1e-9))
        acc = float((np.round(pred) == y[te]).mean())
        err = float(np.abs(np.round(pred) - y[te]).mean())
        return {"r2": round(r2, 3), "acc_round": round(acc, 3),
                "mae_round": round(err, 3)}

    y = np.asarray(labels)
    res = {
        "samples": int(len(y)),
        "label_mean": round(float(y.mean()), 3),
        "label_std": round(float(y.std()), 3),
        "probe_h": ridge_probe(feats_h, y),
        "probe_slots": ridge_probe(feats_s, y),
        "chance_acc": round(float((y == int(np.round(y.mean()))).mean()), 3),
    }
    print("[probe]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
