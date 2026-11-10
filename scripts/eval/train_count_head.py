"""train_count_head.py — R3-b feasibility: can a small CNN count from pixels?

Offline experiment: collect (render -> count-within-radius) samples from the
extended world, train a tiny CNN count head, report held-out accuracy. If
learnable, wire the head online (aux loss) as the consolidation consumer.

Usage:
    MUJOCO_GL=osmesa python scripts/eval/train_count_head.py \
        --samples 12000 --epochs 6 --out /root/count_head_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.extended_3d_world import ExtendedThreeDWorld
from src.models.grounded_facts import count_within_radius
from src.platform import get_device
from src.utils import load_config


class CountCNN(nn.Module):
    def __init__(self, max_count: int = 12) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, max_count + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


def collect(cfg, device, samples: int, seed: int):
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
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    while len(ys) < samples:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        for _ in range(30):
            obs = env._render()  # (128,128,3) uint8
            if obs.dtype != np.uint8:
                obs = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
            x = obs[::2, ::2, :]  # 64x64
            xs.append(x)
            try:
                lb = env._model.body("learner")
                lp = (float(env._data.xpos[lb.id, 0]),
                      float(env._data.xpos[lb.id, 1]))
                pos = []
                for oi in range(int(getattr(env, "_num_objects", 0))):
                    ob = env._model.body(f"obj_{oi}")
                    pos.append((float(env._data.xpos[ob.id, 0]),
                                float(env._data.xpos[ob.id, 1])))
                ys.append(count_within_radius(pos, lp,
                                              float(ec.get("num_sense_radius", 4.0))))
            except Exception:
                xs.pop()
                continue
            env.step(int(rng.randint(0, env.action_space_n)))
            if len(ys) >= samples:
                break
    env.close()
    X = np.stack(xs).astype(np.float32) / 255.0
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/count_head_probe.json")
    args = ap.parse_args()
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    device = get_device()
    print("[probe] collecting", args.samples, "...", flush=True)
    X, y = collect(cfg, device, args.samples, args.seed)
    n = len(y)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = torch.from_numpy(X[tr]).permute(0, 3, 1, 2).to(device)
    ytr = torch.from_numpy(np.clip(y[tr], 0, args.max_count)).to(device)
    Xte = torch.from_numpy(X[te]).permute(0, 3, 1, 2).to(device)
    yte = np.clip(y[te], 0, args.max_count)

    model = CountCNN(max_count=args.max_count).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 128
    for ep in range(args.epochs):
        perm = torch.randperm(len(ytr), device=device)
        tot = 0.0
        for i in range(0, len(ytr), bs):
            ib = perm[i:i + bs]
            logits = model(Xtr[ib])
            loss = F.cross_entropy(logits, ytr[ib])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ib)
        print(f"[probe] epoch {ep+1}/{args.epochs} loss={tot/len(ytr):.4f}",
              flush=True)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        mae = 0.0
        for i in range(0, len(Xte), 256):
            logits = model(Xte[i:i + 256])
            pred = logits.argmax(1).cpu().numpy()
            correct += int((pred == yte[i:i + 256]).sum())
            total += len(pred)
            mae += float(np.abs(pred - yte[i:i + 256]).sum())
    res = {
        "samples": int(n), "label_mean": round(float(y.mean()), 3),
        "label_std": round(float(y.std()), 3),
        "holdout_acc": round(correct / max(1, total), 3),
        "holdout_mae": round(mae / max(1, total), 3),
        "chance_acc": round(float((yte == int(np.round(y.mean()))).mean()), 3),
    }
    print("[probe]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
