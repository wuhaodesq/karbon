"""train_detect_head.py — R3-c: object detection from pixels (matched slots).

Supervised object learning with the environment as the 'caregiver pointing':
a small CNN predicts K=8 object slots (presence, dx, dy relative to the
learner, colour 7, kind 4). Slots are matched to the env's object list
(greedy by position) for training; counting = number of present slots.

Pre-registered offline criteria: position MAE < 0.3 m, count accuracy >= 0.8,
colour accuracy >= 0.8 (held-out episodes).

Usage:
    MUJOCO_GL=osmesa python scripts/eval/train_detect_head.py \
        --samples 12000 --epochs 10 --out /root/detect_head.json
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
from src.utils import load_config

COLS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple")
KINDS = ("box", "sphere", "cylinder", "capsule")
K = 8
IN_FRAME_M = 4.5  # top-down footprint radius proxy


def color_name(rgb) -> str:
    import colorsys
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    h, sat, _v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360.0
    if sat < 0.15:
        return "gray"
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 265:
        return "blue"
    return "purple"


class DetectHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, K * 14)  # pres, dx, dy, col7, kind4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1)).view(-1, K, 14)


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
        num_occluders=int(ec.get("num_occluders", 3)),
        occluder_obs_slots=int(ec.get("occluder_obs_slots", 3)),
        object_crossing_every=int(ec.get("object_crossing_every", 0)),
        object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
        object_spawn_half=float(ec.get("object_spawn_half", 4.0)),
        num_sense_radius=float(ec.get("num_sense_radius", 4.0)),
        camera_pos=(0.0, 0.0, 4.0),
        camera_fovy=90.0,
    )
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    while len(ys) < samples:
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        ep_tmp = []
        for _ in range(25):
            obs = env._render()
            if obs.dtype != np.uint8:
                obs = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
            x = obs[::2, ::2, :]
            try:
                lb = env._model.body("learner")
                lx = float(env._data.xpos[lb.id, 0])
                ly = float(env._data.xpos[lb.id, 1])
                lab = []
                for oi in range(int(getattr(env, "_num_objects", 0))):
                    ob = env._model.body(f"obj_{oi}")
                    ox = float(env._data.xpos[ob.id, 0])
                    oy = float(env._data.xpos[ob.id, 1])
                    d = float(np.hypot(ox - lx, oy - ly))
                    if d > IN_FRAME_M:
                        continue
                    o = env._object_lib[oi]
                    col = color_name(getattr(o, "color", (0.5, 0.5, 0.5)))
                    ci = COLS.index(col) if col in COLS else 0
                    kind = str(getattr(o, "kind", "box"))
                    ki = KINDS.index(kind) if kind in KINDS else 0
                    lab.append([(ox - lx) / 4.0, (oy - ly) / 4.0, ci, ki])
                if not lab:
                    ep_tmp.append((x, np.zeros((0, 4), dtype=np.float32)))
                else:
                    ep_tmp.append((x, np.asarray(lab, dtype=np.float32)))
            except Exception:
                continue
            env.step(int(rng.randint(0, env.action_space_n)))
            if len(ys) + len(ep_tmp) >= samples:
                break
        for x, lab in ep_tmp:
            xs.append(x)
            ys.append(lab)
    env.close()
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/detect_head.json")
    args = ap.parse_args()
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[detect] collecting", args.samples, flush=True)
    xs, ys = collect(cfg, device, args.samples, args.seed)
    n = len(xs)
    X = torch.from_numpy(np.stack(xs).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, te = idx[:ntr], idx[ntr:]
    model = DetectHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 64

    def match(pred_xy, truth_xy):
        # greedy nearest matching (no scipy): returns list of (pi, ti)
        d = ((pred_xy[:, None, :] - truth_xy[None, :, :]) ** 2).sum(-1)
        pairs = []
        used_p, used_t = set(), set()
        for flat in np.argsort(d, axis=None):
            pi, ti = int(flat // len(truth_xy)), int(flat % len(truth_xy))
            if pi in used_p or ti in used_t:
                continue
            used_p.add(pi)
            used_t.add(ti)
            pairs.append((pi, ti))
        return pairs

    for ep in range(args.epochs):
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for i in range(0, len(tr), bs):
            ib = perm[i:i + bs]
            xb = X[tr[ib.cpu().numpy()]].to(device)
            out = model(xb)
            loss = torch.zeros((), device=device)
            for b in range(len(ib)):
                lab = ys[tr[ib[b].item()]]
                m = len(lab)
                pred = out[b]
                pres = pred[:, 0]
                tgt_pres = torch.zeros(K, device=device)
                if m > 0:
                    pairs = match(pred[:, 1:3].detach().cpu().numpy(),
                                  lab[:, :2])
                    for pi, ti in pairs:
                        tgt_pres[pi] = 1.0
                        loss = loss + F.smooth_l1_loss(pred[pi, 1:3],
                                                       torch.from_numpy(lab[ti, :2]).to(device))
                        loss = loss + F.cross_entropy(pred[pi, 3:10].unsqueeze(0),
                                                      torch.tensor([int(lab[ti, 2])], device=device))
                        loss = loss + F.cross_entropy(pred[pi, 10:14].unsqueeze(0),
                                                      torch.tensor([int(lab[ti, 3])], device=device))
                loss = loss + F.binary_cross_entropy_with_logits(pres, tgt_pres)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ib)
        print(f"[detect] epoch {ep+1}/{args.epochs} loss={tot/len(tr):.4f}",
              flush=True)

    model.eval()
    pos_err, cnt_ok, cnt_tot = 0.0, 0, 0
    col_ok, col_tot = 0, 0
    with torch.no_grad():
        for i in range(0, len(te), bs):
            ib = te[i:i + bs]
            xb = X[ib].to(device)
            out = model(xb)
            for b in range(len(ib)):
                lab = ys[ib[b]]
                m = len(lab)
                pred = out[b].cpu().numpy()
                present = np.where(pred[:, 0] > 0.0)[0]
                cnt_tot += 1
                if len(present) == m:
                    cnt_ok += 1
                if m > 0:
                    pairs = match(pred[present, 1:3] if len(present) else
                                  np.zeros((0, 2)), lab[:, :2])
                    for pi, ti in pairs:
                        p = present[pi]
                        pos_err += float(np.hypot(pred[p, 1] - lab[ti, 0],
                                                  pred[p, 2] - lab[ti, 1])) * 4.0
                        if int(np.argmax(pred[p, 3:10])) == int(lab[ti, 2]):
                            col_ok += 1
                        col_tot += 1
    res = {
        "samples": n, "label_mean": round(float(np.mean([len(y) for y in ys])), 2),
        "count_acc": round(cnt_ok / max(1, cnt_tot), 3),
        "pos_mae_m": round(pos_err / max(1, col_tot), 3),
        "color_acc": round(col_ok / max(1, col_tot), 3),
        "criteria": "pos_mae<0.3m, count_acc>=0.8, color_acc>=0.8",
    }
    print("[detect]", json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
