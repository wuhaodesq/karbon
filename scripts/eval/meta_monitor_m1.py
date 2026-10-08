"""Stage 21 M1 — offline meta-monitor validation on the cognitive-event log.

Predicts the agent's OWN verify outcomes from PROCESS features (hypothesis
age, test count, recent outcome window, belief content) — no environment
state, no leakage of the current outcome. Honest protocol: temporal split
(first 70% train / last 30% test), majority baseline, Brier skill.

M1 pass criterion (docs/stage21_design.md): held-out skill_vs_majority > 0.05.

Usage:
    python scripts/eval/meta_monitor_m1.py --events /root/stage21_events.jsonl \
        --out /root/meta_monitor_m1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = ["freshness", "n_probes", "recent_succ", "probe_rate",
            "consec_fail", "lk_x", "lk_y", "obj"]


def build_samples(events_path: str, task: str = "onset") -> tuple[list[list[float]], list[float], list[int]]:
    """Two tasks:

    - ``onset`` (honest, default): pair each probe with its eventual outcome;
      features come from the process state AT PROBE ONSET (outcome not yet
      unfolded). Label = outcome of that probe's verify.
    - ``finalize`` (diagnostic only): predict at verification time — the
      probe-age then reveals *why* the event finalized (arrival=low age,
      deadline=high age), i.e. near-tautological. Kept for contrast.
    """
    xs: list[list[float]] = []
    ys: list[float] = []
    steps: list[int] = []
    recent: deque = deque(maxlen=10)
    probe_times: deque = deque(maxlen=256)
    consec_fail = 0
    lk_x, lk_y, obj = 0.0, 0.0, 0.0
    pending: dict | None = None  # features captured at probe onset
    n_probes_total = 0
    last_verify_step: int | None = None
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("t")
            step = int(e.get("step", 0))
            if t == "propose":
                lk = e.get("lk") or [lk_x, lk_y]
                lk_x, lk_y = float(lk[0]), float(lk[1])
                obj = float(e.get("obj", obj))
            elif t == "probe":
                probe_times.append(step)
                n_probes_total += 1
                lk = e.get("lk") or [lk_x, lk_y]
                lk_x, lk_y = float(lk[0]), float(lk[1])
                # onset features: everything known BEFORE the outcome unfolds
                # (no step/age/current-outcome info)
                recent_succ = (sum(recent) / len(recent)) if recent else 0.5
                rate = float(sum(1 for s in probe_times if step - s <= 2000)) / 20.0
                fresh = float(step - (last_verify_step if last_verify_step is not None
                                      else step))
                pending = {
                    "feat": [fresh, float(n_probes_total), recent_succ, rate,
                             float(consec_fail), lk_x, lk_y, obj],
                    "step": step,
                }
            elif t == "verify":
                ok = float(e.get("ok", 0.0))
                if task == "onset":
                    if pending is not None:
                        xs.append(pending["feat"])
                        ys.append(ok)
                        steps.append(pending["step"])
                        pending = None
                else:  # finalize (diagnostic; circular by construction)
                    lp = probe_times[-1] if probe_times else step
                    age = float(step - lp)
                    recent_succ = (sum(recent) / len(recent)) if recent else 0.5
                    rate = float(sum(1 for s in probe_times if step - s <= 2000)) / 20.0
                    xs.append([age, float(n_probes_total), recent_succ, rate,
                               float(consec_fail), lk_x, lk_y, obj])
                    ys.append(ok)
                    steps.append(step)
                recent.append(ok)
                consec_fail = consec_fail + 1 if ok < 0.5 else 0
                last_verify_step = step
    return xs, ys, steps


def _fit_eval(x: torch.Tensor, y: torch.Tensor, n_train: int,
              epochs: int = 300) -> dict:
    """Train the tiny MLP on the first n_train rows; report honest metrics."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.GELU(),
                          nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        p = model(x[:n_train]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(p, y[:n_train])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_test = torch.sigmoid(model(x[n_train:]).squeeze(-1)).numpy()
    y_test = y[n_train:].numpy()
    m = len(y_test)
    base = float(y_test.mean())
    majority = max(base, 1.0 - base)
    acc = float(((p_test > 0.5) == (y_test > 0.5)).mean())
    brier = float(((p_test - y_test) ** 2).mean())
    return {"n_test": m, "positive_rate": round(base, 3),
            "majority": round(majority, 3), "accuracy": round(acc, 3),
            "skill": round(acc - majority, 3),
            "brier": round(brier, 3),
            "brier_skill": round(1.0 - brier / 0.25, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, default="/root/stage21_events.jsonl")
    ap.add_argument("--out", type=str, default="/root/meta_monitor_m1.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--task", type=str, default="onset",
                    choices=["onset", "finalize"])
    ap.add_argument("--ablation", action="store_true",
                    help="leave-one-out + group ablations on the onset task")
    ap.add_argument("--events_test", type=str, default="",
                    help="cross-log generalization: train on --events, test on this file")
    args = ap.parse_args()

    xs, ys, steps = build_samples(args.events, args.task)
    n = len(xs)
    if n < 200:
        print(f"[m1] too few samples: {n}")
        sys.exit(1)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    n_train = int(n * 0.7)
    mu, sd = x[:n_train].mean(0), x[:n_train].std(0).clamp_min(1e-6)
    x = (x - mu) / sd

    if args.ablation:
        rows = []
        full = _fit_eval(x, y, n_train, args.epochs)
        rows.append(("all", full))
        for i, f in enumerate(FEATURES):
            keep = [j for j in range(len(FEATURES)) if j != i]
            rows.append((f"w/o {f}", _fit_eval(x[:, keep], y, n_train, args.epochs)))
        for label, idx in [("recent-only", [2, 3, 4]),
                           ("belief-only", [5, 6, 7]),
                           ("freshness-only", [0])]:
            rows.append((label, _fit_eval(x[:, idx], y, n_train, args.epochs)))
        print(f"{'variant':>18} {'acc':>6} {'majority':>9} {'skill':>7} {'brier_sk':>9}", flush=True)
        for label, r in rows:
            print(f"{label:>18} {r['accuracy']:>6.3f} {r['majority']:>9.3f} "
                  f"{r['skill']:>7.3f} {r['brier_skill']:>9.3f}", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"ablation": {k: v for k, v in rows}}, f, indent=1)
        print("[m1] saved", args.out, flush=True)
        return

    # cross-log generalization (M1.5): train on --events, test on another log
    if args.events_test:
        xs_b, ys_b, _ = build_samples(args.events_test, args.task)
        xb = (torch.tensor(xs_b, dtype=torch.float32) - mu) / sd  # train stats
        yb = torch.tensor(ys_b, dtype=torch.float32)
        # fit on ALL of log A, evaluate on log B
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.GELU(),
                              nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(args.epochs):
            model.train()
            opt.zero_grad()
            p = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(p, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_b = torch.sigmoid(model(xb).squeeze(-1)).numpy()
        y_b = yb.numpy()
        m = len(y_b)
        base = float(y_b.mean())
        majority = max(base, 1.0 - base)
        acc = float(((p_b > 0.5) == (y_b > 0.5)).mean())
        brier = float(((p_b - y_b) ** 2).mean())
        report = {
            "task": args.task, "train_log": args.events,
            "test_log": args.events_test,
            "n_train_all": n, "n_test": m,
            "test_positive_rate": round(base, 3),
            "majority_baseline": round(majority, 3),
            "accuracy": round(acc, 3),
            "skill_vs_majority": round(acc - majority, 3),
            "brier": round(brier, 3),
            "brier_skill_vs_0.25": round(1.0 - brier / 0.25, 3),
            "features": FEATURES,
        }
        print("[m1] CROSS " + json.dumps(report, indent=1), flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("[m1] saved", args.out, flush=True)
        return

    res = _fit_eval(x, y, n_train, args.epochs)
    report = {
        "task": args.task,
        "events_file": args.events,
        "n_total": n, "n_train": n_train,
        "test_positive_rate": res["positive_rate"],
        "majority_baseline": res["majority"],
        "accuracy": res["accuracy"],
        "skill_vs_majority": res["skill"],
        "brier": res["brier"],
        "brier_skill_vs_0.25": res["brier_skill"],
        "M1_pass_skill>0.05": bool(res["skill"] > 0.05),
        "features": FEATURES,
    }
    print("[m1] " + json.dumps(report, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("[m1] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
