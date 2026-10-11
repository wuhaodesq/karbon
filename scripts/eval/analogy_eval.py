"""Stage 22 M3 — analogy battery (honest record).

Held-out relation transfer: for each demonstration pair (a,b), a DIFFERENT
triple from the same relation family provides the source c and target d;
the engine must return d as the top answer. Reports top-1/top-3 accuracy
per relation vs the random baseline (1/(N-3)).

Usage: python scripts/eval/analogy_eval.py --out /root/m3_analogy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.analogy import AnalogyEngine  # noqa: E402


def _world(seed: int, n_per: int = 30, dim: int = 16, noise: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    R = {
        "above": torch.tensor([0.0, 1.0] + [0.0] * (dim - 2)),
        "older": torch.tensor([1.0, 0.0] + [0.0] * (dim - 2)),
        "left": torch.tensor([0.0, 0.0, 1.0] + [0.0] * (dim - 3)),
    }
    ents, pairs = {}, []
    for rname, rvec in R.items():
        for i in range(n_per):
            base = torch.randn(dim, generator=g)
            a, b = f"{rname}_a{i}", f"{rname}_b{i}"
            ents[a] = base + torch.randn(dim, generator=g) * noise
            ents[b] = base + rvec + torch.randn(dim, generator=g) * noise
            pairs.append((rname, a, b))
    return ents, pairs


def run(seeds=(0, 1, 2)) -> dict:
    per_rel: dict[str, dict] = {}
    for seed in seeds:
        ents, pairs = _world(seed)
        eng = AnalogyEngine(dim=16, max_entities=512)
        for k, v in ents.items():
            eng.add_entity(k, v)
        for _, a, b in pairs:
            eng.add_mapping(a, b)
        rand = 1.0 / max(1, eng.n_entities - 3)
        by_rel: dict[str, list] = {}
        for i, (rname, a, b) in enumerate(pairs):
            fam = [(x, y) for r, x, y in pairs if r == rname]
            j = (i + 1) % len(fam)
            c, d_true = fam[j]
            if c == a and d_true == b:
                continue
            top = eng.analogize(a, b, c, top_k=3)
            top1 = bool(top) and top[0][0] == d_true
            top3 = any(t[0] == d_true for t in top)
            by_rel.setdefault(rname, []).append((top1, top3))
        for rname, rows in by_rel.items():
            agg = per_rel.setdefault(rname, {"n": 0, "top1": 0, "top3": 0})
            agg["n"] += len(rows)
            agg["top1"] += sum(1 for t1, _ in rows if t1)
            agg["top3"] += sum(1 for _, t3 in rows if t3)
        per_rel.setdefault("_random_baseline", rand)
    for rname, agg in per_rel.items():
        if isinstance(agg, dict) and "n" in agg:
            agg["top1_acc"] = round(agg["top1"] / max(1, agg["n"]), 3)
            agg["top3_acc"] = round(agg["top3"] / max(1, agg["n"]), 3)
    return per_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/root/m3_analogy.json")
    args = ap.parse_args()
    r = run()
    print("[m3-analogy] " + json.dumps(r, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print("[m3-analogy] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
