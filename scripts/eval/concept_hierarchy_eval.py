"""Stage 22 M1 — offline record for the concept hierarchy.

Produces the M1 evidence file:
  1. Structure recovery (clean regime): category/super counts vs truth,
     fragment purity, super alignment.
  2. Classification A/B (novel instances, sparse + clean regimes): flat
     vs hierarchical routing across multiple seeds — recorded honestly
     (the pre-registered "> 0.05" bar measured NEGATIVE in realistic
     regimes; nearest-prototype is strong).

Usage: python scripts/eval/concept_hierarchy_eval.py --out /root/m1_concept.json
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

from src.models.concept_hierarchy import ConceptHierarchy  # noqa: E402


def _structure(seed_means=0, n_supers=3, n_cats_per=4, dim=16, spread=0.55):
    g = torch.Generator().manual_seed(seed_means)
    super_means = torch.randn(n_supers, dim, generator=g) * 2.2
    cat_means, cat_super = [], []
    for s in range(n_supers):
        for c in range(n_cats_per):
            cat_means.append(super_means[s] + torch.randn(dim, generator=g) * spread)
            cat_super.append(s)
    return cat_means, cat_super


def _sample(cat_means, per_cat, seed, noise=0.30):
    g = torch.Generator().manual_seed(seed)
    embs, labels = [], []
    for ci, m in enumerate(cat_means):
        for _ in range(per_cat):
            embs.append(m + torch.randn(len(m), generator=g) * noise)
            labels.append(f"c{ci}")
    return torch.stack(embs), labels


def run(args) -> dict:
    report: dict = {}

    # ---- 1. structure recovery (clean regime) ----
    cm, cat_super = _structure()
    x, labels = _sample(cm, per_cat=40, seed=1)
    h = ConceptHierarchy(dim=16, max_nodes=256, cat_dist=2.4, super_dist=6.0)
    h.build(x, labels)
    pure = 0
    for cid in h._cats:
        counts = h._nodes[cid].label_counts
        pure += 1 if max(counts.values()) >= 0.9 * sum(counts.values()) else 0
    same_super = 0
    for sid in h._supers:
        seen = set()
        for cid in h._cats:
            if h._nodes[cid].parent_id == sid:
                top = max(h._nodes[cid].label_counts.items(), key=lambda kv: kv[1])[0]
                seen.add(cat_super[int(top[1:])])
        same_super += 1 if len(seen) == 1 else 0
    report["structure_recovery"] = {
        "true_categories": 12, "found_categories": h.n_categories,
        "true_supers": 3, "found_supers": h.n_supers,
        "fragment_purity": round(pure / max(1, h.n_categories), 3),
        "super_alignment": round(same_super / max(1, h.n_supers), 3),
    }

    # ---- 2. classification A/B (honest record) ----
    ab = []
    for seed_m in (0, 1, 2):
        cm2, _ = _structure(seed_means=seed_m)
        xtr, ltr = _sample(cm2, per_cat=5, seed=10 + seed_m)
        xte, lte = _sample(cm2, per_cat=25, seed=20 + seed_m)
        h2 = ConceptHierarchy(dim=16, max_nodes=256, cat_dist=2.4, super_dist=5.0)
        h2.build(xtr, ltr)
        ah = sum(1 for i in range(len(xte)) if h2.predict(xte[i], True) == lte[i])
        af = sum(1 for i in range(len(xte)) if h2.predict(xte[i], False) == lte[i])
        n = len(xte)
        ab.append({
            "seed": seed_m,
            "hier_acc": round(ah / n, 3), "flat_acc": round(af / n, 3),
            "skill": round((ah - af) / n, 3),
            "n_categories": h2.n_categories, "n_supers": h2.n_supers,
        })
    report["classification_AB"] = ab
    report["note"] = (
        "Pre-registered '> 0.05 over flat' bar measured NEGATIVE (honest): "
        "nearest-prototype baselines are strong; hierarchy gains +0.01..+0.03 "
        "in sparse regimes, <=0 under blurred structure. Value expected in "
        "M2 concept-level deduction, not the classification margin.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/root/m1_concept.json")
    args = ap.parse_args()
    r = run(args)
    print("[m1-concept] " + json.dumps(r, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print("[m1-concept] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
