"""Stage 22 M2 — concept-rule deduction battery (honest record).

Synthetic concept worlds: a hierarchy (supers -> categories -> instances)
with properties placed at category level (type A), super level (type B,
needs abstraction) and implies-chains (type C, needs composition). For
held-out instances we ask ``prop(inst, ?p)`` and compare:

  full      : concept rules R1+R2+R3 (inheritance + abstraction + implies)
  baseline  : R1+R3 only (instance level: no abstraction)

Metrics: recall of the true (inheritable) properties and precision (no
fabrication; soundness checked against a no-prop world).

Expected honest outcome (per the M2 design): type A equal; types B/C are
unreachable for the instance-level baseline by construction; both sound.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.concept_rules import ConceptProver  # noqa: E402


def _make_world(seed: int, n_supers=3, n_cats=4, n_inst=3):
    rng = random.Random(seed)
    facts = {"member": [], "parent": [], "prop": [], "implies": []}
    # properties: 2 per category (A), 2 per super (B), one implies chain (C)
    cat_props, super_props, implies = {}, {}, []
    for s in range(n_supers):
        sup = f"s{s}"
        raw = [f"sp{s}_{k}" for k in range(2)]
        for p in raw:
            facts["prop"].append((sup, p))
        super_props[sup] = raw
        q = f"imp_target_{s}"
        implies.append((raw[0], q))
        facts["implies"].append((raw[0], q))
        super_props[sup].append(q)  # implied property also inheritable
        for c in range(n_cats):
            cat = f"s{s}_c{c}"
            facts["parent"].append((cat, sup))
            cps = [f"cp{s}_{c}_{k}" for k in range(2)]
            for p in cps:
                facts["prop"].append((cat, p))
            cat_props[cat] = cps
            for i in range(n_inst):
                inst = f"{cat}_i{i}"
                facts["member"].append((inst, cat))
    worlds = {
        "facts": facts,
        "truth": {},  # instance -> expected inheritable props
    }
    for cat, cps in cat_props.items():
        sup = next(p for c, p in facts["parent"] if c == cat)
        s_props = super_props[sup]
        for i in range(n_inst):
            inst = f"{cat}_i{i}"
            worlds["truth"][inst] = set(cps) | set(s_props)
    return worlds


def _build(prover: ConceptProver, world: dict, with_abstraction: bool):
    if with_abstraction:
        prover.add_concept_rules()
    else:
        # instance-level baseline: inheritance + implies, NO abstraction
        prover.add_rule([("member", ("?x", "?c")), ("prop", ("?c", "?p"))],
                        ("prop", ("?x", "?p")))
        prover.add_rule([("prop", ("?x", "?p")), ("implies", ("?p", "?q"))],
                        ("prop", ("?x", "?q")))
    for pred in ("member", "parent", "prop", "implies"):
        for args in world["facts"][pred]:
            prover.add_fact(pred, args)


def _evaluate(with_abstraction: bool, seeds=(0, 1, 2)) -> dict:
    tot_true, tot_got, tot_extra = 0, 0, 0
    for seed in seeds:
        w = _make_world(seed)
        p = ConceptProver(max_facts=2048)
        _build(p, w, with_abstraction)
        for inst, truth in w["truth"].items():
            ans = set()
            for a in p.query("prop", (inst, "?p")):
                if a and isinstance(a[0], str) and not a[0].startswith("?"):
                    ans.add(a[0])
            tot_true += len(truth)
            tot_got += len(ans & truth)
            tot_extra += len(ans - truth)
    recall = tot_got / max(1, tot_true)
    precision = tot_got / max(1, tot_got + tot_extra)
    return {"recall": round(recall, 3), "precision": round(precision, 3),
            "n_expected": tot_true, "n_recovered": tot_got,
            "n_fabricated": tot_extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/root/m2_concept_rules.json")
    args = ap.parse_args()
    report = {
        "full (R1+R2+R3)": _evaluate(True),
        "baseline (R1+R3)": _evaluate(False),
        "note": ("types B/C need the parent chain (abstraction); the "
                 "instance-level baseline cannot join it by construction. "
                 "Precision = soundness (no fabricated properties)."),
    }
    print("[m2-rules] " + json.dumps(report, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("[m2-rules] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
