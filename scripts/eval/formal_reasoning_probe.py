"""Formal-reasoning probe (Stage 20 gate instrument v1).

Runs a graded battery of formal deduction tasks against the agent's
SymbolBackend (the kanren/fallback engine that carries its formal
knowledge). Each task uses a FRESH backend with explicitly injected
premises, so the probe measures the formal substrate + integration, not
the policy.

Task groups:
  G1 retrieval  - ground-fact lookup (kanren relation / fallback dict)
  G2 modus_ponens - single-hop rules, incl. wildcard antecedents
  G3 composition - ground Horn clauses + a *multi-hop* task that the
     current matcher cannot serve (no derived-fact chaining). That task is
     marked expected="gap" and reported separately - the honest gap list.
  G4 soundness  - non-derivable queries must stay empty (no false positives)

Optional ``--ckpt`` mode additionally loads the checkpoint's
``symbol_backend_state`` and queries the agent's OWN knowledge (a sample
causal fact and the rule inventory) - connecting the battery to the
agent's life rather than only synthetic tasks.

Usage:
    python scripts/eval/formal_reasoning_probe.py [--ckpt <ckpt>] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.symbol_backend import SymbolBackend


def _fresh_backend() -> SymbolBackend:
    return SymbolBackend(max_facts=512, max_rules=128, max_resolution_steps=200)


def _battery() -> list[dict]:
    return [
        # ---------------- G1: retrieval ----------------
        dict(name="fact_retrieve", group="G1_retrieval", expected="pass",
             facts=[("at", ("a",))], rules=[],
             query=("at", ("a",)), min_answers=1),
        dict(name="fact_wildcard", group="G1_retrieval", expected="pass",
             facts=[("at", ("a",)), ("at", ("b",))], rules=[],
             query=("at", ("_",)), min_answers=2),
        # ---------------- G2: modus ponens ----------------
        dict(name="mp_single", group="G2_modus_ponens", expected="pass",
             facts=[("occluded", ("o1",))],
             rules=[([("occluded", ("o1",))], ("track", ("o1",)), 0.9)],
             query=("track", ("o1",)), min_answers=1),
        dict(name="mp_wildcard_antecedent", group="G2_modus_ponens", expected="pass",
             facts=[("occluded", ("o9",))],
             rules=[([("occluded", ("_",))], ("track", ("o9",)), 0.9)],
             query=("track", ("o9",)), min_answers=1),
        dict(name="mp_two_antecedents", group="G2_modus_ponens", expected="pass",
             facts=[("visible", ("k",)), ("near", ("a", "k"))],
             rules=[([("visible", ("k",)), ("near", ("a", "k"))],
                     ("grasp", ("k",)), 0.8)],
             query=("grasp", ("k",)), min_answers=1),
        # ---------------- G3: composition ----------------
        dict(name="ground_horn_transitive", group="G3_composition", expected="pass",
             facts=[("on", ("a", "b")), ("on", ("b", "c"))],
             rules=[([("on", ("a", "b")), ("on", ("b", "c"))],
                     ("on", ("a", "c")), 0.9)],
             query=("on", ("a", "c")), min_answers=1),
        dict(name="two_hop_needs_derived_fact", group="G3_composition", expected="gap",
             facts=[("on", ("a", "b")), ("on", ("b", "c"))],
             rules=[([("on", ("a", "b")), ("on", ("b", "c"))],
                     ("on", ("a", "c")), 0.9),
                    ([("on", ("a", "c"))], ("finish", ("a",)), 0.9)],
             query=("finish", ("a",)), min_answers=1),
        # ---------------- G4: soundness ----------------
        dict(name="soundness_absent_fact", group="G4_soundness", expected="pass",
             facts=[("at", ("a",))], rules=[],
             query=("at", ("z",)), min_answers=0, max_answers=0),
        dict(name="soundness_missing_antecedent", group="G4_soundness", expected="pass",
             facts=[("visible", ("k",))],
             rules=[([("visible", ("k",)), ("near", ("a", "k"))],
                     ("grasp", ("k",)), 0.8)],
             query=("grasp", ("k",)), min_answers=0, max_answers=0),
    ]


def run_task(task: dict) -> dict:
    be = _fresh_backend()
    for pred, args in task.get("facts", []):
        be.add_fact(pred, args)
    for ifp, thenp, conf in task.get("rules", []):
        be.add_rule(ifp, thenp, conf)
    pred, args = task["query"]
    result = be.query(pred, args)
    n = len(result.answers)
    lo = task.get("min_answers", 0)
    hi = task.get("max_answers", 10**9)
    passed = lo <= n <= hi
    return {
        "name": task["name"],
        "group": task["group"],
        "expected": task["expected"],
        "n_answers": n,
        "passed": bool(passed),
        "chain": result.rule_chain[:2],
    }


def probe_ckpt_knowledge(ckpt_path: str) -> dict:
    """Query the agent's own backend state from a checkpoint."""
    try:
        import torch
        ck = torch.load(ckpt_path, map_location="cpu")
    except Exception as ex:
        return {"error": f"load failed: {type(ex).__name__}: {ex}"}
    st = (ck.get("extra") or {}).get("symbol_backend_state")
    if not isinstance(st, dict):
        return {"error": "no symbol_backend_state in ckpt"}
    be = _fresh_backend()
    be.load_state_dict(st)
    facts_db = getattr(be, "_facts_db", {})
    rules_db = getattr(be, "_rules_db", [])
    out = {
        "facts_total": sum(len(v) for v in facts_db.values()),
        "predicates": sorted(facts_db.keys())[:10],
        "rules": len(rules_db),
    }
    # Query up to 3 real causal facts (agent's own discovered knowledge)
    hits = 0
    checked = 0
    for src, tgt in list(facts_db.get("causes", []))[:3]:
        checked += 1
        res = be.query("causes", (src, tgt))
        if len(res.answers) > 0:
            hits += 1
    out["own_knowledge_queries"] = {"checked": checked, "hits": hits}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--out", type=str, default="/root/formal_reasoning_probe.json")
    args = ap.parse_args()

    results = [run_task(t) for t in _battery()]
    scorable = [r for r in results if r["expected"] == "pass"]
    gaps = [r for r in results if r["expected"] == "gap"]
    report = {
        "tasks": results,
        "pass_rate_scorable": round(
            sum(1 for r in scorable if r["passed"]) / max(1, len(scorable)), 3),
        "n_scorable": len(scorable),
        "gaps": [{"name": r["name"], "passed": r["passed"]} for r in gaps],
        "engine": "kanren" if SymbolBackend().available else "fallback",
    }
    print("[formal] engine:", report["engine"])
    for r in results:
        flag = "PASS" if r["passed"] else ("GAP " if r["expected"] == "gap" else "FAIL")
        print(f"[formal] {flag} {r['group']}/{r['name']}: answers={r['n_answers']}",
              flush=True)
    print(f"[formal] scorable pass rate: {report['pass_rate_scorable']:.3f} "
          f"({report['n_scorable']} tasks); gaps={len(gaps)}")

    if args.ckpt:
        report["ckpt_knowledge"] = probe_ckpt_knowledge(args.ckpt)
        print("[formal] ckpt knowledge:", json.dumps(report["ckpt_knowledge"])[:300],
              flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("[formal] saved", args.out)


if __name__ == "__main__":
    main()
