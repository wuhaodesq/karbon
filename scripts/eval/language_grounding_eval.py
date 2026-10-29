"""language_grounding_eval.py — Stage 23 language L-out battery (M3a).

For each truth-table scene, builds the agent_state dict and asks the FROZEN
LanguageGenerator to describe it. Strict rule-based scoring (v2):

  total_ok:       an explicit "N objects/items/things" total is stated and
                  equals num_objects (and no contradictory totals);
  hidden_ok:      if occluded_now > 0, an explicit hidden-count MUST be
                  stated and match; if 0, no positive hidden claim allowed;
  visible_ok:     if a visible count is stated it equals total - hidden;
  kinds_ok:       all real kinds mentioned, no false kinds;
  contradiction:  visible + hidden != total, or conflicting totals.

  faithfulness = all four ok (per scene); pass requires >= 0.8 AND
  hallucination <= 0.1. Hallucination = false kinds OR any wrong number.

Honesty guards:
  - asserts the generator is in 'llm' mode (template echoes the prompt);
  - raw texts saved for manual review;
  - --stub replaces the LLM with a facts-echoing stub (pipeline self-test
    ONLY, results are marked STUB and never pass criteria).

Usage:
    python scripts/eval/language_grounding_eval.py --scenes /root/lang_scenes.json \
        --out /root/m3a_grounding.json --model /root/autodl-tmp/qwen2.5-7b-instruct
    ... --stub            # scoring-pipeline self-test
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KIND_WORDS = {
    "sphere": ("sphere", "ball", "round object"),
    "box": ("box", "block", "cube", "crate"),
    "cylinder": ("cylinder", "can", "bottle", "tube"),
    "capsule": ("capsule", "pill"),
}
OCCLUSION_WORDS = ("hidden", "occluded", "out of sight", "behind the wall",
                   "behind a wall", "lost sight")
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}


def _num_pat() -> str:
    words = "|".join(WORD_NUMS)
    return rf"(\d{{1,3}}|{words})"


def _to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else WORD_NUMS[tok.lower()]


def _stated_totals(text: str) -> set:
    pat = _num_pat() + r"\s+(?:objects?|things?|items?)(?!\s+(?:hidden|behind))"
    return {_to_int(m) for m in re.findall(pat, text.lower())}


def _stated_hidden(text: str) -> set:
    low = text.lower()
    out = set()
    pat1 = (_num_pat()
            + r"\s+(?:of\s+(?:them|which)\s+)?(?:other\s+|more\s+)?"
              r"(?:objects?\s+)?(?:are\s+|is\s+)?"
              r"(?:hidden|occluded|out of sight|behind\s+(?:the\s+|a\s+)?walls?)")
    pat2 = (r"(?:hidden|occluded)\s+(?:objects?|ones?)?\s*(?:is|are|:)?\s*"
            + _num_pat())
    for pat in (pat1, pat2):
        for m in re.findall(pat, low):
            out.add(_to_int(m))
    return out


def _stated_visible(text: str) -> set:
    pat = _num_pat() + r"\s+(?:objects?\s+|things?\s+)?(?:visible|in\s+front|in\s+view)"
    return {_to_int(m) for m in re.findall(pat, text.lower())}


def _kind_mentions(text: str, kinds: list[str]) -> set:
    low = text.lower()
    return {k for k in kinds if any(w in low for w in KIND_WORDS.get(k, (k,)))}


def _false_kind_mentions(text: str, present: list[str]) -> set:
    low = text.lower()
    return {k for k, words in KIND_WORDS.items()
            if k not in present and any(w in low for w in words)}


def _describe_stub(scene: dict) -> str:
    return (f"I see {scene['num_objects']} objects in total: "
            f"{', '.join(scene['kinds'])}. "
            f"{scene.get('occluded_now', 0)} of them are hidden behind walls.")


def score_scene(scene: dict, text: str) -> dict:
    low = text.lower()
    totals = _stated_totals(text)
    hiddens = _stated_hidden(text)
    visibles = _stated_visible(text)
    kinds_mentioned = _kind_mentions(text, scene["kinds"])
    false_kinds = _false_kind_mentions(text, scene["kinds"])
    occ = int(scene.get("occluded_now", 0))

    total_ok = totals == {scene["num_objects"]}
    if occ > 0:
        hidden_ok = hiddens == {occ}
    else:
        hidden_ok = hiddens <= {0}
    vis_expected = scene["num_objects"] - occ
    visible_ok = visibles == set() or visibles == {vis_expected}
    kinds_ok = kinds_mentioned == set(scene["kinds"])

    contradiction = False
    if len(totals) > 1:
        contradiction = True
    for v in visibles:
        if v + occ != scene["num_objects"]:
            contradiction = True
    if visibles and hiddens:
        v = next(iter(visibles))
        h = next(iter(hiddens))
        if v + h != scene["num_objects"]:
            contradiction = True
    if occ > 0 and not any(w in low for w in OCCLUSION_WORDS):
        hidden_ok = False

    faithful = (total_ok and hidden_ok and visible_ok and kinds_ok
                and not contradiction)
    wrong_numbers = ((bool(totals) and totals != {scene["num_objects"]})
                     or (bool(hiddens) and hiddens != {occ})
                     or (bool(visibles) and visibles != {vis_expected})
                     or contradiction)
    hallucinated = bool(false_kinds) or wrong_numbers
    return {
        "scene": scene["id"], "source": scene.get("source", "env"),
        "text": text,
        "totals": sorted(totals), "hiddens": sorted(hiddens),
        "visibles": sorted(visibles),
        "count_ok": total_ok, "hidden_ok": hidden_ok,
        "visible_ok": visible_ok,
        "false_kinds": sorted(false_kinds), "kinds_ok": kinds_ok,
        "faithful": faithful, "hallucinated": hallucinated,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str, default="/root/lang_scenes.json")
    ap.add_argument("--out", type=str, default="/root/m3a_grounding.json")
    ap.add_argument("--stub", action="store_true",
                    help="pipeline self-test: facts-echo stub, never passes")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF model id or LOCAL path (modelscope download)")
    args = ap.parse_args()

    scenes = json.loads(Path(args.scenes).read_text(encoding="utf-8"))
    gen = None
    if not args.stub:
        from src.models.language_generation import LanguageGenerator
        gen = LanguageGenerator(model_name=args.model)
        mode = str(getattr(gen, "mode", getattr(gen, "_mode", "?")))
        print(f"[lang] generator mode: {mode}")
        if mode != "llm":
            print("[lang] TEMPLATE MODE - aborting (no honest L-out possible)")
            print("[lang] (run this on a machine with the Qwen weights)")
            return

    rows = []
    for sc in scenes:
        state = {
            "confidence": 0.6, "familiarity": 0.5, "progress": 0.4,
            "active_rules": [], "current_task": sc["task_type"],
            "episode_return": 0.0,
            "num_objects": sc["num_objects"],
            "kinds_seen": ", ".join(sc["kinds"]),
            "occluded_now": sc.get("occluded_now", 0),
        }
        text = _describe_stub(sc) if args.stub else gen.describe(state)
        r = score_scene(sc, text)
        rows.append(r)
        print(f"[lang] scene {sc['id']} ({r['source']}): "
              f"total={r['count_ok']} hidden={r['hidden_ok']} "
              f"vis={r['visible_ok']} kinds={r['kinds_ok']} "
              f"faith={r['faithful']} hall={r['hallucinated']} "
              f"| totals={r['totals']} hiddens={r['hiddens']}", flush=True)

    n = max(1, len(rows))
    faith_rate = sum(1 for r in rows if r["faithful"]) / n
    hall_rate = sum(1 for r in rows if r["hallucinated"]) / n
    summary = {
        "mode": "STUB" if args.stub else "llm",
        "n_scenes": len(rows),
        "n_proc": sum(1 for r in rows if r["source"] == "proc"),
        "faithfulness": round(faith_rate, 3),
        "hallucination": round(hall_rate, 3),
        "criterion": "faithfulness>=0.8 AND hallucination<=0.1 (M3a)",
        "pass": (not args.stub) and faith_rate >= 0.8 and hall_rate <= 0.1,
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1),
        encoding="utf-8")
    print("[lang] SUMMARY:", json.dumps(summary))


if __name__ == "__main__":
    main()
