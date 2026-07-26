#!/usr/bin/env python
"""Stage 8+ Language & Metacognition Evaluation.

Evaluates agent language and self-model capabilities from training logs:
1. Reflection diversity — unique n-grams, topic variation
2. Reflection relevance — keyword overlap with scene concepts
3. Self-model calibration — predicted vs actual reward (when available)
4. Behavioral impact — change in mean_ret before/after reflection episodes

Usage:
    .venv/bin/python scripts/eval/run_language_eval.py --log logs/stage8.log
    .venv/bin/python scripts/eval/run_language_eval.py --log logs/stage8.log --out eval_results/language_eval.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def parse_log(log_path: str) -> tuple[list[dict], list[dict]]:
    """Parse stage log into structured reflection entries and step entries."""
    reflections: list[dict] = []
    steps: list[dict] = []

    refl_pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d+) INFO.*"
        r"\[llm_refl\] (.*)"
    )
    step_pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d+) INFO.*"
        r"step=(\d+) ep=(\d+) mean_ret=([\d.-]+)"
    )

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = refl_pat.match(line)
            if m:
                reflections.append({"timestamp": m.group(1), "text": m.group(2)})
                continue
            m = step_pat.match(line)
            if m:
                steps.append({
                    "timestamp": m.group(1),
                    "step": int(m.group(2)),
                    "episode": int(m.group(3)),
                    "mean_ret": float(m.group(4)),
                })
    return reflections, steps


# ---------------------------------------------------------------------------
# 1. Reflection Diversity
# ---------------------------------------------------------------------------

def measure_reflection_diversity(reflections: list[dict]) -> dict:
    """Measure semantic diversity of reflections."""
    if not reflections:
        return {"score": 0.0, "unique_bigrams": 0, "unique_trigrams": 0,
                "total_reflections": 0, "summary": "no reflections"}

    texts = [r["text"].lower() for r in reflections]
    all_words: list[str] = []
    for t in texts:
        all_words.extend(re.findall(r"[a-z]{3,}", t))

    # Bigram diversity
    bigrams = set()
    trigrams = set()
    for t in texts:
        words = re.findall(r"[a-z]{3,}", t)
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i + 1]))
        for i in range(len(words) - 2):
            trigrams.add((words[i], words[i + 1], words[i + 2]))

    # Topic clusters: count occurrences of key cognitive concepts
    topic_keywords = {
        "color": 0, "size": 0, "object": 0, "shape": 0, "position": 0,
        "sort": 0, "categorize": 0, "pattern": 0, "rule": 0,
        "learn": 0, "understand": 0, "recognize": 0, "predict": 0,
        "confuse": 0, "improve": 0, "explore": 0, "strategy": 0,
    }
    for t in texts:
        for kw in topic_keywords:
            if kw in t:
                topic_keywords[kw] += 1

    # Top topics
    top_topics = sorted(topic_keywords.items(), key=lambda x: -x[1])[:8]

    diversity_score = len(trigrams) / max(len(texts), 1)

    return {
        "total_reflections": len(reflections),
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "diversity_score": round(diversity_score, 2),
        "top_topics": {k: v for k, v in top_topics},
        "summary": (
            f"{len(reflections)} reflections, "
            f"{len(bigrams)} unique bigrams, "
            f"top topics: {dict(top_topics[:5])}"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Reflection Grounding (Relevance to Scene)
# ---------------------------------------------------------------------------

def measure_reflection_grounding(reflections: list[dict]) -> dict:
    """Check whether reflections mention concepts that relate to actual scene state.

    Since we don't have ground-truth scene labels, we measure:
    - Object attribute mentions (color/size/shape) → likely from PerceptionProjector
    - Action mentions (push/move/sort) → likely from policy output
    - Metacognitive terms (learn/understand/predict) → self-model awareness
    """
    if not reflections:
        return {"score": 0.0, "summary": "no reflections"}

    texts = [r["text"] for r in reflections]

    object_terms = {"object", "color", "size", "shape", "red", "blue", "green",
                    "yellow", "white", "black", "orange", "purple",
                    "small", "medium", "large", "ball", "block", "cube", "toy"}
    action_terms = {"push", "move", "sort", "categorize", "grab", "touch",
                    "interact", "manipulate", "reach", "pick"}
    meta_terms = {"learn", "understand", "recognize", "predict", "improve",
                  "confuse", "observe", "attention", "aware", "strategy",
                  "focus", "efficient"}

    obj_hit = sum(1 for t in texts if any(w in t.lower().split() for w in object_terms))
    act_hit = sum(1 for t in texts if any(w in t.lower().split() for w in action_terms))
    meta_hit = sum(1 for t in texts if any(w in t.lower().split() for w in meta_terms))

    n = len(texts)
    return {
        "object_grounding_rate": round(obj_hit / n, 3),
        "action_grounding_rate": round(act_hit / n, 3),
        "metacognition_rate": round(meta_hit / n, 3),
        "summary": (
            f"{obj_hit/n:.0%} object-ref, "
            f"{act_hit/n:.0%} action-ref, "
            f"{meta_hit/n:.0%} meta-ref"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Self-Model Calibration (Prediction vs Reality)
# ---------------------------------------------------------------------------

def measure_self_model_calibration(reflections: list[dict], steps: list[dict]) -> dict:
    """Estimate self-model error from episode-level return variance.

    Since SelfModel doesn't log per-episode predictions, we proxy calibration
    by measuring how well the agent's mean_ret stabilizes over the session.
    Lower variance after reflections = better metacognition.
    """
    if len(steps) < 10:
        return {"score": 0.0, "summary": "insufficient step data"}

    # Split into early (no reflections yet) vs late phase
    mid = len(steps) // 2
    early_rets = [s["mean_ret"] for s in steps[:mid]]
    late_rets = [s["mean_ret"] for s in steps[mid:]]

    early_std = _safe_std(early_rets)
    late_std = _safe_std(late_rets)
    early_mean = sum(early_rets) / len(early_rets)
    late_mean = sum(late_rets) / len(late_rets)

    # Score: lower variance = better self-regulation
    stability_score = max(0.0, 1.0 - late_std / max(early_std, 1.0))
    trend = (late_mean - early_mean) / max(early_mean, 0.01)

    return {
        "early_mean_ret": round(early_mean, 2),
        "late_mean_ret": round(late_mean, 2),
        "early_std": round(early_std, 2),
        "late_std": round(late_std, 2),
        "stability_score": round(stability_score, 3),
        "mean_ret_trend": round(trend, 3),
        "summary": (
            f"ret: {early_mean:.1f}→{late_mean:.1f} (trend {trend:+.2f}), "
            f"std: {early_std:.1f}→{late_std:.1f}"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Behavioral Impact
# ---------------------------------------------------------------------------

def measure_behavioral_impact(reflections: list[dict], steps: list[dict]) -> dict:
    """Estimate whether reflections correlate with return changes.

    Compares mean_ret in windows before and after high-reflection periods.
    """
    if len(reflections) < 50 or len(steps) < 10:
        return {"score": 0.0, "summary": "insufficient data"}

    # Count reflections per step window
    refs_by_window: dict[int, int] = collections.defaultdict(int)
    for r in reflections:
        ts = r["timestamp"]
        refs_by_window[ts[:13]] += 1  # group by hour

    # High-reflection periods
    high_refl_hours = {k for k, v in refs_by_window.items() if v >= 5}

    impact_scores: list[float] = []
    for i in range(1, len(steps) - 1):
        if steps[i]["timestamp"][:13] in high_refl_hours:
            delta = steps[i + 1]["mean_ret"] - steps[i]["mean_ret"]
            impact_scores.append(delta)

    if not impact_scores:
        return {"score": 0.0, "summary": "no high-reflection periods found"}

    avg_delta = sum(impact_scores) / len(impact_scores)
    return {
        "high_reflection_periods": len(high_refl_hours),
        "mean_ret_change_after_reflection": round(avg_delta, 2),
        "score": round(abs(avg_delta) / max(_safe_std([s["mean_ret"] for s in steps]), 0.01), 3),
        "summary": (
            f"{len(high_refl_hours)} high-refl periods, "
            f"avg Δret={avg_delta:+.1f} after reflection"
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return float(var ** 0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Language & Metacognition Eval for Stage 8+")
    ap.add_argument("--log", type=str, required=True,
                    help="Path to stage training log (e.g. logs/stage8.log)")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path for JSON report")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[lang_eval] Log not found: {log_path}")
        return 1

    print(f"[lang_eval] Parsing {log_path} ...")
    reflections, steps = parse_log(str(log_path))
    print(f"[lang_eval] Found {len(reflections)} reflections, {len(steps)} step records")

    # --- Run all metrics ---
    results = {}

    # 1. Diversity
    div = measure_reflection_diversity(reflections)
    results["diversity"] = div

    # 2. Grounding
    grd = measure_reflection_grounding(reflections)
    results["grounding"] = grd

    # 3. Self-model
    cal = measure_self_model_calibration(reflections, steps)
    results["calibration"] = cal

    # 4. Behavioral impact
    imp = measure_behavioral_impact(reflections, steps)
    results["behavioral_impact"] = imp

    # --- Composite score ---
    composite = (
        div["diversity_score"] * 0.3
        + (grd["object_grounding_rate"] + grd["metacognition_rate"]) / 2 * 0.3
        + cal["stability_score"] * 0.2
        + min(imp["score"], 1.0) * 0.2
    )
    results["composite_score"] = round(composite, 3)

    # --- Print report ---
    print()
    print("=" * 60)
    print("  Language & Metacognition Evaluation Report")
    print("=" * 60)
    print(f"\n  Composite Score: {composite:.3f}")
    print()
    print(f"  1. Reflection Diversity:   {div['summary']}")
    print(f"  2. Grounding Rate:          {grd['summary']}")
    print(f"  3. Self-Model Calibration:  {cal['summary']}")
    print(f"  4. Behavioral Impact:       {imp['summary']}")
    print()
    print(f"  Top reflection topics: {div['top_topics']}")
    print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[lang_eval] Report saved -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
