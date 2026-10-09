"""Tests for Stage 21 M3 — contradiction detection & self-correction.

M3 criteria: injected-contradiction detection rate > 0.8; false-positive
rate < 0.1 on non-conflicting rules.
"""

from __future__ import annotations

import torch

from src.models.contradiction import ContradictionDetector
from src.models.neural_symbolic import Rule, RuleMemory


D = 64


def _rules_from_memory(mem: RuleMemory) -> dict:
    return dict(mem._rules)


def _make_memory_with_injections(n_normal: int = 20, n_inject: int = 10,
                                 seed: int = 0):
    """Normal rules + one injected conflict per injection (same embedding,
    different action, inserted directly to bypass add()'s merge logic)."""
    torch.manual_seed(seed)
    mem = RuleMemory(max_rules=64, d_model=D)
    normal_ids = []
    for i in range(n_normal):
        r = mem.add(torch.randn(D), action=i % 7, confidence=0.5)
        normal_ids.append(r.id)
    injected = []
    for k in range(n_inject):
        base = mem._rules[normal_ids[k]]
        conflict = Rule(id=10000 + k,
                        condition_embedding=base.condition_embedding.clone(),
                        action=(int(base.action) + 1) % 7, confidence=0.5)
        mem._rules[conflict.id] = conflict
        injected.append((normal_ids[k], conflict.id))
    return mem, injected


def test_injected_detection_rate_and_false_positives():
    det = ContradictionDetector(sim_threshold=0.9)
    found = 0
    total = 0
    fps = 0
    for seed in range(5):
        mem, injected = _make_memory_with_injections(seed=seed)
        rules = _rules_from_memory(mem)
        pairs = det.detect(rules)
        pair_set = {frozenset((p.a_id, p.b_id)) for p in pairs}
        for a, b in injected:
            total += 1
            if frozenset((a, b)) in pair_set:
                found += 1
        # false positives: pairs among NORMAL rules (no injection involved)
        injected_ids = {i for ab in injected for i in ab}
        for p in pairs:
            if p.a_id not in injected_ids and p.b_id not in injected_ids:
                fps += 1
    assert total > 0
    rate = found / total
    fp_rate = fps / max(1, total)
    assert rate > 0.8, f"detection rate {rate:.2f}"
    assert fp_rate < 0.1, f"false-positive rate {fp_rate:.2f}"


def test_same_condition_same_action_not_flagged():
    det = ContradictionDetector(sim_threshold=0.9)
    torch.manual_seed(1)
    e = torch.randn(D)
    rules = {
        0: Rule(id=0, condition_embedding=e.clone(), action=3),
        1: Rule(id=1, condition_embedding=e.clone(), action=3),
    }
    assert det.detect(rules) == []


def test_resolve_lowers_weaker_rule():
    det = ContradictionDetector(sim_threshold=0.9, resolve_factor=0.5)
    torch.manual_seed(2)
    e = torch.randn(D)
    strong = Rule(id=0, condition_embedding=e.clone(), action=1, confidence=0.8)
    weak = Rule(id=1, condition_embedding=e.clone(), action=2, confidence=0.8)
    for _ in range(10):
        strong.update(reward=1.0)   # success_rate 1.0
    for _ in range(10):
        weak.update(reward=-1.0)    # success_rate 0.0
    rules = {0: strong, 1: weak}
    pairs = det.detect(rules)
    assert len(pairs) == 1
    n = det.resolve(rules, pairs)
    assert n == 1
    assert weak.confidence < strong.confidence
    assert weak.confidence <= 0.8 * 0.5 + 1e-6


def test_detect_without_embeddings_is_honest_noop():
    det = ContradictionDetector()
    rules = {0: Rule(id=0, condition_embedding=None, action=1),
             1: Rule(id=1, condition_embedding=None, action=2)}
    assert det.detect(rules) == []


def test_step_summary_bounded():
    det = ContradictionDetector()
    torch.manual_seed(3)
    rules = {}
    for i in range(10):
        rules[i] = Rule(id=i, condition_embedding=torch.randn(D), action=i % 3)
    out = det.step(rules)
    assert set(out.keys()) == {"pairs", "resolved"}
    assert out["resolved"] <= out["pairs"]


def test_resolve_is_once_per_pair():
    """The periodic loop must not re-punish the same conflict every cycle."""
    det = ContradictionDetector(sim_threshold=0.9, resolve_factor=0.5)
    torch.manual_seed(4)
    e = torch.randn(D)
    ra = Rule(id=0, condition_embedding=e.clone(), action=1, confidence=0.8)
    rb = Rule(id=1, condition_embedding=e.clone(), action=2, confidence=0.8)
    rules = {0: ra, 1: rb}
    pairs = det.detect(rules)
    assert len(pairs) == 1
    assert det.resolve(rules, pairs) == 1
    conf_after_first = min(ra.confidence, rb.confidence)
    # second cycle: same pair detected again but must NOT be re-resolved
    pairs2 = det.detect(rules)
    assert len(pairs2) == 1
    assert det.resolve(rules, pairs2) == 0
    assert min(ra.confidence, rb.confidence) == conf_after_first


def test_detect_handles_mixed_dtype_embeddings():
    """2026-10-09 production bug: rule embeddings can be mixed dtype/device;
    the detector normalizes to float-CPU before stacking."""
    det = ContradictionDetector(sim_threshold=0.9)
    e = torch.randn(D)
    rules = {
        0: Rule(id=0, condition_embedding=e.clone().double(), action=1),
        1: Rule(id=1, condition_embedding=e.clone().float(), action=2),
    }
    pairs = det.detect(rules)
    assert len(pairs) == 1  # same direction, different action
