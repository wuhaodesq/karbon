"""Landing/integration tests for the three System-2 modules:
ProgramSynthesizer, ActiveExperimenter, TemporalAbstractor.
"""

from types import SimpleNamespace

from src.models.program_synthesis import (
    ProgramSynthesizer,
    ActiveExperimenter,
    TemporalAbstractor,
)
from src.models.rule_induction import RuleInductionEngine


# --------------------------------------------------------------------------
# ProgramSynthesizer
# --------------------------------------------------------------------------


def test_program_synthesizer_induces_candidates():
    ps = ProgramSynthesizer(min_examples=2)
    examples = [
        {"input": {"exists(s0)": True, "near(s0,s1)": True}, "output": 1},
        {"input": {"exists(s0)": True, "large(s0)": True}, "output": 1},
    ]
    cands = ps.synthesize(examples)
    # Strategy1 (common preds) + Strategy2 (changed preds) -> 2 candidates.
    assert len(cands) == 2
    assert any("exists(s0)" in c["if_predicates"] for c in cands)
    print(f"\n[program] candidates={len(cands)} "
          f"{[c['if_predicates'] for c in cands]}")


def test_program_synthesizer_feeds_rule_engine():
    ps = ProgramSynthesizer(min_examples=2)
    examples = [
        {"input": {"exists(s0)": True, "near(s0,s1)": True}, "output": 1},
        {"input": {"exists(s0)": True, "large(s0)": True}, "output": 1},
    ]
    ps.synthesize(examples)
    re = RuleInductionEngine()
    added = ps.feedback_to_rules(re)
    print(f"\n[program] rules fed={added} engine_rules={len(re)}")
    assert added >= 1
    assert len(re) >= 1


# --------------------------------------------------------------------------
# ActiveExperimenter
# --------------------------------------------------------------------------


def test_active_experimenter_proposes_and_records():
    ae = ActiveExperimenter(test_every_steps=2000)
    fake_causal = SimpleNamespace(_graph=SimpleNamespace(edges={
        ("action_3", "world_state"): SimpleNamespace(source="action_3", target="world_state", strength=0.4),
    }))
    h = ae.propose_experiment(fake_causal, None, rssm_uncertainty=0.3)
    assert h is not None
    assert h["test_action"] == 3
    ae.record_result(h, actual_outcome=0.8, step=2000)
    assert len(ae._test_results) == 1
    assert ae.should_test(4000) is True
    print(f"\n[active] hypothesis='{h['hypothesis']}' results={len(ae._test_results)}")


# --------------------------------------------------------------------------
# TemporalAbstractor
# --------------------------------------------------------------------------


def test_temporal_abstractor_extracts_repeated_pattern():
    ta = TemporalAbstractor(min_occurrences=3, max_sequence_length=4)
    for _ in range(5):
        ta.record_step(["push"], 0.1)
        ta.record_step(["hit"], 0.2)
        ta.record_step(["fall"], 0.3)
    patterns = ta.extract_episode_patterns()
    assert len(patterns) >= 1
    assert any(p["count"] >= 3 for p in patterns)
    assert any("push" in p["sig"] for p in patterns)
    print(f"\n[temporal] patterns={len(patterns)} "
          f"{[(p['sig'], p['count']) for p in patterns[:3]]}")


def test_temporal_abstractor_second_extract_does_not_crash():
    """Regression: dedup check must survive a second extraction."""
    ta = TemporalAbstractor(min_occurrences=3, max_sequence_length=4)
    for _ in range(5):
        ta.record_step(["push"], 0.1)
        ta.record_step(["hit"], 0.2)
        ta.record_step(["fall"], 0.3)
    ta.extract_episode_patterns()
    # Second episode with the same pattern.
    for _ in range(5):
        ta.record_step(["push"], 0.1)
        ta.record_step(["hit"], 0.2)
        ta.record_step(["fall"], 0.3)
    patterns = ta.extract_episode_patterns()  # must not raise
    # Second call reports no NEW patterns (already emitted), but the
    # persistent count in _patterns must have grown across both episodes.
    assert not patterns  # nothing new on 2nd call
    total = sum(info["count"] for info in ta._patterns.values())
    assert total >= 6  # 5 cycles + 5 cycles
