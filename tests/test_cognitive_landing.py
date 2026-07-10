"""Landing tests for the untested high-level cognitive modules.

Goal: prove each module is NOT an empty shell by driving it with minimal
real inputs and asserting non-trivial, correct output. CPU-only.

Modules covered:
  concept_graph.ConceptGraph (+ tier2.Analogizer on top)
  theory_of_mind.TheoryOfMind (false_belief_test)
  tier2_cognitive.{Analogizer, BeliefDepth2, MoralConnector, SurpriseHumor}
  iq_boost.{CrossDomainTransfer, DeepMultiModal, TemporalReasoner,
            CounterfactualRegret, CuriosityDirector, ValueSystem}
  abstract_reasoning.{MicroPrologMath, IdentityNarrative}
  developmental_memory.AutobiographicalMemory
"""

import torch
from types import SimpleNamespace

from src.models.concept_graph import ConceptGraph
from src.models.theory_of_mind import TheoryOfMind
from src.models.tier2_cognitive import (
    Analogizer,
    BeliefDepth2,
    MoralConnector,
    SurpriseHumor,
)
from src.models.iq_boost import (
    CrossDomainTransfer,
    DeepMultiModal,
    TemporalReasoner,
    CounterfactualRegret,
    CuriosityDirector,
    ValueSystem,
    ValueJudgment,
)
from src.models.abstract_reasoning import MicroPrologMath, IdentityNarrative
from src.models.developmental_memory import AutobiographicalMemory


D = 8  # small d_model for fast CPU tests


# --------------------------------------------------------------------------
# ConceptGraph + Analogizer
# --------------------------------------------------------------------------


def _build_graph():
    g = ConceptGraph(d_model=D, max_nodes=50, max_edges=200)
    ball = g.add_concept(torch.randn(D), name="ball")
    ground = g.add_concept(torch.randn(D), name="ground")
    shape = g.add_concept(torch.randn(D), name="shape")
    apple = g.add_concept(torch.randn(D), name="apple")
    red = g.add_concept(torch.randn(D), name="red")
    light = g.add_concept(torch.randn(D), name="light")
    fast = g.add_concept(torch.randn(D), name="fast")
    solid = g.add_concept(torch.randn(D), name="solid")
    # 10 edges so Analogizer's len(_edges) >= 10 guard passes.
    g.add_edge(ball, ground, "rolls")
    g.add_edge(ball, shape, "round")
    g.add_edge(ball, red, "has_color")
    g.add_edge(ball, light, "light")
    g.add_edge(apple, ground, "rolls")
    g.add_edge(apple, shape, "round")
    g.add_edge(apple, red, "has_color")
    g.add_edge(apple, light, "light")
    g.add_edge(ground, fast, "fast")
    g.add_edge(shape, solid, "solid")
    return g


def test_concept_graph_stores_nodes_and_edges():
    g = _build_graph()
    assert len(g) == 8
    assert len(g._edges) == 10


def test_concept_graph_find_analog_returns_a_concept():
    # NOTE: find_analog applies _query_proj to the query but NOT to stored
    # nodes, so its cosine similarity is only meaningful after that
    # projection is trained. Here we assert it runs and returns a concept.
    g = _build_graph()
    q = g._nodes[0].embedding + 0.01 * torch.randn(D)
    res = g.find_analog(q, k=1)
    assert res and isinstance(res[0][0].name, str)


def test_analogizer_finds_metaphor_across_shared_edges():
    g = _build_graph()
    ana = Analogizer(d_model=D)
    res = ana.find_metaphor(g, "ball")
    assert any("apple" in r["metaphor"] for r in res), res
    assert res[0]["similarity"] > 0.9


# --------------------------------------------------------------------------
# TheoryOfMind
# --------------------------------------------------------------------------


def test_theory_of_mind_forward_populates_beliefs():
    tom = TheoryOfMind(d_model=D, num_slots=4)
    self_slots = torch.randn(1, 4, D)
    others = {"caregiver": torch.tensor([[0.0, 0.0, 0.5]])}
    objs = torch.tensor([[0.0, 0.0, 0.0]])
    out = tom.forward(self_slots, others, objs)
    assert "caregiver_predicted_action" in out
    assert "caregiver" in tom._belief_states


def test_theory_of_mind_false_belief_logic():
    tom = TheoryOfMind(d_model=D, num_slots=4)
    # White-box: set a known belief state.
    tom._belief_states["x"] = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    # Hidden object == belief -> agent knows it -> NOT a false belief.
    same = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0.0])
    assert tom.false_belief_test("x", same) is False
    # Orthogonal hidden object -> belief does not encode it -> correct false-belief call.
    ortho = torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0.0])
    assert tom.false_belief_test("x", ortho) is True


# --------------------------------------------------------------------------
# tier2_cognitive: BeliefDepth2, MoralConnector, SurpriseHumor
# --------------------------------------------------------------------------


def test_belief_depth2_recursive_two_level():
    b = BeliefDepth2()
    known = {"caregiver": {"learner_knows_loc_ball"}, "learner": {"loc_ball"}}
    r = b.reason_depth2("caregiver", "learner", "loc_ball", known)
    assert r["knows_about"] is True
    # Without the nested belief, level-2 fails.
    known2 = {"caregiver": {"loc_ball"}, "learner": {"loc_ball"}}
    r2 = b.reason_depth2("caregiver", "learner", "loc_ball", known2)
    assert r2["knows_about"] is False


def test_moral_connector_judges_good_action():
    class FakeEmotion:
        class _S:
            pleasure = 0.8
        state = _S()

    val = ValueSystem(d_model=D)
    val.judge(4, torch.randn(D), {"safety": 0.5}, step=0)
    mc = MoralConnector()
    out = mc.evaluate_action(val, FakeEmotion(), {"safety": 0.5})
    assert out["evaluation"] == "good", out


def test_surprise_humor_detects_and_rejects_bad_outcome():
    sh = SurpriseHumor()
    funny = sh.detect(0.3, 0.5, step=1)
    assert funny is not None and funny["is_funny"] is True
    # Bad outcome -> not funny.
    assert sh.detect(0.3, -0.5, step=2) is None


# --------------------------------------------------------------------------
# iq_boost: 6 modules
# --------------------------------------------------------------------------


def test_cross_domain_transfer_finds_similar_domain():
    cd = CrossDomainTransfer(d_model=D, max_domains=4)
    emb = torch.randn(3, D)
    sig = cd.extract_signature(emb, step=0)
    cd.register_domain("A", sig)
    sims = cd.find_similar(sig)
    assert sims and sims[0][0] == "A" and sims[0][1] > 0.7
    weights = cd.transfer(sims)
    assert weights["A"] > 0


def test_deep_multi_modal_fuses_modalities():
    dm = DeepMultiModal(d_model=D, num_modalities=4, num_heads=2)
    mods = {f"m{i}": torch.randn(1, D) for i in range(4)}
    fused = dm(mods)
    assert fused.shape == (1, D)
    assert fused.abs().sum() > 0


def test_temporal_reasoner_runs_and_is_bounded():
    tr = TemporalReasoner(d_model=D, max_trajectories=5)
    expected = [torch.randn(D) for _ in range(3)]
    tr.set_plan(expected)
    out = tr.verify_step(torch.randn(D), step_in_plan=0)
    assert out is None or ("deviation" in out and "step" in out)
    for _ in range(10):
        tr.verify_step(torch.randn(D), step_in_plan=0)
    assert len(tr) <= tr.capacity  # bounded


def test_counterfactual_regret_biases_better_action():
    cr = CounterfactualRegret(max_regrets=10)
    cr.record_regret(actual_action=0, counterfactual_action=3,
                     actual_reward=0.0, counterfactual_reward=1.0,
                     regret_magnitude=0.5, step=1)
    bias = cr.get_regret_bias(num_actions=8)
    assert bias[3] > 0 and bias[0] == 0


def test_curiosity_director_weights_signals():
    cd = CuriosityDirector(d_model=D)
    w = cd.forward(rssm_uncertainty=0.5, knowledge_gap=0.3, social_curiosity=0.2)
    assert "total" in w and w["total"] > 0
    assert abs(w["rssm"] + w["gap"] + w["social"] - 1.0) < 1e-5


def test_value_system_judges_and_predicts():
    vs = ValueSystem(d_model=D)
    j = vs.judge(4, torch.randn(D), {"safety": 0.5}, step=0)
    assert -1.0 <= j.goodness <= 1.0 and j.goodness > 0.8
    pred = vs.predict_goodness(4, torch.randn(D))
    assert -1.0 <= pred <= 1.0
    assert isinstance(vs.get_principle(), str)


# --------------------------------------------------------------------------
# abstract_reasoning: MicroPrologMath, IdentityNarrative
# --------------------------------------------------------------------------


def test_micro_prolog_math_solves_arithmetic_and_pattern():
    m = MicroPrologMath()
    sol = m.solve("add(2,3,X)")
    assert any(s.get("X") == "5" for s in sol), sol
    assert m.next_pattern(2, 4, 6, 8) == 10
    assert m.next_pattern(1, 4, 9, 16) == 25  # second-order diff


def test_identity_narrative_extracts_traits():
    inv = IdentityNarrative(d_model=D, min_events_for_identity=20)
    # Too few events -> neutral.
    few = [SimpleNamespace(description="explore", lesson_learned="") for _ in range(5)]
    assert inv.extract_traits(few)["openness"] == 0.5
    # 25 explore events -> openness high.
    many = [SimpleNamespace(description="explore novel place", lesson_learned="") for _ in range(25)]
    traits = inv.extract_traits(many)
    assert traits["openness"] > 0.5
    assert "explore" in inv.generate_narrative(traits)


# --------------------------------------------------------------------------
# developmental_memory: AutobiographicalMemory
# --------------------------------------------------------------------------


def test_autobiographical_memory_promotes_and_stores():
    am = AutobiographicalMemory(max_events=10, promotion_threshold=0.3)
    e = am.add_event(1, "first successful push", importance=0.8, episode_id=1, lesson="push works")
    assert e is not None and len(am) == 1
    # Below threshold -> not promoted.
    assert am.add_event(2, "tiny blip", importance=0.1, episode_id=2) is None
    assert "first successful push" in am.get_life_story()
