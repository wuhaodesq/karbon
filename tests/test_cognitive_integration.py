"""Integration evals — pin cognitive abilities to measurable numbers.

Unlike test_cognitive_landing.py (does each module run?), these assert
END-TO-END behavior across module boundaries and report pass metrics:
  - Schema extraction from a realistic multi-episode stream
  - Causal2Prolog chained (transitive) reasoning
  - Theory-of-Mind false-belief logic pass rate
  - Metaphor (Analogy) recovery from a ConceptGraph

All CPU-only, deterministic (seeded).
"""

import torch

from src.models.rule_induction import RuleInductionEngine
from src.models.neuro_symbolic_bridge import SchemaDetector, Causal2Prolog
from src.models.causal_discovery import CausalDiscovery
from src.models.abstract_reasoning import MicroPrologMath
from src.models.theory_of_mind import TheoryOfMind
from src.models.concept_graph import ConceptGraph
from src.models.tier2_cognitive import Analogizer

torch.manual_seed(0)
D = 8


def test_schema_extraction_from_multi_episode_stream():
    """Feed a realistic stream of SEPARATE episodes; expect schemas to emerge."""
    r = RuleInductionEngine(num_slots=4, max_rules=128, induction_min_positive=3)
    sigs = [
        {"exists(s0)": True, "near(s0,s1)": True},
        {"exists(s1)": True, "large(s1)": True},
        {"exists(s2)": True, "moving(s2)": True},
    ]
    # Repeat each signature 8x across separate episodes (margin vs decay).
    for sig in sigs:
        for _ in range(8):
            r.record_episode([sig], [2], outcome=1.0)
            r.induce_rules()

    n_rules = len(r)
    sd = SchemaDetector(min_rule_count=3)
    schemas = sd.extract(r, step=0)

    print(f"\n[schema] rules={n_rules} schemas={len(schemas)} "
          f"best={sd.get_best_schema()}")
    assert n_rules >= 3, f"expected >=3 rules, got {n_rules}"
    assert len(schemas) >= 1, "no schema extracted from rule stream"
    assert "Action" in schemas[0]["description"]


def test_causal2prolog_chained_reasoning():
    """Causal edges -> Prolog facts -> transitive (2-hop) query resolves."""
    # Build a causal chain: push->move, move->hit.
    fake = type("F", (), {"_graph": type("G", (), {
        "edges": {
            ("push", "move"): type("E", (), {"strength": 0.9})(),
            ("move", "hit"): type("E", (), {"strength": 0.9})(),
        }
    })()})()

    mm = MicroPrologMath()
    n = Causal2Prolog(min_strength=0.3).feed_to_math(mm, fake)
    # Flat facts are queryable.
    assert any("causes(push, move)" in s for s in [str(x) for x in mm._facts.get("causes", [])]) or n >= 2

    # Add a transitive rule so the engine can CHAIN two causes.
    mm._add_axiom("causes_chain(X, Z) :- causes(X, Y), causes(Y, Z)")
    sols = mm.solve("causes_chain(push, Z)")
    chained = [s for s in sols if s.get("Z") == "hit"]

    print(f"\n[causal] prolog_facts={n} chain_push->hit={bool(chained)}")
    assert n >= 2
    assert chained, "transitive causal chain did not resolve"


def test_theory_of_mind_false_belief_pass_rate():
    """Run controlled false-belief scenarios; logic must be correct 100%."""
    tom = TheoryOfMind(d_model=D, num_slots=4)
    obj = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])

    def scenario(agent_knows_object: bool) -> bool:
        if agent_knows_object:
            tom._belief_states["caregiver"] = obj.clone().unsqueeze(0)
        else:
            tom._belief_states["caregiver"] = torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0]).unsqueeze(0)
        # Agent "knows object" => NOT a false belief (returns False).
        # Agent "doesn't know" => correct false-belief call (returns True).
        return tom.false_belief_test("caregiver", obj)

    n = 20
    correct = 0
    for i in range(n):
        knows = (i % 2 == 0)
        expected = (not knows)  # False-belief test True iff agent doesn't know
        if scenario(knows) == expected:
            correct += 1

    rate = correct / n
    print(f"\n[tom] false_belief logic pass rate = {rate*100:.0f}% ({correct}/{n})")
    assert rate == 1.0, f"false-belief logic wrong: {rate}"


def test_metaphor_recovery_from_concept_graph():
    """ConceptGraph with shared structure -> Analogizer recovers the metaphor."""
    g = ConceptGraph(d_model=D, max_nodes=50, max_edges=200, merge_similarity=0.999)
    ball = g.add_concept(torch.randn(D), name="ball")
    ground = g.add_concept(torch.randn(D), name="ground")
    shape = g.add_concept(torch.randn(D), name="shape")
    apple = g.add_concept(torch.randn(D), name="apple")
    red = g.add_concept(torch.randn(D), name="red")
    light = g.add_concept(torch.randn(D), name="light")
    fast = g.add_concept(torch.randn(D), name="fast")
    solid = g.add_concept(torch.randn(D), name="solid")
    for a, b, rel in [
        (ball, ground, "rolls"), (ball, shape, "round"), (ball, red, "has_color"), (ball, light, "light"),
        (apple, ground, "rolls"), (apple, shape, "round"), (apple, red, "has_color"), (apple, light, "light"),
        (ground, fast, "fast"), (shape, solid, "solid"),
    ]:
        g.add_edge(a, b, rel)

    ana = Analogizer(d_model=D)
    res = ana.find_metaphor(g, "ball")
    found_apple = any("apple" in r["metaphor"] for r in res)

    print(f"\n[metaphor] edges={len(g._edges)} candidates={len(res)} "
          f"recovered_apple={found_apple} top_sim={res[0]['similarity']:.2f}" if res else "")
    assert found_apple, f"metaphor not recovered; got {res}"
    assert res[0]["similarity"] >= 0.9
