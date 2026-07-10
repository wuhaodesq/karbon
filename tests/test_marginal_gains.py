"""Landing tests for the two marginal-gain modules:
CompositionalTester, LearningProgressTracker.

(Knowledge-gap detection lives in src.intrinsic.knowledge_gap and is tested
separately; it is intentionally NOT duplicated in marginal_gains.)
"""

import torch

from src.models.concept_graph import ConceptGraph
from src.models.marginal_gains import (
    CompositionalTester,
    LearningProgressTracker,
)


# ------------------------------------------------------------- compositional


def _graph_with_composable_nodes():
    g = ConceptGraph(d_model=8)
    ids = []
    for i in range(4):
        emb = torch.zeros(8)
        emb[i] = 1.0
        ids.append(g.add_concept(emb, name=f"c{i}", source=f"mod{i}", step=i))
    # Give node0 and node1 different (non-overlapping) edge relations so that
    # their combined relation set is strictly larger than either alone.
    g.add_edge(ids[0], ids[2], relation="has_color", confidence=0.9)
    g.add_edge(ids[1], ids[3], relation="has_shape", confidence=0.9)
    return g


def test_compositional_not_enough_nodes():
    ct = CompositionalTester(min_known_nodes=4)
    g = ConceptGraph(d_model=8)
    res = ct.test(g)
    assert res["passed"] is False
    assert res["nodes"] == 0


def test_compositional_detects_novel_combination():
    ct = CompositionalTester(min_known_nodes=4)
    g = _graph_with_composable_nodes()
    res = ct.test(g)
    assert res["passed"] is True
    assert res["novel_combination"] >= 1
    assert ct.score > 0.0
    assert "passed" in ct.summary()


def test_compositional_none_graph_is_safe():
    ct = CompositionalTester()
    res = ct.test(None)
    assert res["passed"] is False


# ------------------------------------------------------------- lp tracker


def test_lp_tracker_first_update_no_signal():
    lp = LearningProgressTracker()
    res = lp.update(1.0, step=0)
    assert res["is_flat"] is False
    assert res["curiosity_boost"] == 0.0


def test_lp_tracker_detects_plateau_and_boosts():
    lp = LearningProgressTracker(flat_threshold=0.01, boost_amount=0.2)
    boosted = False
    for i in range(20):
        res = lp.update(5.0, step=i)  # perfectly flat return
        if res["curiosity_boost"] > 0:
            boosted = True
    assert boosted is True
    assert lp.is_stuck is True


def test_lp_tracker_rising_return_not_flat():
    lp = LearningProgressTracker(flat_threshold=0.01)
    res = {}
    for i in range(20):
        res = lp.update(float(i), step=i)  # steadily rising
    assert res["is_flat"] is False
    assert res["lp"] > 0.0


def test_lp_tracker_window_bounded():
    lp = LearningProgressTracker(window_size=10)
    for i in range(50):
        lp.update(float(i), step=i)
    assert len(lp._return_history) <= 10
