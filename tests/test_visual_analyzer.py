"""Tests for :mod:`src.models.visual_analyzer` (VisualAnalyzer)."""

import torch

from src.models.visual_analyzer import VisualAnalyzer


class _FakeConceptGraph:
    """Minimal stand-in for ConceptGraph add_concept/add_edge."""

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_concept(self, embedding=None, name="", source="", step=0):
        self.nodes.append((name, source))
        return len(self.nodes) - 1

    def add_edge(self, source_id, target_id, relation, confidence=0.5,
                 source_module="", step=0):
        self.edges.append((source_id, target_id, relation))


def test_forward_shapes():
    m = VisualAnalyzer()
    slots = torch.randn(2, 7, 128)
    out = m(slots)
    assert out["color"].shape == (2, 7, 8)
    assert out["shape"].shape == (2, 7, 4)
    assert out["size"].shape == (2, 7, 3)
    assert out["texture"].shape == (2, 7, 3)
    assert out["motion"].shape == (2, 7, 3)


def test_describe_slot_no_nameerror():
    m = VisualAnalyzer()
    slots = torch.randn(1, 7, 128)
    desc = m.describe_slot(slots, 0)
    assert isinstance(desc, str)
    assert desc.startswith("a ")
    assert "object," in desc


def test_describe_scene_empty():
    m = VisualAnalyzer()
    slots = torch.zeros(1, 7, 128)  # all below the 0.1 norm threshold
    assert m.describe_scene(slots) == "I don't see any objects."


def test_describe_scene_lists_active_slots():
    m = VisualAnalyzer()
    slots = torch.randn(1, 7, 128)
    out = m(slots)
    desc = m.describe_scene(slots)
    assert desc.startswith("I see ")
    # 7 active slots -> 7 "object," phrases
    assert desc.count("object,") == 7


def test_feed_to_graph_skips_empty_slots():
    m = VisualAnalyzer()
    slots = torch.randn(1, 7, 128)
    out = m(slots)
    slots2 = slots.clone()
    slots2[0, 5] = 0.0
    slots2[0, 6] = 0.0
    cg = _FakeConceptGraph()
    added = m.feed_to_graph(out, slots2, cg, step=10)
    assert added == 5
    assert len(cg.nodes) == 10  # one object node + one attr node each
    assert len(cg.edges) == 5


def test_feed_to_graph_returns_zero_when_graph_none():
    m = VisualAnalyzer()
    slots = torch.randn(1, 7, 128)
    out = m(slots)
    assert m.feed_to_graph(out, slots, None, step=0) == 0


def test_motion_still_when_identical():
    m = VisualAnalyzer()
    slots = torch.randn(1, 3, 128)
    m(slots)
    out = m(slots)  # identical frame -> diff 0 -> still
    assert bool((out["motion"][0].argmax(dim=-1) == 0).all())


def test_motion_fast_on_large_change():
    m = VisualAnalyzer()
    s1 = torch.randn(1, 3, 128)
    m(s1)
    s2 = s1 + 5.0  # large change between frames
    out = m(s2)
    assert bool((out["motion"][0].argmax(dim=-1) == 2).all())
