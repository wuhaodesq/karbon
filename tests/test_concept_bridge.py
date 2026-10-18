"""Tests for Stage 22 M4 — ConceptBridge (concept stack over live experience)."""

from __future__ import annotations

from src.models.concept_bridge import ConceptBridge, attribute_embedding


class _Obj:
    """ObjectDef-like stub (name/kind/color/size protocol only)."""

    def __init__(self, name, kind, color_r=0.5, sx=0.06):
        self.name = name
        self.kind = kind
        self.color = (color_r, 0.4, 0.6, 1.0)
        self.size = (sx, sx, sx)


def _world(bridge: ConceptBridge, n_sphere=6, n_box=6):
    for i in range(n_sphere):
        bridge.observe_object(f"sp{i}", "sphere", i % 8, 1)
    for i in range(n_box):
        bridge.observe_object(f"bx{i}", "box", i % 8, 2)


def test_categories_form_from_instances():
    b = ConceptBridge()
    _world(b)
    s = b.summary()
    assert s["instances"] == 12
    assert s["categories"] >= 2  # spheres vs boxes separate
    assert s["facts"] > 0


def test_movable_generalizes_to_category_and_novel_instance():
    b = ConceptBridge(min_cat_movable=2)
    _world(b)
    # physics experience: two spheres actually moved by contact
    b.record_movable("sp0")
    b.record_movable("sp1")
    # a novel (still unseen individually) sphere instance inherits movable
    b.observe_object("sp_new", "sphere", 3, 1)
    assert b.novel_movable("sp_new") is True
    # a box does NOT (its category never moved)
    assert b.novel_movable("bx0") is False


def test_single_movable_does_not_overgeneralize():
    b = ConceptBridge(min_cat_movable=2)
    _world(b)
    b.record_movable("sp0")  # only ONE member moved
    b.novel_movable("sp1")
    # sp1 itself was observed but never moved; category prop not earned yet
    assert b.novel_movable("sp1") is False


def test_bounded_instances():
    b = ConceptBridge(max_instances=4)
    for i in range(10):
        b.observe_object(f"o{i}", "sphere", 0, 0)
    assert len(b._seen) <= 4  # Axiom 1


def test_state_dict_roundtrip_preserves_deduction():
    b = ConceptBridge(min_cat_movable=2)
    _world(b)
    b.record_movable("sp0")
    b.record_movable("sp1")
    sd = b.state_dict()
    b2 = ConceptBridge(min_cat_movable=2)
    b2.load_state_dict(sd)
    b2.observe_object("sp_new", "sphere", 3, 1)  # seen but never moved
    assert b2.novel_movable("sp_new") is True
    # and an unobserved-but-new box is not movable
    b2.observe_object("bx_new", "box", 5, 2)
    assert b2.novel_movable("bx_new") is False


def test_attribute_embedding_frame():
    e = attribute_embedding("sphere", 3, 1)
    assert e.shape == (16,)
    assert float(e.sum()) == 3.0  # kind + color + size one-hots


def test_feed_episode_end_to_end():
    """The exact train.py call path: objects + grasp_carry_events."""
    b = ConceptBridge(min_cat_movable=2)
    objs = [_Obj(f"sp{i}", "sphere", color_r=i / 8.0) for i in range(4)]
    objs += [_Obj(f"bx{i}", "box", color_r=i / 8.0) for i in range(4)]
    info = {"grasp_carry_events": [
        {"obj_id": 0, "carry_distance": 0.31},
        {"obj_id": 1, "carry_distance": 0.02},   # tiny nudge: NOT movable
        {"obj_id": 2, "carry_distance": 0.50},
    ]}
    b.feed_episode(objs, info)
    assert len(b) == 8
    s = b.summary()
    assert s["movable_facts"] == 2          # sp0 + sp2 only
    assert s["categories"] >= 2
    # novel sphere instance inherits the category-level property
    b.observe_object("sp_new", "sphere", 3, 1)
    assert b.novel_movable("sp_new") is True
    b.observe_object("bx_new", "box", 1, 1)
    assert b.novel_movable("bx_new") is False


def test_feed_episode_idempotent_and_capped():
    b = ConceptBridge(max_instances=4)
    objs = [_Obj(f"o{i}", "sphere") for i in range(10)]
    info = {"grasp_carry_events": [{"obj_id": 0, "carry_distance": 0.9}]}
    b.feed_episode(objs, info)
    b.feed_episode(objs, info)  # same episode twice: no double counting
    assert len(b) <= 4
    assert b.summary()["instances"] <= 4


def test_feed_episode_tolerates_empty_and_missing_info():
    b = ConceptBridge()
    b.feed_episode([], {})
    b.feed_episode(None, {"grasp_carry_events": [{"obj_id": 99, "carry_distance": 1.0}]})
    assert len(b) == 0
