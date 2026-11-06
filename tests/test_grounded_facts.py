"""Tests for Stage 26 R2: assertion generator + bounded grounded fact store."""
import random

from src.models.assertion_verifier import SceneFacts, SceneObject, verify
from src.models.grounded_facts import (
    AssertionGenerator,
    GroundedFactStore,
    ingest_scene,
)


def _facts(occluded=()):
    objs = [
        SceneObject(0, "box", "red"),
        SceneObject(1, "box", "blue"),
        SceneObject(2, "sphere", "green"),
        SceneObject(3, "cylinder", "red"),
    ]
    return SceneFacts(objects=objs, occluded_ids=frozenset(occluded))


def test_generator_deterministic_and_verifiable():
    g1 = AssertionGenerator(seed=3)
    g2 = AssertionGenerator(seed=3)
    a = g1.generate(_facts(), n=20)
    b = g2.generate(_facts(), n=20)
    assert a == b
    # every generated assertion must get a defined verdict
    for text in a:
        assert verify(text, _facts())["status"] in ("true", "false",
                                                    "unverifiable")


def test_store_dedupe_and_merge():
    st = GroundedFactStore(capacity=100)
    st.ingest("the red box", "true", step=1)
    st.ingest("the red box", "true", step=2)
    st.ingest("the red box", "true", step=3)
    assert len(st) == 1
    t = st.telemetry()
    assert t["accepted_total"] == 3
    assert st.state_dict()["facts"]["exists:box:red"][0] == 3


def test_store_rejects_false_and_unverifiable():
    st = GroundedFactStore(capacity=100)
    st.ingest("the purple sphere", "false", step=1)
    st.ingest("the weather is nice", "unverifiable", step=1)
    t = st.telemetry()
    assert t["rejected_total"] == 1
    assert t["unverifiable_total"] == 1
    assert len(st) == 0 and t["recent_rejected"]


def test_store_bounded_and_prunes():
    st = GroundedFactStore(capacity=16)
    for i in range(200):
        st.ingest(f"the box number{i}", "true", step=i)
    assert len(st) <= 16


def test_store_state_roundtrip():
    st = GroundedFactStore(capacity=100)
    st.ingest("the red box", "true", step=1)
    st.ingest("the cylinder is hidden", "true", step=2)
    st2 = GroundedFactStore(capacity=100)
    st2.load_state_dict(st.state_dict())
    assert len(st2) == len(st)
    t1, t2 = st.telemetry(), st2.telemetry()
    assert t1["accepted_total"] == t2["accepted_total"]


def test_ingest_scene_counts_match_verdicts():
    st = GroundedFactStore(capacity=100)
    gen = AssertionGenerator(seed=5)
    stats = ingest_scene(st, _facts(occluded=[1]), gen, step=10, n=40)
    assert sum(stats.values()) == 40
    t = st.telemetry()
    assert t["accepted_total"] + t["rejected_total"] + t["unverifiable_total"] == 40
