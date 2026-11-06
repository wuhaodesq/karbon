"""Tests for the Stage 26 R1 assertion verifier (pure logic)."""
from src.models.assertion_verifier import (
    SceneFacts,
    SceneObject,
    verify,
)


def _facts(occluded=()):
    objs = [
        SceneObject(0, "box", "red"),
        SceneObject(1, "box", "blue"),
        SceneObject(2, "sphere", "green"),
        SceneObject(3, "cylinder", "red"),
        SceneObject(4, "capsule", "red"),
    ]
    return SceneFacts(objects=objs, occluded_ids=frozenset(occluded))


def test_count_true_and_false():
    f = _facts()
    assert verify("there are five objects", f)["status"] == "true"
    assert verify("there are two boxes", f)["status"] == "true"
    assert verify("there are four boxes", f)["status"] == "false"
    assert verify("5 objects", f)["status"] == "true"
    assert verify("there are seven objects", f)["status"] == "false"


def test_existence():
    f = _facts()
    assert verify("the red box", f)["status"] == "true"
    assert verify("a green sphere", f)["status"] == "true"
    assert verify("the blue sphere", f)["status"] == "false"
    assert verify("the purple cylinder", f)["status"] == "false"


def test_occlusion_claims():
    f = _facts(occluded=[1])
    assert verify("the blue box is hidden", f)["status"] == "true"
    assert verify("the red box is hidden", f)["status"] == "false"
    assert verify("something is hidden behind the wall", f)["status"] == "true"
    f2 = _facts(occluded=[])
    assert verify("something is hidden", f2)["status"] == "false"


def test_unverifiable_cases():
    f = _facts()
    assert verify("", f)["status"] == "unverifiable"
    assert verify("the weather is nice", f)["status"] == "unverifiable"
    assert verify("the box is left of the sphere", f)["status"] == "unverifiable"


def test_count_of_kind_with_colour():
    f = _facts()
    assert verify("there are two red objects", f)["status"] == "false"
    assert verify("there are three red objects", f)["status"] == "true"


def test_facts_from_env_contract():
    class _O:
        def __init__(self, k, c):
            self.kind = k
            self.color = c

    class _E:
        objects = [_O("box", (0.9, 0.1, 0.1)), _O("sphere", (0.1, 0.1, 0.9))]
        _active_occlusions_3d = {"occ_1": {"truly_occluded": True}}

    f = SceneFacts.from_env(_E())
    assert len(f.objects) == 2
    assert f.objects[0].color == "red" and f.objects[1].color == "blue"
    assert f.occluded_ids == frozenset({1})
