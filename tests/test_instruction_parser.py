"""Unit tests for the M3b instruction parser (pure logic, no MuJoCo)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "instruction_battery",
    str(Path(__file__).resolve().parents[1] / "scripts" / "eval"
        / "instruction_battery.py"))
ib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ib)


def _facts():
    objs = [
        {"id": 0, "kind": "box", "color": "red", "xy": (5.0, 0.0)},
        {"id": 1, "kind": "box", "color": "blue", "xy": (1.0, 1.0)},
        {"id": 2, "kind": "box", "color": "blue", "xy": (-2.0, 0.0)},
        {"id": 3, "kind": "sphere", "color": "green", "xy": (0.0, 3.0)},
        {"id": 4, "kind": "cylinder", "color": "red", "xy": (0.0, -3.0)},
    ]
    return {"objects": objs, "by_id": {o["id"]: o for o in objs},
            "occluded": [1], "last_known": {1: (1.5, 2.5)}}


def test_color_names():
    assert ib.color_name((0.9, 0.1, 0.1)) == "red"
    assert ib.color_name((0.1, 0.9, 0.1)) == "green"
    assert ib.color_name((0.1, 0.1, 0.9)) == "blue"
    assert ib.color_name((0.9, 0.9, 0.1)) == "yellow"


def test_parse_kind_only():
    r = ib.parse_instruction("go to the box", _facts(), (0.0, 0.0))
    assert r["ids"] == [0, 1, 2] and r["mode"] == "live"


def test_parse_kind_color():
    r = ib.parse_instruction("go to the red box", _facts(), (0.0, 0.0))
    assert r["ids"] == [0]


def test_parse_memory_anchor():
    r = ib.parse_instruction("go to the object that disappeared",
                             _facts(), (0.0, 0.0))
    assert r["ids"] == [1] and r["mode"] == "memory"
    assert r["anchor"] == (1.5, 2.5)


def test_disappeared_is_not_the_color_red():
    # substring matching made "disappeaRED" parse as the color red (B17)
    f = _facts()
    assert f["by_id"][1]["color"] == "blue"
    r = ib.parse_instruction("go to the object that disappeared", f, (0.0, 0.0))
    assert r["ids"] == [1]


def test_parse_nearest():
    r = ib.parse_instruction("go to the nearest box", _facts(), (0.0, 0.0))
    assert r["ids"] == [1]


def test_make_trials_binding():
    import random
    trials = ib.make_trials(_facts(), (0.0, 0.0), random.Random(0))
    assert len(trials) == len(ib.TEMPLATES)
    mem = [t for t in trials if t.get("mode") == "memory"]
    assert mem and all(t["acceptable"] == [1] for t in mem)
    live = [t for t in trials if t.get("mode") == "live"]
    assert live and all("text" in t for t in live)


def test_make_trials_no_occlusion_skips():
    import random
    f = _facts()
    f["occluded"] = []
    f["last_known"] = {}
    trials = ib.make_trials(f, (0.0, 0.0), random.Random(0))
    mem = [t for t in trials if t["template"] in ib.MEM_TEMPLATES]
    assert mem and all("skip" in t for t in mem)
