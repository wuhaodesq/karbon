"""Tests for the Stage 25 M-L1 InstructionChannel (pure logic)."""
from src.models.instruction_channel import (
    COLORS,
    KINDS,
    VEC_DIM,
    Instruction,
    InstructionChannel,
    build_vector,
    color_name,
)


def _objects():
    return [
        {"id": 0, "kind": "box", "color": "red"},
        {"id": 1, "kind": "box", "color": "blue"},
        {"id": 2, "kind": "sphere", "color": "green"},
        {"id": 3, "kind": "cylinder", "color": "red"},
    ]


def test_color_names():
    assert color_name((0.9, 0.1, 0.1)) == "red"
    assert color_name((0.1, 0.9, 0.1)) == "green"
    assert color_name((0.1, 0.1, 0.9)) == "blue"


def test_vector_layout():
    v = build_vector(Instruction("box", "red", "go to the red box"), 2.0, -4.0)
    assert len(v) == VEC_DIM
    assert v[0] == 1.0 and v[4] == 1.0  # kind box + valid
    assert v[5 + COLORS.index("red")] == 1.0
    assert abs(v[12] - 0.5) < 1e-9 and abs(v[13] + 1.0) < 1e-9


def test_sampling_resolvable_and_control():
    ch = InstructionChannel(seed=0)
    smp = ch.sample(_objects())
    assert smp is not None
    chosen, control, cid, ctrl_id = smp
    assert chosen.kind in KINDS and chosen.color in COLORS
    assert cid in (0, 1, 2, 3)
    assert ctrl_id in (0, 1, 2, 3)
    # M-L1c: the control must be a DIFFERENT present object (a same-target
    # fallback made every pair a tie and collapsed the win rate to ~0.08).
    assert ctrl_id != cid


def test_sampling_empty_returns_none():
    ch = InstructionChannel(seed=0)
    assert ch.sample([]) is None
    assert ch.start([], 0) is False


def test_window_arrival_bonus_once():
    ch = InstructionChannel(seed=1, window_steps=10, reach_radius=0.8,
                            bonus=3.0)
    assert ch.start(_objects(), 0) is True
    r1 = ch.step(1, chosen_dist=2.0, control_dist=2.0)
    assert r1["bonus"] == 0.0
    r2 = ch.step(2, chosen_dist=0.5, control_dist=0.5)
    assert r2["bonus"] == 3.0 and r2["chosen_arrived"]
    r3 = ch.step(3, chosen_dist=0.4, control_dist=0.4)
    assert r3["bonus"] == 0.0  # one-shot
    assert ch.telemetry()["arrived_chosen"] == 1
    assert ch.telemetry()["arrived_control"] == 1


def test_window_expiry_and_restart():
    ch = InstructionChannel(seed=2, window_steps=5)
    ch.start(_objects(), 0)
    for st in range(1, 5):
        ch.step(st, 9.9, 9.9)
    r = ch.step(5, 9.9, 9.9)
    assert r["expired"] is True
    r2 = ch.step(6, 0.1, 0.1)
    assert r2["bonus"] == 0.0 and r2["expired"] is True
    assert ch.start(_objects(), 7) is True


def test_state_roundtrip():
    ch = InstructionChannel(seed=3, window_steps=10)
    ch.start(_objects(), 0)
    ch.step(1, 0.5, 2.0)
    st = ch.state_dict()
    ch2 = InstructionChannel(seed=99, window_steps=10)
    ch2.load_state_dict(st)
    assert ch2.telemetry()["n_windows"] == ch.telemetry()["n_windows"]
    assert ch2.telemetry()["arrived_chosen"] == ch.telemetry()["arrived_chosen"]
    assert ch2.active == ch.active



def test_time_to_arrival_tracking():
    ch = InstructionChannel(seed=4, window_steps=100, reach_radius=0.5,
                            bonus=3.0)
    ch.start(_objects(), 0)
    ch.step(1, chosen_dist=2.0, control_dist=2.6)
    ch.step(2, chosen_dist=1.0, control_dist=2.4)
    ch.step(3, chosen_dist=0.4, control_dist=2.0)   # chosen arrives at t=3
    ch.step(4, chosen_dist=0.3, control_dist=0.4)   # control arrives at t=4
    t = ch.telemetry()
    assert t["arrived_chosen"] == 1 and t["arrived_control"] == 1
    assert t["t_chosen"] == 3.0 and t["t_control"] == 4.0
    assert abs(t["d0_chosen"] - 2.0) < 1e-9
    assert abs(t["d0_control"] - 2.6) < 1e-9


def test_tighter_radius_blocks_ambient_arrival():
    ch = InstructionChannel(seed=5, window_steps=50, reach_radius=0.5)
    ch.start(_objects(), 0)
    r = ch.step(1, chosen_dist=0.7, control_dist=0.7)  # inside old 0.8, not 0.5
    assert r["chosen_arrived"] is False and r["bonus"] == 0.0




def test_paired_first_bonus_and_win_rate():
    ch = InstructionChannel(seed=6, window_steps=20, reach_radius=0.5,
                            bonus=3.0)
    ch.start(_objects(), 0)
    ch.step(1, chosen_dist=0.4, control_dist=2.0)   # chosen first -> bonus
    ch.step(2, chosen_dist=0.4, control_dist=0.4)   # control arrives later
    t = ch.telemetry()
    assert t["arrived_chosen"] == 1 and t["arrived_control"] == 1
    # window ends -> pair recorded, then the win rate is available
    for st in range(3, 22):
        ch.step(st, 0.4, 0.4)
    t2 = ch.telemetry()
    assert t2["pairs"] == 1
    assert t2["win_rate"] == 1.0


def test_control_first_pays_nothing():
    ch = InstructionChannel(seed=7, window_steps=20, reach_radius=0.5,
                            bonus=3.0)
    ch.start(_objects(), 0)
    r1 = ch.step(1, chosen_dist=2.0, control_dist=0.4)  # control first
    assert r1["bonus"] == 0.0
    r2 = ch.step(2, chosen_dist=0.4, control_dist=0.4)  # chosen later
    assert r2["bonus"] == 0.0  # paired-first rule: control already arrived
    for s in range(3, 22):
        ch.step(s, 0.4, 0.4)
    assert ch.telemetry()["win_rate"] == 0.0




def test_shaping_caps_and_reset():
    ch = InstructionChannel(seed=8, window_steps=100, reach_radius=0.5,
                            bonus=3.0)
    ch.start(_objects(), 0)
    assert ch.shaping(3.0, gain=0.5, per_step_cap=0.02) == 0.0  # first read
    r = ch.shaping(2.0, gain=0.5, per_step_cap=0.02)  # progress 1.0
    assert abs(r - 0.02) < 1e-9  # per-step cap
    # total cap: simulate many steps
    total = r
    d = 2.0
    for _ in range(1000):
        d -= 0.5
        total += ch.shaping(d, gain=0.5, per_step_cap=0.02, total_cap=0.05)
    assert abs(total - 0.05) < 1e-9
    # a new window resets the accumulator
    ch.start(_objects(), 200)
    assert ch.shaping(1.0, gain=0.5, per_step_cap=0.02) == 0.0

