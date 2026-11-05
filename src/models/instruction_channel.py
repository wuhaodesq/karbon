"""Instruction channel — Stage 25 M-L1: instructions as a training signal.

A bounded, deterministic session that
  1. samples an instruction window ("go to the red box") from the objects
     PRESENT in the scene (resolvable by construction),
  2. encodes it as a fixed-size semantic vector (kind + colour + target
     offset) for the actor's zero-init projection (goal plumbing reuse),
  3. judges arrival ENV-SIDE (distance <= reach_radius, one-shot bonus per
     window, <=1 per window — B10 rule),
  4. runs a within-run control arm (uniform random instruction, NO reward)
     for the pre-registered instruction-matching comparison.

Parsing is the already-validated template lexicon (M3b); raw-text encoding
is deliberately M-L2 (disclosure in stage25_language_training_design.md).

语言进入行为回路: 指令窗口采样/语义向量/环境侧到达判定/同运行对照臂。
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

KINDS = ("box", "sphere", "cylinder", "capsule")
COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple")

TEMPLATES = (
    "go to the {color} {kind}",
    "approach the {color} {kind}",
    "find the {color} {kind}",
    "move to the {color} {kind}",
    "go to the {kind}",
    "find the {kind}",
)

VEC_DIM = 21


@dataclass
class Instruction:
    kind: str
    color: str
    text: str

    @property
    def key(self) -> str:
        return f"{self.color}:{self.kind}"


def color_name(rgb: tuple[float, ...]) -> str:
    """RGB (0..1) -> nearest colour name (hue bins; same as the M3b battery)."""
    import colorsys

    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360.0
    if s < 0.15:
        return "gray"
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 265:
        return "blue"
    return "purple"


def build_vector(instr: Instruction, dx: float, dy: float) -> list[float]:
    """Semantic vector (21 dims; kind 0-3, valid 4, colour 5-11, dx/dy 12-13)."""
    v = [0.0] * VEC_DIM
    if instr.kind in KINDS:
        v[KINDS.index(instr.kind)] = 1.0
    v[4] = 1.0
    if instr.color in COLORS:
        v[5 + COLORS.index(instr.color)] = 1.0
    v[12] = max(-1.0, min(1.0, float(dx) / 4.0))
    v[13] = max(-1.0, min(1.0, float(dy) / 4.0))
    return v


class InstructionChannel:
    """Bounded instruction-window session (chosen + control arms)."""

    def __init__(
        self,
        seed: int = 0,
        window_steps: int = 400,
        reach_radius: float = 0.5,
        bonus: float = 3.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._window = max(2, int(window_steps))
        self._radius = float(reach_radius)
        self._bonus = float(bonus)
        self._step = 0
        self._active = False
        self._start = 0
        self._chosen: Instruction | None = None
        self._control: Instruction | None = None
        self._target_id = -1
        self._control_id = -1
        self._awarded = False
        self._n_windows = 0
        self._arrived_chosen = 0
        self._arrived_control = 0
        # bounded arrival-time buffers (instrument v2: time-to-arrival is the
        # primary metric; the 0.8 m arrival rate hit a ceiling - both arms
        # ~95% by ambient proximity, an instrument flaw, B16 family).
        self._time_chosen: deque[int] = deque(maxlen=64)
        self._time_control: deque[int] = deque(maxlen=64)
        # M-L1b: PAIRED per-window outcomes (both arms start from the same
        # agent position at the same moment, so the paired comparison is
        # d0-controlled; raw time is dominated by start distance).
        self._pairs: deque[dict] = deque(maxlen=64)
        self._start_d_chosen: deque[float] = deque(maxlen=64)
        self._start_d_control: deque[float] = deque(maxlen=64)

    # -------------------------------------------------------------- sampling
    def sample(self, objects: list[dict]) -> tuple[Instruction, Instruction,
                                                   int, int] | None:
        """Sample (chosen, control, chosen_id, control_id) from present pairs.

        ``objects``: [{"id": int, "kind": str, "color": str}]. Returns None
        when no resolvable pair exists (caller skips the window).
        """
        pairs: dict[tuple[str, str], list[int]] = {}
        for o in objects:
            k, c = str(o.get("kind", "")), str(o.get("color", ""))
            if k in KINDS and c in COLORS:
                pairs.setdefault((k, c), []).append(int(o["id"]))
        if not pairs:
            return None
        keys = sorted(pairs)
        ck, cc = keys[self._rng.randrange(len(keys))]
        ch_id = pairs[(ck, cc)][self._rng.randrange(len(pairs[(ck, cc)]))]
        # control: uniform over ALL (kind, color) pairs, not just present ones
        # (a random instruction the agent was never asked to follow).
        rk = KINDS[self._rng.randrange(len(KINDS))]
        rc = COLORS[self._rng.randrange(len(COLORS))]
        cands = pairs.get((rk, rc), [])
        ctrl_id = (cands[self._rng.randrange(len(cands))] if cands
                   else ch_id)  # unresolvable control falls back to a real id
        tpl = TEMPLATES[self._rng.randrange(len(TEMPLATES))]
        if "{color}" not in tpl:
            chosen = Instruction(ck, cc, f"go to the {ck}")
            control = Instruction(rk, rc, f"go to the {rk}")
        else:
            chosen = Instruction(ck, cc, tpl.format(kind=ck, color=cc))
            control = Instruction(rk, rc, tpl.format(kind=rk, color=rc))
        return chosen, control, ch_id, ctrl_id

    # -------------------------------------------------------------- window
    def start(self, objects: list[dict], step: int) -> bool:
        """Open a window; returns False if nothing resolvable (no window)."""
        smp = self.sample(objects)
        if smp is None:
            return False
        self._chosen, self._control, self._target_id, self._control_id = smp
        self._active = True
        self._start = int(step)
        self._awarded = False
        self._paid = False
        self._ctrl_counted = False
        self._got_start = False
        self._pair_recorded = False
        self._chosen_t: int | None = None
        self._control_t: int | None = None
        self._n_windows += 1
        return True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def chosen(self) -> Instruction | None:
        return self._chosen

    @property
    def target_id(self) -> int:
        return self._target_id

    @property
    def control_id(self) -> int:
        return self._control_id

    @property
    def expired(self) -> bool:
        return self._active and (self._step - self._start) >= self._window

    # ---------------------------------------------------------------- step
    def step(self, step: int, chosen_dist: float, control_dist: float) -> dict:
        """Advance; returns the one-shot bonus (chosen arm) and flags."""
        self._step = int(step)
        if self.expired:
            self._active = False
        if not self._active:
            return {"bonus": 0.0, "chosen_arrived": False,
                    "control_arrived": False, "expired": True}
        if not getattr(self, "_got_start", False):
            self._start_d_chosen.append(float(chosen_dist))
            self._start_d_control.append(float(control_dist))
            self._got_start = True
        chosen_arrived = float(chosen_dist) <= self._radius
        control_arrived = float(control_dist) <= self._radius
        bonus = 0.0
        if chosen_arrived and self._chosen_t is None:
            self._chosen_t = int(step) - int(self._start)
            self._arrived_chosen += 1
            self._time_chosen.append(self._chosen_t)
            self._awarded = True
            # M-L1b: pay only if the CHOSEN target was reached BEFORE the
            # arbitrary control target (paired-first rule; the control may
            # not have arrived yet, which still counts as chosen-first).
            if not self._paid and not self._ctrl_counted:
                bonus = self._bonus
                self._paid = True
        if control_arrived and not self._ctrl_counted:
            self._ctrl_counted = True
            self._control_t = int(step) - int(self._start)
            self._arrived_control += 1
            self._time_control.append(self._control_t)
        # M-L1b throughput: record the pair as soon as BOTH arms have
        # arrived (previously only at window expiry -> ~0.5 pairs/min).
        if (self._chosen_t is not None and self._control_t is not None
                and not getattr(self, "_pair_recorded", False)):
            self._pairs.append({
                "t_ch": int(self._chosen_t),
                "t_ctl": int(self._control_t),
                "d0_ch": (float(self._start_d_chosen[-1])
                           if self._start_d_chosen else -1.0),
                "d0_ctl": (float(self._start_d_control[-1])
                            if self._start_d_control else -1.0),
            })
            self._pair_recorded = True
        return {"bonus": bonus, "chosen_arrived": chosen_arrived,
                "control_arrived": control_arrived, "expired": False}

    # ----------------------------------------------------------- telemetry
    def telemetry(self) -> dict[str, Any]:
        def _mean(q) -> float | None:
            return round(float(sum(q) / len(q)), 1) if len(q) else None
        return {
            "active": self._chosen.text if self._chosen else None,
            "control": self._control.text if self._control else None,
            "n_windows": self._n_windows,
            "arrived_chosen": self._arrived_chosen,
            "arrived_control": self._arrived_control,
            "t_chosen": _mean(self._time_chosen),
            "t_control": _mean(self._time_control),
            "d0_chosen": _mean(self._start_d_chosen),
            "d0_control": _mean(self._start_d_control),
            "bonus_paid": bool(getattr(self, "_paid", False)),
            "pairs": len(self._pairs),
            "win_rate": (round(sum(1 for p in self._pairs
                                     if p["t_ch"] < p["t_ctl"])
                                 / max(1, len(self._pairs)), 3)
                          if self._pairs else None),
        }

    # --------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step, "active": self._active, "start": self._start,
            "chosen": (self._chosen.__dict__ if self._chosen else None),
            "control": (self._control.__dict__ if self._control else None),
            "target_id": self._target_id, "control_id": self._control_id,
            "awarded": self._awarded, "paid": bool(getattr(self, "_paid", False)),
            "ctrl_counted": bool(getattr(self, "_ctrl_counted", False)),
            "n_windows": self._n_windows,
            "arrived_chosen": self._arrived_chosen,
            "arrived_control": self._arrived_control,
            "time_chosen": list(self._time_chosen),
            "time_control": list(self._time_control),
            "start_d_chosen": list(self._start_d_chosen),
            "start_d_control": list(self._start_d_control),
            "got_start": bool(getattr(self, "_got_start", False)),
            "pairs": list(self._pairs),
            "pair_recorded": bool(getattr(self, "_pair_recorded", False)),
            "chosen_t": self._chosen_t,
            "control_t": self._control_t,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = int(state.get("step", 0))
        self._active = bool(state.get("active", False))
        self._start = int(state.get("start", 0))
        cd = state.get("chosen")
        self._chosen = Instruction(**cd) if cd else None
        cd = state.get("control")
        self._control = Instruction(**cd) if cd else None
        self._target_id = int(state.get("target_id", -1))
        self._control_id = int(state.get("control_id", -1))
        self._awarded = bool(state.get("awarded", False))
        self._paid = bool(state.get("paid", False))
        self._ctrl_counted = bool(state.get("ctrl_counted", False))
        self._n_windows = int(state.get("n_windows", 0))
        self._arrived_chosen = int(state.get("arrived_chosen", 0))
        self._arrived_control = int(state.get("arrived_control", 0))
        self._time_chosen.clear()
        self._time_chosen.extend(int(x) for x in state.get("time_chosen", [])[-64:])
        self._time_control.clear()
        self._time_control.extend(int(x) for x in state.get("time_control", [])[-64:])
        self._start_d_chosen.clear()
        self._start_d_chosen.extend(float(x) for x in state.get("start_d_chosen", [])[-64:])
        self._start_d_control.clear()
        self._start_d_control.extend(float(x) for x in state.get("start_d_control", [])[-64:])
        self._got_start = bool(state.get("got_start", False))
        self._pairs.clear()
        for p in state.get("pairs", [])[-64:]:
            self._pairs.append(dict(p))
        self._pair_recorded = bool(state.get("pair_recorded", False))
        self._chosen_t = state.get("chosen_t")
        self._control_t = state.get("control_t")
        rs = state.get("rng_state")
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except (TypeError, ValueError):
                pass
