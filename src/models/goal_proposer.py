"""Goal proposer — Stage 24: drives -> self-proposed goals -> feedback.

The agent PROPOSES its own goals from internal signals (homeostatic drive
proxies, learning-progress gaps, pending hypotheses) instead of only picking
externally-defined tasks (M2/B15). Bounded and deterministic:

  propose()      -> up to ``proposals`` candidate goals (utility-clamped)
  arbitrate()    -> one active goal; anti-monopoly penalty
  record_outcome -> EMA utility table (bounded capacity, pruned by count)
  GoalTracker    -> env-side achievement evaluators (chosen + random control)
  GoalSession    -> trainer-side driver: proxies -> proposal -> tracking -> bonus
  state_dict()/load_state_dict() for checkpointing

Goal kinds map to existing machinery at wiring time:
  explore  -> novelty zones (room/cell visits)
  practice -> current-task success events (competence drive / LP slope)
  approach -> sibling / open-space (social / safety drives)
  test     -> hypothesis-probe rising edges
  idle     -> low-motion windows (rest drive)

驱动 -> 自主目标提案 (有界/确定性/可存档)。达成判定在环境侧 (房间/成功事件/
距离/探针/运动窗口), 随机臂作同运行对照。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

KINDS = ("explore", "practice", "approach", "test", "idle")

_DRIVE_TO_KIND = {
    "curiosity": ("explore", "novel_zone"),
    "competence": ("practice", "current_task"),
    "social": ("approach", "sibling"),
    "safety": ("approach", "open_space"),
    "rest": ("idle", "stillness"),
}


@dataclass
class Goal:
    """A proposed goal (bounded fields only)."""

    kind: str
    target: str
    utility: float
    source: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}"


class GoalProposer:
    """Bounded self-proposal + arbitration + outcome feedback.

    Args:
        seed: RNG seed (deterministic proposal/arbitration).
        proposals: max candidate goals per proposal round (<= 8).
        capacity: max tracked utility keys (<= 64).
        history: arbitration history length (deque, <= 64).
        monopoly_k: consecutive identical choices that trigger suppression.
        ema: feedback EMA rate for utility updates.
        explore_eps: epsilon exploration in arbitration.
    """

    MAX_PROPOSALS = 8
    MAX_CAPACITY = 64
    MAX_HISTORY = 64

    def __init__(
        self,
        seed: int = 0,
        proposals: int = 8,
        capacity: int = 64,
        history: int = 32,
        monopoly_k: int = 12,
        ema: float = 0.1,
        explore_eps: float = 0.1,
    ) -> None:
        self._rng = random.Random(seed)
        self._max_proposals = max(1, min(int(proposals), self.MAX_PROPOSALS))
        self._capacity = max(4, min(int(capacity), self.MAX_CAPACITY))
        self._history: list[str] = []
        self._history_cap = max(4, min(int(history), self.MAX_HISTORY))
        self._monopoly_k = max(2, int(monopoly_k))
        self._ema = float(ema)
        self._explore_eps = float(explore_eps)
        # key -> [ema_utility, count]; bounded by _capacity (prune by count)
        self._utility: dict[str, list[float]] = {}
        self._last_choice: str | None = None
        self._proposed_total = 0

    # ------------------------------------------------------------ proposal
    def propose(
        self,
        drive_levels: dict[str, float],
        lp_slopes: dict[str, float] | None = None,
        pending_hypotheses: int = 0,
    ) -> list[Goal]:
        """Drive levels (0..1) -> candidate goals, utility-sorted, bounded."""
        goals: dict[str, Goal] = {}

        def learn(key: str) -> float:
            entry = self._utility.get(key)
            return float(entry[0]) if entry else 0.0

        for drive, level in drive_levels.items():
            if drive not in _DRIVE_TO_KIND:
                continue
            kind, target = _DRIVE_TO_KIND[drive]
            urgency = max(0.0, min(1.0, 1.0 - float(level)))
            u = max(0.0, min(1.0, 0.7 * urgency + 0.3 * learn(f"{kind}:{target}")))
            key = f"{kind}:{target}"
            if key not in goals or u > goals[key].utility:
                goals[key] = Goal(kind, target, u, source=drive)

        for task, slope in (lp_slopes or {}).items():
            s = max(0.0, min(1.0, float(slope)))
            if s <= 0.0:
                continue
            key = f"practice:{task}"
            u = max(0.0, min(1.0, 0.6 * s + 0.4 * learn(key)))
            if key not in goals or u > goals[key].utility:
                goals[key] = Goal("practice", str(task), u, source="lp")

        if pending_hypotheses > 0:
            key = "test:hypothesis"
            u = min(1.0, 0.2 + 0.15 * float(pending_hypotheses)
                    + 0.3 * learn(key))
            goals[key] = Goal("test", "hypothesis", max(0.0, u),
                              source="hypothesis")

        ranked = sorted(goals.values(), key=lambda g: g.utility, reverse=True)
        ranked = ranked[: self._max_proposals]
        self._proposed_total += len(ranked)
        return ranked

    # --------------------------------------------------------- arbitration
    def arbitrate(self, goals: list[Goal]) -> Goal | None:
        """Pick one goal; suppress a goal that monopolizes consecutive
        windows so exploration of alternatives stays possible."""
        if not goals:
            return None
        mono_key = None
        if len(self._history) >= self._monopoly_k:
            tail = self._history[-self._monopoly_k:]
            if len(set(tail)) == 1:
                mono_key = tail[0]
        scored = []
        for g in goals:
            u = g.utility
            if mono_key is not None and g.key == mono_key:
                u *= 0.2
            scored.append((u, g))
        if self._rng.random() < self._explore_eps:
            choice = self._rng.choice(goals)
        else:
            choice = max(scored, key=lambda p: p[0])[1]
        self._history.append(choice.key)
        if len(self._history) > self._history_cap:
            del self._history[: len(self._history) - self._history_cap]
        self._last_choice = choice.key
        return choice

    # ------------------------------------------------------------ feedback
    def record_outcome(self, key: str, achieved: bool) -> None:
        """Outcome -> EMA utility (bounded table, prune lowest-count)."""
        entry = self._utility.get(key)
        target = 1.0 if achieved else 0.0
        if entry is None:
            self._utility[key] = [target, 1.0]
        else:
            entry[0] = (1.0 - self._ema) * entry[0] + self._ema * target
            entry[1] += 1.0
        if len(self._utility) > self._capacity:
            keep = sorted(self._utility.items(),
                          key=lambda kv: kv[1][1], reverse=True)[: self._capacity]
            self._utility = dict(keep)

    # --------------------------------------------------------------- state
    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._utility)

    @property
    def last_choice(self) -> str | None:
        return self._last_choice

    def telemetry(self) -> dict[str, Any]:
        return {
            "utility_keys": len(self._utility),
            "proposed_total": self._proposed_total,
            "last_choice": self._last_choice,
            "history_len": len(self._history),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "utility": {k: list(v) for k, v in self._utility.items()},
            "history": list(self._history),
            "last_choice": self._last_choice,
            "proposed_total": self._proposed_total,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._utility = {str(k): [float(v[0]), float(v[1])]
                         for k, v in state.get("utility", {}).items()}
        self._history = [str(x) for x in state.get("history", [])][-self._history_cap:]
        self._last_choice = state.get("last_choice")
        self._proposed_total = int(state.get("proposed_total", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.setstate(rng_state)
            except (TypeError, ValueError):
                pass


TRACK_KINDS = ("explore", "practice", "approach", "test", "idle")


class GoalTracker:
    """Bounded achievement evaluator for one active goal pair (chosen+random).

    Achievement is judged on ENV-SIDE facts (position zones, success events,
    sibling distance, danger, probe edges, motion windows) — never on agent
    self-report. Both arms share the same evaluator; the random arm is the
    within-run control (no reward, no feedback).
    """

    def __init__(self, ttl: int = 400, zone_history: int = 128,
                 idle_k: int = 60, approach_r: float = 1.2) -> None:
        self._ttl = max(1, int(ttl))
        self._zones: deque[tuple[int, int]] = deque(maxlen=max(8, int(zone_history)))
        self._idle_k = max(10, int(idle_k))
        self._approach_r = float(approach_r)
        self._steps = 0
        self._active = False
        self._chosen: Goal | None = None
        self._random: Goal | None = None
        self._start_step = 0
        self._idle_run = 0
        self._prev_probe = False
        self._start_facts: dict = {}
        self._hit = {"chosen": False, "random": False}

    # --------------------------------------------------------------- lifecycle
    def start(self, chosen: Goal | None, random_goal: Goal | None,
              xy: tuple[float, float], facts: dict | None = None) -> None:
        self._active = True
        self._chosen = chosen
        self._random = random_goal
        self._start_step = self._steps
        self._idle_run = 0
        self._hit = {"chosen": False, "random": False}
        _f = facts or {}
        self._start_facts = {
            "sibling_dist": float(_f.get("sibling_dist", 9.9)),
            "danger": float(_f.get("danger", 0.0)),
        }
        self._zones.append((int(xy[0] // 1), int(xy[1] // 1)))

    def stop(self) -> None:
        self._active = False
        self._chosen = None
        self._random = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def chosen(self) -> Goal | None:
        return self._chosen

    @property
    def random_goal(self) -> Goal | None:
        return self._random

    @property
    def expired(self) -> bool:
        return self._active and (self._steps - self._start_step) >= self._ttl

    # ------------------------------------------------------------------ step
    def _hit_for(self, goal: Goal | None, facts: dict) -> bool:
        if goal is None:
            return False
        kind, target = goal.kind, goal.target
        if kind == "explore":
            cell = (int(float(facts.get("x", 0.0)) // 1),
                    int(float(facts.get("y", 0.0)) // 1))
            return cell not in self._zones
        if kind == "practice":
            return (bool(facts.get("success", False))
                    and str(facts.get("task_id", "")) == str(target))
        if kind == "approach":
            # ARRIVAL/ESCAPE delta events, not instantaneous thresholds:
            # ambient proximity must NOT count (it made chosen==control).
            start = getattr(self, "_start_facts", {})
            if target == "open_space":
                return (float(start.get("danger", 0.0)) > 0.4
                        and float(facts.get("danger", 0.0)) < 0.15)
            return (float(start.get("sibling_dist", 9.9)) > 2.5
                    and float(facts.get("sibling_dist", 9.9)) < self._approach_r)
        if kind == "test":
            return bool(facts.get("probe_active", False)) and not self._prev_probe
        if kind == "idle":
            return self._idle_run >= self._idle_k
        return False

    def step(self, facts: dict) -> dict:
        """Advance one step; returns achievement flags per arm."""
        self._steps += 1
        x = float(facts.get("x", 0.0))
        y = float(facts.get("y", 0.0))
        cell = (int(x // 1), int(y // 1))
        if not self._active:
            self._zones.append(cell)
            self._prev_probe = bool(facts.get("probe_active", False))
            return {"chosen": False, "random": False}
        movement = float(facts.get("movement", 0.0))
        self._idle_run = self._idle_run + 1 if movement < 0.05 else 0
        out = {}
        for arm, goal in (("chosen", self._chosen), ("random", self._random)):
            if not self._hit[arm] and self._hit_for(goal, facts):
                self._hit[arm] = True
            out[arm] = self._hit[arm]
        self._zones.append(cell)
        self._prev_probe = bool(facts.get("probe_active", False))
        return out

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self._steps, "active": self._active,
            "hit": dict(self._hit), "idle_run": self._idle_run,
            "prev_probe": self._prev_probe,
            "zones": list(self._zones),
            "chosen": (self._chosen.key if self._chosen else None),
            "random": (self._random.key if self._random else None),
            "start_step": self._start_step,
            "start_facts": dict(getattr(self, "_start_facts", {})),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._steps = int(state.get("steps", 0))
        self._active = bool(state.get("active", False))
        self._hit = dict(state.get("hit", {"chosen": False, "random": False}))
        self._idle_run = int(state.get("idle_run", 0))
        self._prev_probe = bool(state.get("prev_probe", False))
        self._zones.clear()
        for z in state.get("zones", [])[-self._zones.maxlen:]:
            self._zones.append((int(z[0]), int(z[1])))
        self._start_step = int(state.get("start_step", 0))
        self._start_facts = dict(state.get("start_facts", {}))
        self._chosen = None
        self._random = None


class GoalSession:
    """Trainer-side session: drive proxies -> proposal -> tracking -> bonus.

    M2a (inject=False): achievement rates of the chosen arm vs the random
    within-run control are measured WITHOUT any reward change.
    M2b (inject=True): one-shot bonus on chosen-arm achievement (<=1 per
    window, B10-safe). Random arm never gets reward or feedback.
    """

    def __init__(self, proposer: GoalProposer, seed: int = 0, ttl: int = 400,
                 inject: bool = False, bonus: float = 0.2) -> None:
        self.proposer = proposer
        self.tracker = GoalTracker(ttl=ttl)
        self._rng = random.Random(seed)
        self._inject = bool(inject)
        self._bonus = float(bonus)
        self._ema = {"novelty": 0.0, "success": 0.0}
        self._n_proposals = 0
        self._achieved = {"chosen": 0, "random": 0}
        self._expired = 0
        self._awarded = False
        self._awarded_rand = False
        self._achieved_kinds: dict[str, int] = {}

    @property
    def inject(self) -> bool:
        return self._inject

    @property
    def active_goal(self):
        return self.tracker.chosen

    # --------------------------------------------------------- drive proxies
    def drive_proxies(self, facts: dict) -> dict[str, float]:
        nov = max(0.0, min(1.0, float(facts.get("novelty", 0.0))))
        succ = 1.0 if facts.get("success", False) else 0.0
        self._ema["novelty"] = 0.995 * self._ema["novelty"] + 0.005 * nov
        self._ema["success"] = 0.95 * self._ema["success"] + 0.05 * succ
        mov = max(0.0, min(1.0, float(facts.get("movement", 0.0)) / 0.05))
        sib = float(facts.get("sibling_dist", 9.9))
        danger = max(0.0, min(1.0, float(facts.get("danger", 0.0))))

        def clip01(v: float) -> float:
            return max(0.2, min(1.0, v))

        return {
            "curiosity": clip01(1.0 - 3.0 * self._ema["novelty"]),
            "competence": clip01(1.0 - 2.5 * self._ema["success"]),
            "social": clip01(1.0 - sib / 3.0),
            "safety": clip01(1.0 - danger),
            "rest": clip01(1.0 - mov),
        }

    # ------------------------------------------------------------------ tick
    def tick(self, facts: dict, due: bool = False,
             lp_slopes: dict[str, float] | None = None,
             pending_hypotheses: int = 0) -> dict:
        if self.tracker.expired:
            self._expired += 1
            if self.tracker.chosen is not None:
                self.proposer.record_outcome(self.tracker.chosen.key, False)
            self.tracker.stop()
        drove = False
        if due and not self.tracker.active:
            drives = self.drive_proxies(facts)
            goals = self.proposer.propose(drives, lp_slopes, pending_hypotheses)
            if goals:
                chosen = self.proposer.arbitrate(goals)
                rand_goal = self._rng.choice(goals)
                xy = (float(facts.get("x", 0.0)), float(facts.get("y", 0.0)))
                self.tracker.start(chosen, rand_goal, xy, facts)
                self._n_proposals += 1
                # B18: the per-window award latch MUST reset when a new
                # window starts, otherwise every later achievement is
                # swallowed (first telemetry: 1/473) — regression-tested.
                self._awarded = False
                self._awarded_rand = False
                drove = True
        hits = self.tracker.step(facts)
        bonus = 0.0
        if hits["chosen"] and self.tracker.chosen is not None and not self._awarded:
            self._achieved["chosen"] += 1
            _kind = self.tracker.chosen.kind
            self._achieved_kinds[_kind] = self._achieved_kinds.get(_kind, 0) + 1
            if len(self._achieved_kinds) > 8:
                _drop = min(self._achieved_kinds, key=self._achieved_kinds.get)
                del self._achieved_kinds[_drop]
            self.proposer.record_outcome(self.tracker.chosen.key, True)
            if self._inject:
                bonus = self._bonus * float(self.tracker.chosen.utility)
            self._awarded = True
        if hits["random"] and not self._awarded_rand:
            self._achieved["random"] += 1
            self._awarded_rand = True
        if not self.tracker.active:
            self._awarded = False
            self._awarded_rand = False
        return {"bonus": bonus, "chosen_achieved": bool(hits["chosen"]),
                "random_achieved": bool(hits["random"]), "proposed": drove}

    # --------------------------------------------------------------- telemetry
    def telemetry(self) -> dict[str, Any]:
        t = self.tracker
        return {
            "active": t.chosen.key if t.chosen else None,
            "control": t.random_goal.key if t.random_goal else None,
            "n_proposals": self._n_proposals,
            "achieved_chosen": self._achieved["chosen"],
            "achieved_random": self._achieved["random"],
            "achieved_kinds": dict(self._achieved_kinds),
            "expired": self._expired,
            "inject": self._inject,
            "ema": {k: round(v, 4) for k, v in self._ema.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker.state_dict(),
            "ema": dict(self._ema),
            "n_proposals": self._n_proposals,
            "achieved": dict(self._achieved),
            "achieved_kinds": dict(self._achieved_kinds),
            "expired": self._expired,
            "awarded": self._awarded,
            "awarded_rand": self._awarded_rand,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tracker.load_state_dict(state.get("tracker", {}))
        self._ema = {str(k): float(v) for k, v in state.get("ema", {}).items()} \
            or {"novelty": 0.0, "success": 0.0}
        self._n_proposals = int(state.get("n_proposals", 0))
        ach = state.get("achieved", {})
        self._achieved = {"chosen": int(ach.get("chosen", 0)),
                          "random": int(ach.get("random", 0))}
        _ak = state.get("achieved_kinds", {})
        self._achieved_kinds = {str(k): int(v) for k, v in _ak.items()}
        self._expired = int(state.get("expired", 0))
        self._awarded = bool(state.get("awarded", False))
        self._awarded_rand = bool(state.get("awarded_rand", False))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.setstate(rng_state)
            except (TypeError, ValueError):
                pass
