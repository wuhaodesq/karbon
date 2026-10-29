"""Goal proposer — Stage 24 M1: drives -> self-proposed goals -> feedback.

The agent PROPOSES its own goals from internal signals (homeostatic drive
levels, learning-progress gaps, pending hypotheses) instead of only picking
externally-defined tasks (M2/B15). Bounded and deterministic:

  propose()      -> up to ``proposals`` candidate goals (utility-clamped)
  arbitrate()    -> one active goal; anti-monopoly penalty (no goal may
                    monopolize consecutive windows)
  record_outcome -> EMA utility table (bounded capacity, pruned by count)
  state_dict()/load_state_dict() for checkpointing

Goal kinds map to existing machinery at wiring time:
  explore  -> novelty/coverage zones (Extended world rooms)
  practice -> current-task interaction (competence drive / LP slope)
  approach -> sibling / open-space (social / safety drives)
  test     -> hypothesis-probe pending questions
  idle     -> rest drive (low-motion window)

驱动 -> 自主目标提案 (有界/确定性/可存档)。这是"内驱->自主目标"闭环的
提案与仲裁层; 奖励注入与达成判定在 train.py 接线 (下一步)。
"""

from __future__ import annotations

import random
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
