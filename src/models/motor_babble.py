"""Stage 23 — Motor babbling intrinsic drive (action-sequence novelty).

Developmental rationale: infant motor babbling (0-6m) is spontaneous,
intrinsically-driven motor exploration that builds the sensorimotor
repertoire; its novelty income makes constant-action policies
self-penalizing (no freezing). Three-layer schedule:

  1. strong early   : coef decays exponentially toward a floor
  2. lifelong floor : adults keep exploratory practice (small residual)
  3. reactivation   : when the action repertoire collapses (low unique
                      fraction in a window), the drive temporarily doubles
                      — "boredom -> explore", like adults facing new skills.

Bounded by construction: n-gram table capped (max_entries, LRU-ish
eviction by min count), all counters bounded, serializable.
"""

from __future__ import annotations

import math
from collections import deque


class MotorBabble:
    """Bounded action n-gram novelty drive."""

    def __init__(
        self,
        n_actions: int = 12,
        n: int = 3,
        max_entries: int = 512,
        coef: float = 0.3,
        floor: float = 0.02,
        tau: float = 1_500_000.0,
        reactivate: bool = True,
        window: int = 2000,
        min_unique_frac: float = 0.3,
    ) -> None:
        self._n_actions = max(1, int(n_actions))
        self._n = max(1, int(n))
        self.max_entries = max(8, int(max_entries))
        self._coef0 = float(coef)
        self._floor = float(floor)
        self._tau = max(1.0, float(tau))
        self._reactivate = bool(reactivate)
        self._window = max(64, int(window))
        self._min_unique_frac = float(min_unique_frac)
        self._buf: deque = deque(maxlen=self._n)  # BOUNDS-OK: n fixed
        self._counts: dict[tuple, int] = {}
        self._win_actions: deque = deque(maxlen=self._window)  # BOUNDS-OK
        self._t0: int | None = None
        self._bonus_sum = 0.0
        self._steps = 0
        self._boost_windows = 0
        self._last_boost = 1.0

    # ------------------------------------------------------------- protocol

    @property
    def capacity(self) -> int:
        """Bounded-storage contract (HealthChecker / AGENTS §4)."""
        return self.max_entries

    def __len__(self) -> int:
        return len(self._counts)

    # ------------------------------------------------------------- schedule

    def coef_at(self, global_step: int) -> float:
        if self._t0 is None:
            self._t0 = int(global_step)
        elapsed = max(0, int(global_step) - self._t0)
        return self._floor + (self._coef0 - self._floor) * math.exp(
            -elapsed / self._tau)

    # ---------------------------------------------------------------- step

    def step(self, action: int, global_step: int) -> float:
        """Record one executed action; return this step's novelty bonus."""
        a = int(action)
        self._buf.append(a)
        self._win_actions.append(a)
        self._steps += 1

        # n-gram count-based novelty
        novelty = 0.0
        if len(self._buf) == self._n:
            key = tuple(self._buf)
            c = self._counts.get(key, 0) + 1
            if key not in self._counts and len(self._counts) >= self.max_entries:
                # evict the least-visited entry (bounded table)
                try:
                    worst = min(self._counts, key=self._counts.get)
                    del self._counts[worst]
                except ValueError:
                    pass
            self._counts[key] = c
            novelty = 1.0 / math.sqrt(float(c))

        # reactivation: repertoire collapse -> temporary boost
        boost = 1.0
        if self._reactivate and len(self._win_actions) >= self._window:
            uniq = len(set(self._win_actions))
            if uniq / float(self._n_actions) < self._min_unique_frac:
                boost = 2.0
                self._boost_windows += 1
        self._last_boost = boost

        bonus = self.coef_at(global_step) * novelty * boost
        self._bonus_sum += bonus
        return bonus

    # ---------------------------------------------------------------- info

    @property
    def summary(self) -> dict:
        uniq = len(set(self._win_actions)) if self._win_actions else 0
        return {
            "steps": self._steps,
            "ngrams": len(self._counts),
            "coef": round(self.coef_at(
                (self._t0 or 0) + self._steps), 4),
            "bonus_mean": round(self._bonus_sum / max(1, self._steps), 5),
            "uniq_actions": uniq,
            "boost": self._last_boost,
        }

    # ------------------------------------------------------------ state

    def state_dict(self) -> dict:
        return {
            "counts": {tuple(k): int(v) for k, v in self._counts.items()},
            "t0": self._t0,
            "bonus_sum": self._bonus_sum,
            "steps": self._steps,
            "boost_windows": self._boost_windows,
        }

    def load_state_dict(self, state: dict) -> None:
        self._counts = {tuple(k): int(v) for k, v in
                        state.get("counts", {}).items()}
        self._t0 = state.get("t0")
        self._bonus_sum = float(state.get("bonus_sum", 0.0))
        self._steps = int(state.get("steps", 0))
        self._boost_windows = int(state.get("boost_windows", 0))
