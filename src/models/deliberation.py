"""Stage 21 M4 — DeliberationGate: budget-neutral deliberation control.

Second-order state (recent failure pressure) modulates WHEN the agent
spends its probe budget:

- stuck state (recent failures) -> deliberate more (higher effective
  probe epsilon: try more hypotheses),
- rolling state -> deliberate less (exploit what works).

Equal-budget by construction: the pressure is centered by its own rolling
mean, so the long-run average multiplier is ~1 (the gate redistributes
probes across moments, it does not inflate the total). The A/B against a
fixed rate is a training-window experiment (see docs/stage21_design.md
M4); this module provides the mechanism + bounded bookkeeping.
"""

from __future__ import annotations

from collections import deque


class DeliberationGate:
    """Process-state -> effective probe-epsilon (budget-neutral)."""

    def __init__(self, window: int = 2000, gain: float = 0.5,
                 floor: float = 0.02, cap: float = 0.5) -> None:
        self.window = int(window)
        self.gain = float(gain)
        self.floor = float(floor)
        self.cap = float(cap)
        self._pressures: deque = deque(maxlen=self.window)  # BOUNDS-OK
        self._mods = 0
        self._sum_factor = 0.0

    def modulate(self, eps_base: float, pressure: float) -> float:
        """Return the effective epsilon for this step.

        ``pressure`` in [0, 1]: 1 = recently failing (stuck), 0 = rolling.
        """
        p = float(max(0.0, min(1.0, pressure)))
        self._pressures.append(p)
        pm = sum(self._pressures) / len(self._pressures)
        factor = 1.0 + self.gain * (p - pm)
        self._mods += 1
        self._sum_factor += factor
        return float(max(self.floor, min(self.cap, float(eps_base) * factor)))

    @property
    def summary(self) -> dict:
        n = max(1, len(self._pressures))
        return {
            "mods": self._mods,
            "pressure_mean": round(sum(self._pressures) / n, 4),
            "avg_factor": round(self._sum_factor / max(1, self._mods), 4),
        }
