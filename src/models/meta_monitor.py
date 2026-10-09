"""Stage 21 — MetaMonitor: online recursive metacognition (M2).

Second-order monitor of the agent's OWN reasoning process. M1 (offline)
validated the capability: predicting probe outcomes from pre-outcome
process features (recent-outcome window carries the signal; +0.154
within-log, +0.15/+0.29 cross-policy, see docs/stage21_design.md).

M2 wires it ONLINE:
- ``on_probe(step)``: capture onset features (freshness, recent success
  rate, consecutive failures, probe density) and the head's prediction.
- ``on_outcome(step, ok)``: online BCE training step (capacity-bounded
  buffers); updates rolling skill statistics (accuracy vs the window's
  majority baseline) so the capability is measured live.

All state is bounded (Axioms): rolling windows only, small head
(4->16->1), stats counters. State persists in checkpoints.
"""

from __future__ import annotations

import json
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMonitor(nn.Module):
    """Online second-order monitor over the hypothesis-loop event stream."""

    N_FEATURES = 4

    def __init__(
        self,
        window: int = 8,
        skill_window: int = 2000,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self._window = int(window)
        self._skill_window = int(skill_window)
        self.head = nn.Sequential(
            nn.Linear(self.N_FEATURES, 16), nn.GELU(), nn.Linear(16, 1))
        self._opt = torch.optim.Adam(self.head.parameters(), lr=float(lr))

        # rolling process state (bounded)
        self._outcomes: deque = deque(maxlen=self._window)  # BOUNDS-OK
        self._probe_steps: deque = deque(maxlen=256)  # BOUNDS-OK
        self._last_outcome_step: int | None = None
        self._consec_fail = 0
        self._n_probes = 0

        # rolling skill stats (bounded windows)
        self._preds: deque = deque(maxlen=self._skill_window)  # BOUNDS-OK
        self._correct: deque = deque(maxlen=self._skill_window)  # BOUNDS-OK
        self._outcome_win: deque = deque(maxlen=self._skill_window)  # BOUNDS-OK
        self._updates = 0
        self._loss_last = 0.0
        self._pending: torch.Tensor | None = None

    # ------------------------------------------------------------ features

    def _features(self, step: int) -> torch.Tensor:
        fresh = float(step - (self._last_outcome_step
                              if self._last_outcome_step is not None else step))
        recent = (sum(self._outcomes) / len(self._outcomes)
                  if self._outcomes else 0.5)
        rate = sum(1 for s in self._probe_steps if step - s <= 2000) / 20.0
        return torch.tensor(
            [fresh, recent, float(self._consec_fail), float(rate)],
            dtype=torch.float32)

    # ------------------------------------------------------------ online API

    def on_probe(self, step: int) -> float:
        """Capture onset features; return the head's P(success) prediction."""
        self._probe_steps.append(int(step))
        self._n_probes += 1
        x = self._features(int(step)).unsqueeze(0)
        with torch.no_grad():
            p = float(torch.sigmoid(self.head(x)).reshape(-1)[0].item())
        self._pending = x[0]
        return p

    def on_outcome(self, step: int, ok: float) -> float:
        """Train on the realized outcome; update rolling skill statistics."""
        ok_f = float(ok)
        loss_val = 0.0
        if self._pending is not None:
            x = self._pending.unsqueeze(0)
            y = torch.tensor([[ok_f]], dtype=torch.float32)
            with torch.enable_grad():
                logit = self.head(x)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
            loss_val = float(loss.item())
            self._loss_last = loss_val
            self._updates += 1
            with torch.no_grad():
                pred = float(torch.sigmoid(self.head(x)).reshape(-1)[0].item())
            self._preds.append(1.0 if pred > 0.5 else 0.0)
            self._correct.append(1.0 if (pred > 0.5) == (ok_f > 0.5) else 0.0)
            self._pending = None
        self._outcomes.append(ok_f)
        self._outcome_win.append(ok_f)
        self._consec_fail = self._consec_fail + 1 if ok_f < 0.5 else 0
        self._last_outcome_step = int(step)
        return loss_val

    # ------------------------------------------------------------ reporting

    @property
    def recent_pressure(self) -> float:
        """0..1: recent failure pressure (1 = stuck, 0 = rolling).

        Derived from the bounded outcome window; 0.5 when no history yet
        (neutral pressure, budget-neutral for the deliberation gate).
        """
        if not self._outcomes:
            return 0.5
        return max(0.0, min(1.0, 1.0 - sum(self._outcomes) / len(self._outcomes)))

    @property
    def skill(self) -> dict:
        """Rolling honest skill estimate (accuracy vs majority baseline).

        Majority baseline comes from the OUTCOME window (what is being
        predicted), not from the prediction rate.
        """
        n = len(self._correct)
        if n < 100:
            return {"n": n, "accuracy": 0.0, "majority": 0.0, "skill": 0.0,
                    "updates": self._updates}
        acc = sum(self._correct) / n
        base = sum(self._outcome_win) / max(1, len(self._outcome_win))
        maj = max(base, 1.0 - base)
        return {"n": n, "accuracy": round(acc, 4),
                "majority": round(maj, 4),
                "skill": round(acc - maj, 4),
                "updates": self._updates,
                "loss": round(self._loss_last, 4)}

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        sd = super().state_dict(*args, **kwargs)
        sd["_updates"] = int(self._updates)
        sd["_consec_fail"] = int(self._consec_fail)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        sd = dict(state_dict)
        self._updates = int(sd.pop("_updates", 0))
        self._consec_fail = int(sd.pop("_consec_fail", 0))
        return super().load_state_dict(sd, strict=strict)
