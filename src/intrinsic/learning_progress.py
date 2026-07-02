"""Learning Progress (LP) intrinsic motivation.

An alternative to RND / ICM that avoids the noisy-TV trap. Instead of using
raw prediction error as the reward signal, we use the *decrease* in prediction
error over a sliding time window — i.e., the *learning progress*.

Intuition (Oudeyer, Schmidhuber): novel states may give high prediction error,
but if that error refuses to decrease (e.g., pure noise), the agent is not
learning and should move on. Real learnable structure gives *decreasing* error.

Per-task LP for task ``k``:

.. code-block:: text

    LP_k(t) = mean(err_k over [t-W, t-W/2])  -  mean(err_k over [t-W/2, t])

where ``err_k`` is a sliding window of prediction losses on task ``k``.
Positive LP = still learning; near-zero = plateau; negative = forgetting.

Bounded: for each task, we keep a fixed-length ring buffer of recent errors.
No unbounded history growth (Axiom 1).

Learning Progress：把 novelty 换成"预测误差下降速度"作为动机。
每个任务维护定容误差 ring buffer。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LPConfig:
    """Configuration for :class:`LearningProgressTracker`.

    - ``window_size``: total sliding window (steps of samples).
    - ``min_samples_for_signal``: minimum window fill before returning non-zero LP.
    - ``smoothing``: exponential smoothing factor on returned LP; 0 disables.
    """

    window_size: int = 64
    min_samples_for_signal: int = 8
    smoothing: float = 0.0


class LearningProgressTracker:
    """Bounded per-task learning-progress tracker.

    Usage:

    .. code-block:: python

        lp = LearningProgressTracker(config=LPConfig(window_size=32))
        for step, (task_id, err) in enumerate(stream):
            lp.push(task_id, err)
            if step % 100 == 0:
                priorities = lp.priorities()   # for curriculum sampling

    Bounded: per-task ring buffer of size ``window_size``. No unbounded state.
    """

    def __init__(self, config: LPConfig | None = None) -> None:
        self.config = config or LPConfig()
        if self.config.window_size < 4:
            raise ValueError("window_size must be at least 4")
        if self.config.window_size % 2 != 0:
            raise ValueError("window_size must be even (halved into two halves)")
        # Per-task deques; each capped at window_size (Axiom 1).
        self._errors: dict[int, deque[float]] = {}
        # Smoothed LP snapshot per task.
        self._smoothed_lp: dict[int, float] = {}

    # ---------------------------------------------------- Bounded protocol
    # This is not a single bounded container but a *dict of bounded ones*.
    # We expose (max_tasks, size) via capacity/len for HealthChecker if needed.

    @property
    def capacity(self) -> int:
        """Max total samples across all tasks."""
        # Approximation: per-task cap × number of active tasks.
        # Callers should register tasks with a known bound if strict enforcement matters.
        return max(1, len(self._errors)) * self.config.window_size

    def __len__(self) -> int:
        return sum(len(dq) for dq in self._errors.values())

    # -------------------------------------------------------------- push

    def push(self, task_id: int, error: float) -> None:
        """Record a prediction error for ``task_id`` at the current step."""
        dq = self._errors.get(task_id)
        if dq is None:
            dq = deque(maxlen=self.config.window_size)   # BOUNDS-OK: maxlen bounded
            self._errors[task_id] = dq
        dq.append(float(error))

    def push_batch(self, task_id: int, errors: Iterable[float]) -> None:
        for e in errors:
            self.push(task_id, e)

    # ------------------------------------------------------------- query

    def learning_progress(self, task_id: int) -> float:
        """LP = mean(older half) - mean(newer half).

        Positive value → error is decreasing → still learning.
        Returns 0 if we don't have enough samples yet.
        """
        dq = self._errors.get(task_id)
        if dq is None or len(dq) < self.config.min_samples_for_signal:
            return 0.0
        arr = np.array(dq, dtype=np.float64)
        n = len(arr)
        half = n // 2
        older = arr[:half].mean()
        newer = arr[half:].mean()
        raw_lp = float(older - newer)  # positive = improving

        if self.config.smoothing > 0.0:
            prev = self._smoothed_lp.get(task_id, 0.0)
            smoothed = self.config.smoothing * prev + (1 - self.config.smoothing) * raw_lp
            self._smoothed_lp[task_id] = smoothed
            return smoothed
        return raw_lp

    def priorities(self, tasks: Iterable[int] | None = None) -> dict[int, float]:
        """Return per-task priority (max(|LP|, epsilon)) for curriculum sampling.

        Uses ``|LP|`` because "still improving" and "getting worse (forgetting)"
        are both signals worth attending to. Follows Oudeyer's ACL scheme.
        """
        if tasks is None:
            tasks = list(self._errors.keys())
        out: dict[int, float] = {}
        for t in tasks:
            out[t] = abs(self.learning_progress(t)) + 1e-6
        return out

    def normalize_priorities(self, tasks: Iterable[int] | None = None) -> dict[int, float]:
        """Softmax-normalized priorities: sum to 1 across all tasks."""
        prios = self.priorities(tasks)
        if not prios:
            return {}
        total = sum(prios.values())
        if total == 0:
            k = len(prios)
            return {t: 1.0 / k for t in prios}
        return {t: p / total for t, p in prios.items()}

    # ---------------------------------------------------------- diagnostics

    def known_tasks(self) -> list[int]:
        return list(self._errors.keys())

    def sample_count(self, task_id: int) -> int:
        return len(self._errors.get(task_id, ()))

    def snapshot(self) -> dict:
        """Full snapshot for debugging / checkpointing."""
        return {
            "tasks": {
                t: {
                    "n_samples": len(dq),
                    "lp": self.learning_progress(t),
                    "mean_error": float(np.mean(dq)) if dq else 0.0,
                }
                for t, dq in self._errors.items()
            },
            "capacity": self.capacity,
            "total_samples": len(self),
        }

    # ---------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "errors": {t: list(dq) for t, dq in self._errors.items()},
            "smoothed_lp": dict(self._smoothed_lp),
        }

    def load_state_dict(self, state: dict) -> None:
        # Config compatibility (window_size, etc.) — validate.
        if state["config"]["window_size"] != self.config.window_size:
            raise ValueError("window_size mismatch on load")
        self._errors.clear()
        for t, lst in state["errors"].items():
            dq: deque[float] = deque(maxlen=self.config.window_size)  # BOUNDS-OK: maxlen bounded
            dq.extend(lst)
            self._errors[int(t)] = dq
        self._smoothed_lp = dict(state["smoothed_lp"])

    def reset(self, task_id: int | None = None) -> None:
        """Clear one task's history, or all if ``task_id`` is None."""
        if task_id is None:
            self._errors.clear()
            self._smoothed_lp.clear()
        else:
            self._errors.pop(task_id, None)
            self._smoothed_lp.pop(task_id, None)
