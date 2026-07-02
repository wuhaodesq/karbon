"""Sleep Consolidation Loop.

Stage 6's core deliverable. A periodic "offline" pass that:

1. Trims low-value replay buffer entries.
2. Merges / prunes skill-library entries.
3. Distills TTT inner "slow-W" into fixed weights (placeholder hook).
4. Consolidates the Online EWC Fisher with the current parameters.

The loop is *bounded* — every operation has a max-work bound (Axiom 1).
It is *state-serializable* (Axiom 6) via ``state_dict``.

Design uses a **step-based trigger**: consolidate once per ``every_n_steps``.
Each consolidation call is idempotent-safe if triggered externally.

睡眠固化循环：周期性做修剪 / 合并 / 蒸馏 / EWC 固化，
每步工作量都是常数 O(1) —— 有界。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """When and how often to run each consolidation task.

    All periods are in *training steps*. Set to 0 to disable.
    """

    replay_trim_every: int = 10_000
    skills_merge_every: int = 20_000
    ttt_distill_every: int = 20_000
    ewc_consolidate_every: int = 100_000
    # Global gate — nothing runs before this many steps have elapsed:
    warmup_steps: int = 1_000


@dataclass
class ConsolidationCounters:
    """How many times each sub-task ran. Debugging aid; also verifies boundedness."""

    replay_trim_runs: int = 0
    skills_merge_runs: int = 0
    ttt_distill_runs: int = 0
    ewc_consolidate_runs: int = 0
    last_wall_time: float = 0.0
    total_wall_seconds: float = 0.0


ConsolidationTask = Callable[[], None]
"""A no-arg callable that performs one consolidation action."""


class SleepConsolidationLoop:
    """Bounded periodic consolidation orchestrator.

    Callers register optional tasks:

    .. code-block:: python

        loop = SleepConsolidationLoop(ConsolidationConfig())
        loop.set_replay_trim(lambda: replay.trim_low_value())
        loop.set_skills_merge(lambda: skill_lib.merge_similar())
        loop.set_ttt_distill(lambda: distill_slow_W_into_fixed())
        loop.set_ewc_consolidate(lambda: ewc.consolidate(...))

        for step in training_loop():
            loop.tick(step)

    Each ``tick`` decides which tasks to fire based on step index and
    respective periods. ``every=0`` disables that task.

    All state (config + counters) is bounded and serializable.
    """

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()
        self._counters = ConsolidationCounters()

        self._replay_trim: Optional[ConsolidationTask] = None
        self._skills_merge: Optional[ConsolidationTask] = None
        self._ttt_distill: Optional[ConsolidationTask] = None
        self._ewc_consolidate: Optional[ConsolidationTask] = None

    # ------------------------------------------------- registration API

    def set_replay_trim(self, task: ConsolidationTask) -> None:
        self._replay_trim = task

    def set_skills_merge(self, task: ConsolidationTask) -> None:
        self._skills_merge = task

    def set_ttt_distill(self, task: ConsolidationTask) -> None:
        self._ttt_distill = task

    def set_ewc_consolidate(self, task: ConsolidationTask) -> None:
        self._ewc_consolidate = task

    # ------------------------------------------------------------- tick

    def tick(self, step: int) -> list[str]:
        """Advance one training step. Returns the list of tasks that fired.

        Emits nothing until ``warmup_steps`` have elapsed.
        """
        if step < self.config.warmup_steps:
            return []

        fired: list[str] = []
        t_start = time.time()

        if self._should_fire(step, self.config.replay_trim_every) and self._replay_trim:
            self._run("replay_trim", self._replay_trim)
            self._counters.replay_trim_runs += 1
            fired.append("replay_trim")

        if self._should_fire(step, self.config.skills_merge_every) and self._skills_merge:
            self._run("skills_merge", self._skills_merge)
            self._counters.skills_merge_runs += 1
            fired.append("skills_merge")

        if self._should_fire(step, self.config.ttt_distill_every) and self._ttt_distill:
            self._run("ttt_distill", self._ttt_distill)
            self._counters.ttt_distill_runs += 1
            fired.append("ttt_distill")

        if self._should_fire(step, self.config.ewc_consolidate_every) and self._ewc_consolidate:
            self._run("ewc_consolidate", self._ewc_consolidate)
            self._counters.ewc_consolidate_runs += 1
            fired.append("ewc_consolidate")

        if fired:
            dt = time.time() - t_start
            self._counters.last_wall_time = dt
            self._counters.total_wall_seconds += dt
        return fired

    @staticmethod
    def _should_fire(step: int, period: int) -> bool:
        if period <= 0:
            return False
        return step > 0 and (step % period == 0)

    def _run(self, name: str, task: ConsolidationTask) -> None:
        try:
            task()
        except Exception:
            logger.exception("Consolidation task %s failed", name)
            # Deliberately swallow — sleep should not crash the trainer.
            # Callers can inspect logs.

    # ------------------------------------------------- diagnostics

    def counters(self) -> ConsolidationCounters:
        return self._counters

    def summary(self) -> dict:
        return {
            "replay_trim_runs": self._counters.replay_trim_runs,
            "skills_merge_runs": self._counters.skills_merge_runs,
            "ttt_distill_runs": self._counters.ttt_distill_runs,
            "ewc_consolidate_runs": self._counters.ewc_consolidate_runs,
            "total_wall_seconds": self._counters.total_wall_seconds,
        }

    # ------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "counters": self._counters.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["config"].items():
            setattr(self.config, k, v)
        for k, v in state["counters"].items():
            setattr(self._counters, k, v)
