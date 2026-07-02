"""Auto Curriculum: LP-driven task sampling.

Stage 5's core deliverable. Maintains a bounded task pool and a
LearningProgress tracker. Each ``sample_task`` call picks a task whose
priority is proportional to |LP|, so the agent naturally moves from tasks
where it's still improving to new tasks when a task plateaus.

Bounded:
- Task pool has a fixed maximum size ``max_tasks``.
- Task templates are stored as small metadata dicts, no growth over time.
- LP tracker is bounded per :mod:`src.intrinsic.learning_progress`.

Auto Curriculum：LP-驱动任务采样。任务池定容；LP tracker 定容。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.intrinsic.learning_progress import LearningProgressTracker, LPConfig


@dataclass
class TaskTemplate:
    """A curriculum task template.

    ``spec`` is a domain-specific dict (env params, difficulty knobs, etc.).
    ``difficulty`` is a hint used for logging/plots; not used in sampling.
    """

    id: int
    spec: dict
    difficulty: float = 0.0
    tag: str = ""


@dataclass
class AutoCurriculumConfig:
    max_tasks: int = 100
    lp_window_size: int = 64
    lp_min_samples: int = 8
    exploration_epsilon: float = 0.1     # prob of uniform sample regardless of LP
    smoothing: float = 0.5


class AutoCurriculum:
    """LP-driven bounded curriculum.

    Usage:

    .. code-block:: python

        curr = AutoCurriculum(AutoCurriculumConfig(max_tasks=50))
        curr.add_task(TaskTemplate(id=0, spec={"grid": 5}))
        curr.add_task(TaskTemplate(id=1, spec={"grid": 7}))

        for step in range(N):
            task = curr.sample_task()
            err = train_on_task(task)          # returns some prediction error
            curr.report_error(task.id, err)

    Bounded:
    - task pool ≤ max_tasks (Axiom 1; oldest evicted when full)
    - LP tracker window_size (per task) bounded
    """

    def __init__(
        self,
        config: AutoCurriculumConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or AutoCurriculumConfig()
        if self.config.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self._tasks: dict[int, TaskTemplate] = {}
        self._insertion_order: list[int] = []       # FIFO eviction
        self._lp = LearningProgressTracker(LPConfig(
            window_size=self.config.lp_window_size,
            min_samples_for_signal=self.config.lp_min_samples,
            smoothing=self.config.smoothing,
        ))
        self._rng = rng or random.Random(0)

    # ---------------------------------------------------- Bounded protocol

    @property
    def capacity(self) -> int:
        return self.config.max_tasks

    def __len__(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------- tasks

    def add_task(self, task: TaskTemplate) -> Optional[TaskTemplate]:
        """Insert a task; evict oldest if at capacity. Returns the evicted task
        (or None if none evicted).
        """
        if task.id in self._tasks:
            # Overwrite existing
            self._tasks[task.id] = task
            return None
        evicted: Optional[TaskTemplate] = None
        if len(self._tasks) >= self.config.max_tasks:
            # FIFO evict — could plug in a smarter policy in Stage 5+.
            oldest_id = self._insertion_order.pop(0)
            evicted = self._tasks.pop(oldest_id, None)
            self._lp.reset(oldest_id)
        self._tasks[task.id] = task
        self._insertion_order.append(task.id)
        return evicted

    def get_task(self, task_id: int) -> Optional[TaskTemplate]:
        return self._tasks.get(task_id)

    def known_tasks(self) -> list[int]:
        return list(self._tasks.keys())

    # ------------------------------------------------------------- reports

    def report_error(self, task_id: int, error: float) -> None:
        """Record a prediction error for task ``task_id``.

        The LP tracker aggregates these; sampling uses the resulting |LP|.
        """
        if task_id not in self._tasks:
            raise KeyError(f"unknown task {task_id}")
        self._lp.push(task_id, float(error))

    def report_error_batch(self, task_id: int, errors: list[float]) -> None:
        for e in errors:
            self.report_error(task_id, e)

    # --------------------------------------------------- sampling

    def sample_task(self) -> TaskTemplate:
        """Pick the next task to train on.

        With probability ``exploration_epsilon`` sample uniformly;
        otherwise sample proportional to normalized |LP|.
        """
        if not self._tasks:
            raise RuntimeError("Curriculum has no tasks")

        # Exploration branch
        if self._rng.random() < self.config.exploration_epsilon:
            tid = self._rng.choice(list(self._tasks.keys()))
            return self._tasks[tid]

        # LP-weighted sampling
        probs = self._lp.normalize_priorities(self._tasks.keys())
        # If everyone's LP≈0, this becomes uniform; still fine.
        tids = list(probs.keys())
        weights = [probs[t] for t in tids]
        chosen = self._rng.choices(tids, weights=weights, k=1)[0]
        return self._tasks[chosen]

    def sample_batch(self, batch_size: int) -> list[TaskTemplate]:
        return [self.sample_task() for _ in range(batch_size)]

    # -------------------------------------------------------- summary

    def summary(self) -> dict:
        return {
            "num_tasks": len(self._tasks),
            "capacity": self.capacity,
            "lp_by_task": {
                t: self._lp.learning_progress(t) for t in self._tasks
            },
            "priorities": self._lp.normalize_priorities(self._tasks.keys()),
        }

    # ---------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "tasks": {t: {"spec": tk.spec, "difficulty": tk.difficulty, "tag": tk.tag}
                      for t, tk in self._tasks.items()},
            "insertion_order": list(self._insertion_order),
            "lp": self._lp.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._tasks = {
            int(t): TaskTemplate(id=int(t), spec=info["spec"],
                                 difficulty=info["difficulty"], tag=info["tag"])
            for t, info in state["tasks"].items()
        }
        self._insertion_order = [int(t) for t in state["insertion_order"]]
        self._lp.load_state_dict(state["lp"])
