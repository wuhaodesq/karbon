"""Knowledge ledger — per-task symbolic-knowledge state drives task choice.

发展逻辑: 自主学习者应练习"既未掌握、又非毫无进展"的任务 (zone of
proximal development)。账本把 agent 自身的符号状态聚合成每个任务的
"可练习度" (practice headroom):

- ``ema_ret``:    滚动 episode 回报 -> mastery 信号 (会了就不必反复练)
- ``ema_margin``: 神经规则匹配边距 (投影后余弦) -> 该任务状态分布对已知
                  规则的"熟悉度"; 低 = 知识未覆盖 (不会 + 不熟 = 该练)
- ``n``:          观测次数 (未观测任务给足探索权重)

``preference()`` 输出候选任务上的采样权重; 与叙事偏好 (v4) 乘性组合,
两条信号共同决定"学什么" — 数据分布路线, 而非动作层微调。

Bounded: fixed capacity (least-observed task evicted). Serializable
(Axiom 1/6).
"""

from __future__ import annotations


class KnowledgeLedger:
    """Bounded per-task knowledge state -> practice preference weights."""

    def __init__(
        self,
        capacity: int = 16,
        alpha: float = 0.2,
        ret_scale: float = 100.0,
        floor: float = 0.05,
    ) -> None:
        self._capacity = int(capacity)
        self._alpha = float(alpha)
        self._ret_scale = float(ret_scale)
        self._floor = float(floor)
        self._tasks: dict[int, dict[str, float]] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------------ record

    def _ensure(self, task_id: int) -> dict[str, float]:
        st = self._tasks.get(task_id)
        if st is None:
            if len(self._tasks) >= self._capacity:
                # Evict the least-observed task (Axiom 1/2)
                worst = min(self._tasks, key=lambda k: self._tasks[k]["n"])
                del self._tasks[worst]
            st = {"n": 0.0, "ema_ret": 0.0, "ema_margin": 0.0, "margin_n": 0.0}
            self._tasks[task_id] = st
        return st

    def record_episode(self, task_id: int, ep_ret: float) -> None:
        """Update the task's rolling return with one episode outcome."""
        st = self._ensure(int(task_id))
        a = self._alpha
        st["ema_ret"] = (1.0 - a) * st["ema_ret"] + a * float(ep_ret)
        st["n"] += 1.0

    def record_margin(self, task_id: int, margin: float) -> None:
        """Update the task's rolling rule-match margin (cosine, may be < 0)."""
        st = self._ensure(int(task_id))
        a = self._alpha
        st["ema_margin"] = (1.0 - a) * st["ema_margin"] + a * float(margin)
        st["margin_n"] += 1.0

    # -------------------------------------------------------------- preference

    def practice_headroom(self, task_id: int) -> float:
        """In [0, 1]: high when neither mastered nor hopeless.

        Unseen tasks return 1.0 (full exploration weight) so unknown ground
        gets sampled at least once.
        """
        st = self._tasks.get(int(task_id))
        if st is None or st["n"] <= 0.0:
            return 1.0
        mastery = min(1.0, max(0.0, st["ema_ret"] / max(self._ret_scale, 1e-6)))
        familiarity = min(1.0, max(0.0, (st["ema_margin"] + 1.0) * 0.5))
        return (1.0 - mastery) * (1.0 - 0.5 * familiarity)

    def preference(self, task_ids: "list[int]") -> "list[float] | None":
        """Sampling weights over candidate tasks, or None with no data.

        Weight = floor + headroom; normalized. Mastered tasks keep only the
        floor (still selectable, never banned).
        """
        if not self._tasks or not task_ids:
            return None
        w = [self._floor + self.practice_headroom(t) for t in task_ids]
        s = float(sum(w))
        if s <= 0.0:
            return None
        return [x / s for x in w]

    # ------------------------------------------------------------------- state

    def state_dict(self) -> dict:
        return {
            "capacity": self._capacity,
            "alpha": self._alpha,
            "ret_scale": self._ret_scale,
            "floor": self._floor,
            "tasks": {str(k): dict(v) for k, v in self._tasks.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._capacity = int(state.get("capacity", self._capacity))
        self._alpha = float(state.get("alpha", self._alpha))
        self._ret_scale = float(state.get("ret_scale", self._ret_scale))
        self._floor = float(state.get("floor", self._floor))
        tasks = state.get("tasks", {})
        self._tasks = {int(k): dict(v) for k, v in tasks.items()}
