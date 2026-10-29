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
        novelty_weight: float = 0.0,
        lp_selection: bool = False,
    ) -> None:
        self._capacity = int(capacity)
        self._alpha = float(alpha)
        self._ret_scale = float(ret_scale)
        self._floor = float(floor)
        self._novelty_weight = float(novelty_weight)
        self._lp_selection = bool(lp_selection)
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
            st = {"n": 0.0, "ema_ret": 0.0, "ema_margin": 0.0,
                  "margin_n": 0.0, "ema_int": 0.0,
                  "ema_ret_fast": 0.0, "ema_ret_slow": 0.0}
            self._tasks[task_id] = st
        return st

    def record_episode(self, task_id: int, ep_ret: float) -> None:
        """Update the task's rolling return with one episode outcome."""
        st = self._ensure(int(task_id))
        a = self._alpha
        r = float(ep_ret)
        st["ema_ret"] = (1.0 - a) * st["ema_ret"] + a * r
        # Stage 23 M3+ (B15): learning-progress signals. Fast/slow EMAs give
        # a scale-free per-task progress estimate (fast - slow)/slow — the
        # classic intrinsic-curriculum criterion (Oudeyer): practice what is
        # IMPROVING, not what merely PAYS LESS (the level-based criterion
        # over-practiced low-payout sparse scenes; band drifted 0.65->0.50).
        if st["n"] <= 0.0:
            # seed both EMAs with the first return (avoids a fake progress
            # spike from a zero warmup on flat task series)
            st["ema_ret_fast"] = r
            st["ema_ret_slow"] = r
        else:
            # .get with fallback: resume states saved before the LP upgrade
            # do not contain the fast/slow EMAs (KeyError observed live on
            # 2026-10-27 after resuming the 38.35M checkpoint).
            st["ema_ret_fast"] = 0.5 * st.get("ema_ret_fast", r) + 0.5 * r
            st["ema_ret_slow"] = 0.95 * st.get("ema_ret_slow", r) + 0.05 * r
        st["n"] += 1.0

    def record_margin(self, task_id: int, margin: float) -> None:
        """Update the task's rolling rule-match margin (cosine, may be < 0)."""
        st = self._ensure(int(task_id))
        a = self._alpha
        st["ema_margin"] = (1.0 - a) * st["ema_margin"] + a * float(margin)
        st["margin_n"] += 1.0

    def record_intrinsic(self, task_id: int, value: float) -> None:
        """Update the task's rolling intrinsic (novelty/curiosity) signal.

        Stage 23 M2: per-episode intrinsic return is a first-party signal of
        "this task still teaches me something" — blended into preference()
        when ``novelty_weight > 0``.
        """
        st = self._ensure(int(task_id))
        a = self._alpha
        st["ema_int"] = (1.0 - a) * st.get("ema_int", 0.0) + a * float(value)

    # -------------------------------------------------------------- preference

    def practice_headroom(self, task_id: int) -> float:
        """In [0, 1]: high when neither mastered nor hopeless.

        Unseen tasks return 1.0 (full exploration weight) so unknown ground
        gets sampled at least once.
        """
        st = self._tasks.get(int(task_id))
        if st is None or st["n"] <= 0.0:
            return 1.0
        mastery = min(1.0, max(0.0, st["ema_ret"] / self._reference_return()))
        familiarity = min(1.0, max(0.0, (st["ema_margin"] + 1.0) * 0.5))
        return (1.0 - mastery) * (1.0 - 0.5 * familiarity)

    def _reference_return(self) -> float:
        """Stage 23 M2.5 (bug-ledger B13): mastery must be RELATIVE to the
        current reward regime. The absolute ret_scale=100 assumption broke
        once rich-environment returns reached 1200-2400 -> every task read
        mastery=1 -> weights collapsed to the floor (uniform sampling), i.e.
        the zone-of-proximal-development signal was dead. With >=2 observed
        tasks, use their mean return as the reference; a single-task ledger
        falls back to the absolute scale.
        """
        vals = [st["ema_ret"] for st in self._tasks.values() if st["n"] > 0.0]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            if mean > 1e-6:
                return mean
        return max(self._ret_scale, 1e-6)

    def learning_progress(self, task_id: int) -> float | None:
        """Scale-free per-task progress: (fast - slow) / |slow|.

        None for unseen tasks (caller applies the exploration weight).
        """
        st = self._tasks.get(int(task_id))
        if st is None or st["n"] <= 0.0:
            return None
        fast = float(st.get("ema_ret_fast", st["ema_ret"]))
        slow = float(st.get("ema_ret_slow", st["ema_ret"]))
        return (fast - slow) / max(abs(slow), 1e-6)

    def preference(self, task_ids: "list[int]") -> "list[float] | None":
        """Sampling weights over candidate tasks, or None with no data.

        Weight = floor + headroom; normalized. Mastered tasks keep only the
        floor (still selectable, never banned).
        """
        if not self._tasks or not task_ids:
            return None
        if self._lp_selection:
            # LP mode: unseen -> exploration weight 1.0; improving -> up to
            # 1.0 by relative progress; stalled/declining -> floor.
            progs = [self.learning_progress(t) for t in task_ids]
            known = [p for p in progs if p is not None]
            mx = max([max(0.0, p) for p in known], default=0.0)
            w = []
            for prog in progs:
                if prog is None:
                    w.append(1.0)
                elif prog > 0.0 and mx > 1e-9:
                    w.append(self._floor
                             + (1.0 - self._floor) * (prog / mx))
                else:
                    w.append(self._floor)
        else:
            w = [self._floor + self.practice_headroom(t) for t in task_ids]
        # Stage 23 M2: intrinsic-novelty boost. Per-task intrinsic EMA is
        # normalized by the batch max (bounded [0,1]); tasks that keep paying
        # curiosity get extra practice weight. novelty_weight=0 -> unchanged.
        if self._novelty_weight > 0.0:
            iv = [max(0.0, float(self._tasks.get(int(t), {}).get("ema_int", 0.0)))
                  for t in task_ids]
            mx = max(iv) if iv else 0.0
            if mx > 1e-9:
                w = [wi * (1.0 + self._novelty_weight * (ni / mx))
                     for wi, ni in zip(w, iv)]
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
            "novelty_weight": self._novelty_weight,
            "lp_selection": self._lp_selection,
            "tasks": {str(k): dict(v) for k, v in self._tasks.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._capacity = int(state.get("capacity", self._capacity))
        self._alpha = float(state.get("alpha", self._alpha))
        self._ret_scale = float(state.get("ret_scale", self._ret_scale))
        self._floor = float(state.get("floor", self._floor))
        self._novelty_weight = float(state.get("novelty_weight",
                                               self._novelty_weight))
        self._lp_selection = bool(state.get("lp_selection", self._lp_selection))
        tasks = state.get("tasks", {})
        self._tasks = {int(k): dict(v) for k, v in tasks.items()}
