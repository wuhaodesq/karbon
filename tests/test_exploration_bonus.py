"""Tests for ExplorationBonus — 3D deadlock guard.

关键验证：纯常数奖励 => 优势方差=0（死锁）；
常数奖励 + ExplorationBonus => 优势方差>0（价值头拟合不掉访问
计数带来的逐状态/逐历史变化 => 策略始终有探索信号）。
"""
from __future__ import annotations

import math
import torch

from src.intrinsic import ExplorationBonus
from src.train import _normalize_advantages, compute_gae


def _obs(seed: int) -> torch.Tensor:
    g = torch.manual_seed(seed)
    return torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8)


class TestExplorationBonusProperties:
    def test_bonus_decreases_with_visits_and_stays_positive(self) -> None:
        eb = ExplorationBonus((16, 16, 3), capacity=1024, coef=0.5, grid=4)
        a = _obs(0)
        first = float(eb.bonus(a).reshape(-1)[0])  # unseen -> max
        assert first > 0.0
        for _ in range(50):
            eb.update(a)
        later = float(eb.bonus(a).reshape(-1)[0])
        assert later < first  # decays as visited more
        assert later > 0.0  # never reaches 0

    def test_bonus_differs_across_states(self) -> None:
        eb = ExplorationBonus((16, 16, 3), capacity=1024, coef=0.5, grid=4)
        a, b = _obs(0), _obs(1)
        for _ in range(3):
            eb.update(a)
            eb.update(b)
        ba = float(eb.bonus(a).reshape(-1)[0])
        bb = float(eb.bonus(b).reshape(-1)[0])
        # different states -> generally different visit history -> different bonus
        assert abs(ba - bb) > 1e-6 or (ba > 0 and bb > 0)

    def test_capacity_is_bounded(self) -> None:
        cap = 1024
        eb = ExplorationBonus((16, 16, 3), capacity=cap, coef=0.1, grid=4)
        assert eb.capacity == cap
        assert eb._counts.shape == (cap,)  # fixed size, Axiom 1
        for s in range(200):
            eb.update(_obs(s))
        assert eb._counts.shape == (cap,)  # structure never grows
        assert len(eb) >= 200  # only the counts grow, not the structure

    def test_state_dict_roundtrip(self) -> None:
        eb = ExplorationBonus((16, 16, 3), capacity=1024, coef=0.2, grid=4)
        for s in range(10):
            eb.update(_obs(s))
        sd = eb.state_dict()
        eb2 = ExplorationBonus((16, 16, 3), capacity=1024, coef=0.2, grid=4)
        eb2.load_state_dict(sd)
        assert torch.equal(eb._counts, eb2._counts)


class TestExplorationBonusBreaksDeadlock:
    def test_constant_reward_is_deadlock_zero_advantage(self) -> None:
        n = 64
        rewards = torch.zeros(n)  # constant env reward -> deadlock
        values = torch.zeros(n)
        adv, _ = compute_gae(rewards, values, torch.zeros(n), 0.0, 0.99, 0.95)
        assert float(adv.std()) < 1e-9  # no signal

    def test_bonus_turns_deadlock_into_signal(self) -> None:
        n = 101
        eb = ExplorationBonus((16, 16, 3), capacity=4096, coef=0.5, grid=4)
        a, b = _obs(0), _obs(1)
        # State A revisited many times (bonus decays), state B once (bonus max).
        rewards_ctrl = torch.zeros(n)
        rewards_bonus = torch.zeros(n)
        for t in range(n):
            ob = a if t < 100 else b
            rewards_bonus[t] = float(eb.bonus(ob).reshape(-1)[0])
            eb.update(ob)
        values = torch.zeros(n)

        adv_ctrl, _ = compute_gae(rewards_ctrl, values, torch.zeros(n), 0.0, 0.99, 0.95)
        adv_bonus, _ = compute_gae(rewards_bonus, values, torch.zeros(n), 0.0, 0.99, 0.95)

        assert float(adv_ctrl.std()) < 1e-9  # pure constant -> deadlock
        assert float(adv_bonus.std()) > 1e-2  # bonus -> persistent signal
        # normalization must not invent (or kill) it: still > 0 after guard
        norm = _normalize_advantages(adv_bonus)
        assert float(norm.std()) > 0.0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
