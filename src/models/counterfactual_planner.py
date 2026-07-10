"""Counterfactual Planning — validate plans via RSSM before executing.

The core System 2 capability missing from the current architecture:
    "Before I act, let me imagine what would happen."

Architecture:
    LongRangePlanner produces plan -> RSSM imagines each step
    -> predict reward for plan via the world model's reward head
    -> compare with alternatives -> pick the plan with highest predicted
    reward -> execute

Reward is produced by the world model's own reward head (Dreamer-style),
trained on real experience. This grounds planning in objective environment
reward, decoupled from (and more stable than) the policy's value estimate,
so System 2 can surface plans the current policy would not choose.

Zero GPU, zero retraining of the policy. Uses existing RSSM +
LongRangePlanner. 反事实规划：行动前用世界模型模拟后果，挑选最优方案。

Bounded guarantees (Axiom 1):
    Prediction/actual history is capped by a fixed-capacity deque; the
    module never grows without bound.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Capacity for the rolling (predicted, actual) history used by planning_accuracy.
_HISTORY_CAPACITY = 64


class CounterfactualPlanner(nn.Module):
    """Validate plans via RSSM imagination before executing.

    Loop:
        1. LongRangePlanner proposes candidate action sequences
        2. For each sequence, RSSM imagines the outcome (state trajectory)
        3. Score by: predicted reward (world-model reward head) - uncertainty
        4. Select best sequence -> execute first action
        5. After execution, compare actual vs predicted -> update planner confidence

    This transforms planning from "blind tree search" to "model-based verification".
    """

    def __init__(
        self,
        num_actions: int = 8,          # adaptive: matches env action space
        num_candidates: int = 5,
        max_imagine_steps: int = 8,
        min_confidence: float = 0.3,
        uncertainty_penalty: float = 0.1,
    ) -> None:
        super().__init__()
        self._num_actions = num_actions
        self._num_candidates = num_candidates
        self._max_steps = max_imagine_steps
        self._min_conf = min_confidence
        self._uncertainty_penalty = uncertainty_penalty

        self._best_plan: list[int] = []
        # Rolling paired history of (predicted_total_reward, actual_reward).
        # Capped deque -> bounded (Axiom 1).
        self._pred_reward: deque[float] = deque(maxlen=_HISTORY_CAPACITY)
        self._actual_reward: deque[float] = deque(maxlen=_HISTORY_CAPACITY)
        self._last_predicted: float | None = None

    def evaluate_plan(
        self,
        plan: list[int],
        wm: Any,                     # RSSM
        wm_state: Any,               # initial RSSMState
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate one plan via RSSM imagination + world-model reward head.

        Returns ``(score, first_reward)`` where ``score`` is the predicted
        total reward for the whole plan (minus an uncertainty penalty) and
        ``first_reward`` is the predicted reward of the first action — the one
        actually executed next, used for apples-to-apples validation against
        the single observed reward.

        Falls back to decoder-norm proxy when the world model has no reward head.
        """
        if wm is None or wm_state is None:
            return 0.0, 0.0

        use_wm_reward = hasattr(wm, "predict_reward")
        state = wm_state
        total_predicted = 0.0
        uncertainty = 0.0
        first_reward = 0.0
        first = True

        for step_action in plan[:self._max_steps]:
            action_onehot = F.one_hot(
                torch.tensor([step_action]), self._num_actions,
            ).float().to(device)
            try:
                state, prior = wm.imagine_step(state, action_onehot)
                if use_wm_reward:
                    with torch.no_grad():
                        step_reward = float(wm.predict_reward(state).item())
                else:
                    decoded = wm.decode(state)
                    step_reward = float(decoded.norm().item()) * 0.01
                if first:
                    first_reward = step_reward
                    first = False
                total_predicted += step_reward
                if hasattr(prior, 'stddev'):
                    uncertainty += float(prior.stddev.mean().item())
            except Exception:
                break

        score = total_predicted - self._uncertainty_penalty * uncertainty
        return score, first_reward

    def select_best(
        self,
        planner: Any,                # LongRangePlanner
        wm: Any,                     # RSSM
        wm_state: Any,               # initial state
        device: torch.device,
    ) -> list[int] | None:
        """Generate candidate plans and select the best via RSSM simulation.

        Returns best action sequence, or None if no good plan found.
        """
        if planner is None:
            return None

        # Generate candidates: original plan + random variants
        candidates: list[list[int]] = []

        # Original plan (from LongRangePlanner's last plan or random)
        if hasattr(planner, '_current_plan') and planner._current_plan:
            candidates.append(list(planner._current_plan))

        # Random variants
        for _ in range(self._num_candidates - len(candidates)):
            rand_plan = [np.random.randint(0, self._num_actions)
                        for _ in range(min(4, self._max_steps))]
            candidates.append(rand_plan)

        # Evaluate all candidates
        scored: list[tuple[list[int], float, float]] = []
        for plan in candidates:
            score, first_r = self.evaluate_plan(plan, wm, wm_state, device)
            scored.append((plan, score, first_r))

        if not scored:
            return None

        # Select best by total predicted reward
        scored.sort(key=lambda x: -x[1])
        best_plan, best_score, best_first = scored[0]

        # Only accept if score is meaningfully positive
        if best_score < self._min_conf:
            return None

        self._best_plan = best_plan
        # Record the first-step predicted reward (the action actually taken)
        # so validation compares against a single observed reward.
        self._last_predicted = best_first
        logger.debug(
            "[cf_plan] best score=%.3f vs runner-up=%.3f (n=%d)",
            best_score,
            scored[1][1] if len(scored) > 1 else 0.0,
            len(scored),
        )
        return best_plan

    def validate_outcome(
        self, actual_reward: float, step: int,
    ) -> float:
        """Compare actual reward with the last predicted plan reward.

        Returns a per-step planning accuracy in [0, 1].
        """
        if self._last_predicted is None:
            return 0.5
        pred = self._last_predicted
        self._pred_reward.append(pred)
        self._actual_reward.append(actual_reward)
        self._last_predicted = None
        error = abs(actual_reward - pred)
        return max(0.0, 1.0 - error)

    @property
    def planning_accuracy(self) -> float:
        if not self._pred_reward:
            return 0.5
        errors = [abs(p - a) for p, a in zip(self._pred_reward, self._actual_reward)]
        return max(0.0, 1.0 - float(np.mean(errors)))

    def summary(self) -> dict:
        return {
            "accuracy": f"{self.planning_accuracy:.2%}",
            "plans_evaluated": len(self._actual_reward),
            "best_plan_len": len(self._best_plan),
        }
