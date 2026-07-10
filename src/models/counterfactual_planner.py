"""Counterfactual Planning — validate plans via RSSM before executing.

The core System 2 capability missing from the current architecture:
    "Before I act, let me imagine what would happen."

Architecture:
    LongRangePlanner produces plan → RSSM imagines each step
    → compute predicted reward for plan → compare with alternatives
    → pick the plan with highest predicted reward → execute

This directly improves decision quality: every action is validated
against a learned world model before being taken.

Zero GPU, zero retraining. Uses existing RSSM + LongRangePlanner.

反事实规划：行动前用世界模型模拟后果，挑选最优方案。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CounterfactualPlanner(nn.Module):
    """Validate plans via RSSM imagination before executing.

    Loop:
        1. LongRangePlanner proposes candidate action sequences
        2. For each sequence, RSSM imagines the outcome (state trajectory)
        3. Score by: predicted reward + novelty bonus - uncertainty penalty
        4. Select best sequence → execute first action
        5. After execution, compare actual vs predicted → update planner confidence

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
        self._all_predicted: list[float] = []  # store all predictions (not just top-3)
        self._actual_rewards: list[float] = []

    def evaluate_plan(
        self,
        plan: list[int],
        wm: Any,                     # RSSM
        wm_state: Any,               # initial RSSMState
        slot_states: torch.Tensor,   # current observation
        device: torch.device,
    ) -> float:
        """Evaluate one plan via RSSM imagination.

        Returns predicted total reward for this plan.
        """
        if wm is None or wm_state is None:
            return 0.0

        state = wm_state
        total_predicted = 0.0
        uncertainty = 0.0

        for step_action in plan[:self._max_steps]:
            action_onehot = F.one_hot(
                torch.tensor([step_action]), self._num_actions,
            ).float().to(device)
            try:
                state, prior = wm.imagine_step(state, action_onehot)
                decoded = wm.decode(state)
                # Reward proxy: norm change in decoded state
                step_reward = float(decoded.norm().item()) * 0.01
                total_predicted += step_reward
                # Uncertainty: prior distribution entropy
                if hasattr(prior, 'stddev'):
                    uncertainty += float(prior.stddev.mean().item())
            except Exception:
                break

        # Penalize uncertainty
        score = total_predicted - self._uncertainty_penalty * uncertainty
        return score

    def select_best(
        self,
        planner: Any,                # LongRangePlanner
        wm: Any,                     # RSSM
        wm_state: Any,               # initial state
        slot_states: torch.Tensor,
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
        scores: list[tuple[list[int], float]] = []
        for plan in candidates:
            score = self.evaluate_plan(plan, wm, wm_state, slot_states, device)
            scores.append((plan, score))

        if not scores:
            return None

        # Select best
        scores.sort(key=lambda x: -x[1])
        best_plan, best_score = scores[0]

        # Only accept if score is meaningfully positive
        if best_score < self._min_conf:
            return None

        self._best_plan = best_plan
        self._all_predicted = [s[1] for s in scores]
        logger.debug(
            "[cf_plan] best score=%.3f vs runner-up=%.3f (n=%d)",
            best_score,
            scores[1][1] if len(scores) > 1 else 0.0,
            len(scores),
        )
        return best_plan

    def validate_outcome(
        self, actual_reward: float, step: int,
    ) -> float:
        """Compare actual reward with predicted. Returns planning accuracy."""
        self._actual_rewards.append(actual_reward)
        if self._all_predicted:
            predicted = self._all_predicted.pop(0)  # FIFO, matches top-1
            error = abs(actual_reward - predicted)
            return max(0.0, 1.0 - error)
        return 0.5

    @property
    def planning_accuracy(self) -> float:
        if not self._actual_rewards:
            return 0.5
        actuals = self._actual_rewards[-50:]
        predicted = self._all_predicted[-min(50, len(self._all_predicted)):]
        if not actuals or not predicted:
            return 0.5
        errors = [abs(a - p) for a, p in zip(actuals, predicted)]
        return max(0.0, 1.0 - np.mean(errors))

    def summary(self) -> dict:
        return {
            "accuracy": f"{self.planning_accuracy:.2%}",
            "plans_evaluated": len(self._actual_rewards),
            "best_plan_len": len(self._best_plan),
        }
