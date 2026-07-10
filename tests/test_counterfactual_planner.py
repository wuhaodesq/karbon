"""Landing tests for CounterfactualPlanner (System 2: validate-before-act)."""

import numpy as np
import torch
from types import SimpleNamespace

from src.models.counterfactual_planner import CounterfactualPlanner


class FakeWM:
    """Minimal RSSM: state advances by action magnitude; decode = state.
    No reward head -> planner falls back to decoder-norm proxy.
    """

    def imagine_step(self, state, action_onehot):
        a = action_onehot.argmax(dim=-1).float().unsqueeze(-1)  # (1,1)
        new = state + a * 2.0
        prior = SimpleNamespace(stddev=torch.ones(1, 4) * 0.1)
        return new, prior

    def decode(self, state):
        return state


class FakeWMWithReward:
    """RSSM with a reward head: higher state norm -> higher reward."""

    def imagine_step(self, state, action_onehot):
        a = action_onehot.argmax(dim=-1).float().unsqueeze(-1)  # (1,1)
        new = state + a * 2.0
        prior = SimpleNamespace(stddev=torch.ones(1, 4) * 0.1)
        return new, prior

    def decode(self, state):
        return state

    def predict_reward(self, state):
        return state.norm(dim=-1)


def _state():
    return torch.zeros(1, 4)


def _dev():
    return torch.device("cpu")


def test_counterfactual_planner_scores_higher_action_plans():
    cp = CounterfactualPlanner(min_confidence=0.01)
    wm = FakeWM()
    hi, _ = cp.evaluate_plan([7, 7, 7, 7], wm, _state(), _dev())
    lo, _ = cp.evaluate_plan([0, 0, 0, 0], wm, _state(), _dev())
    assert hi > lo  # bigger actions -> bigger predicted reward
    print(f"\n[cf_plan] score hi={hi:.2f} lo={lo:.2f}")


def test_counterfactual_planner_uses_wm_reward_head():
    cp = CounterfactualPlanner(min_confidence=0.01)
    wm = FakeWMWithReward()
    hi, _ = cp.evaluate_plan([7, 7, 7, 7], wm, _state(), _dev())
    lo, _ = cp.evaluate_plan([0, 0, 0, 0], wm, _state(), _dev())
    assert hi > lo  # reward head scores bigger imagined states higher
    print(f"\n[cf_plan] wm-reward hi={hi:.2f} lo={lo:.2f}")


def test_counterfactual_planner_selects_best_and_validates():
    np.random.seed(0)
    cp = CounterfactualPlanner(num_candidates=5, max_imagine_steps=8, min_confidence=0.3)
    wm = FakeWM()
    planner = SimpleNamespace(_current_plan=[7, 7, 7, 7])  # known strong plan
    plan = cp.select_best(planner, wm, _state(), _dev())

    assert plan is not None
    # Selected plan must score at least as well as the weak plan.
    sel, _ = cp.evaluate_plan(plan, wm, _state(), _dev())
    lo, _ = cp.evaluate_plan([0, 0, 0, 0], wm, _state(), _dev())
    assert sel >= lo
    print(f"\n[cf_plan] selected={plan} score={sel:.2f}")


def test_counterfactual_planner_none_when_no_planner():
    cp = CounterfactualPlanner()
    assert cp.select_best(None, FakeWM(), _state(), _dev()) is None


def test_counterfactual_planner_records_first_step_reward():
    # Validation must compare the first-step predicted reward (the action
    # actually executed) against the single observed reward, NOT the summed
    # total over the whole imagined plan.
    cp = CounterfactualPlanner()
    wm = FakeWMWithReward()
    cp.select_best(
        SimpleNamespace(_current_plan=[7, 7, 7, 7]), wm, _state(), _dev()
    )
    assert cp._last_predicted is not None
    # action 7 -> one imagine step -> state ~ (14,14,14,14) -> norm = 28,
    # whereas the summed total would be ~112. Assert we stored the 1-step value.
    assert 20.0 < cp._last_predicted < 36.0
    print(f"\n[cf_plan] recorded first-step reward={cp._last_predicted:.2f}")


def test_counterfactual_planner_validate_outcome_accuracy():
    cp = CounterfactualPlanner()
    # No prediction on record -> default 0.5
    assert cp.planning_accuracy == 0.5

    cp._last_predicted = 0.8
    acc = cp.validate_outcome(actual_reward=0.8, step=0)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0  # exact match
    assert cp.planning_accuracy == 1.0

    # Mismatch -> lower accuracy (history is bounded, not unbounded)
    cp._last_predicted = 1.0
    cp.validate_outcome(actual_reward=0.0, step=1)
    assert cp.planning_accuracy < 1.0
    assert len(cp._pred_reward) <= 64
    print(f"\n[cf_plan] validate accuracy={cp.planning_accuracy:.2f}")
