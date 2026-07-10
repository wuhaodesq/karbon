"""Landing tests for CounterfactualPlanner (System 2: validate-before-act)."""

import numpy as np
import torch
from types import SimpleNamespace

from src.models.counterfactual_planner import CounterfactualPlanner


class FakeWM:
    """Minimal RSSM: state advances by action magnitude; decode = state."""

    def imagine_step(self, state, action_onehot):
        a = action_onehot.argmax(dim=-1).float().unsqueeze(-1)  # (1,1)
        new = state + a * 2.0
        prior = SimpleNamespace(stddev=torch.ones(1, 4) * 0.1)
        return new, prior

    def decode(self, state):
        return state


def _state():
    return torch.zeros(1, 4)


def _slots():
    return torch.zeros(1, 4, 4)


def _dev():
    return torch.device("cpu")


def test_counterfactual_planner_scores_higher_action_plans():
    cp = CounterfactualPlanner(min_confidence=0.01)
    wm = FakeWM()
    hi = cp.evaluate_plan([7, 7, 7, 7], wm, _state(), _slots(), _dev())
    lo = cp.evaluate_plan([0, 0, 0, 0], wm, _state(), _slots(), _dev())
    assert hi > lo  # bigger actions -> bigger predicted reward
    print(f"\n[cf_plan] score hi={hi:.2f} lo={lo:.2f}")


def test_counterfactual_planner_selects_best_and_validates():
    np.random.seed(0)
    cp = CounterfactualPlanner(num_candidates=5, max_imagine_steps=8, min_confidence=0.3)
    wm = FakeWM()
    planner = SimpleNamespace(_current_plan=[7, 7, 7, 7])  # known strong plan
    plan = cp.select_best(planner, wm, _state(), _slots(), _dev())

    assert plan is not None
    # Selected plan must score at least as well as the weak plan.
    sel = cp.evaluate_plan(plan, wm, _state(), _slots(), _dev())
    lo = cp.evaluate_plan([0, 0, 0, 0], wm, _state(), _slots(), _dev())
    assert sel >= lo
    print(f"\n[cf_plan] selected={plan} score={sel:.2f}")


def test_counterfactual_planner_none_when_no_planner():
    cp = CounterfactualPlanner()
    assert cp.select_best(None, FakeWM(), _state(), _slots(), _dev()) is None


def test_counterfactual_planner_validate_outcome_accuracy():
    cp = CounterfactualPlanner()
    cp._all_predicted = [0.8]
    acc = cp.validate_outcome(actual_reward=0.8, step=0)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0  # exact match
    # After popping the only prediction, accuracy falls back to default 0.5.
    assert cp.planning_accuracy == 0.5
    print(f"\n[cf_plan] validate accuracy={acc:.2f}")
