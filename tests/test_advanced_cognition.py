"""Tests for advanced cognitive modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.advanced_cognition import (
    BehaviorCloningHead,
    CounterfactualImagination,
    CounterfactualResult,
    Hypothesis,
    HypothesisTester,
    MetaLearner,
)


D_MODEL = 64
NUM_ACTIONS = 7


# =====================================================================
# HypothesisTester
# =====================================================================


def test_hypothesis_update_positive():
    h = Hypothesis(
        id=0, condition_embedding=torch.randn(D_MODEL),
        predicted_action=2, confidence=0.5,
    )
    h.update(result=1.0, decay=0.9)
    assert h.confidence > 0.5
    assert h.tested is True
    assert h.test_count == 1


def test_hypothesis_tester_capacity_bounded():
    ht = HypothesisTester(d_model=D_MODEL, num_actions=NUM_ACTIONS, max_hypotheses=8)
    for i in range(20):
        ht.propose_hypothesis(
            condition_embedding=torch.randn(D_MODEL),
            predicted_action=i % NUM_ACTIONS,
        )
    assert len(ht) <= ht.capacity  # Axiom 1


def test_hypothesis_tester_propose_and_get_probe():
    ht = HypothesisTester(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    ht.propose_hypothesis(
        condition_embedding=torch.randn(D_MODEL),
        predicted_action=3,
        description="IF see key THEN pick up",
    )
    probe_action = ht.get_probe_action()
    assert probe_action == 3


def test_hypothesis_tester_feedback_updates():
    ht = HypothesisTester(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    ht.propose_hypothesis(torch.randn(D_MODEL), 3)
    ht.get_probe_action()  # activate
    ht.feedback(result=1.0, decay=0.9)
    verified = ht.get_verified_rules(min_confidence=0.5, min_tests=1)
    assert len(verified) == 1
    assert verified[0].confidence > 0.5


def test_hypothesis_tester_verified_rules_filtered():
    ht = HypothesisTester(d_model=D_MODEL, num_actions=NUM_ACTIONS, max_hypotheses=16)
    # Add + test a hypothesis with positive results
    ht.propose_hypothesis(torch.randn(D_MODEL), 2, "rule A")
    for _ in range(5):
        ht.get_probe_action()
        ht.feedback(result=1.0)
    # Add but don't test another
    ht.propose_hypothesis(torch.randn(D_MODEL), 4, "rule B")
    verified = ht.get_verified_rules(min_confidence=0.7, min_tests=3)
    assert len(verified) == 1
    assert verified[0].description == "rule A"


def test_hypothesis_tester_should_probe():
    ht = HypothesisTester(d_model=D_MODEL, num_actions=NUM_ACTIONS, probe_epsilon=0.5)
    h = torch.randn(D_MODEL)
    result = ht.should_probe(h)
    assert isinstance(result, bool)


def test_hypothesis_tester_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    ht = HypothesisTester(d_model=D_MODEL, max_hypotheses=8)
    assert isinstance(ht, BoundedComponent)


# =====================================================================
# CounterfactualImagination
# =====================================================================


def test_counterfactual_result_regret():
    r = CounterfactualResult(actual_reward=0.3, imagined_reward=0.8)
    assert r.regret == 0.5  # imagined was better → positive regret

    r2 = CounterfactualResult(actual_reward=0.8, imagined_reward=0.3)
    assert r2.regret == -0.5  # actual was better → negative regret


def test_counterfactual_imagination_init():
    ci = CounterfactualImagination(max_imagination_steps=5, num_alternatives=3)
    assert ci.max_steps == 5


def test_counterfactual_imagination_compute_regret():
    """Should return one result per alternative action."""
    ci = CounterfactualImagination(max_imagination_steps=3)

    # Create a mock world model
    from src.models import RSSM, RSSMConfig
    wm = RSSM(RSSMConfig(
        obs_dim=16, action_dim=NUM_ACTIONS, z_dim=8, h_dim=16,
        embed_dim=8, hidden=16, max_rollout_steps=5,
    ))
    state = wm.initial_state(batch_size=1, device=torch.device("cpu"))

    results = ci.compute_regret(
        world_model=wm,
        initial_state=state,
        actual_action=0,
        actual_reward=0.5,
        num_actions=NUM_ACTIONS,
    )
    # Should have NUM_ACTIONS - 1 alternatives
    assert len(results) == NUM_ACTIONS - 1
    for r in results:
        assert isinstance(r, CounterfactualResult)
        assert r.actual_reward == 0.5


# =====================================================================
# BehaviorCloningHead
# =====================================================================


def test_bc_loss_shape():
    bc = BehaviorCloningHead(bc_coef=0.3)
    logits = torch.randn(4, NUM_ACTIONS)
    expert = torch.tensor([0, 2, 1, 3])
    loss = bc.loss(logits, expert)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_bc_coef_decays():
    bc = BehaviorCloningHead(bc_coef=0.5, decay_per_step=0.1)
    initial = bc.current_coef
    for _ in range(5):
        bc.step()
    assert bc.current_coef < initial


def test_bc_loss_gradient():
    bc = BehaviorCloningHead(bc_coef=0.5)
    logits = torch.randn(4, NUM_ACTIONS, requires_grad=True)
    expert = torch.tensor([0, 2, 1, 3])
    loss = bc.loss(logits, expert)
    loss.backward()
    assert logits.grad is not None


def test_bc_no_growing_state():
    bc = BehaviorCloningHead(bc_coef=0.3)
    for _ in range(100):
        bc.step()
        bc.loss(torch.randn(4, NUM_ACTIONS), torch.tensor([0, 1, 2, 3]))
    # Just step count — no accumulation
    assert bc.summary()["step_count"] == 100


def test_bc_summary():
    bc = BehaviorCloningHead(bc_coef=0.3)
    s = bc.summary()
    assert s["initial_coef"] == 0.3
    assert s["step_count"] == 0


# =====================================================================
# MetaLearner
# =====================================================================


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_meta_learner_init_from_model():
    model = TinyModel()
    ml = MetaLearner(model, ema_decay=0.9)
    assert ml.has_meta is True
    meta = ml.get_meta_init()
    # TinyModel has 4 trainable params: fc1.weight, fc1.bias, fc2.weight, fc2.bias
    assert len(meta) == 4


def test_meta_learner_consolidate_updates():
    model = TinyModel()
    ml = MetaLearner(model, ema_decay=0.5)
    meta_before = ml.get_meta_init()

    # Simulate training: change model weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ml.consolidate(model)
    meta_after = ml.get_meta_init()

    # Meta params should have moved toward the new weights
    for key in meta_before:
        assert not torch.allclose(meta_before[key], meta_after[key])


def test_meta_learner_initialize_model():
    model1 = TinyModel()
    ml = MetaLearner(model1, ema_decay=0.9)

    # Train model1
    with torch.no_grad():
        for p in model1.parameters():
            p.add_(1.0)
    ml.consolidate(model1)

    # Create model2 and initialize from meta
    model2 = TinyModel()
    ml.initialize_model(model2)

    # model2 should now have meta-averaged weights
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if n1 in ml._meta_params:
            # p2 should be close to meta params (not the original model2 init)
            meta_p = ml._meta_params[n1]
            assert torch.allclose(p2, meta_p, atol=1e-6)


def test_meta_learner_state_dict_roundtrip():
    model = TinyModel()
    ml = MetaLearner(model, ema_decay=0.8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ml.consolidate(model)
    state = ml.state_dict()

    ml2 = MetaLearner(model, ema_decay=0.5)
    ml2.load_state_dict(state)
    assert ml2._ema_decay == 0.8
    for key in ml._meta_params:
        torch.testing.assert_close(ml._meta_params[key], ml2._meta_params[key])


def test_meta_learner_no_growing_state():
    """Meta params should be same size as model params regardless of #tasks."""
    model = TinyModel()
    ml = MetaLearner(model, ema_decay=0.9)
    size_before = sum(v.numel() for v in ml._meta_params.values())

    for _ in range(20):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        ml.consolidate(model)

    size_after = sum(v.numel() for v in ml._meta_params.values())
    assert size_before == size_after  # Axiom 1: no growth


def test_meta_learner_summary():
    model = TinyModel()
    ml = MetaLearner(model, ema_decay=0.9)
    s = ml.summary()
    assert s["ema_decay"] == 0.9
    assert s["has_meta"] is True
    assert s["num_params"] > 0
