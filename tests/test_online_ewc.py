"""Tests for :mod:`src.continual.online_ewc`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.continual import OnlineEWC, OnlineEWCConfig


class TinyNet(nn.Module):
    def __init__(self, d_in=4, d_out=2, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_batches(n_batches, batch_size, d_in=4, d_out=2, seed=0):
    torch.manual_seed(seed)
    for _ in range(n_batches):
        x = torch.randn(batch_size, d_in)
        y = torch.randint(0, d_out, (batch_size,))
        yield x, y


def _loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits, y)


# =====================================================================
# Structural / bounded properties
# =====================================================================


def test_ewc_state_shapes_match_model_params():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model)
    for name, p in model.named_parameters():
        assert p.shape == ewc._fisher[name].shape
        assert p.shape == ewc._anchor[name].shape


def test_ewc_penalty_zero_before_consolidate():
    model = TinyNet()
    ewc = OnlineEWC(model)
    pen = ewc.penalty(model)
    assert float(pen.item()) == 0.0
    assert not ewc.has_consolidated()


def test_ewc_state_size_constant_across_consolidations():
    """Whether we consolidate 1 or 100 times, memory footprint doesn't grow."""
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model, OnlineEWCConfig(gamma=0.9))

    def _state_size(e):
        return sum(t.numel() for t in e._fisher.values()) + sum(
            t.numel() for t in e._anchor.values()
        )

    size_before = _state_size(ewc)
    for _ in range(20):
        ewc.consolidate(model, _make_batches(4, 8), _loss_fn, num_batches=4)
    size_after = _state_size(ewc)
    assert size_before == size_after  # Axiom 1


# =====================================================================
# Consolidation logic
# =====================================================================


def test_consolidate_marks_as_ready():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model)
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)
    assert ewc.has_consolidated()


def test_consolidate_fisher_positive():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model, OnlineEWCConfig(gamma=1.0))  # no decay
    ewc.consolidate(model, _make_batches(5, 8), _loss_fn, num_batches=5)
    for name, f in ewc._fisher.items():
        assert (f >= 0).all(), f"negative Fisher entries for {name}"
        assert f.sum() > 0, f"Fisher zero for {name}"


def test_gamma_decays_older_fisher():
    """Two consolidations at gamma=0.5 → older is halved before being added."""
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model, OnlineEWCConfig(gamma=0.5))

    # First consolidation
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)
    first_fisher = {k: v.clone() for k, v in ewc._fisher.items()}

    # Second consolidation (with the same data seed, so contribution ~ equal)
    ewc.consolidate(model, _make_batches(3, 4, seed=42), _loss_fn, num_batches=3)

    for name in first_fisher:
        # After second consolidate: F_new = 0.5 * F1 + F2 → strictly ≥ 0.5*F1
        assert (ewc._fisher[name] >= 0.5 * first_fisher[name] - 1e-6).all()


# =====================================================================
# Penalty behavior
# =====================================================================


def test_penalty_zero_at_anchor():
    """If θ hasn't moved since consolidation, penalty is exactly 0."""
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model)
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)
    pen = ewc.penalty(model)
    assert abs(float(pen.item())) < 1e-6


def test_penalty_grows_when_theta_moves():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model, OnlineEWCConfig(lambda_reg=10.0))
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)

    # Move parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    pen = ewc.penalty(model)
    assert float(pen.item()) > 0.0


def test_penalty_gradient_flows_to_model():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model, OnlineEWCConfig(lambda_reg=1.0))
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)

    # Zero grads, then compute penalty and back-prop
    model.zero_grad(set_to_none=True)
    pen = ewc.penalty(model)
    pen.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no penalty grad for {name}"
        assert p.grad.abs().sum() > 0


# =====================================================================
# Persistence
# =====================================================================


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    model = TinyNet()
    ewc = OnlineEWC(model)
    ewc.consolidate(model, _make_batches(3, 4), _loss_fn, num_batches=3)
    state = ewc.state_dict()

    model2 = TinyNet()
    ewc2 = OnlineEWC(model2)
    ewc2.load_state_dict(state)
    for name in ewc._fisher:
        torch.testing.assert_close(ewc._fisher[name], ewc2._fisher[name])
        torch.testing.assert_close(ewc._anchor[name], ewc2._anchor[name])
    assert ewc2.has_consolidated()


def test_summary_shape():
    model = TinyNet()
    ewc = OnlineEWC(model)
    ewc.consolidate(model, _make_batches(2, 4), _loss_fn, num_batches=2)
    s = ewc.summary()
    assert set(s.keys()) >= {"num_params_tracked", "fisher_l1_total", "has_consolidated"}
