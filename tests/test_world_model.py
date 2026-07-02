"""Tests for :mod:`src.models.world_model`."""

from __future__ import annotations

import pytest
import torch

from src.models import RSSM, RSSMConfig, RSSMState


def _make_rssm(obs_dim=16, action_dim=6, z_dim=8, h_dim=16, max_T=8, free_nats=1.0):
    return RSSM(RSSMConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        z_dim=z_dim,
        h_dim=h_dim,
        embed_dim=16,
        hidden=32,
        max_rollout_steps=max_T,
        kl_free_nats=free_nats,
    ))


def test_initial_state_shape():
    m = _make_rssm()
    s = m.initial_state(batch_size=3, device=torch.device("cpu"))
    assert s.h.shape == (3, 16)
    assert s.z.shape == (3, 8)


def test_observe_step_produces_valid_state():
    torch.manual_seed(0)
    m = _make_rssm()
    s0 = m.initial_state(2, torch.device("cpu"))
    a = torch.zeros(2, 6)  # dummy one-hot
    a[:, 0] = 1
    obs = torch.randn(2, 16)
    s1, prior, posterior = m.observe_step(s0, a, obs)
    assert s1.h.shape == (2, 16)
    assert s1.z.shape == (2, 8)
    assert prior.mean.shape == (2, 8)
    assert posterior.mean.shape == (2, 8)


def test_imagine_step_advances_state():
    torch.manual_seed(0)
    m = _make_rssm()
    s0 = m.initial_state(2, torch.device("cpu"))
    a = torch.zeros(2, 6); a[:, 1] = 1
    s1, prior = m.imagine_step(s0, a)
    assert s1.h.shape == (2, 16)
    assert prior.mean.shape == (2, 8)


def test_decode_shape():
    m = _make_rssm()
    s = m.initial_state(2, torch.device("cpu"))
    recon = m.decode(s)
    assert recon.shape == (2, 16)


def test_compute_loss_shape_and_positive():
    torch.manual_seed(0)
    m = _make_rssm(max_T=4)
    B, T = 2, 4
    obs = torch.randn(B, T, 16)
    actions = torch.zeros(B, T, 6); actions[..., 0] = 1
    out = m.compute_loss(obs, actions)
    assert set(out.keys()) == {"loss", "recon_loss", "kl_loss"}
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert out["recon_loss"].item() >= 0
    assert out["kl_loss"].item() >= 0


def test_compute_loss_gradient_flows():
    torch.manual_seed(0)
    # Use free_nats=0 so KL clamp doesn't zero out prior gradients
    m = _make_rssm(max_T=4, free_nats=0.0)
    obs = torch.randn(2, 3, 16)
    actions = torch.zeros(2, 3, 6); actions[..., 2] = 1
    out = m.compute_loss(obs, actions)
    out["loss"].backward()
    grads_by_module = {}
    for name, p in m.named_parameters():
        top = name.split(".")[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads_by_module[top] = True
    for expected in ("encoder", "decoder", "recurrent", "prior_dist", "posterior_dist"):
        assert grads_by_module.get(expected, False), f"no grad reaching {expected}"


def test_compute_loss_rejects_over_max_rollout():
    m = _make_rssm(max_T=4)
    obs = torch.randn(1, 5, 16)  # too long
    actions = torch.zeros(1, 5, 6); actions[..., 0] = 1
    with pytest.raises(ValueError):
        m.compute_loss(obs, actions)


def test_imagine_bounded_length():
    m = _make_rssm(max_T=4)
    s0 = m.initial_state(1, torch.device("cpu"))
    too_long_actions = torch.zeros(1, 5, 6); too_long_actions[..., 0] = 1
    with pytest.raises(ValueError):
        m.imagine(s0, too_long_actions)


def test_imagine_produces_trajectory():
    torch.manual_seed(0)
    m = _make_rssm(max_T=8)
    s0 = m.initial_state(2, torch.device("cpu"))
    actions = torch.zeros(2, 5, 6); actions[..., 3] = 1
    traj = m.imagine(s0, actions)
    assert len(traj) == 5
    for s in traj:
        assert isinstance(s, RSSMState)
        assert s.h.shape == (2, 16)


def test_rssm_param_count_reasonable():
    m = _make_rssm()
    n = m.num_parameters()
    # ~5–50k params for these toy sizes
    assert 1000 < n < 100_000, f"unexpected param count: {n}"
