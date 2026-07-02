"""Tests for :mod:`src.memory.generative_replay`."""

from __future__ import annotations

import torch

from src.memory import GenerativeReplayConfig, GenerativeReplayVAE


def _mk(obs_dim=8, latent=4, hidden=16):
    return GenerativeReplayVAE(GenerativeReplayConfig(
        obs_dim=obs_dim, latent_dim=latent, hidden=hidden, lr=1e-2, kl_weight=1.0,
    ))


def test_forward_shapes():
    torch.manual_seed(0)
    vae = _mk()
    x = torch.randn(4, 8)
    recon, mean, logvar = vae.forward(x)
    assert recon.shape == (4, 8)
    assert mean.shape == (4, 4)
    assert logvar.shape == (4, 4)


def test_kl_nonnegative_at_prior():
    torch.manual_seed(0)
    vae = _mk()
    mean = torch.zeros(4, 4)
    logvar = torch.zeros(4, 4)
    kl = vae._kl_gaussian(mean, logvar)
    torch.testing.assert_close(kl, torch.zeros(4))


def test_loss_and_metrics_shape():
    torch.manual_seed(0)
    vae = _mk()
    x = torch.randn(6, 8)
    loss, m = vae.loss(x)
    assert loss.dim() == 0
    assert "recon" in m and "kl" in m
    assert torch.isfinite(loss)


def test_update_reduces_loss_on_stationary_data():
    """Overfit a single tiny batch — loss should decrease."""
    torch.manual_seed(0)
    vae = _mk(obs_dim=8, latent=4, hidden=32)
    torch.manual_seed(42)
    batch = torch.randn(8, 8)
    initial = vae.update(batch)["loss"]
    for _ in range(100):
        vae.update(batch)
    final = vae.update(batch)["loss"]
    assert final < initial * 0.8, f"loss did not drop: {initial} → {final}"


def test_sample_shape():
    torch.manual_seed(0)
    vae = _mk()
    s = vae.sample(5)
    assert s.shape == (5, 8)


def test_reconstruct_shape():
    torch.manual_seed(0)
    vae = _mk()
    x = torch.randn(3, 8)
    r = vae.reconstruct(x)
    assert r.shape == x.shape


def test_no_growing_state():
    """After many updates, param count is unchanged (Axiom 1)."""
    vae = _mk()
    p0 = vae.num_parameters()
    for _ in range(30):
        vae.update(torch.randn(4, 8))
        vae.sample(2)
        vae.reconstruct(torch.randn(2, 8))
    p1 = vae.num_parameters()
    assert p0 == p1


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    vae = _mk()
    for _ in range(5):
        vae.update(torch.randn(4, 8))
    state = vae.state_dict()

    torch.manual_seed(1234)
    vae2 = _mk()
    vae2.load_state_dict(state)

    x = torch.randn(2, 8)
    torch.manual_seed(0)
    r1 = vae.reconstruct(x)
    torch.manual_seed(0)
    r2 = vae2.reconstruct(x)
    torch.testing.assert_close(r1, r2)


def test_summary_and_num_params():
    vae = _mk(obs_dim=10, latent=3, hidden=8)
    s = vae.summary()
    assert s["obs_dim"] == 10
    assert s["latent_dim"] == 3
    assert s["num_params"] == vae.num_parameters() > 0
