"""Tests for :mod:`src.intrinsic.rnd`."""

from __future__ import annotations

import numpy as np
import torch

from src.intrinsic import RND, RNDConfig, RNDNet, RunningMeanStd


OBS_SHAPE = (4, 4, 3)


def _make_obs(n: int = 4, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(n, *OBS_SHAPE), dtype=np.uint8)
    return torch.from_numpy(arr)


# =====================================================================
# RunningMeanStd
# =====================================================================


def test_rms_updates_correctly():
    rms = RunningMeanStd(shape=())
    x = torch.arange(10, dtype=torch.float32)  # mean 4.5
    rms.update(x)
    assert abs(float(rms.mean) - 4.5) < 1e-4


def test_rms_stable_over_many_batches():
    torch.manual_seed(0)
    rms = RunningMeanStd(shape=())
    all_vals = []
    for _ in range(20):
        batch = torch.randn(64) * 3.0 + 7.0
        rms.update(batch)
        all_vals.append(batch)
    concat = torch.cat(all_vals)
    assert abs(float(rms.mean) - float(concat.mean())) < 0.2
    assert abs(float(rms.std()) - float(concat.std(unbiased=False))) < 0.2


def test_rms_roundtrip():
    rms = RunningMeanStd()
    rms.update(torch.randn(10))
    state = rms.state_dict()
    rms2 = RunningMeanStd()
    rms2.load_state_dict(state)
    assert float(rms.mean) == float(rms2.mean)
    assert float(rms.var) == float(rms2.var)


# =====================================================================
# RNDNet
# =====================================================================


def test_rndnet_shape():
    net = RNDNet(OBS_SHAPE, embed_dim=32)
    y = net(_make_obs(4))
    assert y.shape == (4, 32)


# =====================================================================
# RND full module
# =====================================================================


def test_rnd_target_is_frozen_forever():
    rnd = RND(OBS_SHAPE)
    before = {n: p.clone() for n, p in rnd.target.named_parameters()}
    for _ in range(5):
        rnd.update(_make_obs(4))
    for n, p in rnd.target.named_parameters():
        assert p.requires_grad is False
        torch.testing.assert_close(before[n], p)


def test_rnd_predictor_moves_toward_target():
    """After many updates on the same batch, predictor loss should decrease."""
    torch.manual_seed(0)
    rnd = RND(OBS_SHAPE, config=RNDConfig(lr=1e-2, embed_dim=32))
    obs = _make_obs(8)
    losses = [rnd.update(obs) for _ in range(50)]
    # Should decrease overall
    assert losses[-1] < losses[0] * 0.8, f"losses did not decrease: {losses[0]} → {losses[-1]}"


def test_rnd_intrinsic_reward_no_grad_side_effect():
    """`intrinsic_reward` must not build a graph or update anything."""
    rnd = RND(OBS_SHAPE)
    before_pred = {n: p.clone() for n, p in rnd.predictor.named_parameters()}
    r = rnd.intrinsic_reward(_make_obs(4))
    assert r.shape == (4,)
    assert not r.requires_grad
    for n, p in rnd.predictor.named_parameters():
        torch.testing.assert_close(before_pred[n], p)


def test_rnd_novel_state_higher_reward_after_training():
    """After training on obs_A repeatedly, novel obs_B should get higher reward."""
    torch.manual_seed(0)
    rnd = RND(OBS_SHAPE, config=RNDConfig(lr=1e-2, embed_dim=32))
    obs_A = _make_obs(4, seed=0)
    obs_B = _make_obs(4, seed=999)  # different distribution

    # Train hard on A
    for _ in range(100):
        rnd.update(obs_A)

    r_A = float(rnd.intrinsic_reward(obs_A).mean())
    r_B = float(rnd.intrinsic_reward(obs_B).mean())
    assert r_B > r_A, f"novel should be more surprising: A={r_A}, B={r_B}"


def test_rnd_normalized_reward_and_clip():
    torch.manual_seed(0)
    rnd = RND(OBS_SHAPE, config=RNDConfig(reward_clip=2.0))
    obs = _make_obs(16)
    r = rnd.normalized_reward(obs)
    assert r.shape == (16,)
    assert (r.abs() <= 2.0 + 1e-5).all()


def test_rnd_state_dict_roundtrip():
    torch.manual_seed(0)
    rnd = RND(OBS_SHAPE, config=RNDConfig(embed_dim=16))
    # Train a few steps so predictor differs from init
    for _ in range(3):
        rnd.update(_make_obs(4))

    state = rnd.rnd_state_dict()

    rnd2 = RND(OBS_SHAPE, config=RNDConfig(embed_dim=16))
    rnd2.load_rnd_state_dict(state)

    obs = _make_obs(4, seed=42)
    r1 = rnd.intrinsic_reward(obs)
    r2 = rnd2.intrinsic_reward(obs)
    torch.testing.assert_close(r1, r2)


def test_rnd_has_no_growing_state():
    """RND does not maintain any collection that grows over calls (Axiom 1).

    Verify by counting attribute types before and after many updates.
    """
    rnd = RND(OBS_SHAPE)
    param_sizes_before = {n: p.numel() for n, p in rnd.named_parameters()}
    for _ in range(30):
        rnd.update(_make_obs(4))
        rnd.intrinsic_reward(_make_obs(4))
    param_sizes_after = {n: p.numel() for n, p in rnd.named_parameters()}
    assert param_sizes_before == param_sizes_after
    # RunningMeanStd is bounded scalars
    assert isinstance(rnd.reward_rms.mean, torch.Tensor)
    assert rnd.reward_rms.mean.numel() <= 1
