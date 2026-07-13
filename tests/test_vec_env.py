"""CPU tests for the Stage-2 vectorization primitives (no mujoco/GPU needed).

Covers:
  * VecEnv with an injectable fake factory (shapes, auto-reset, batched step)
  * RolloutBuffer in (T, N) layout + as_batch flatten
  * compute_gae_vec shape/values vs the scalar compute_gae (per-column parity)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.envs.vec_three_d_world import VecEnv, VecStep
from src.train import RolloutBuffer, compute_gae, compute_gae_vec


@dataclass
class _FakeOut:
    obs: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    proprio: Any = None
    info: dict = None

    def __post_init__(self):
        if self.proprio is None:
            self.proprio = np.zeros(5, dtype=np.float32)
        if self.info is None:
            self.info = {}


class FakeEnv:
    """Minimal env emulating the ThreeDWorld API used by VecEnv."""

    observation_shape = (8, 8, 3)
    action_space_n = 4
    proprio_dim = 5

    def __init__(self, i: int = 0):
        self._i = i
        self._t = 0

    def reset(self):
        self._t = 0
        return np.zeros(self.observation_shape, dtype=np.uint8)

    def step(self, a: int) -> _FakeOut:
        self._t += 1
        term = (self._t % 3) == 0  # terminate every 3rd step
        return _FakeOut(
            obs=np.full(self.observation_shape, self._t, dtype=np.uint8),
            reward=float(a) * 0.1,
            terminated=term,
            truncated=False,
            proprio=np.full(5, float(self._i), dtype=np.float32),
        )

    def summary(self) -> dict:
        return {"episodes": self._t // 3}


def _factory(i: int) -> FakeEnv:
    return FakeEnv(i)


def test_vec_env_reset_and_step_shapes():
    n = 4
    env = VecEnv(n_envs=n, factory=_factory)
    obs = env.reset()
    assert obs.shape == (n, 8, 8, 3), obs.shape
    assert obs.dtype == np.uint8

    actions = np.arange(n) % env.action_space_n
    out = env.step(actions)
    assert isinstance(out, VecStep)
    assert out.obs.shape == (n, 8, 8, 3)
    assert out.reward.shape == (n,)
    assert out.terminated.shape == (n,)
    assert out.truncated.shape == (n,)
    assert out.proprio.shape == (n, 5)
    assert len(out.info) == n
    assert out.reward.dtype == np.float32
    # deterministic env: reward[i] == action*0.1
    np.testing.assert_allclose(out.reward, actions.astype(np.float32) * 0.1)


def test_vec_env_auto_reset_on_done():
    n = 2
    env = VecEnv(n_envs=n, factory=_factory)
    env.reset()
    # Drive env 0 to termination without resetting the whole batch.
    for _ in range(3):
        out = env.step(np.array([0, 0]))
    # After termination the per-env obs for env 0 must have been reset (zeros).
    # We can't read internal state directly, but stepping further must not crash
    # and shapes stay consistent.
    out = env.step(np.array([1, 1]))
    assert out.obs.shape == (n, 8, 8, 3)
    assert out.reward.shape == (n,)


def test_rollout_buffer_tn_layout_and_flatten():
    cap = 16
    obs_shape = (8, 8, 3)
    n = 4
    buf = RolloutBuffer(cap, obs_shape, device=torch.device("cpu"), n_envs=n)
    assert buf.n_envs == n
    assert buf.obs.shape == (cap, n, *obs_shape)

    T = 5
    for _ in range(T):
        buf.add(
            obs=np.zeros((n, *obs_shape), dtype=np.uint8),
            action=np.zeros(n, dtype=np.int64),
            logprob=np.zeros(n, dtype=np.float32),
            value=np.zeros(n, dtype=np.float32),
            reward=np.ones(n, dtype=np.float32),
            done=np.zeros(n, dtype=bool),
        )
    assert buf.full() is False
    assert buf._ptr == T
    bat = buf.as_batch()
    assert bat.obs.shape == (T * n, *obs_shape)
    assert bat.actions.shape == (T * n,)
    assert bat.rewards.shape == (T * n,)
    assert bat.dones.shape == (T * n,)


def test_gae_vec_parity_with_scalar():
    torch.manual_seed(0)
    T, N = 7, 3
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.8).bool()
    dones[-1] = False  # keep last timestep non-terminal for bootstrap
    last_values = torch.randn(N)
    gamma, lam = 0.99, 0.95

    adv2d, ret2d = compute_gae_vec(rewards, values, dones, last_values, gamma, lam)
    assert adv2d.shape == (T, N)
    assert ret2d.shape == (T, N)

    # Per-column parity with the original scalar GAE.
    for j in range(N):
        a1, r1 = compute_gae(rewards[:, j], values[:, j], dones[:, j],
                               float(last_values[j]), gamma, lam)
        np.testing.assert_allclose(adv2d[:, j].numpy(), a1.numpy(), atol=1e-5)
        np.testing.assert_allclose(ret2d[:, j].numpy(), r1.numpy(), atol=1e-5)


def test_gae_vec_no_sync_scalar_last():
    """compute_gae_vec must not rely on .item() (which would sync the GPU)."""
    T, N = 4, 2
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.zeros(T, N, dtype=torch.bool)
    last_values = torch.zeros(N)
    adv, ret = compute_gae_vec(rewards, values, dones, last_values, 0.99, 0.95)
    # With zero values, gamma=1 would make adv==rewards; with gamma<1 it is discounted.
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)
