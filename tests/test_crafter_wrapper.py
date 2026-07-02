"""Tests for :mod:`src.envs.crafter_wrapper`.

Crafter isn't a hard dependency (Stage-3+); tests are skipped if not installed.
Where possible we exercise import-safety and a fake-crafter fallback so the
tests still validate wrapper logic.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _crafter_available() -> bool:
    try:
        import crafter  # noqa: F401
        return True
    except ImportError:
        return False


# =====================================================================
# Import safety
# =====================================================================


def test_wrapper_module_imports_without_crafter():
    """The wrapper module should import fine even if crafter isn't installed;
    the ImportError should surface only at construction time."""
    from src.envs import crafter_wrapper  # noqa: F401


def test_wrapper_raises_import_error_if_crafter_missing(monkeypatch):
    """If crafter is missing, instantiation must raise ImportError with a hint."""
    monkeypatch.setitem(sys.modules, "crafter", None)  # simulate missing
    from src.envs import CrafterWrapper
    with pytest.raises(ImportError, match="crafter"):
        CrafterWrapper()


# =====================================================================
# Fake-crafter test: verify wrapper mechanics
# =====================================================================


class _FakeCrafterEnv:
    """Minimal in-test replacement for crafter.Env."""

    class _Space:
        n = 22

    action_space = _Space()

    def __init__(self, **kwargs):
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self):
        self._t = 0
        return self._rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)

    def step(self, action):
        self._t += 1
        obs = self._rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        reward = 0.1 if action == 0 else 0.0
        done = self._t >= 5
        info = {"achievement": action}
        return obs, reward, done, info


def _install_fake_crafter(monkeypatch):
    mod = types.ModuleType("crafter")
    mod.Env = _FakeCrafterEnv
    monkeypatch.setitem(sys.modules, "crafter", mod)


def test_wrapper_reset_shape(monkeypatch):
    _install_fake_crafter(monkeypatch)
    from src.envs import CrafterWrapper
    env = CrafterWrapper(seed=0)
    obs = env.reset()
    assert obs.shape == (16, 16, 3)
    assert obs.dtype == np.uint8


def test_wrapper_step_returns_crafterstep(monkeypatch):
    _install_fake_crafter(monkeypatch)
    from src.envs import CrafterWrapper
    env = CrafterWrapper(seed=0)
    env.reset()
    step = env.step(3)
    assert step.obs.shape == (16, 16, 3)
    assert isinstance(step.reward, float)
    assert isinstance(step.terminated, bool)
    assert isinstance(step.truncated, bool)
    assert isinstance(step.info, dict)


def test_wrapper_auto_reset_on_done(monkeypatch):
    _install_fake_crafter(monkeypatch)
    from src.envs import CrafterWrapper
    env = CrafterWrapper(seed=0, auto_reset=True)
    env.reset()
    # 5 steps triggers done in fake env
    for _ in range(5):
        step = env.step(0)
    # After done + auto-reset, current_length reset to 0
    assert env._current_length == 0
    # Episode return recorded
    assert env.summary()["episodes"] >= 1


def test_wrapper_action_space_and_obs_shape(monkeypatch):
    _install_fake_crafter(monkeypatch)
    from src.envs import CrafterWrapper
    env = CrafterWrapper()
    env.reset()
    assert env.action_space_n == 22
    assert env.observation_shape == (16, 16, 3)


def test_wrapper_episode_history_bounded(monkeypatch):
    _install_fake_crafter(monkeypatch)
    from src.envs import CrafterWrapper
    env = CrafterWrapper()
    env.reset()
    for _ in range(20 * 5):   # 20 episodes of 5 steps each
        env.step(0)
    ep = env.episode_returns
    # Well under bound of 1024 (Axiom 1)
    assert len(ep) < 1024
    assert env.summary()["episodes"] > 0


# =====================================================================
# Real Crafter (only if installed)
# =====================================================================


@pytest.mark.skipif(not _crafter_available(), reason="crafter not installed")
def test_real_crafter_smoke():  # pragma: no cover
    from src.envs import CrafterWrapper
    env = CrafterWrapper(seed=0)
    obs = env.reset()
    assert obs.dtype == np.uint8
    step = env.step(0)
    assert step.obs.dtype == np.uint8
    env.close()
