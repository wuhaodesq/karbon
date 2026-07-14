"""Vectorized environment wrapper (Stage-2 throughput).

Holds ``n_envs`` independent ``ThreeDWorld`` instances and exposes a
batched interface so the rollout loop can issue ONE actor-critic
forward over ``(N, 3, H, W)`` instead of N serial forwards.

Why this matters (measured on RTX 4090D, phase-1 3D):
  PROF per_step=17.6ms  env=1.6(9%)  model=6.8(39%)  cog=0.7(4%)
  other=8.3(47%)
The 85% wall is the GPU forward (called once per single obs,
batch=1). Batched N-obs forward amortizes that overhead and
fills the GPU.

NOTE (first cut): the batched hot path (actor-critic + world-model
curiosity + intention + expl-bonus + replay + PPO) runs under N>1.
Single-env-only cognitive blocks (homeostatic drives, emotion,
number-sense/rule predicates, knowledge-gap, concept-graph, memory,
creativity, LLM fusion, RND, skills/symbolic/reflection episode
hooks, causal intervention, cross-modal bridge) are guarded with
``n_envs == 1`` and skipped when N>1. Set ``env.num_envs: 1``
to get the full single-env module coverage back; keep 8 for max
throughput.

Design (Axiom 1 — bounded):
  - Fixed ``n_envs`` at construction; no unbounded growth.
  - Each sub-env capacity is whatever its constructor declares.
  - Backend is serial (N mujoco steps in the main process). A
    Subproc backend can replace this later for env-parallel speed
    without touching the training loop (interface is identical).

CPU-testable: the factory is injectable, so a fake env (no mujoco)
can drive the whole vectorized rollout in ``tests/test_vec_env.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class VecStep:
    """Batched result of ``VecEnv.step``."""

    obs: np.ndarray          # (N, H, W, C) uint8
    reward: np.ndarray       # (N,) float32
    terminated: np.ndarray  # (N,) bool
    truncated: np.ndarray  # (N,) bool
    proprio: np.ndarray     # (N, proprio_dim) float32
    info: list[dict]       # per-env info dicts


def _as_proprio(obj: Any, dim: int) -> np.ndarray:
    raw = getattr(obj, "proprio", None)
    if raw is None:
        return np.zeros(dim, dtype=np.float32)
    return np.asarray(raw, dtype=np.float32).reshape(-1)[:dim]


class VecEnv:
    """Generic serial vectorized env wrapper.

    ``factory(i)`` must return an env with the same API as
    ``ThreeDWorld``: ``reset() -> obs (H,W,C) uint8``,
    ``step(int) -> step_out(.obs/.reward/.terminated/.truncated/
    .proprio)``, ``observation_shape``, ``action_space_n``,
    ``proprio_dim``, ``summary()``.
    """

    def __init__(self, n_envs: int, factory: Callable[[int], Any]) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = int(n_envs)
        self._envs = [factory(i) for i in range(self.n_envs)]
        self.observation_shape = tuple(self._envs[0].observation_shape)
        self.action_space_n = int(self._envs[0].action_space_n)
        self._proprio_dim = int(getattr(self._envs[0], "proprio_dim", 12))
        self._obs = np.zeros(
            (self.n_envs, *self.observation_shape), dtype=np.uint8
        )
        self._reset_all()

    def _reset_all(self) -> None:
        for i, e in enumerate(self._envs):
            self._obs[i] = np.asarray(e.reset())

    def reset(self) -> np.ndarray:
        self._reset_all()
        return self._obs.copy()

    @property
    def obs(self) -> np.ndarray:
        return self._obs

    def step(self, actions: np.ndarray) -> VecStep:
        actions = np.asarray(actions).reshape(self.n_envs)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        proprio = np.zeros((self.n_envs, self._proprio_dim), dtype=np.float32)
        info: list[dict] = [{} for _ in range(self.n_envs)]
        nxt = np.zeros_like(self._obs)
        for i, e in enumerate(self._envs):
            out = e.step(int(actions[i]))
            nxt[i] = np.asarray(out.obs)
            rewards[i] = float(getattr(out, "reward", 0.0))
            terminated[i] = bool(getattr(out, "terminated", False))
            truncated[i] = bool(getattr(out, "truncated", False))
            proprio[i] = _as_proprio(out, self._proprio_dim)
            info[i] = dict(getattr(out, "info", {}) or {})
            if terminated[i] or truncated[i]:
                nxt[i] = np.asarray(e.reset())
        self._obs = nxt
        return VecStep(
            obs=self._obs.copy(),
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            proprio=proprio,
            info=info,
        )

    def summary(self, i: int = 0) -> dict:
        return self._envs[i].summary()

    def close(self) -> None:
        for env in self._envs:
            if hasattr(env, 'close'):
                env.close()

    def episodes(self, i: int = 0) -> int:
        return int(self._envs[i].summary().get("episodes", 0))


class VecThreeDWorld(VecEnv):
    """``VecEnv`` backed by N ``ThreeDWorld`` instances."""

    def __init__(self, n_envs: int = 1, **kwargs: Any) -> None:
        from src.envs.three_d_world import ThreeDWorld

        def _make(i: int) -> ThreeDWorld:
            kw = dict(kwargs)
            kw.setdefault("seed", 42 + i)
            return ThreeDWorld(**kw)

        super().__init__(n_envs, _make)
