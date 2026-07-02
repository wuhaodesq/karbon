"""MiniGrid environment wrapper.

Provides a thin, deterministic, image-based wrapper around ``gymnasium`` +
``minigrid`` so the training loop sees a consistent (obs, action, reward, done)
tuple regardless of the underlying env version.

MiniGrid 环境的最小封装，屏蔽 gymnasium/minigrid 版本细节，提供确定的接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvStep:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class MiniGridWrapper:
    """Deterministic MiniGrid wrapper with ``uint8`` image observations.

    - Uses ``ImgObsWrapper`` so ``obs`` is a plain HxWxC ``uint8`` array.
    - ``done = terminated or truncated``.
    - Auto-resets on ``done`` if ``auto_reset=True``.

    ``uint8`` 观测 + 自动重置 + 屏蔽 API 变化。
    """

    def __init__(
        self,
        env_id: str = "MiniGrid-Empty-5x5-v0",
        seed: int | None = None,
        max_episode_steps: int | None = None,
        auto_reset: bool = True,
    ) -> None:
        try:
            import gymnasium as gym
            from minigrid.wrappers import ImgObsWrapper
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "minigrid + gymnasium are required. Run: pip install minigrid gymnasium"
            ) from exc

        kwargs: dict[str, Any] = {}
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        self._env = ImgObsWrapper(gym.make(env_id, **kwargs))

        self._seed = seed
        self._auto_reset = auto_reset
        self._last_obs: np.ndarray | None = None
        self._episode_returns: list[float] = []
        self._current_return: float = 0.0
        self._episode_lengths: list[int] = []
        self._current_length: int = 0

    # ------------------------------------------------------------ Gym-like API

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed if seed is not None else self._seed)
        self._last_obs = np.asarray(obs, dtype=np.uint8)
        self._current_return = 0.0
        self._current_length = 0
        return self._last_obs

    def step(self, action: int) -> EnvStep:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._current_return += float(reward)
        self._current_length += 1
        done = bool(terminated) or bool(truncated)
        if done:
            self._episode_returns.append(self._current_return)
            self._episode_lengths.append(self._current_length)
            if self._auto_reset:
                obs, _info = self._env.reset()
            self._current_return = 0.0
            self._current_length = 0
        self._last_obs = np.asarray(obs, dtype=np.uint8)
        return EnvStep(
            obs=self._last_obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self._env.close()

    # -------------------------------------------------------------- properties

    @property
    def action_space_n(self) -> int:
        n = getattr(self._env.action_space, "n", None)
        if n is None:  # pragma: no cover
            raise RuntimeError("Unsupported action space (expected Discrete)")
        return int(n)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        # ImgObsWrapper: HxWxC uint8
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        return tuple(self._last_obs.shape)

    @property
    def episode_returns(self) -> list[float]:
        """Return-list of completed episodes since construction."""
        return list(self._episode_returns)

    @property
    def episode_lengths(self) -> list[int]:
        return list(self._episode_lengths)

    def summary(self) -> dict:
        return {
            "episodes": len(self._episode_returns),
            "mean_return": float(np.mean(self._episode_returns)) if self._episode_returns else 0.0,
            "mean_length": float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0,
            "last_return": self._episode_returns[-1] if self._episode_returns else 0.0,
        }
