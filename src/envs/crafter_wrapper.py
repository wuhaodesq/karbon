"""Crafter environment wrapper.

Deterministic wrapper mirroring the MiniGrid one so the training loop sees a
uniform (obs, action, reward, done) interface. Crafter is Stage-3-onwards
territory (world model + skills) — MiniGrid is fine for Stage 0–2.

Crafter is imported lazily so it isn't a hard install requirement on Windows
where the wheel can be flaky.

Crafter 环境的最小封装，接口与 MiniGridWrapper 对齐。
Stage 3 起启用；懒导入避免 Windows 下 pip 安装失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CrafterStep:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class CrafterWrapper:
    """Minimal Crafter wrapper with ``uint8`` image observations.

    Crafter is a Minecraft-inspired 2D sandbox with a 22-action space and a
    diverse achievement structure. Ideal environment for Stage 3–5 experiments.

    Args:
        seed: PRNG seed. If ``None``, uses Crafter default.
        area: Crafter map size (e.g., ``(64, 64)``). Passed only if supported
            by the installed Crafter version.
        length: Max episode length (Crafter default is 10_000).
        auto_reset: If True, ``step()`` implicitly resets on ``done``.

    Attributes:
        action_space_n: number of discrete actions (typically 17-22)
        observation_shape: (H, W, C) — Crafter default is (64, 64, 3)
    """

    def __init__(
        self,
        seed: int | None = 0,
        area: tuple[int, int] | None = None,
        length: int | None = None,
        auto_reset: bool = True,
    ) -> None:
        try:
            import crafter  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "crafter is not installed. Install with:\n"
                "  pip install crafter\n"
                "It's declared as a Stage 3+ dependency."
            ) from exc

        # Crafter's constructor signature varies across versions; be tolerant.
        kwargs: dict[str, Any] = {}
        if area is not None:
            kwargs["area"] = area
        if length is not None:
            kwargs["length"] = length
        if seed is not None:
            kwargs["seed"] = seed

        try:
            self._env = crafter.Env(**kwargs)
        except TypeError:
            # Older versions don't accept some kwargs; fall back to defaults.
            self._env = crafter.Env()

        self._auto_reset = auto_reset
        self._last_obs: np.ndarray | None = None

        # Episode bookkeeping — bounded lists (fixed-len ring, Axiom 1)
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_return: float = 0.0
        self._current_length: int = 0

    # ---------------------------------------------------- Gym-like API

    def reset(self) -> np.ndarray:
        # crafter.Env.reset returns just obs in most releases
        obs = self._env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self._last_obs = np.asarray(obs, dtype=np.uint8)
        self._current_return = 0.0
        self._current_length = 0
        return self._last_obs

    def step(self, action: int) -> CrafterStep:
        out = self._env.step(int(action))
        # Some versions: (obs, reward, done, info)
        # Others:        (obs, reward, terminated, truncated, info)
        if len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)

        self._current_return += float(reward)
        self._current_length += 1
        if done:
            self._episode_returns.append(self._current_return)
            self._episode_lengths.append(self._current_length)
            # Bound the on-memory return history — Axiom 1
            if len(self._episode_returns) > 1024:
                self._episode_returns = self._episode_returns[-1024:]
                self._episode_lengths = self._episode_lengths[-1024:]
            if self._auto_reset:
                obs = self.reset()
                self._current_return = 0.0
                self._current_length = 0

        self._last_obs = np.asarray(obs, dtype=np.uint8)
        return CrafterStep(
            obs=self._last_obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info) if info else {},
        )

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if close_fn:
            close_fn()

    # ---------------------------------------------------- properties

    @property
    def action_space_n(self) -> int:
        n = getattr(self._env.action_space, "n", None)
        if n is None:  # pragma: no cover
            raise RuntimeError("Unsupported action space (expected Discrete)")
        return int(n)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        return tuple(self._last_obs.shape)

    @property
    def episode_returns(self) -> list[float]:
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
