"""Intention Achievement Curiosity.

Phase 1+ intrinsic motivation improvement C: replaces blunt RND with
intention-based curiosity.

Core idea (Oudeyer & Kaplan, 2007; Pathak et al. ICM):
    1. Agent takes action with an intended outcome in mind.
    2. After acting, compare RSSM-predicted next state (prior) vs.
       actual observed state (posterior via real obs).
    3. Large prediction error → agent's world model is wrong here →
       high curiosity → explore more.

This is more targeted than RND because it's action-conditioned:
the agent isn't just curious about "novel states", but about states
where its own model of cause-and-effect fails.

Bounded: fixed-size RSSM latent, no growing memory.

意图达成好奇心：比较"我以为会发生什么" vs "实际发生了什么"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model import RSSM, RSSMState


@dataclass
class IntentionConfig:
    """Configuration for :class:`IntentionCuriosity`.

    - ``error_mode``: "kl" (posterior vs prior KL divergence) or
      "mse" (predicted obs vs actual obs MSE).
    - ``coef``: intrinsic reward multiplier.
    - ``ema_decay``: smoothing for running mean/std normalization.
    - ``reward_clip``: clip normalized reward (0 = no clip).
    - ``min_steps_before_active``: warmup steps before using intention signal.
    """

    error_mode: str = "kl"       # "kl" | "mse"
    coef: float = 0.5
    ema_decay: float = 0.99
    reward_clip: float = 5.0
    min_steps_before_active: int = 1000


class IntentionCuriosity(nn.Module):
    """Intention-achievement based intrinsic motivation.

    Uses RSSM to compute the gap between the prior (what the agent
    predicted would happen) and the posterior (what actually happened
    given the real observation). This gap = intention error = intrinsic reward.

    Natural boundedness: fixed RSSM size, no growing data structures.
    """

    def __init__(
        self,
        config: IntentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or IntentionConfig()
        self._total_steps = 0

        # Running statistics for reward normalization
        self.register_buffer("_r_mean", torch.tensor(0.0))
        self.register_buffer("_r_var", torch.tensor(1.0))
        self.register_buffer("_r_count", torch.tensor(1e-4))

    @property
    def capacity(self) -> int:
        return 1  # fixed-size, no growing

    def __len__(self) -> int:
        return self._total_steps

    # -------------------------------------------------------- compute

    def compute_intention_error(
        self,
        wm: RSSM,
        prev_state: RSSMState,
        action_onehot: torch.Tensor,
        real_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intention error: prior vs posterior divergence.

        Args:
            wm: trained RSSM world model.
            prev_state: latent state before the action.
            action_onehot: (B, action_dim) one-hot action.
            real_obs: (B, obs_dim) actual observed next state.

        Returns:
            (B,) tensor of per-sample intention errors.
        """
        # Prior: what the agent thought would happen
        prior_state, prior_dist = wm.imagine_step(prev_state, action_onehot)
        # Posterior: what actually happened (given real obs)
        posterior_state, _, posterior_dist = wm.observe_step(
            prev_state, action_onehot, real_obs
        )

        if self.config.error_mode == "kl":
            # KL(posterior || prior): how much the actual observation
            # changes the agent's belief. High KL = big surprise.
            kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
            error = kl.sum(dim=-1)  # sum over latent dimensions
        else:
            # MSE between prior-predicted obs and actual obs
            pred_obs = wm.decode(prior_state)
            error = F.mse_loss(pred_obs, real_obs, reduction="none").mean(dim=-1)

        return error

    def intention_reward(
        self,
        wm: RSSM,
        prev_state: RSSMState,
        action_onehot: torch.Tensor,
        real_obs: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute normalized intention-based intrinsic reward.

        Args:
            wm: RSSM world model.
            prev_state: RSSMState before the action.
            action_onehot: one-hot encoded action.
            real_obs: actual observation after action.
            normalize: if True, divide by running std.

        Returns:
            (B,) tensor of intrinsic rewards.
        """
        raw = self.compute_intention_error(wm, prev_state, action_onehot, real_obs)

        if not normalize:
            return raw * self.config.coef

        # Online running mean/std update (Welford-style)
        batch_mean = raw.mean()
        batch_var = raw.var(unbiased=False)
        batch_n = raw.shape[0]

        delta = batch_mean - self._r_mean
        tot = self._r_count + batch_n
        new_mean = self._r_mean + delta * batch_n / tot
        self._r_var = (
            self._r_var * self._r_count
            + batch_var * batch_n
            + delta.pow(2) * self._r_count * batch_n / tot
        ) / tot
        self._r_mean = new_mean
        self._r_count = tot

        std = self._r_var.clamp(min=1e-8).sqrt()
        normalized = raw / (std + 1e-8)

        if self.config.reward_clip > 0:
            normalized = normalized.clamp(-self.config.reward_clip, self.config.reward_clip)

        return normalized * self.config.coef

    # -------------------------------------------------------- lifecycle

    def step(self) -> None:
        self._total_steps += 1

    def is_active(self) -> bool:
        return self._total_steps >= self.config.min_steps_before_active

    # -------------------------------------------------------- persistence

    def summary(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "reward_mean": float(self._r_mean.item()),
            "reward_std": float(self._r_var.clamp(min=0).sqrt().item()),
        }

    def state_dict(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "r_mean": self._r_mean,
            "r_var": self._r_var,
            "r_count": self._r_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._total_steps = int(state.get("total_steps", 0))
        self._r_mean = state.get("r_mean", torch.tensor(0.0))
        self._r_var = state.get("r_var", torch.tensor(1.0))
        self._r_count = state.get("r_count", torch.tensor(1e-4))
