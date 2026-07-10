"""Social Curiosity.

Phase 3+ intrinsic motivation improvement: curiosity about social agents.

Core idea:
    1. Predict caregiver's next action from observation.
    2. Prediction error = social curiosity = intrinsic reward.
    3. Also predicts "what will happen after caregiver acts" —
       RSSM-based social outcome prediction error.

This drives the agent to pay attention to and learn from social
interactions (Phase 3 imitation, Phase 4 language grounding).

Bounded: fixed-size predictor networks, no growing data.

社交好奇心：预测看护者行为 → 预测误差 = 好奇信号。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model import RSSM, RSSMState


@dataclass
class SocialCuriosityConfig:
    """Configuration for :class:`SocialCuriosity`.

    - ``action_predictor_hidden``: hidden dim for caregiver action predictor.
    - ``social_coef``: weight of social curiosity in total reward.
    - ``ema_decay``: running mean/std normalization decay.
    - ``reward_clip``: clip normalized reward.
    """

    action_predictor_hidden: int = 64
    social_coef: float = 0.3
    ema_decay: float = 0.99
    reward_clip: float = 5.0


class SocialCuriosity(nn.Module):
    """Curiosity about what other agents (caregiver) will do.

    Learns to predict caregiver actions from agent's own observation.
    When the prediction is wrong, the agent experiences social curiosity
    — driving it to pay attention to social interactions.

    Bounded: fixed-size MLP predictor.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        config: SocialCuriosityConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SocialCuriosityConfig()
        cfg = self.config

        self._num_actions = num_actions
        self._total_predictions = 0

        # Predict caregiver action from encoded observation
        self._predictor = nn.Sequential(
            nn.Linear(obs_dim, cfg.action_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_predictor_hidden, cfg.action_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_predictor_hidden, num_actions),
        )

        # Predict social outcome: what RSSM state will result?
        self._outcome_predictor = nn.Sequential(
            nn.Linear(obs_dim + num_actions, cfg.action_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_predictor_hidden, obs_dim),
        )

        # Running statistics
        self.register_buffer("_r_mean", torch.tensor(0.0))
        self.register_buffer("_r_var", torch.tensor(1.0))
        self.register_buffer("_r_count", torch.tensor(1e-4))

    @property
    def capacity(self) -> int:
        return 1  # fixed-size

    def __len__(self) -> int:
        return self._total_predictions

    # -------------------------------------------------------- prediction

    def predict_caregiver_action(
        self,
        obs_embedding: torch.Tensor,  # (B, obs_dim)
    ) -> torch.Tensor:
        """Predict what the caregiver will do next.

        Returns:
            (B, num_actions) logits.
        """
        return self._predictor(obs_embedding)

    def predict_social_outcome(
        self,
        obs_embedding: torch.Tensor,   # (B, obs_dim)
        caregiver_action_onehot: torch.Tensor,  # (B, num_actions)
    ) -> torch.Tensor:
        """Predict what will happen after caregiver acts.

        Returns:
            (B, obs_dim) predicted next observation.
        """
        x = torch.cat([obs_embedding, caregiver_action_onehot], dim=-1)
        return self._outcome_predictor(x)

    # -------------------------------------------------------- reward

    def social_reward(
        self,
        obs_embedding: torch.Tensor,        # (B, obs_dim) agent's observation
        caregiver_action: torch.Tensor,      # (B,) int actions
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute social curiosity reward.

        High reward when caregiver's action is surprising (hard to predict).

        Args:
            obs_embedding: agent's encoded observation.
            caregiver_action: ground-truth caregiver action.
            normalize: if True, divide by running std.

        Returns:
            (B,) intrinsic social reward.
        """
        logits = self.predict_caregiver_action(obs_embedding)
        # Cross-entropy: low prob for true action = high surprise
        ce_loss = F.cross_entropy(
            logits, caregiver_action.to(torch.long), reduction="none"
        )
        raw = ce_loss

        self._total_predictions += obs_embedding.shape[0]

        if not normalize:
            return raw * self.config.social_coef

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

        return normalized * self.config.social_coef

    # -------------------------------------------------------- training

    def update(
        self,
        obs_embedding: torch.Tensor,
        caregiver_action: torch.Tensor,
        actual_next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Train social predictors on observed data.

        Returns:
            dict with ``action_loss`` and ``outcome_loss``.
        """
        # Action prediction loss
        logits = self.predict_caregiver_action(obs_embedding)
        action_loss = F.cross_entropy(logits, caregiver_action.to(torch.long))

        # Outcome prediction loss
        action_onehot = F.one_hot(
            caregiver_action.clamp(0, self._num_actions - 1).to(torch.long),
            num_classes=self._num_actions,
        ).float()
        pred_outcome = self.predict_social_outcome(obs_embedding, action_onehot)
        outcome_loss = F.mse_loss(pred_outcome, actual_next_obs)

        return {
            "action_loss": float(action_loss.item()),
            "outcome_loss": float(outcome_loss.item()),
        }

    # -------------------------------------------------------- persistence

    def summary(self) -> dict:
        return {
            "total_predictions": self._total_predictions,
            "reward_mean": float(self._r_mean.item()),
            "reward_std": float(self._r_var.clamp(min=0).sqrt().item()),
        }

    def state_dict(self) -> dict:
        return {
            "total_predictions": self._total_predictions,
            "predictor": self._predictor.state_dict(),
            "outcome_predictor": self._outcome_predictor.state_dict(),
            "r_mean": self._r_mean,
            "r_var": self._r_var,
            "r_count": self._r_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._total_predictions = int(state.get("total_predictions", 0))
        self._predictor.load_state_dict(state["predictor"])
        if "outcome_predictor" in state:
            self._outcome_predictor.load_state_dict(state["outcome_predictor"])
        self._r_mean = state.get("r_mean", torch.tensor(0.0))
        self._r_var = state.get("r_var", torch.tensor(1.0))
        self._r_count = state.get("r_count", torch.tensor(1e-4))
