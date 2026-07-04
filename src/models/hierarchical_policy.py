"""Hierarchical RL: sub-goal generation + goal-conditioned action.

Instead of a flat policy (obs → action), this adds a **meta-controller**
that generates abstract sub-goals ("go to the key", "open the door"), and a
**controller** that produces primitive actions conditioned on the sub-goal.

Architecture (simplified — no separate training algorithm needed):

    obs → Hybrid backbone → hidden state h
                              ↓
                    ┌─────────┴──────────┐
                    ↓                    ↓
            sub-goal head          action head (FiLM-conditioned)
            (predicts target       (outputs primitive action
             hidden state)          given h + sub-goal)
                    ↓                    ↓
            sub-goal embedding     action logits
            g = f(h)               a = π(h | g)

The sub-goal head is trained via a **self-supervised auxiliary objective**:
predict the hidden state N steps in the future. This makes sub-goals
meaningful without needing a separate RL algorithm (like HIRO/Option-Critic).

    loss_subgoal = || sub_goal_head(h_t) - h_{t+k} ||²

The action head is conditioned on the sub-goal via FiLM:
    action_logits = action_head(FiLM(h, sub_goal))

Bounded: sub-goal head is (d_model → d_model) ≈ 150k params.
FiLM layer is 2 × d_model² ≈ 300k params. Axiom 1 satisfied.

层次化策略：高层生成子目标（"去找钥匙"），底层执行原始动作。
不需要单独的层次化 RL 算法——用自监督辅助损失训练子目标头。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_backbone import HybridBackbone
from .language_encoder import FiLMLayer
from .vision_encoder import CNNEncoder, VisionEncoder

logger = logging.getLogger(__name__)


# =====================================================================
# Sub-goal head
# =====================================================================


class SubGoalHead(nn.Module):
    """Predicts a target hidden state (sub-goal) from the current state.

    The sub-goal is a vector in the same space as the backbone's hidden state.
    It represents "where the agent should be in a few steps".

    Trained via self-supervised loss: predict the future hidden state.
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Generate a sub-goal embedding from the current hidden state."""
        return self.net(hidden_state)

    def auxiliary_loss(
        self,
        current_hidden: torch.Tensor,
        future_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised: predict where the hidden state will be in k steps.

        Args:
            current_hidden: (B, d_model) — hidden state at time t.
            future_hidden: (B, d_model) — hidden state at time t+k.
        """
        predicted_goal = self.forward(current_hidden)
        return F.mse_loss(predicted_goal, future_hidden.detach())


# =====================================================================
# Goal-conditioned action head
# =====================================================================


class GoalConditionedActionHead(nn.Module):
    """Action head conditioned on a sub-goal via FiLM.

    Instead of directly mapping hidden state → action, this head first
    modulates the hidden state with the sub-goal (FiLM), then maps to action.

        action_logits = action_head(FiLM(h, g))

    This lets the sub-goal steer WHAT actions are preferred.
    """

    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.film = FiLMLayer(d_vis=d_model, d_lang=d_model)
        self.action_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_state: torch.Tensor,
        sub_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value) conditioned on the sub-goal."""
        conditioned = self.film(hidden_state, sub_goal)
        return self.action_head(conditioned), self.value_head(conditioned).squeeze(-1)


# =====================================================================
# Full hierarchical actor-critic
# =====================================================================


class HierarchicalActorCritic(nn.Module):
    """Two-level policy: meta-controller (sub-goals) + controller (actions).

    Pipeline:
        obs → encoder → Hybrid backbone → hidden h
        h → SubGoalHead → sub-goal g
        (h, g) → GoalConditionedActionHead → action + value

    The sub-goal is regenerated every ``sub_goal_every`` steps; between
    regenerations, the same sub-goal is reused (temporal abstraction).

    Bounded: all components are fixed-size. Sub-goal is a single (d_model,)
    vector. No accumulation. Axiom 1 satisfied.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        d_model: int = 384,
        n_layers: int = 3,
        n_heads: int = 4,
        swa_window: int = 16,
        ttt_mini_batch: int = 8,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        use_vision_encoder: bool = False,
        vision_model_name: str = "dinov2_vits14",
        sub_goal_every: int = 8,
    ) -> None:
        super().__init__()
        # Snap d_model
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        if d_model % 2 != 0:
            d_model += 1
        self.d_model = d_model
        self._sub_goal_every = max(1, int(sub_goal_every))

        # Encoder
        if use_vision_encoder:
            try:
                self.encoder = VisionEncoder(d_model=d_model, model_name=vision_model_name)
            except (RuntimeError, ValueError):
                self.encoder = CNNEncoder(obs_shape, d_model=d_model)
        else:
            self.encoder = CNNEncoder(obs_shape, d_model=d_model)

        # Hybrid backbone
        swa_window = max(2, int(swa_window))
        ttt_mini_batch = max(1, min(int(ttt_mini_batch), swa_window))
        self.backbone = HybridBackbone(
            d_model=d_model, n_layers=int(n_layers), vocab_size=0,
            n_heads=int(n_heads), swa_window_size=swa_window,
            ttt_mini_batch=ttt_mini_batch, max_seq_len=4096,
            ffn_hidden_mult=int(ffn_hidden_mult), dropout=float(dropout),
        )

        # Hierarchical heads
        self.sub_goal_head = SubGoalHead(d_model=d_model)
        self.action_head = GoalConditionedActionHead(d_model=d_model, num_actions=num_actions)

        # Cached sub-goal (regenerated every N steps)
        self.register_buffer("_cached_sub_goal", torch.zeros(d_model), persistent=False)
        self._step_in_goal = 0

    def forward(self, obs_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Uses cached sub-goal if within refresh window."""
        feats = self.encoder(obs_u8)  # (B, d_model)
        seq = feats.unsqueeze(1)       # (B, 1, d_model)
        seq_out = self.backbone(seq)    # (B, 1, d_model)
        h = seq_out.squeeze(1)         # (B, d_model)

        # Regenerate sub-goal if it's time
        if self._step_in_goal == 0:
            with torch.no_grad():
                new_goal = self.sub_goal_head(h.mean(dim=0, keepdim=True))
                self._cached_sub_goal.copy_(new_goal.squeeze(0))
        self._step_in_goal = (self._step_in_goal + 1) % self._sub_goal_every

        # Use cached sub-goal for all batch elements
        sub_goal = self._cached_sub_goal.unsqueeze(0).expand(h.shape[0], -1)
        return self.action_head(h, sub_goal)

    def compute_sub_goal_loss(
        self,
        obs_current: torch.Tensor,
        obs_future: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised loss: predict future hidden state.

        Args:
            obs_current: (B, H, W, C) observations at time t.
            obs_future: (B, H, W, C) observations at time t+k.
        """
        with torch.no_grad():
            h_future = self._encode_to_hidden(obs_future)

        h_current = self._encode_to_hidden(obs_current)
        return self.sub_goal_head.auxiliary_loss(h_current, h_future)

    def _encode_to_hidden(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Encode observation to hidden state (no sub-goal, no action)."""
        feats = self.encoder(obs_u8)
        seq = feats.unsqueeze(1)
        seq_out = self.backbone(seq)
        return seq_out.squeeze(1)

    def get_sub_goal(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Get the current sub-goal for a given observation."""
        h = self._encode_to_hidden(obs_u8)
        return self.sub_goal_head(h)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, sub_goal_every={self._sub_goal_every}"
        )
