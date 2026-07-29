"""Hierarchical RL: sub-goal generation + goal-conditioned action.

Two-level architecture for Stage 11+:

    Manager (high-level): outputs sub-goal every K steps, receives env reward.
    Worker  (low-level):  outputs action every step, receives goal-progress reward.

This separates navigation from planning, solving the seesaw problem where
training one overwrites the other.

Architecture:

    obs → encoder → backbone → hidden h
                                  ↓
                     ┌────────────┴────────────┐
                     ↓                         ↓
             Manager head                Worker head
             (sub-goal + M-value)         (FiLM(h, g) → action + W-value)
                     ↓                         ↓
             sub-goal g                  action logits
             (cached K steps)            + worker value

Training signals:
  - Worker: env_reward + intrinsic_reward (-||h_t - g||²) → PPO
  - Manager: accumulated env_reward over K steps → separate PPO
  - Auxiliary: sub-goal head predicts future hidden state (self-supervised)

Bounded (Axiom 1): all components fixed-size, capacity declared at init.
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
# Sub-goal head (Manager's policy output)
# =====================================================================


class SubGoalHead(nn.Module):
    """Generates a sub-goal vector from the current hidden state.

    The sub-goal represents "where the agent should be in a few steps".
    Trained via both:
    1. Self-supervised loss: predict future hidden state
    2. Manager PPO: sub-goals that lead to high env reward get reinforced
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
# Manager head (high-level value + sub-goal)
# =====================================================================


class ManagerHead(nn.Module):
    """Manager: produces sub-goal vector and estimates value (env return).

    The manager operates at a lower temporal resolution (every K steps).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.sub_goal = SubGoalHead(d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self, hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sub_goal, manager_value)."""
        sg = self.sub_goal(hidden_state)
        v = self.value_head(hidden_state).squeeze(-1)
        return sg, v


# =====================================================================
# Goal-conditioned action head (Worker)
# =====================================================================


class GoalConditionedActionHead(nn.Module):
    """Worker: action head conditioned on a sub-goal via FiLM.

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
        """Returns (action_logits, worker_value) conditioned on the sub-goal."""
        conditioned = self.film(hidden_state, sub_goal)
        return self.action_head(conditioned), self.value_head(conditioned).squeeze(-1)


# =====================================================================
# Full hierarchical actor-critic
# =====================================================================


class HierarchicalActorCritic(nn.Module):
    """Two-level policy: Manager (sub-goals) + Worker (actions).

    Pipeline:
        obs → encoder → Hybrid backbone → hidden h
        h → ManagerHead → sub-goal g + manager_value
        (h, g) → GoalConditionedActionHead → action_logits + worker_value

    The sub-goal is regenerated every ``sub_goal_every`` steps; between
    regenerations, the same sub-goal is reused (temporal abstraction).

    Forward returns ``(logits, worker_value)`` — same interface as
    ``HybridActorCritic`` for backward compat with eval scripts.
    Manager value is accessible via ``last_manager_value``.

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
        sub_goal_every: int = 10,
        use_slot_attention: bool = False,
        slot_num_slots: int = 7,
        slot_dim: int = 128,
        slot_num_iterations: int = 3,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        if d_model % 2 != 0:
            d_model += 1
        self.d_model = d_model
        self._sub_goal_every = max(1, int(sub_goal_every))
        self.num_actions = num_actions
        self.obs_shape = tuple(obs_shape)

        # Encoder (reuse the same encoder variants as HybridActorCritic)
        self.use_slots = use_slot_attention
        self.use_vision = use_vision_encoder
        if use_slot_attention:
            if slot_dim != d_model:
                raise ValueError(
                    f"SlotAttention requires slot_dim == d_model, got "
                    f"slot_dim={slot_dim} d_model={d_model}."
                )
            from src.models.slot_attention import SlotAttention
            self.encoder = SlotAttention(
                d_model=d_model,
                num_slots=slot_num_slots,
                slot_dim=slot_dim,
                num_iterations=slot_num_iterations,
            )
        elif use_vision_encoder:
            try:
                self.encoder = VisionEncoder(
                    d_model=d_model,
                    model_name=vision_model_name,
                    freeze=True,
                )
            except (RuntimeError, ValueError):
                self.encoder = CNNEncoder(obs_shape, d_model=d_model)
                self.use_vision = False
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

        # Manager head (sub-goal + manager value)
        self.manager = ManagerHead(d_model=d_model)

        # Worker head (action + worker value, FiLM-conditioned)
        self.worker = GoalConditionedActionHead(d_model=d_model, num_actions=num_actions)

        # Cached sub-goal (regenerated every N steps)
        # Note: plain tensor attr avoids copy_() inplace version conflicts.
        # Persisted via custom state_dict/load_state_dict overrides.
        self._cached_sub_goal = torch.zeros(d_model, dtype=torch.float32)
        self._step_in_goal = 0

        # Last outputs stored for access by training loop
        self._last_hidden: torch.Tensor | None = None
        self._last_manager_value: torch.Tensor | None = None
        self._last_sub_goal: torch.Tensor | None = None
        self._last_slots: torch.Tensor | None = None

    def forward(
        self, obs_u8: torch.Tensor, return_hidden: bool = False,
        skill_delta: "Any | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (action_logits, worker_value).

        When ``return_hidden=True``, returns (logits, worker_value, hidden).
        """
        # Encode
        if self.use_slots:
            seq = self.encoder(obs_u8)  # (B, num_slots, d_model)
        elif self.use_vision:
            feats = self.encoder(obs_u8)
            seq = feats.unsqueeze(1)
        else:
            feats = self.encoder(obs_u8)  # CNNEncoder handles permute internally
            seq = feats.unsqueeze(1)

        seq_out = self.backbone(seq)
        self._last_slots = seq

        if self.use_slots:
            h = seq_out.mean(dim=1)  # (B, d_model)
        else:
            h = seq_out.squeeze(1)  # (B, d_model)
        self._last_hidden = h

        # Manager: regenerate sub-goal at period boundary
        # Device guard: ensure cached sub-goal is on the same device as h
        if self._cached_sub_goal.device != h.device:
            self._cached_sub_goal = self._cached_sub_goal.to(h.device)
        if self._step_in_goal == 0:
            sg, mgr_v = self.manager(h)
            self._cached_sub_goal = sg.mean(dim=0).detach()  # new tensor, no inplace
            self._last_manager_value = mgr_v
            self._last_sub_goal = sg
        else:
            with torch.no_grad():
                sg = self._cached_sub_goal.unsqueeze(0).expand(h.shape[0], -1)
                _, mgr_v = self.manager(h)  # no grad needed outside period
            self._last_manager_value = mgr_v
            self._last_sub_goal = sg

        self._step_in_goal = (self._step_in_goal + 1) % self._sub_goal_every

        # Worker: M2 skill-injection residual (optional)
        if skill_delta is not None:
            h = h + skill_delta.apply(h)

        action_logits, worker_value = self.worker(h, sg)

        if return_hidden:
            return action_logits, worker_value, h
        return action_logits, worker_value

    @property
    def manager_value(self) -> torch.Tensor:
        """Last manager value estimate (for manager buffer)."""
        if self._last_manager_value is None:
            return torch.zeros(1, device=self._cached_sub_goal.device)
        return self._last_manager_value.detach()

    @property
    def current_sub_goal(self) -> torch.Tensor:
        """Current cached sub-goal vector."""
        return self._cached_sub_goal.detach()

    def compute_intrinsic_reward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Worker intrinsic reward: negative distance to sub-goal in latent space.

        Args:
            obs_u8: (B, H, W, C) uint8 observations.

        Returns:
            (B,) intrinsic rewards — higher when closer to sub-goal.
        """
        with torch.no_grad():
            h = self._encode_to_hidden(obs_u8)
            sg = self._cached_sub_goal.unsqueeze(0).expand(h.shape[0], -1)
            # Negative MSE: [-inf, 0], higher = closer to goal
            return -F.mse_loss(h, sg, reduction='none').mean(dim=-1)

    def _encode_to_hidden(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Encode observation to hidden state (no sub-goal, no action)."""
        if self.use_slots:
            seq = self.encoder(obs_u8)
        elif self.use_vision:
            feats = self.encoder(obs_u8)
            seq = feats.unsqueeze(1)
        else:
            feats = self.encoder(obs_u8)  # CNNEncoder handles permute internally
            seq = feats.unsqueeze(1)
        seq_out = self.backbone(seq)
        if self.use_slots:
            return seq_out.mean(dim=1)
        return seq_out.squeeze(1)

    def get_sub_goal(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Get the current sub-goal for a given observation."""
        h = self._encode_to_hidden(obs_u8)
        sg, _ = self.manager(h)
        return sg

    def compute_sub_goal_loss(
        self,
        obs_current: torch.Tensor,
        obs_future: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised loss: predict future hidden state."""
        with torch.no_grad():
            h_future = self._encode_to_hidden(obs_future)
        h_current = self._encode_to_hidden(obs_current)
        return self.manager.sub_goal.auxiliary_loss(h_current, h_future)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, sub_goal_every={self._sub_goal_every}, "
            f"use_slots={self.use_slots}"
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["_cached_sub_goal"] = self._cached_sub_goal
        sd["_step_in_goal"] = torch.tensor(self._step_in_goal, dtype=torch.long)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        sd = dict(state_dict)
        if "_cached_sub_goal" in sd:
            self._cached_sub_goal = sd.pop("_cached_sub_goal")
        if "_step_in_goal" in sd:
            self._step_in_goal = int(sd.pop("_step_in_goal").item())
        return super().load_state_dict(sd, strict=strict)
