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
        proprio_dim: int = 0,  # Stage 20f: proprio (incl. occluder slots) -> policy
        belief_slots: int = 0,  # Stage 20s: internal object-permanence belief head
        use_gru: bool = False,  # Stage 20v: recurrent temporal memory
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
        self.proprio_dim = int(proprio_dim)
        self.belief_slots = int(belief_slots)

        # Stage 19: symbol-bias callback (set by train.py after the
        # NarrativeLoopController is created). Returns a (num_actions,)
        # logit bias tensor or None. Keeps the model decoupled from the
        # narrative module (no circular import, no ckpt pollution).
        self._symbol_bias_fn: "Any | None" = None

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

        # Stage 20f: proprio injection — the occluder memory slots (last_known
        # offsets) live in the proprio vector; without this the policy never
        # sees them (model input was image-only). MLP maps proprio (pos, vel,
        # touch, joints, yaw, grasp, slots) into a residual on h, so both the
        # manager (sub-goal) and worker (action) see the tracking target.
        if self.proprio_dim > 0:
            self.proprio_mlp = nn.Sequential(
                nn.Linear(self.proprio_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.proprio_mlp = None

        # Stage 20s: internal object-permanence belief head. Predicts the
        # occluded-object last_known offset from the policy hidden state h so
        # the agent can recall position WITHOUT the env proprio slot cue. The
        # training loop fades out the env slot (occluder_slot_fade_*), forcing
        # the belief to be internalized. Output = (dx,dy,dist)/4 per slot.
        if self.belief_slots > 0:
            _bout = 3 * self.belief_slots
            self.belief_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, _bout),
            )
        else:
            self.belief_head = None

        # Stage 20v: GRU recurrent layer for temporal memory. The pure
        # feedforward policy (CNN→transformer→MLP) cannot track objects
        # across frames — it sees only the current observation. GRU adds
        # a recurrent state that implicitly maintains a belief about the
        # environment's state (including hidden object positions) across
        # steps. Applied AFTER backbone, BEFORE proprio injection so the
        # belief_head also benefits from temporal context.
        self.use_gru = bool(use_gru)
        if self.use_gru:
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            self._gru_state: torch.Tensor | None = None  # (1, B, d_model)
        else:
            self.gru = None
            self._gru_state = None

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
        self._last_belief: torch.Tensor | None = None  # 20t: belief from h_raw
        # Stage 19-FiLM: optional narration→hidden modulation hook (set by
        # train.py from ThoughtActionLoop.modulate). In the graph on purpose
        # so the FiLM projection learns from the PPO objective.
        self._film_fn: "Any | None" = None

    def forward(
        self, obs_u8: torch.Tensor, return_hidden: bool = False,
        skill_delta: "Any | None" = None,
        proprio: "torch.Tensor | None" = None,
        proprio_hook: "Any | None" = None,
        update_gru: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (action_logits, worker_value).

        When ``return_hidden=True``, returns (logits, worker_value, hidden).
        ``proprio``: (B, proprio_dim) float vector; None-safe (no injection
        when absent, e.g. off-policy callbacks that have no env step).
        ``proprio_hook``: optional callable ``(proprio, belief) -> proprio``
        applied after belief computation but before proprio injection.
        Stage 20t uses this to blend belief output into the proprio slot.
        ``update_gru``: False for non-rollout calls (skill retrieval, PPO
        mini-batch, off-policy) so the persistent GRU state is not advanced
        by auxiliary forward passes (Stage 20v).
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

        # Stage 20v: GRU temporal memory. Applied after backbone, before
        # belief/proprio so both benefit from temporal context. During
        # rollout the GRU state is carried forward (detached); during PPO
        # update the stored state is passed via _gru_state_override.
        if self.gru is not None:
            _state = getattr(self, '_gru_state_override', None)
            if _state is None:
                _state = self._gru_state
            if _state is None:
                _state = torch.zeros(
                    1, h.shape[0], self.d_model, dtype=torch.float32, device=h.device)
            elif _state.device != h.device:
                _state = _state.to(h.device)
            elif _state.shape[1] != h.shape[0]:
                # batch-size change (e.g. PPO mini-batch vs rollout single).
                # Slice the BATCH dim (dim 1), not the num_layers dim (dim 0
                # is always 1, so [:1] was a no-op and expand crashed when
                # shrinking 32->1). Stage 20hyp fix.
                _state = _state[:, :1].expand(1, h.shape[0], -1).contiguous()
            h_seq = h.unsqueeze(1)  # (B, 1, d_model)
            h_gru, new_state = self.gru(h_seq, _state)
            h = h_gru.squeeze(1)  # (B, d_model)
            # During rollout (no_grad + update_gru), update the persistent state.
            # During PPO (grad enabled), the override is consumed and cleared.
            if getattr(self, '_gru_state_override', None) is not None:
                self._gru_state_override = None  # consumed
            elif update_gru and not torch.is_grad_enabled():
                self._gru_state = new_state.detach()

        # Stage 19-FiLM: narration-conditioned modulation of the hidden state
        # (identity when no thought is cached). Kept differentiable so the
        # FiLM projection receives PPO gradients.
        if self._film_fn is not None:
            try:
                h = self._film_fn(h)
            except Exception:
                pass  # legit: never break the forward pass on narration failure

        # Stage 20t: compute belief from h_raw (BEFORE proprio injection) so
        # the belief head cannot cheat by reading the env slot cue. The
        # belief output is then blended into proprio via proprio_hook.
        if self.belief_head is not None:
            self._last_belief = self.belief_head(h)
        else:
            self._last_belief = None

        # 20t: allow training loop to modify proprio (e.g. blend belief)
        if proprio_hook is not None:
            proprio = proprio_hook(proprio, self._last_belief)

        # Stage 20f: proprio residual injection BEFORE manager/worker so the
        # sub-goal can be conditioned on the target as well. no_grad-free by
        # design (part of the differentiable policy path); must be a plain
        # add (no inplace) to keep autograd graphs valid.
        if proprio is not None and self.proprio_mlp is not None:
            h = h + self.proprio_mlp(proprio.float())

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
            if skill_delta.A.device != h.device:
                skill_delta = skill_delta.to(h.device)
            h = h + skill_delta.apply(h)

        action_logits, worker_value = self.worker(h, sg)

        # Stage 19: apply kanren symbol bias (detached, no grad)
        if self._symbol_bias_fn is not None:
            try:
                _bias = self._symbol_bias_fn()
                if _bias is not None:
                    action_logits = action_logits + _bias.to(action_logits.device)
            except Exception as _sb:
                # Never let a broken bias hook break the forward pass
                pass

        if return_hidden:
            return action_logits, worker_value, h
        return action_logits, worker_value

    def reset_gru_state(self, batch_size: int = 1, device: "torch.device | None" = None) -> None:
        """Reset GRU hidden state (call at episode boundary)."""
        if self.gru is not None:
            dev = device or self._cached_sub_goal.device
            self._gru_state = torch.zeros(1, batch_size, self.d_model, device=dev)

    def get_gru_state(self) -> "torch.Tensor | None":
        """Return current GRU state (for buffer storage during rollout)."""
        if self._gru_state is not None:
            return self._gru_state.detach()
        return None

    @property
    def manager_value(self) -> torch.Tensor:
        """Last manager value estimate (for manager buffer)."""
        if self._last_manager_value is None:
            return torch.zeros(1, device=self._cached_sub_goal.device)
        return self._last_manager_value.detach()

    def set_symbol_bias_fn(self, fn: "Any | None") -> None:
        """Attach the Stage 19 symbol-bias callback (from train.py)."""
        self._symbol_bias_fn = fn

    def set_film_fn(self, fn: "Any | None") -> None:
        """Attach the Stage 19-FiLM narration hook h -> h' (from train.py).

        Called on the hidden state every forward; identity when no thought
        is cached. Differentiable (FiLM projection learns from PPO).
        """
        self._film_fn = fn

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
