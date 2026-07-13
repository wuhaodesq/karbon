"""Model Growth v2 — Autonomous Architecture Expansion During Development.

The ModelGrower v1 (model_growth.py) only recorded growth events but didn't
perform the actual architecture surgery. v2 implements autonomous growth:

1. When should_grow() triggers → the grower actually adds layers to the backbone
2. KnowledgeDistiller transfers knowledge from old→new network
3. Growth budgets are phased by developmental clock

Key design choice:
    Growth is NOT "just add layers". It's a surgery that must:
    a. Expand the backbone (more TTT-Hybrid blocks)
    b. Distill old model knowledge into the larger model
    c. Adjust optimizer state to match new parameter size
    d. Maintain Bounded axioms (capacity declared before growth)

模型生长 v2：自主在发育过程中添加骨干层并蒸馏知识。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GrowthConfigV2:
    """Configuration for autonomous model growth.

    - initial_layers: starting number of TTT-Hybrid blocks
    - max_layers: hard cap (Axiom 1)
    - min_steps_between_growths: prevents rapid oscillation
    - distill_steps: how many SGD steps to distill after growth
    - distill_lr: learning rate for distillation
    """
    initial_layers: int = 2
    max_layers: int = 20
    min_steps_between_growths: int = 100_000
    distill_steps: int = 256
    distill_lr: float = 1e-3
    grow_trigger_lp_threshold: float = 0.05
    grow_trigger_coverage: float = 0.3


def _carry_over_adam_momentum(
    old_optimizer: "torch.optim.Optimizer",
    new_optimizer: "torch.optim.Optimizer",
    old_model: "nn.Module",
    new_model: "nn.Module",
) -> None:
    """Preserve Adam momentum (exp_avg / exp_avg_sq / step) for every
    parameter whose name matches between the old and new models.

    Newly-added (non-matching) parameters — the randomly-initialized
    growth layers — keep fresh optimizer slots and re-learn from scratch.
    This is what stops growth from wiping the policy optimizer's accumulated
    state (which would otherwise re-set learning after every expansion).
    """
    old_state = old_optimizer.state_dict()
    new_state = new_optimizer.state_dict()
    old_names = {
        n: i
        for i, (n, p) in enumerate(
            (nm, p) for nm, p in old_model.named_parameters() if p.requires_grad
        )
    }
    new_names = {
        n: i
        for i, (n, p) in enumerate(
            (nm, p) for nm, p in new_model.named_parameters() if p.requires_grad
        )
    }
    for name, old_idx in old_names.items():
        if name not in new_names:
            continue
        new_idx = new_names[name]
        for state_key in ("exp_avg", "exp_avg_sq", "step"):
            # NOTE: do NOT gate on ``new_idx < len(new_state["state"])``.
            # The new optimizer is fresh, so its ``state`` is empty; gating
            # on its length (``new_idx < 0``) makes the copy a silent no-op
            # — which is exactly why growth used to wipe the policy optimizer's
            # accumulated momentum. We instead create the entry on the
            # fresh optimizer and copy the old Adam slots into it.
            if (
                old_idx < len(old_state["state"])
                and state_key in old_state["state"].get(old_idx, {})
            ):
                if new_idx not in new_state["state"]:
                    new_state["state"][new_idx] = {}
                new_state["state"][new_idx][state_key] = (
                    old_state["state"][old_idx][state_key].clone()
                )
    new_optimizer.load_state_dict(new_state)


class ModelGrowerV2(nn.Module):
    """Autonomous model growth with knowledge distillation.

    When triggered, this module:
    1. Freezes the current model (becomes the "teacher")
    2. Creates a larger model with +n_layers_to_add backbone blocks
    3. Runs knowledge distillation (teacher→student) for distill_steps
    4. Replaces the trainable model with the distilled larger version
    5. Expands optimizer state to match new parameters

    Bounded: max_layers and max_params are declared at construction.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        swa_window: int = 8,
        ttt_mini_batch: int = 4,
        ffn_hidden_mult: int = 4,
        config: GrowthConfigV2 | None = None,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._swa_window = swa_window
        self._ttt_mini_batch = ttt_mini_batch
        self._ffn_hidden_mult = ffn_hidden_mult
        self._cfg = config or GrowthConfigV2()

        self._current_layers = self._cfg.initial_layers
        self._last_growth_step = -self._cfg.min_steps_between_growths
        self._growth_count = 0
        self._growth_history: list[dict] = []

    @property
    def capacity(self) -> int:
        """Max number of layers this grower can produce."""
        return self._cfg.max_layers

    def __len__(self) -> int:
        return self._current_layers

    def should_grow(
        self,
        step: int,
        learning_progress: float,
        coverage_ratio: float,
    ) -> bool:
        """Check if growth should trigger."""
        if self._current_layers >= self._cfg.max_layers:
            return False
        if step - self._last_growth_step < self._cfg.min_steps_between_growths:
            return False
        if coverage_ratio < self._cfg.grow_trigger_coverage:
            return False
        if learning_progress > self._cfg.grow_trigger_lp_threshold:
            return False
        return True

    def grow(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        n_layers_to_add: int = 1,
    ) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
        """Perform architecture surgery and return new (model, optimizer).

        Steps:
        1. Freeze old model as teacher
        2. Create larger model
        3. Distill teacher knowledge into student
        4. Return new model + optimizer

        Args:
            model: current HybridActorCritic model.
            optimizer: current Adam optimizer.
            step: global step count.
            n_layers_to_add: how many backbone blocks to add.

        Returns:
            (new_model, new_optimizer, growth_record_dict)
        """
        from copy import deepcopy

        old_layers = self._current_layers
        new_layers = min(old_layers + n_layers_to_add, self._cfg.max_layers)

        logger.info(
            "ModelGrower: growing from %d → %d layers (step %d)",
            old_layers, new_layers, step,
        )

        # 1. Clone old model as teacher (frozen)
        teacher = deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # 2. Create new model with more layers
        new_model = self._create_larger_model(model, new_layers)
        new_model.train()

        # 3. Knowledge distillation
        self._distill(teacher, new_model, optimizer)

        # 4. Create new optimizer for expanded parameters
        new_optimizer = torch.optim.Adam(
            [p for p in new_model.parameters() if p.requires_grad],
            lr=optimizer.param_groups[0]['lr'],
        )
        # P2: preserve old Adam momentum for matching parameters (so growth
        # does not wipe the policy optimizer's accumulated state).
        _carry_over_adam_momentum(optimizer, new_optimizer, model, new_model)

        self._current_layers = new_layers
        self._last_growth_step = step
        self._growth_count += 1

        record = {
            "step": step,
            "old_layers": old_layers,
            "new_layers": new_layers,
            "growth_count": self._growth_count,
        }
        self._growth_history.append(record)

        return new_model, new_optimizer, record

    def _create_larger_model(self, model: nn.Module, new_n_layers: int) -> nn.Module:
        """Create a new HybridActorCritic with more backbone layers.

        Copies encoder and heads from old model, creates larger backbone.
        """
        from src.train import HybridActorCritic, HybridBackbone

        device = next(model.parameters()).device
        num_actions = model.policy_head.out_features if hasattr(model.policy_head, 'out_features') else 7
        obs_shape = getattr(model, "obs_shape", (64, 64, 3))  # P3: carry real obs shape

        new_model = HybridActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            d_model=self._d_model,
            n_layers=new_n_layers,
            n_heads=self._n_heads,
            swa_window=self._swa_window,
            ttt_mini_batch=self._ttt_mini_batch,
            ffn_hidden_mult=self._ffn_hidden_mult,
            use_slot_attention=getattr(model, 'use_slots', False),
            slot_num_slots=getattr(model.encoder, 'num_slots', 7) if hasattr(model, 'encoder') else 7,
        ).to(device)

        # Copy encoder weights (unchanged)
        if hasattr(model, 'encoder') and hasattr(new_model, 'encoder'):
            new_model.encoder.load_state_dict(model.encoder.state_dict(), strict=False)

        # Copy early backbone layers (same as before)
        state_old = model.backbone.state_dict()
        state_new = new_model.backbone.state_dict()
        for k in state_new:
            if k in state_old and state_new[k].shape == state_old[k].shape:
                state_new[k] = state_old[k].clone()
        new_model.backbone.load_state_dict(state_new, strict=False)

        # Copy heads (unchanged)
        new_model.policy_head.load_state_dict(model.policy_head.state_dict())
        new_model.value_head.load_state_dict(model.value_head.state_dict())

        return new_model

    def _distill(
        self,
        teacher: nn.Module,
        student: nn.Module,
        _optimizer: torch.optim.Optimizer,  # for learning rate reference
        batch_size: int = 16,
    ) -> None:
        """Knowledge distillation: student mimics teacher on synthetic data.

        Uses random input + MSE between teacher and student outputs.
        Bounded: exactly distill_steps SGD steps.
        """
        device = next(student.parameters()).device
        opt = torch.optim.Adam(
            [p for p in student.parameters() if p.requires_grad],
            lr=self._cfg.distill_lr,
        )

        for _ in range(self._cfg.distill_steps):
            # Random observations (simulate diverse inputs)
            x = torch.randint(0, 256, (batch_size, 64, 64, 3),
                             dtype=torch.uint8, device=device)

            with torch.no_grad():
                t_logits, t_values = teacher(x)
            s_logits, s_values = student(x)

            # P2: preserve the teacher's POLICY distribution (KL on softmax),
            # not just raw logit magnitude, so the grown model keeps the
            # agent's learned action preferences instead of resetting them.
            # This is what makes growth non-disruptive to an otherwise-stuck
            # policy.
            kl = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                0.5 * F.mse_loss(s_logits, t_logits)
                + 0.5 * F.mse_loss(s_values, t_values)
                + 1.0 * kl
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    def summary(self) -> dict:
        return {
            "current_layers": self._current_layers,
            "max_layers": self._cfg.max_layers,
            "growth_count": self._growth_count,
            "can_grow": self._current_layers < self._cfg.max_layers,
            "history": self._growth_history[-5:],
        }

    def state_dict(self) -> dict:
        return {
            "current_layers": self._current_layers,
            "last_growth_step": self._last_growth_step,
            "growth_count": self._growth_count,
            "growth_history": self._growth_history,
            "config": {
                "initial_layers": self._cfg.initial_layers,
                "max_layers": self._cfg.max_layers,
                "min_steps_between_growths": self._cfg.min_steps_between_growths,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_layers = int(state["current_layers"])
        self._last_growth_step = int(state["last_growth_step"])
        self._growth_count = int(state["growth_count"])
        self._growth_history = state["growth_history"]
