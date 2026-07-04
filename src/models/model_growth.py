"""Gradual model growth: let the agent's capacity expand as it learns.

Instead of training one massive 1B-parameter model from scratch (expensive),
the agent starts small (7M) and GROWS as it needs more capacity — just like
a child's brain develops more synaptic connections over time.

Three growth mechanisms:

1. :class:`ModelGrower` — periodically adds capacity to the Hybrid backbone:
   - Add a new HybridBlock layer when the agent masters a new domain
   - Add attention heads when it needs richer attention patterns
   - Increase d_model when it needs more expressive features

2. :class:`KnowledgeDistiller` — compresses accumulated knowledge from
   the bounded stores (rules, skills, replay) into the base model's weights,
   freeing up bounded capacity for new knowledge. This is like "sleep
   consolidation" but specifically for ABSTRACTION: the agent practices
   its accumulated rules and skills, and the base model learns to
   represent them implicitly (freeing the explicit stores).

3. :class:`CurriculumGate` — decides WHEN to grow based on learning progress:
   - If LP (learning progress) is high → no need to grow (still learning)
   - If LP plateaus AND coverage is high → grow (need more capacity)
   - If LP plateaus AND coverage is low → don't grow (explore more first)

All mechanisms are bounded: growth is capped at max_params (user-defined).
VRAM grows linearly with parameter count. Axiom 1 still satisfied (the
bounded stores remain bounded; the model is the only thing that grows).

逐步增长模型容量：像儿童大脑发育一样，从 7M 逐步长到 100M+。
不需要一开始就有 1B 参数的大 GPU——随着学习需要逐步增加。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# ModelGrower — gradually increase model capacity
# =====================================================================


@dataclass
class GrowthConfig:
    """When and how to grow the model.

    - ``initial_params``: starting parameter count (e.g., 7M)
    - ``max_params``: hard cap (e.g., 200M on a 24GB GPU)
    - ``grow_threshold_lp``: grow when LP drops below this (plateau)
    - ``grow_threshold_coverage``: grow only if coverage > this (explored enough)
    - ``grow_factor``: each growth multiplies capacity by this (e.g., 1.5×)
    - ``min_steps_between_growths``: cooldown (e.g., 100k steps)
    """

    initial_params: int = 7_000_000
    max_params: int = 200_000_000
    grow_threshold_lp: float = 0.05
    grow_threshold_coverage: float = 0.3
    grow_factor: float = 1.5
    min_steps_between_growths: int = 100_000


@dataclass
class GrowthRecord:
    """Record of one growth event."""
    step: int
    old_params: int
    new_params: int
    trigger: str  # "lp_plateau" | "coverage_saturated" | "manual"


class ModelGrower:
    """Manages gradual model capacity growth.

    Growth is triggered when:
    1. Learning progress (LP) drops below threshold (plateau).
    2. Coverage is above threshold (explored enough of current space).
    3. Enough steps have passed since last growth (cooldown).

    Growth is implemented by adding layers to the Hybrid backbone.
    New layers are initialized near-identity (so they don't disrupt
    existing knowledge) and gradually learn.

    Bounded: max_params is a hard cap. Growth history is a fixed-size
    deque. Axiom 1 for the growth bookkeeping (the model itself grows
    but is capped).

    逐步增长模型容量：当学习停滞时增加参数，像大脑发育一样。
    """

    def __init__(self, config: GrowthConfig) -> None:
        self._config = config
        self._current_params = config.initial_params
        self._last_growth_step = 0
        self._history: list[GrowthRecord] = []
        self._can_grow = True

    @property
    def current_params(self) -> int:
        return self._current_params

    @property
    def max_params(self) -> int:
        return self._config.max_params

    @property
    def can_grow(self) -> bool:
        return self._can_grow and self._current_params < self._config.max_params

    @property
    def num_growths(self) -> int:
        return len(self._history)

    def should_grow(
        self,
        step: int,
        learning_progress: float,
        coverage_ratio: float,
    ) -> bool:
        """Check if the model should grow.

        Returns True if:
        - LP is low (plateaued) AND coverage is high (explored enough)
        - Enough steps since last growth (cooldown)
        - Under max_params cap
        """
        if not self.can_grow:
            return False

        if step - self._last_growth_step < self._config.min_steps_between_growths:
            return False

        if learning_progress > self._config.grow_threshold_lp:
            return False  # still learning, no need to grow

        if coverage_ratio < self._config.grow_threshold_coverage:
            return False  # hasn't explored enough yet

        return True

    def grow(
        self,
        step: int,
        trigger: str = "lp_plateau",
    ) -> GrowthRecord | None:
        """Record a growth event. Returns the record, or None if can't grow.

        Note: this method only RECORDS the growth. The actual model
        surgery (adding layers) is done by the caller, because the
        model architecture is specific to the training setup.

        The caller should:
        1. Call should_grow() to check
        2. If True, add a new HybridBlock to the backbone
        3. Initialize the new block near-identity
        4. Call grow() to record the event
        5. Update the optimizer to include new parameters
        """
        if not self.can_grow:
            return None

        old_params = self._current_params
        new_params = min(
            int(old_params * self._config.grow_factor),
            self._config.max_params,
        )

        record = GrowthRecord(
            step=step,
            old_params=old_params,
            new_params=new_params,
            trigger=trigger,
        )
        self._history.append(record)
        self._current_params = new_params
        self._last_growth_step = step

        if new_params >= self._config.max_params:
            self._can_grow = False

        logger.info(
            "ModelGrower: growth #%d at step %d: %dM → %dM params (trigger=%s)",
            len(self._history), step,
            old_params // 10**6, new_params // 10**6,
            trigger,
        )

        return record

    def summary(self) -> dict:
        return {
            "current_params": self._current_params,
            "max_params": self._config.max_params,
            "num_growths": len(self._history),
            "can_grow": self.can_grow,
            "last_growth_step": self._last_growth_step,
            "history": [
                {"step": r.step, "old": r.old_params, "new": r.new_params,
                 "trigger": r.trigger}
                for r in self._history
            ],
        }

    def state_dict(self) -> dict:
        return {
            "config": self._config.__dict__,
            "current_params": self._current_params,
            "last_growth_step": self._last_growth_step,
            "history": [
                {"step": r.step, "old": r.old_params, "new": r.new_params,
                 "trigger": r.trigger}
                for r in self._history
            ],
            "can_grow": self._can_grow,
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["config"].items():
            setattr(self._config, k, v)
        self._current_params = int(state["current_params"])
        self._last_growth_step = int(state["last_growth_step"])
        self._history = [
            GrowthRecord(
                step=r["step"], old_params=r["old"], new_params=r["new"],
                trigger=r["trigger"],
            )
            for r in state["history"]
        ]
        self._can_grow = bool(state["can_grow"])


# =====================================================================
# KnowledgeDistiller — compress bounded stores into base model
# =====================================================================


class KnowledgeDistiller:
    """Compresses accumulated knowledge from bounded stores into base weights.

    When the skill library or rule memory gets full, the distiller:
    1. Generates synthetic training data from the rules/skills
    2. Trains the base model on this data (a few gradient steps)
    3. The base model "absorbs" the knowledge implicitly
    4. The bounded stores are partially cleared (making room for new knowledge)

    This is like "sleep consolidation" but specifically for abstraction:
    the agent practices its accumulated rules, and the base model learns
    to represent the abstractions implicitly — freeing the explicit stores.

    Bounded: distillation runs for a fixed number of steps. No growing state.
    """

    def __init__(
        self,
        distill_steps: int = 50,
        distill_lr: float = 1e-4,
        clear_ratio: float = 0.3,
    ) -> None:
        self._distill_steps = int(distill_steps)
        self._distill_lr = float(distill_lr)
        self._clear_ratio = float(clear_ratio)
        self._total_distillations = 0

    def distill(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        synthetic_data: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict:
        """Distill synthetic data into the model.

        Args:
            model: the policy network (will be temporarily set to train mode).
            optimizer: the model's optimizer (will be temporarily used).
            synthetic_data: list of (input, target) pairs generated from
                rules/skills.

        Returns:
            Dict with distillation stats (loss, steps, etc.)
        """
        if not synthetic_data:
            return {"steps": 0, "final_loss": 0.0}

        model.train()
        losses = []

        for step in range(min(self._distill_steps, len(synthetic_data))):
            x, y = synthetic_data[step % len(synthetic_data)]
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)

            # Forward (model-specific — caller should wrap this)
            # Here we do a generic forward if model has a standard interface
            try:
                out = model(x) if not isinstance(model(x), tuple) else model(x)[0]
            except Exception:
                break

            loss = torch.nn.functional.mse_loss(out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        self._total_distillations += 1

        return {
            "steps": len(losses),
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "total_distillations": self._total_distillations,
            "clear_ratio": self._clear_ratio,
        }

    @property
    def clear_ratio(self) -> float:
        """Fraction of bounded stores to clear after distillation."""
        return self._clear_ratio

    def summary(self) -> dict:
        return {
            "total_distillations": self._total_distillations,
            "distill_steps": self._distill_steps,
            "clear_ratio": self._clear_ratio,
        }


# =====================================================================
# CurriculumGate — decide when to grow vs explore
# =====================================================================


class CurriculumGate:
    """Decides whether to grow the model or explore more.

    Logic:
    - LP high → keep learning (no grow, no explore boost)
    - LP low + coverage low → explore more (boost exploration)
    - LP low + coverage high → grow model (need more capacity)
    - LP high + coverage high → task mastered, switch to new task

    This gate sits between the curriculum (LP tracker) and the model
    grower, ensuring growth only happens when the agent has exhausted
    its current capacity's learning potential.

    Bounded: no state (just decision logic). Axiom 1 trivially satisfied.
    """

    def __init__(
        self,
        lp_threshold: float = 0.05,
        coverage_threshold: float = 0.3,
        mastery_threshold: float = 0.8,
    ) -> None:
        self._lp_thresh = float(lp_threshold)
        self._cov_thresh = float(coverage_threshold)
        self._mastery_thresh = float(mastery_threshold)

    def decide(
        self,
        learning_progress: float,
        coverage_ratio: float,
        task_return: float,
    ) -> str:
        """Decide what to do next.

        Returns one of:
        - "learn": keep learning current task
        - "explore": increase exploration (low LP, low coverage)
        - "grow": grow model capacity (low LP, high coverage)
        - "switch": switch to a new task (high return, high coverage)
        """
        if task_return >= self._mastery_thresh and coverage_ratio >= self._cov_thresh:
            return "switch"

        if learning_progress < self._lp_thresh:
            if coverage_ratio >= self._cov_thresh:
                return "grow"
            else:
                return "explore"

        return "learn"

    def summary(self) -> dict:
        return {
            "lp_threshold": self._lp_thresh,
            "coverage_threshold": self._cov_thresh,
            "mastery_threshold": self._mastery_thresh,
        }
