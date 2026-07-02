"""Online Elastic Weight Consolidation (Online EWC).

Kirkpatrick et al. 2017 (original EWC); Schwarz et al. 2018 (Online EWC).

Standard EWC keeps a Fisher-diagonal matrix for *each* previous task, so
memory grows linearly with #tasks. **Online EWC** keeps a single
exponentially-weighted running Fisher and a corresponding "anchor" set of
parameters. Memory is O(#params), constant over tasks — Axiom 1 satisfied.

Update rules:

.. code-block:: text

    On end of task t:
      F_t = diag(∂L_t/∂θ)² averaged over a data batch
      F_online ← γ · F_online + F_t          (γ ∈ (0,1], typically 0.95)
      θ_anchor ← θ                            (or a running average)

    EWC penalty added to training loss on subsequent tasks:
      L_reg(θ) = (λ/2) · Σ_i F_online_i · (θ_i - θ_anchor_i)²

Bounded: F_online and θ_anchor are each 1× the model's parameter count.
No per-task storage.

Online EWC：单份 Fisher 累积 + 单份 anchor。避免 O(tasks) 增长。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OnlineEWCConfig:
    """Configuration.

    - ``lambda_reg``: overall regularization weight in the loss.
    - ``gamma``: exponential decay for the accumulated Fisher. gamma=1 means
      Fisher never forgets; gamma<1 means older tasks fade.
    - ``update_anchor_mode``: how to update the anchor when consolidating:
      ``"replace"``  → θ_anchor := θ (Schwarz-style)
      ``"ema"``      → θ_anchor := α·θ_anchor + (1-α)·θ
    - ``anchor_ema_alpha``: only used if mode="ema".
    """

    lambda_reg: float = 1.0
    gamma: float = 0.95
    update_anchor_mode: str = "replace"
    anchor_ema_alpha: float = 0.9


class OnlineEWC:
    """Bounded EWC state — one Fisher, one anchor, both O(#params).

    Owns two ``dict[str, Tensor]`` mirrors of the model parameters. Never
    grows over time (Axiom 1). Serialization supported (Axiom 6).

    Usage:

    .. code-block:: python

        ewc = OnlineEWC(model, OnlineEWCConfig())

        # ... train on task A ...
        ewc.consolidate(model, data_loader_task_a, loss_fn)

        # ... train on task B; add EWC penalty to loss:
        loss = task_loss + ewc.penalty(model)

    Bounded:
    - Fisher dict = one tensor per parameter (same shape as param).
    - Anchor dict = one tensor per parameter.
    - No history of prior Fishers / anchors kept.
    """

    def __init__(
        self,
        model: nn.Module,
        config: OnlineEWCConfig | None = None,
    ) -> None:
        self.config = config or OnlineEWCConfig()
        if self.config.update_anchor_mode not in ("replace", "ema"):
            raise ValueError(f"unknown anchor mode {self.config.update_anchor_mode}")

        self._param_names: list[str] = []
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._param_names.append(name)
            self._fisher[name] = torch.zeros_like(p, requires_grad=False)
            self._anchor[name] = p.detach().clone()

        self._has_consolidated_once = False

    # ---------------------------------------------------- consolidation

    def consolidate(
        self,
        model: nn.Module,
        data_batches: Iterable[torch.Tensor] | Iterable[tuple],
        loss_fn: Callable[[nn.Module, object], torch.Tensor],
        num_batches: int = 32,
    ) -> None:
        """Estimate the Fisher for the just-finished task and roll into ``F_online``.

        Args:
            model: the model whose current θ we consolidate around.
            data_batches: iterable yielding batches. Passed as-is to ``loss_fn``.
            loss_fn: callable ``(model, batch) -> scalar tensor``. This is the
                objective whose gradient defines the Fisher (typically the
                negative log-likelihood).
            num_batches: how many batches to average over. Larger = smoother
                Fisher estimate but slower.

        Fisher estimator (empirical, diagonal):
            F_i ≈ E_batch[ (∂L/∂θ_i)² ]
        """
        model.eval()
        accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(self._fisher[name]) for name in self._param_names
        }
        count = 0

        # Iterate up to num_batches
        it = iter(data_batches)
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()

            for name, p in model.named_parameters():
                if name not in accum or p.grad is None:
                    continue
                accum[name] += p.grad.detach().pow(2)
            count += 1

        if count == 0:
            logger.warning("consolidate() got zero batches — Fisher not updated")
            return

        for name in self._param_names:
            new_fisher = accum[name] / count
            # Exponential decay accumulation
            self._fisher[name] = self.config.gamma * self._fisher[name] + new_fisher

        # Update anchor
        if self.config.update_anchor_mode == "replace":
            for name, p in model.named_parameters():
                if name in self._anchor:
                    self._anchor[name] = p.detach().clone()
        else:  # ema
            alpha = self.config.anchor_ema_alpha
            for name, p in model.named_parameters():
                if name in self._anchor:
                    self._anchor[name] = (
                        alpha * self._anchor[name] + (1 - alpha) * p.detach()
                    )

        self._has_consolidated_once = True
        # Clear model grads so callers don't accidentally step on the Fisher-grads
        model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------- penalty

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC regularization term as a scalar tensor.

        L_reg = (λ/2) · Σ_i F_i · (θ_i - θ*_i)²
        """
        if not self._has_consolidated_once:
            # No task has been consolidated yet — penalty is zero.
            # Return a scalar that carries gradient wrt model params (via a
            # zero-multiplied param) so callers can freely add it to their loss.
            return torch.tensor(0.0, requires_grad=False)

        losses: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if name not in self._fisher:
                continue
            F = self._fisher[name]
            anchor = self._anchor[name].to(p.device)
            loss = (F.to(p.device) * (p - anchor).pow(2)).sum()
            losses.append(loss)
        total = torch.stack(losses).sum() * (self.config.lambda_reg / 2.0)
        return total

    # ---------------------------------------------------- diagnostics

    def has_consolidated(self) -> bool:
        return self._has_consolidated_once

    def summary(self) -> dict:
        f_total = sum(f.abs().sum().item() for f in self._fisher.values())
        f_params = sum(f.numel() for f in self._fisher.values())
        return {
            "num_params_tracked": f_params,
            "fisher_l1_total": f_total,
            "has_consolidated": self._has_consolidated_once,
            "lambda_reg": self.config.lambda_reg,
            "gamma": self.config.gamma,
        }

    # ---------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "fisher": {k: v.detach().cpu() for k, v in self._fisher.items()},
            "anchor": {k: v.detach().cpu() for k, v in self._anchor.items()},
            "has_consolidated_once": self._has_consolidated_once,
        }

    def load_state_dict(self, state: dict) -> None:
        # Config fields — validate names but accept updated values
        for k, v in state["config"].items():
            setattr(self.config, k, v)
        self._fisher = {k: v.clone() for k, v in state["fisher"].items()}
        self._anchor = {k: v.clone() for k, v in state["anchor"].items()}
        self._param_names = list(self._fisher.keys())
        self._has_consolidated_once = bool(state["has_consolidated_once"])
