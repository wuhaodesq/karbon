"""Tests for ModelGrowerV2 — non-disruptive growth.

Verifies the two fixes applied to the growth path:
1. Distillation now preserves the teacher's POLICY distribution (KL on
   softmax logits), so a grown model keeps the agent's learned
   action preferences instead of resetting them.
2. Optimizer momentum for matching parameters is carried over to the
   new optimizer (so growth does not zero out the optimizer state).

测试 ModelGrowerV2:蒸馏保留策略分布 + 优化器动量跨生长保留。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_growth_v2 import (
    GrowthConfigV2,
    ModelGrowerV2,
    _carry_over_adam_momentum,
)
from src.train import HybridActorCritic


class _TinyAC(nn.Module):
    """Minimal obs -> (logits[8], value) stand-in for HybridActorCritic."""

    def __init__(self, n_act: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 32),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(32, n_act)
        self.value_head = nn.Linear(32, 1)

    def forward(self, obs: torch.Tensor):
        x = obs.permute(0, 3, 1, 2).float() / 255.0
        h = self.body(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def _kl_policy(a: nn.Module, b: nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        pa = torch.softmax(a(x)[0], dim=-1)
        pb = torch.softmax(b(x)[0], dim=-1)
    return float(F.kl_div(pb.log(), pa, reduction="batchmean").item())


class TestGrowerV2DistillPreservesPolicy:
    def test_distill_moves_student_toward_teacher(self) -> None:
        """After growth distill, the student's policy should be close to the
        frozen teacher's (KL small). This is the non-disruptive property."""
        torch.manual_seed(0)
        grower = ModelGrowerV2(config=GrowthConfigV2(distill_steps=80, distill_lr=1e-2))
        teacher = _TinyAC()
        student = _TinyAC()
        with torch.no_grad():
            for p in teacher.parameters():
                p.normal_(0.0, 0.5)  # a "learned" teacher policy
            # student starts at a different (random) policy
        x = torch.randint(0, 256, (8, 8, 8, 3), dtype=torch.uint8)
        kl_before = _kl_policy(teacher, student, x)
        grower._distill(teacher, student, None)
        kl_after = _kl_policy(teacher, student, x)
        assert kl_after < kl_before
        assert kl_after < 0.3  # student now matches teacher's policy


class TestGrowerV2MomentumCarryover:
    def test_optimizer_momentum_carried_on_matching_params(self) -> None:
        """The momentum-copy helper must preserve Adam momentum
        (exp_avg / exp_avg_sq / step) for params whose names match
        between old and new models, so growth does not wipe the
        policy optimizer's accumulated state."""
        torch.manual_seed(1)
        a = _TinyAC()
        b = _TinyAC()
        opt_a = torch.optim.Adam(a.parameters(), lr=1e-3)
        # Populate Adam state on `a`.
        for _ in range(5):
            x = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
            logits, value = a(x)
            loss = F.mse_loss(logits, torch.randn_like(logits)) + value.mean()
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
        assert 0 in opt_a.state_dict()["state"] and "exp_avg" in opt_a.state_dict()["state"][0]

        opt_b = torch.optim.Adam(b.parameters(), lr=1e-3)
        # Before carry-over, `b`'s optimizer has no momentum state.
        assert 0 not in opt_b.state_dict()["state"]

        _carry_over_adam_momentum(opt_a, opt_b, a, b)

        # After carry-over, a matching param in `b` has momentum.
        b_state = opt_b.state_dict()
        assert 0 in b_state["state"], "optimizer state lost for matched param"
        carried = b_state["state"][0].get("exp_avg")
        assert carried is not None
        assert torch.isfinite(carried).all()


class TestGrowerV2ObsShapeCarryover:
    def test_created_model_keeps_real_obs_shape(self) -> None:
        """_create_larger_model must use the *real* observation shape from the
        source model, not a hardcoded (64,64,3). A mismatch (e.g. a
        4-channel or non-square obs) would build an encoder whose first conv
        in_channels is wrong and silently corrupt the grown network."""
        torch.manual_seed(0)
        obs_shape = (64, 48, 4)  # deliberately not 64x64x3
        model = HybridActorCritic(
            obs_shape=obs_shape, num_actions=8, d_model=32, n_layers=2,
        )
        assert model.obs_shape == obs_shape

        grower = ModelGrowerV2(d_model=32, n_heads=4, config=GrowthConfigV2())
        grown = grower._create_larger_model(model, new_n_layers=3)
        assert grown.obs_shape == obs_shape


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
