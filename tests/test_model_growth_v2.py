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


class TestGrowerV2PlateauLP:
    def test_lp_zero_on_plateau_at_peak(self) -> None:
        """On a plateau at the historical peak, headroom ≈ 0 → grower should fire."""
        grower = ModelGrowerV2(config=GrowthConfigV2(grow_trigger_coverage=0.0))
        assert grower.plateau_lp(50.0) == 0.0   # first sample seeds rmax
        assert grower.plateau_lp(100.0) == 0.0  # new peak
        assert grower.plateau_lp(100.0) == 0.0  # plateau at peak -> no headroom

    def test_lp_positive_when_below_running_max(self) -> None:
        """Any return below the running max leaves positive headroom → hold."""
        grower = ModelGrowerV2(config=GrowthConfigV2())
        grower.plateau_lp(10.0)  # rmax = 10
        assert grower.plateau_lp(5.0) > 0.0

    def test_spike_is_forgotten_so_growth_can_refire(self) -> None:
        """Regression for the resume-spike bug: a one-off inflated return on
        the first step after a checkpoint resume (e.g. 113 after the model's
        sustained plateau is ~101) must NOT pin ``rmax`` forever. With the
        decaying running max, the spike fades over a handful of growth-check
        calls and the trigger line (0.95 × rmax) drops back to the sustained
        level, so a future 3→4 layer growth can still fire.

        Without the decay (raw running max), rmax would stay 113 and growth
        would require mean_return ≥ 0.95×113 ≈ 107.5 — unreachable at a 101
        plateau, permanently blocking further growth.
        """
        cfg = GrowthConfigV2(rmax_decay=0.98)
        grower = ModelGrowerV2(config=cfg)
        grower.plateau_lp(101.0)        # sustained plateau seeds rmax = 101
        grower.plateau_lp(113.0)        # transient resume spike -> rmax = 113
        assert grower._rmax == 113.0
        # Sustained plateau returns; spike must decay back toward 101.
        rmax_after = None
        for _ in range(8):
            rmax_after = grower.plateau_lp(101.0)
        assert rmax_after < 107.5, "spike not forgotten; growth would be blocked"
        # Once back at the sustained level, plateau_lp ≈ 0 -> grower fires.
        lp = grower.plateau_lp(101.0)
        assert lp <= 0.05

    def test_grower_fires_on_plateau_with_coverage(self) -> None:
        """Real-breakthrough fix: corrected LP lets the grower trigger on a
        plateau when coverage is sufficient (was previously always blocked)."""
        cfg = GrowthConfigV2(
            min_steps_between_growths=0,
            grow_trigger_lp_threshold=0.05,
            grow_trigger_coverage=0.15,
        )
        grower = ModelGrowerV2(config=cfg)
        grower.plateau_lp(100.0)
        lp = grower.plateau_lp(100.0)  # plateau at peak
        assert lp <= 0.05
        assert grower.should_grow(step=1_000_000, learning_progress=lp, coverage_ratio=0.5)

    def test_grower_blocked_when_coverage_low(self) -> None:
        cfg = GrowthConfigV2(min_steps_between_growths=0, grow_trigger_coverage=0.15)
        grower = ModelGrowerV2(config=cfg)
        grower.plateau_lp(100.0)
        lp = grower.plateau_lp(100.0)
        assert not grower.should_grow(
            step=1_000_000, learning_progress=lp, coverage_ratio=0.05,
        )

    def test_old_formula_was_over_eager(self) -> None:
        """Regression guard: the old ``lp = 1.0 - mean_return`` (≈-99 at
        mean_return≈100) is <= threshold, so ``should_grow`` would return
        True on EVERY eligible step (constant, un-plateaued growth) — not
        "blocked". The actual reason growth never fired was ``coverage is
        None`` (this config lacked a top-level ``coverage:`` section, so the
        whole growth-check block was skipped). plateau_lp fixes the
        over-eager behavior so growth only triggers on a genuine plateau."""
        old_lp = 1.0 - 100.0
        cfg = GrowthConfigV2(min_steps_between_growths=0, grow_trigger_coverage=0.0)
        grower = ModelGrowerV2(config=cfg)
        assert old_lp <= cfg.grow_trigger_lp_threshold  # would NOT block -> over-eager
        assert grower.should_grow(
            step=1_000_000, learning_progress=old_lp, coverage_ratio=1.0,
        )


class TestGrowerV2ResumeLoad:
    def test_load_state_restores_layers_and_arms_warmup(self) -> None:
        """On resume, load_state_dict must restore the grower's layer count and
        growth bookkeeping, AND arm a post-resume warmup so the inflated
        first-step mean_return cannot immediately force a growth. This is the
        fix for the desync bug where a fresh grower (2 layers) on a 3/4-layer
        model would do a no-op growth or silently drop a layer."""
        cfg = GrowthConfigV2(resume_warmup_calls=3)
        grower = ModelGrowerV2(config=cfg)
        state = grower.state_dict()
        state["current_layers"] = 4
        state["growth_count"] = 2
        state["last_growth_step"] = 9_000_000
        state["rmax"] = 113.0
        grower.load_state_dict(state)
        assert grower._current_layers == 4
        assert grower._growth_count == 2
        assert grower._warmup_remaining == 3  # fresh warmup armed on load
        # During warmup, should_grow must NOT fire even on a perfect plateau.
        for _ in range(3):
            assert grower.should_grow(
                step=12_000_000, learning_progress=0.0, coverage_ratio=1.0) is False
        # After warmup it can fire.
        assert grower.should_grow(
            step=12_000_000, learning_progress=0.0, coverage_ratio=1.0) is True

    def test_warmup_default_zero_for_fresh_grower(self) -> None:
        """A freshly constructed grower (no resume) has no warmup, so growth
        can trigger on the very first eligible plateau as before."""
        grower = ModelGrowerV2(config=GrowthConfigV2(min_steps_between_growths=0))
        assert grower._warmup_remaining == 0
        assert grower.should_grow(
            step=1_000_000, learning_progress=0.0, coverage_ratio=1.0) is True


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
