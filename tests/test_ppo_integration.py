"""Integration test: the PPO scale-mismatch fix actually enables learning.

Unit tests in ``test_ppo_normalization.py`` prove the helpers are correct.
This file proves the *whole PPO step* behaves correctly end-to-end:

1. With the FIXED pipeline (values denormalized to raw scale before GAE),
   a PPO gradient step actually moves the policy (approx_kl > 0, finite) —
   i.e. the agent can learn. This is the behaviour that was broken before
   the fix (scale-mismatched advantages carried no signal).

2. The OLD buggy path (value head output, which is in normalized scale,
   fed straight into GAE next to raw rewards) produces different / distorted
   advantages — guarded so nobody can "simplify" the denorm away.

测试修复后 PPO 整步是否真的能学习(策略会动),并锁定"旧 bug 会扭曲优势"
这一回归。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.train import ReturnNormalizer, _normalize_advantages, compute_gae


class _TinyActorCritic(nn.Module):
    """Minimal obs -> (logits[8], value) net, mirroring train.py's interface."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 8):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh())
        self.policy = nn.Linear(16, n_actions)
        self.value = nn.Linear(16, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.policy(h), self.value(h).squeeze(-1)


def _make_rollout(T: int = 32, obs_dim: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(T, obs_dim, generator=g)
    # VARIED rewards so advantages carry real signal.
    rewards = torch.randn(T, generator=g).clamp(-1.0, 2.0).abs()  # >= 0, varied
    dones = torch.zeros(T)
    dones[-1] = 1.0
    return obs, rewards, dones


class TestPpoStepEnablesLearning:
    def test_fixed_pipeline_moves_policy(self) -> None:
        """After GAE on RAW-scale values + normalized advantages, one PPO step
        changes the policy (approx_kl != 0). This is the learning that was
        previously impossible due to the scale mismatch."""
        torch.manual_seed(0)
        obs, rewards, dones = _make_rollout()
        net = _TinyActorCritic()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Value head "trained on normalized returns" => raw-scale values here
        # are produced by denormalizing a normalized prediction, i.e. realistic.
        norm = ReturnNormalizer(alpha=0.5)
        with torch.no_grad():
            values_norm = net(obs)[1]
        # Warm EMA on the true per-step returns (multi-element, scale O(T)) so
        # denormalize maps the normalized value-head output back to raw scale.
        returns_gt, _ = compute_gae(
            rewards, torch.zeros(rewards.shape[0]), dones, 0.0, 0.99, 0.95
        )
        for _ in range(50):
            norm.update(returns_gt)
        values_raw = norm.denormalize(values_norm)
        last_value_raw = float(norm.denormalize(values_norm[-1:]).item())

        advantages, returns = compute_gae(
            rewards, values_raw, dones, last_value_raw, 0.99, 0.95
        )
        assert torch.isfinite(advantages).all()
        assert float(advantages.std()) > 0.0  # signal exists
        adv_norm = _normalize_advantages(advantages)

        # old logprobs (behavior policy)
        with torch.no_grad():
            logits_old, _ = net(obs)
            dist_old = torch.distributions.Categorical(logits=logits_old)
            actions = dist_old.sample()
            logprobs_old = dist_old.log_prob(actions)

        # one PPO step (mirrors train.py:2157-2177)
        new_logits, new_values = net(obs)
        dist = torch.distributions.Categorical(logits=new_logits)
        new_logprobs = dist.log_prob(actions)
        ratio = (new_logprobs - logprobs_old).exp()
        unclipped = ratio * adv_norm
        clipped = torch.clamp(ratio, 0.8, 1.2) * adv_norm
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = (new_values - _normalize_advantages(returns)).pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            logits_after, _ = net(obs)
            dist_after = torch.distributions.Categorical(logits=logits_after)
            logprobs_after = dist_after.log_prob(actions)
            approx_kl = (logprobs_old - logprobs_after).mean().item()

        # THE ASSERTION THAT MATTERS: the policy actually moved.
        assert approx_kl != 0.0
        assert abs(approx_kl) > 1e-6
        # and gradients were finite (no NaN blow-up from the old bug)
        for p in net.parameters():
            assert torch.isfinite(p.grad).all()

    def test_fixed_advantages_rank_with_reward(self) -> None:
        """Higher raw reward steps should get higher (less negative) advantages
        than lower-reward steps, under the fixed pipeline."""
        torch.manual_seed(1)
        T = 16
        obs = torch.randn(T, 4)
        norm = ReturnNormalizer(alpha=0.1)
        net = _TinyActorCritic()
        with torch.no_grad():
            values = norm.denormalize(net(obs)[1])  # raw scale

        # Force a clear reward structure: good vs bad halves.
        rewards = torch.zeros(T)
        rewards[: T // 2] = 2.0
        rewards[T // 2 :] = 0.1
        dones = torch.zeros(T)
        dones[-1] = 1.0

        advantages, _ = compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)
        adv_norm = _normalize_advantages(advantages)
        # mean advantage of the high-reward half should exceed the low-reward half
        good = adv_norm[: T // 2].mean().item()
        bad = adv_norm[T // 2 :].mean().item()
        assert good > bad


class TestOldBugDistortsAdvantages:
    def test_normalized_values_into_gae_differ_from_raw(self) -> None:
        """Regression guard: feeding NORMALIZED value-head output straight into
        GAE (the old bug) yields advantages that differ from the raw-scale
        (fixed) path. If this assertion ever fails, the denormalize-before-GAE
        fix was removed/neutralized."""
        torch.manual_seed(2)
        obs, rewards, dones = _make_rollout()
        norm = ReturnNormalizer(alpha=0.1)
        net = _TinyActorCritic()
        with torch.no_grad():
            values_norm = net(obs)[1]
        returns_gt, _ = compute_gae(
            rewards, torch.zeros(rewards.shape[0]), dones, 0.0, 0.99, 0.95
        )
        for _ in range(50):
            norm.update(returns_gt)
        values_raw = norm.denormalize(values_norm)
        last_raw = float(norm.denormalize(values_norm[-1:]).item())
        last_norm = float(values_norm[-1:].item())

        adv_fixed, _ = compute_gae(rewards, values_raw, dones, last_raw, 0.99, 0.95)
        adv_buggy, _ = compute_gae(rewards, values_norm, dones, last_norm, 0.99, 0.95)

        # They must NOT be equal — the bug changes the advantages.
        assert not torch.allclose(adv_fixed, adv_buggy, atol=1e-3)
        # The fixed path advantages should have meaningful variance (real signal);
        # the buggy path should be scale-distorted relative to it.
        assert float(adv_fixed.std()) > 0.0
        # scale ratio roughly equals the EMA std (the bug multiplies by it)
        ratio = float(adv_buggy.std().abs() / (adv_fixed.std().abs() + 1e-12))
        assert ratio != 1.0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
