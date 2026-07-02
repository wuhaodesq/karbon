"""Random Network Distillation (RND) intrinsic motivation.

Burda et al. 2018 — Exploration by Random Network Distillation.

Two small networks:

- **Target**: randomly initialized, frozen forever. Maps obs → embedding.
- **Predictor**: same architecture, trained to imitate the target on observed
  states.

Intrinsic reward at state s = squared distance between predictor(s) and target(s).
States the agent has seen a lot → predictor matches → low reward. Novel states
→ mismatch → high reward.

RND has no ``memory`` per se (unlike ICM), so it's naturally bounded —
Axiom 1 satisfied by construction.

内在动机：目标网冻结不动，预测网学去逼近目标网，两者输出差 = 好奇心 reward。
天然有界（无记忆存储）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class RunningMeanStd:
    """Numerically stable running mean/std tracker.

    Used to normalize the intrinsic reward (recommended by Burda et al.).
    Bounded: fixed-size state (mean, var, count) — Axiom 1 OK.

    数值稳定的在线均值/方差跟踪。用于归一化内在 reward。
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(self.mean.dtype)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def std(self) -> torch.Tensor:
        return self.var.clamp(min=1e-8).sqrt()

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = float(state["count"])


class RNDNet(nn.Module):
    """A small CNN → MLP embedding net used by both target and predictor.

    Input:  (B, H, W, C) uint8 image observations (MiniGrid style).
    Output: (B, embed_dim) float.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        embed_dim: int = 128,
        conv_channels: tuple[int, int] = (16, 32),
        hidden: int = 128,
    ) -> None:
        super().__init__()
        h, w, c = obs_shape
        c1, c2 = conv_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(c, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(c2 * h * w, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) uint8 → (B, C, H, W) float
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        return self.mlp(self.encoder(x))


@dataclass
class RNDConfig:
    """Configuration for :class:`RND`.

    - ``embed_dim``: dimensionality of the target/predictor output space.
    - ``lr``: predictor optimizer learning rate.
    - ``reward_clip``: clip normalized reward to ±this (0 → no clip).
    """

    embed_dim: int = 128
    lr: float = 1e-4
    reward_clip: float = 5.0


class RND(nn.Module):
    """Random Network Distillation for intrinsic motivation.

    Composed of:
    - A frozen ``target`` net (no gradients, buffers only).
    - A trainable ``predictor`` net.
    - A :class:`RunningMeanStd` for reward normalization.

    Usage:
        ``intrinsic_reward = rnd.intrinsic_reward(obs)`` — no side effect.
        ``loss = rnd.update(obs)`` — one predictor step; returns MSE loss scalar.

    None of the state grows over time (Axiom 1).
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        config: RNDConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RNDConfig()
        self.target = RNDNet(obs_shape, embed_dim=self.config.embed_dim)
        self.predictor = RNDNet(obs_shape, embed_dim=self.config.embed_dim)

        # Freeze target permanently
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optim = torch.optim.Adam(self.predictor.parameters(), lr=self.config.lr)
        self.reward_rms = RunningMeanStd(shape=())

    def _target_embed(self, obs: torch.Tensor) -> torch.Tensor:
        self.target.eval()
        with torch.no_grad():
            return self.target(obs)

    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-example intrinsic reward (no gradients, no side effects).

        Returns:
            (B,) tensor.
        """
        tgt = self._target_embed(obs)
        with torch.no_grad():
            pred = self.predictor(obs)
        r = (pred - tgt).pow(2).mean(dim=-1)
        return r

    def normalized_reward(self, obs: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Reward divided by running std; optionally clip; optionally update stats.

        Args:
            obs: batched observations.
            update_stats: if True, update the running stats with this batch's rewards.
        """
        r = self.intrinsic_reward(obs)
        if update_stats:
            self.reward_rms.update(r)
        std = self.reward_rms.std().to(r.device)
        r_norm = r / std
        if self.config.reward_clip > 0:
            r_norm = r_norm.clamp(-self.config.reward_clip, self.config.reward_clip)
        return r_norm

    def update(self, obs: torch.Tensor) -> float:
        """One predictor training step on the given batch. Returns the loss.

        Frozen target guarantees ``target`` weights never change (Axiom 6:
        no drift over time).
        """
        tgt = self._target_embed(obs)
        pred = self.predictor(obs)
        loss = (pred - tgt.detach()).pow(2).mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return float(loss.item())

    # ------------------------------------------------------- persistence

    def rnd_state_dict(self) -> dict:
        """Composite state for checkpointing (Axiom 6)."""
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optim": self.optim.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
        }

    def load_rnd_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.optim.load_state_dict(state["optim"])
        self.reward_rms.load_state_dict(state["reward_rms"])
