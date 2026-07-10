"""Recurrent State-Space Model (RSSM) — Dreamer-style world model.

Stage 3's core deliverable. Skeleton scaffold sized for CPU-friendly toy
experiments; the same architecture scales to Crafter on GPU.

**Formulation** (Hafner et al., Dreamer / DreamerV2 / DreamerV3):

At each time step ``t``, the latent state consists of:
- **Deterministic** part ``h_t`` (recurrent hidden state, GRU-updated).
- **Stochastic** part ``z_t`` — Gaussian with mean/logstd predicted from ``h_t``.

Two heads produce the stochastic latent:
- **Prior**  ``p(z_t | h_t)`` — sampled during "imagination" without obs.
- **Posterior** ``q(z_t | h_t, e_t)`` — conditioned on the encoded observation
  ``e_t = Encoder(o_t)``. Used during "learning" from real experience.

Recurrent update:
    ``h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))``

Reconstruction head:
    ``ô_t = Decoder(h_t, z_t)``

For Stage 3 we only need forward, KL loss between posterior and prior, and
reconstruction loss. Full Dreamer would add reward + continue heads.

**Bounded guarantees**:
- All rollout tensors are pre-allocated with a max-length declared at
  construction (Axiom 1).
- Truncated BPTT: unroll length is a fixed hyperparameter, never grows.

RSSM 世界模型：确定 h + 随机 z 隐状态，GRU 递归。
posterior + prior + decoder 三头。展开长度固定，有界。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =====================================================================
# Encoder / Decoder (small toy MLPs; sufficient for gridworld obs)
# =====================================================================


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.GELU()]
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObsEncoder(nn.Module):
    """Encoder for flat vector observations.

    For gridworld images, callers should flatten HxWxC → vector before passing.
    """

    def __init__(self, obs_dim: int, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = _MLP(obs_dim, embed_dim, hidden=hidden, depth=2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class ObsDecoder(nn.Module):
    """Decoder ``(h, z) → ô``."""

    def __init__(self, h_dim: int, z_dim: int, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = _MLP(h_dim + z_dim, obs_dim, hidden=hidden, depth=2)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h, z], dim=-1))


# =====================================================================
# Prior / Posterior networks
# =====================================================================


class LatentDistribution(nn.Module):
    """Produce Gaussian (mean, std) from a feature vector."""

    def __init__(self, in_dim: int, z_dim: int, hidden: int = 128, min_std: float = 0.1) -> None:
        super().__init__()
        self.trunk = _MLP(in_dim, hidden, hidden=hidden, depth=1)
        self.mean_head = nn.Linear(hidden, z_dim)
        self.std_head = nn.Linear(hidden, z_dim)
        self.min_std = float(min_std)

    def forward(self, x: torch.Tensor) -> Normal:
        h = self.trunk(x)
        mean = self.mean_head(h)
        std = F.softplus(self.std_head(h)) + self.min_std
        return Normal(mean, std)


# =====================================================================
# RSSM
# =====================================================================


@dataclass
class RSSMConfig:
    obs_dim: int
    action_dim: int
    z_dim: int = 32
    h_dim: int = 64
    embed_dim: int = 64
    hidden: int = 128
    max_rollout_steps: int = 15   # bounded — Axiom 1
    kl_free_nats: float = 1.0
    reward_loss_weight: float = 1.0  # weight of the reward-prediction term


@dataclass
class RSSMState:
    """Latent state at a single timestep."""

    h: torch.Tensor           # (B, h_dim)
    z: torch.Tensor           # (B, z_dim)
    z_dist_mean: torch.Tensor
    z_dist_std: torch.Tensor


class RSSM(nn.Module):
    """Recurrent State-Space Model (Dreamer-style, minimal).

    Provides three primitives:

    - :meth:`observe` — one step conditioned on a real observation (posterior).
    - :meth:`imagine` — one step without observation (prior only), rolls forward.
    - :meth:`compute_loss` — reconstruction + KL over an observed rollout.

    Bounded: unroll length is capped by ``config.max_rollout_steps``.
    """

    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config
        c = config

        self._reward_loss_weight = float(c.reward_loss_weight)
        self.encoder = ObsEncoder(c.obs_dim, c.embed_dim, hidden=c.hidden)
        self.decoder = ObsDecoder(c.h_dim, c.z_dim, c.obs_dim, hidden=c.hidden)

        # GRU cell: input = z + action_onehot, hidden = h
        self.recurrent = nn.GRUCell(c.z_dim + c.action_dim, c.h_dim)

        # Prior:      p(z | h)
        # Posterior:  q(z | h, e)
        self.prior_dist = LatentDistribution(c.h_dim, c.z_dim, hidden=c.hidden)
        self.posterior_dist = LatentDistribution(
            c.h_dim + c.embed_dim, c.z_dim, hidden=c.hidden
        )

        # Reward predictor:  r̂_t = Reward(h_t, z_t)  (Dreamer-style).
        # Grounds planning in objective environment reward rather than the
        # policy's value estimate, so System 2 can discover plans the current
        # policy would not choose. Fixed-size MLP -> bounded (Axiom 1).
        self.reward_head = _MLP(c.h_dim + c.z_dim, 1, hidden=c.hidden, depth=1)

    # ---------------------------------------------------- initial state

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.config.h_dim, device=device)
        z = torch.zeros(batch_size, self.config.z_dim, device=device)
        z_mean = torch.zeros_like(z)
        z_std = torch.ones_like(z)
        return RSSMState(h=h, z=z, z_dist_mean=z_mean, z_dist_std=z_std)

    # ---------------------------------------------------- one-step "observe"

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action_onehot: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[RSSMState, Normal, Normal]:
        """Advance state using both prior and posterior. Returns (new_state, prior, posterior).

        posterior samples the actual z for use downstream; prior is used only
        for the KL loss.
        """
        # Recurrent update: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        gru_input = torch.cat([prev_state.z, prev_action_onehot], dim=-1)
        h_t = self.recurrent(gru_input, prev_state.h)

        # Prior
        prior = self.prior_dist(h_t)

        # Posterior (uses encoded obs)
        e_t = self.encoder(obs)
        posterior = self.posterior_dist(torch.cat([h_t, e_t], dim=-1))

        z_t = posterior.rsample()
        new_state = RSSMState(
            h=h_t, z=z_t,
            z_dist_mean=posterior.mean,
            z_dist_std=posterior.stddev,
        )
        return new_state, prior, posterior

    # ------------------------------------------------------ one-step "imagine"

    def imagine_step(
        self,
        prev_state: RSSMState,
        prev_action_onehot: torch.Tensor,
    ) -> tuple[RSSMState, Normal]:
        """Advance state without observation (prior sample). Returns (state, prior)."""
        gru_input = torch.cat([prev_state.z, prev_action_onehot], dim=-1)
        h_t = self.recurrent(gru_input, prev_state.h)
        prior = self.prior_dist(h_t)
        z_t = prior.rsample()
        return (
            RSSMState(h=h_t, z=z_t, z_dist_mean=prior.mean, z_dist_std=prior.stddev),
            prior,
        )

    def decode(self, state: RSSMState) -> torch.Tensor:
        return self.decoder(state.h, state.z)

    # ------------------------------------------------------ reward

    def predict_reward(self, state: RSSMState) -> torch.Tensor:
        """Predict instantaneous reward from a latent state. Returns (B,)."""
        return self.reward_head(torch.cat([state.h, state.z], dim=-1)).squeeze(-1)

    # ------------------------------------------------------ full-sequence loss

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq_onehot: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction + KL (+ reward) loss over a rollout.

        When ``reward_seq`` (B, T, 1) is provided, an MSE reward-prediction
        loss is added so the world model learns objective environment reward.

        Rollout length ``T`` must not exceed ``config.max_rollout_steps``.
        """
        B, T, obs_dim = obs_seq.shape
        if T > self.config.max_rollout_steps:
            raise ValueError(
                f"rollout length {T} exceeds max_rollout_steps="
                f"{self.config.max_rollout_steps}"
            )
        if obs_dim != self.config.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {obs_dim}, cfg {self.config.obs_dim}")

        device = obs_seq.device
        state = self.initial_state(B, device)

        recon_losses: list[torch.Tensor] = []
        kl_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []

        for t in range(T):
            state, prior, posterior = self.observe_step(
                state, action_seq_onehot[:, t, :], obs_seq[:, t, :]
            )
            recon = self.decode(state)
            recon_loss = F.mse_loss(recon, obs_seq[:, t, :])
            recon_losses.append(recon_loss)

            # KL(posterior || prior) per element, then free-nats floor
            kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1)
            kl = torch.clamp(kl, min=self.config.kl_free_nats).mean()
            kl_losses.append(kl)

            if reward_seq is not None:
                r_target = reward_seq[:, t].squeeze(-1)
                # Reward from the posterior state (what the agent was in).
                r_pred = self.predict_reward(state)
                reward_losses.append(F.mse_loss(r_pred, r_target))
                # Reward from the prior (imagined) state. At planning time
                # rewards are predicted from imagine_step (prior) states, not
                # posterior ones, so training on both narrows the train/serve
                # gap (Dreamer-style).
                prior_state, _ = self.imagine_step(state, action_seq_onehot[:, t, :])
                r_pred_prior = self.predict_reward(prior_state)
                reward_losses.append(F.mse_loss(r_pred_prior, r_target))

        recon_loss_total = torch.stack(recon_losses).mean()
        kl_loss_total = torch.stack(kl_losses).mean()
        total = recon_loss_total + kl_loss_total
        out: dict[str, torch.Tensor] = {
            "loss": total,
            "recon_loss": recon_loss_total,
            "kl_loss": kl_loss_total,
        }
        if reward_losses:
            reward_loss_total = torch.stack(reward_losses).mean()
            total = total + self._reward_loss_weight * reward_loss_total
            out["reward_loss"] = reward_loss_total
        out["loss"] = total
        return out

    # --------------------------------------------- imagined rollout

    def imagine(
        self,
        initial_state: RSSMState,
        action_seq_onehot: torch.Tensor,   # (B, T, action_dim)
    ) -> list[RSSMState]:
        """Roll the world model forward without observations.

        Length capped at ``max_rollout_steps``. Returns the trajectory of
        latent states (bounded list).
        """
        B, T, _ = action_seq_onehot.shape
        if T > self.config.max_rollout_steps:
            raise ValueError("imagine exceeds max_rollout_steps")
        state = initial_state
        trajectory: list[RSSMState] = []
        for t in range(T):
            state, _ = self.imagine_step(state, action_seq_onehot[:, t, :])
            trajectory.append(state)
        return trajectory

    # ---------------------------------------------------- stats

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
