"""Generative Replay via VAE.

Stage 6 anti-forgetting mechanism. Instead of keeping actual past observations
in a growing buffer, we train a small VAE on the observation distribution and
sample synthetic "rehearsal" observations from it during consolidation.

The VAE size is fixed (Axiom 1); its capacity is the same regardless of how
many observations it has ingested — memory does not grow over time.

Reference: Shin et al. 2017, "Continual Learning with Deep Generative Replay".

生成式回放：训一个小 VAE 学观测分布，用它合成"回忆样本"替代显式历史存储。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenerativeReplayConfig:
    obs_dim: int
    latent_dim: int = 16
    hidden: int = 64
    lr: float = 1e-3
    kl_weight: float = 1.0


class _MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mean(h), self.logvar(h)


class _MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GenerativeReplayVAE(nn.Module):
    """A small MLP VAE used for generative replay.

    Public API:
        - :meth:`update(obs)` — one training step on a batch of real observations.
        - :meth:`sample(n)` — draw ``n`` synthetic observations from prior.
        - :meth:`reconstruct(obs)` — encode-decode round trip.

    Bounded state: only the (encoder, decoder) parameters + optimizer state.
    No growing buffer of observations.
    """

    def __init__(self, config: GenerativeReplayConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _MLPEncoder(config.obs_dim, config.latent_dim, config.hidden)
        self.decoder = _MLPDecoder(config.latent_dim, config.obs_dim, config.hidden)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr)

    # ---------------------------------------------------- core VAE

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(obs)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(obs)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    # ---------------------------------------------------- losses

    @staticmethod
    def _kl_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL[N(mean, exp(logvar)) || N(0, 1)]  per element, then sum over latent dim
        return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)

    def loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        recon, mean, logvar = self.forward(obs)
        recon_loss = F.mse_loss(recon, obs, reduction="none").sum(dim=-1)
        kl = self._kl_gaussian(mean, logvar)
        total = (recon_loss + self.config.kl_weight * kl).mean()
        return total, {
            "recon": float(recon_loss.mean().item()),
            "kl": float(kl.mean().item()),
        }

    # ---------------------------------------------------- update

    def update(self, obs: torch.Tensor) -> dict:
        loss, metrics = self.loss(obs)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return {"loss": float(loss.item()), **metrics}

    # ---------------------------------------------------- sample & reconstruct

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Draw ``n`` synthetic observations from prior N(0, I)."""
        z = torch.randn(n, self.config.latent_dim, device=self._device())
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, obs: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self.forward(obs)
        return recon

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------------------------------------------------- diagnostics

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> dict:
        return {
            "num_params": self.num_parameters(),
            "obs_dim": self.config.obs_dim,
            "latent_dim": self.config.latent_dim,
        }
