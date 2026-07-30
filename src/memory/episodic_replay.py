"""Episodic Replay Memory — wraps EpisodicMemory + cold-tier transition store.

Stage 13 deliverable: bridges the gap between embedding-level EpisodicMemory
(which stores compressed obs_embeddings) and the full-transition storage needed
for PPO training. High-surprise transitions are archived in a ColdShardTier
alongside their episodic metadata, enabling content-based replay sampling.

Architecture:
    EpisodicMemory (embedding, in-memory)   ← for similarity retrieval
    ColdShardTier (full Transition, on SSD)  ← for PPO training samples
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.memory.bounded_replay import ColdShardTier, Transition
from src.models.developmental_memory import EpisodicMemory

logger = logging.getLogger(__name__)


class EpisodicReplayMemory:
    """Surprise-gated episodic storage with full-transition cold archive.

    Two-tier storage:
        - **Hot** (in-memory via EpisodicMemory): compressed obs_embeddings
          for fast similarity retrieval.
        - **Cold** (SSD via ColdShardTier): full (obs, action, reward,
          next_obs, done) transitions for PPO replay.

    When a surprising transition arrives, both tiers are written:
        embedding + metadata → EpisodicMemory
        full Transition      → ColdShardTier (only if surprise is high enough)
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        cold_dir: str | Path,
        cold_max_shards: int = 32,
        cold_shard_size: int = 4096,
        store_threshold: float = 1.5,
    ) -> None:
        self.episodic = episodic
        self._cold = ColdShardTier(
            archive_dir=Path(cold_dir),
            max_shards=cold_max_shards,
            shard_size=cold_shard_size,
        )
        self._store_threshold = float(store_threshold)
        self._total_cold_stored = 0

    @property
    def capacity(self) -> int:
        return self.episodic.capacity + self._cold.capacity

    def __len__(self) -> int:
        return len(self.episodic) + len(self._cold)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        obs_embedding: torch.Tensor,
        surprise: float,
        global_step: int,
        episode_id: int,
        tags: list[str] | None = None,
    ) -> None:
        """Store a transition if surprising enough.

        1. Always tries episodic (embedding-level) storage
        2. If surprise exceeds threshold, also archives full Transition to cold
        """
        entry = self.episodic.store(
            obs_embedding=obs_embedding,
            action=action,
            reward=reward,
            surprise=surprise,
            global_step=global_step,
            episode_id=episode_id,
            tags=tags,
        )

        if entry is not None and surprise > self._store_threshold:
            tr = Transition(
                obs=obs,
                action=int(action),
                reward=float(reward),
                next_obs=next_obs,
                done=bool(done),
                priority=float(surprise),
                meta={"step": int(global_step), "episode": int(episode_id)},
            )
            self._cold.add(tr)
            self._total_cold_stored += 1

    def sample(
        self, batch_size: int, device: torch.device, obs_shape: tuple[int, ...], num_actions: int
    ) -> dict[str, torch.Tensor] | None:
        """Sample a batch of full transitions from cold storage.

        Returns dict with keys ``obs``, ``action``, ``reward``, ``next_obs``,
        ``done``, or ``None`` if insufficient data.

        These tensors can be fed directly into the PPO update or world-model training.
        """
        n_cold = len(self._cold)
        if n_cold < batch_size:
            return None

        indices = np.random.choice(n_cold, size=batch_size, replace=False)
        all_transitions: list[Transition] = list(self._cold.iter_all())
        sampled = [all_transitions[i] for i in indices]

        obs_list = np.stack([t.obs for t in sampled], axis=0) if sampled[0].obs.ndim > 1 else np.array([t.obs for t in sampled], dtype=np.float32)
        action_list = np.array([t.action for t in sampled], dtype=np.int64)
        reward_list = np.array([t.reward for t in sampled], dtype=np.float32)
        next_obs_list = np.stack([t.next_obs for t in sampled], axis=0) if sampled[0].next_obs.ndim > 1 else np.array([t.next_obs for t in sampled], dtype=np.float32)
        done_list = np.array([t.done for t in sampled], dtype=np.float32)

        return {
            "obs": torch.as_tensor(obs_list, dtype=torch.uint8 if obs_list.dtype == np.uint8 else torch.float32, device=device),
            "action": torch.as_tensor(action_list, dtype=torch.long, device=device),
            "reward": torch.as_tensor(reward_list, dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(next_obs_list, dtype=torch.uint8 if next_obs_list.dtype == np.uint8 else torch.float32, device=device),
            "done": torch.as_tensor(done_list, dtype=torch.float32, device=device),
        }

    def flush(self) -> None:
        """Force-flush cold tier (ensures all pending transitions are on disk)."""
        self._cold.flush()

    def summary(self) -> dict:
        s = self.episodic.summary()
        s.update({
            "cold_stored": self._total_cold_stored,
            "cold_on_disk": len(self._cold),
        })
        return s
