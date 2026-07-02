"""Bounded three-tier replay buffer.

Implements Axioms 1 (bounded), 2 (eviction before growth), 3 (hierarchical
storage), 6 (serializable).

Three tiers:
- **Hot (GPU)**: fixed-capacity ring buffer of recent transitions in device tensors.
- **Warm (CPU)**: larger ring buffer in CPU RAM; demoted from hot when hot is full.
- **Cold (SSD)**: on-disk archive, shard files under ``data/replay/``.
  Cold tier is *append-then-truncate*: total shards bounded by ``cold_capacity_shards``.

Sampling supports Prioritized Experience Replay (PER) across the hot+warm tiers.

三层有界回放缓冲：GPU 热层（环形）/ CPU 温层（环形）/ SSD 冷层（分片+封顶）。
支持 PER 采样。所有层严格容量上限。
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =====================================================================
# Transition dataclass — the atomic unit
# =====================================================================


@dataclass
class Transition:
    """A single environment transition.

    ``obs`` / ``next_obs`` are stored as ``uint8`` when they are images to keep
    memory low. ``priority`` is used by PER sampling (higher = more likely).
    """

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    priority: float = 1.0
    meta: dict = field(default_factory=dict)


# =====================================================================
# Hot tier: fixed-capacity GPU ring buffer of torch tensors
# =====================================================================


class HotRingTier:
    """Fixed-capacity ring buffer on a torch device (typically GPU).

    Storage layout: struct-of-arrays. Axiom 1 enforced by construction —
    tensors are pre-allocated, never resized.

    Hot 层：GPU 上定容环形存储，SoA 布局，预分配。
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> None:
        self._capacity = int(capacity)
        if self._capacity <= 0:
            raise ValueError("capacity must be positive")
        self._device = torch.device(device)
        self._obs_shape = tuple(obs_shape)

        self.obs = torch.zeros((self._capacity, *obs_shape), dtype=torch.uint8, device=self._device)
        self.next_obs = torch.zeros(
            (self._capacity, *obs_shape), dtype=torch.uint8, device=self._device
        )
        self.action = torch.zeros(self._capacity, dtype=torch.long, device=self._device)
        self.reward = torch.zeros(self._capacity, dtype=torch.float32, device=self._device)
        self.done = torch.zeros(self._capacity, dtype=torch.float32, device=self._device)
        self.priority = torch.ones(self._capacity, dtype=torch.float32, device=self._device)

        self._ptr = 0
        self._size = 0

    # ---------------------------------------------------------- API basics

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device

    def __len__(self) -> int:
        return self._size

    def full(self) -> bool:
        return self._size >= self._capacity

    # -------------------------------------------------------- read/write

    def add(self, tr: Transition) -> Transition | None:
        """Add a transition. Returns the *evicted* transition if buffer was full.

        添加一个 transition。若已满，返回被淘汰的旧 transition（可降级到 warm 层）。
        """
        evicted: Transition | None = None
        if self._size >= self._capacity:
            evicted = self._read_index(self._ptr)

        self.obs[self._ptr] = torch.from_numpy(tr.obs).to(self._device)
        self.next_obs[self._ptr] = torch.from_numpy(tr.next_obs).to(self._device)
        self.action[self._ptr] = tr.action
        self.reward[self._ptr] = tr.reward
        self.done[self._ptr] = float(tr.done)
        self.priority[self._ptr] = tr.priority

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return evicted

    def _read_index(self, i: int) -> Transition:
        return Transition(
            obs=self.obs[i].detach().cpu().numpy().copy(),
            action=int(self.action[i].item()),
            reward=float(self.reward[i].item()),
            next_obs=self.next_obs[i].detach().cpu().numpy().copy(),
            done=bool(self.done[i].item() > 0.5),
            priority=float(self.priority[i].item()),
        )

    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Return indices for a batch. Uniform. Use ``sample_batch`` for PER."""
        if self._size == 0:
            raise ValueError("HotRingTier empty")
        return rng.integers(0, self._size, size=batch_size)

    def gather(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        """Gather a batch by indices — returns tensors on the tier's device."""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self._device)
        return {
            "obs": self.obs.index_select(0, idx),
            "action": self.action.index_select(0, idx),
            "reward": self.reward.index_select(0, idx),
            "next_obs": self.next_obs.index_select(0, idx),
            "done": self.done.index_select(0, idx),
            "priority": self.priority.index_select(0, idx),
        }

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """PER support: update priorities after TD-error computation."""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self._device)
        pr = torch.as_tensor(new_priorities, dtype=torch.float32, device=self._device)
        self.priority.index_copy_(0, idx, pr)

    def state_dict(self) -> dict:
        """Serialize (Axiom 6). Small tensors moved to CPU."""
        return {
            "capacity": self._capacity,
            "obs_shape": self._obs_shape,
            "ptr": self._ptr,
            "size": self._size,
            "obs": self.obs[: self._size].detach().cpu().numpy(),
            "next_obs": self.next_obs[: self._size].detach().cpu().numpy(),
            "action": self.action[: self._size].detach().cpu().numpy(),
            "reward": self.reward[: self._size].detach().cpu().numpy(),
            "done": self.done[: self._size].detach().cpu().numpy(),
            "priority": self.priority[: self._size].detach().cpu().numpy(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self._capacity or tuple(state["obs_shape"]) != self._obs_shape:
            raise ValueError("Incompatible HotRingTier state")
        n = int(state["size"])
        for name in ("obs", "next_obs", "action", "reward", "done", "priority"):
            src = torch.from_numpy(state[name]).to(self._device, dtype=getattr(self, name).dtype)
            getattr(self, name)[:n] = src
        self._ptr = int(state["ptr"])
        self._size = n


# =====================================================================
# Warm tier: CPU-side ring of Transition objects
# =====================================================================


class WarmRingTier:
    """Fixed-capacity CPU ring buffer storing full Transition objects.

    Slightly slower per-op than Hot (Python objects), but capacity is typically
    10× larger. Demotion from Hot goes here first; further demotion goes to Cold.

    Warm 层：CPU 侧定容环形，10× Hot 容量，Python 对象存 Transition。
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._buf: list[Transition | None] = [None] * self._capacity
        self._ptr = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def full(self) -> bool:
        return self._size >= self._capacity

    def add(self, tr: Transition) -> Transition | None:
        evicted = self._buf[self._ptr] if self._size >= self._capacity else None
        self._buf[self._ptr] = tr
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return evicted

    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if self._size == 0:
            raise ValueError("WarmRingTier empty")
        return rng.integers(0, self._size, size=batch_size)

    def gather(self, indices: np.ndarray) -> list[Transition]:
        out: list[Transition] = []
        for i in indices:
            tr = self._buf[int(i)]
            if tr is None:
                raise IndexError(f"WarmRingTier index {i} is None")
            out.append(tr)
        return out

    def priorities(self) -> np.ndarray:
        pr = np.zeros(self._size, dtype=np.float32)
        for i in range(self._size):
            tr = self._buf[i]
            if tr is not None:
                pr[i] = tr.priority
        return pr

    def state_dict(self) -> dict:
        return {
            "capacity": self._capacity,
            "ptr": self._ptr,
            "size": self._size,
            "buf": self._buf[: self._size],
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self._capacity:
            raise ValueError("Incompatible WarmRingTier state")
        n = int(state["size"])
        self._buf = list(state["buf"]) + [None] * (self._capacity - n)
        self._ptr = int(state["ptr"])
        self._size = n


# =====================================================================
# Cold tier: SSD-backed shard archive with a bounded shard count
# =====================================================================


class ColdShardTier:
    """SSD archive of shards. Each shard is a pickle of a list of Transitions.

    Shard count is bounded by ``max_shards`` (Axiom 1). When exceeded, the
    oldest shard is deleted.

    Cold 层：SSD 分片归档，分片数上限。超过则删最老分片。
    """

    SHARD_NAME_FMT = "shard_{idx:08d}.pkl"

    def __init__(
        self,
        archive_dir: Path,
        max_shards: int,
        shard_size: int = 4096,
    ) -> None:
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_shards = int(max_shards)
        self._shard_size = int(shard_size)
        # Pending items waiting to be flushed to a shard.
        self._pending: list[Transition] = []
        self._known: list[int] = self._discover_shards()

    # -------------------------------------------------- capacity accounting

    @property
    def capacity(self) -> int:
        """Approximate transition capacity (bounded)."""
        return self._max_shards * self._shard_size

    def __len__(self) -> int:
        return len(self._known) * self._shard_size + len(self._pending)

    # ------------------------------------------------------- discovery/IO

    def _discover_shards(self) -> list[int]:
        idxs: list[int] = []
        for p in sorted(self._dir.glob("shard_*.pkl")):
            try:
                idxs.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
        return sorted(idxs)

    def _next_shard_idx(self) -> int:
        return (self._known[-1] + 1) if self._known else 0

    def _shard_path(self, idx: int) -> Path:
        return self._dir / self.SHARD_NAME_FMT.format(idx=idx)

    def _flush_shard(self) -> None:
        if not self._pending:
            return
        idx = self._next_shard_idx()
        path = self._shard_path(idx)
        with path.open("wb") as f:
            pickle.dump(self._pending, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._known.append(idx)
        self._pending = []
        # Axiom 2: evict oldest shard(s) if over capacity
        while len(self._known) > self._max_shards:
            oldest = self._known.pop(0)
            try:
                self._shard_path(oldest).unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------- add/sample

    def add(self, tr: Transition) -> None:
        """Buffer for shard flushing. Never blocks growth beyond capacity."""
        self._pending.append(tr)
        if len(self._pending) >= self._shard_size:
            self._flush_shard()

    def flush(self) -> None:
        """Force flush pending items to a partial shard."""
        if self._pending:
            self._flush_shard()

    def load_shard(self, idx: int) -> list[Transition]:
        with self._shard_path(idx).open("rb") as f:
            return pickle.load(f)

    def iter_all(self) -> Iterator[Transition]:
        for idx in self._known:
            yield from self.load_shard(idx)
        yield from self._pending

    def known_shards(self) -> list[int]:
        return list(self._known)


# =====================================================================
# Top-level: BoundedReplayBuffer — orchestrates all three tiers
# =====================================================================


@dataclass
class ReplayBudget:
    """Per-tier capacity budget."""

    hot_capacity: int
    warm_capacity: int
    cold_max_shards: int
    cold_shard_size: int = 4096


class BoundedReplayBuffer:
    """Three-tier bounded replay buffer.

    - New transitions enter the **hot** tier.
    - Hot-tier evictions are demoted to **warm**.
    - Warm-tier evictions are demoted to **cold** (SSD).
    - Cold-tier is capped by ``cold_max_shards`` (oldest deleted).

    Sampling: uniformly picks a tier weighted by size, then samples within
    the tier. PER weighting via ``sample_prioritized`` is supported for hot+warm.

    Bounded: total transitions across all tiers ≤ capacity (Axiom 1).
    Eviction-before-growth: every tier's ``add`` returns the evicted item
    which is passed to the next tier (Axiom 2).
    """

    def __init__(
        self,
        budget: ReplayBudget,
        obs_shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        archive_dir: Path | None = None,
        seed: int = 0,
    ) -> None:
        self._budget = budget
        self._obs_shape = tuple(obs_shape)
        self._device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self.hot = HotRingTier(budget.hot_capacity, obs_shape, self._device)
        self.warm = WarmRingTier(budget.warm_capacity)

        if archive_dir is None:
            archive_dir = Path("data/replay")
        self.cold = ColdShardTier(
            archive_dir=archive_dir,
            max_shards=budget.cold_max_shards,
            shard_size=budget.cold_shard_size,
        )

    # ------------------------------------------------- bounded protocol

    @property
    def capacity(self) -> int:
        """Sum of all tier capacities. Axiom 1."""
        return self.hot.capacity + self.warm.capacity + self.cold.capacity

    def __len__(self) -> int:
        return len(self.hot) + len(self.warm) + len(self.cold)

    # ------------------------------------------------------------ add

    def add(self, tr: Transition) -> None:
        """Insert transition; demote through tiers as needed (Axiom 2)."""
        demoted_hot = self.hot.add(tr)
        if demoted_hot is None:
            return
        demoted_warm = self.warm.add(demoted_hot)
        if demoted_warm is None:
            return
        # Falls to cold; ColdShardTier handles its own capacity trimming.
        self.cold.add(demoted_warm)

    # --------------------------------------------------------- sampling

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Uniform sample from hot ∪ warm. Cold tier not sampled online.

        Returns a batch of tensors on the buffer's device.

        从 hot ∪ warm 均匀采样。Cold 层不参与在线采样（供离线 replay/consolidation）。
        """
        hot_n, warm_n = len(self.hot), len(self.warm)
        total = hot_n + warm_n
        if total == 0:
            raise ValueError("BoundedReplayBuffer empty (hot+warm)")
        # Weighted split by tier size
        n_hot = int(round(batch_size * hot_n / total)) if total > 0 else 0
        n_warm = batch_size - n_hot

        parts: list[dict[str, torch.Tensor]] = []
        if n_hot > 0 and hot_n > 0:
            idx = self.hot.sample_indices(n_hot, self._rng)
            parts.append(self.hot.gather(idx))
        if n_warm > 0 and warm_n > 0:
            idx = self.warm.sample_indices(n_warm, self._rng)
            warm_batch = self.warm.gather(idx)
            parts.append(self._transitions_to_tensors(warm_batch))

        return self._concat(parts)

    def sample_prioritized(
        self,
        batch_size: int,
        alpha: float = 0.6,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """PER sampling from hot tier only (fast path).

        Returns (batch, indices, importance_weights).

        从 hot 层做 PER 采样（快路径）。返回 (batch, indices, IS 权重)。
        """
        if len(self.hot) == 0:
            raise ValueError("Hot tier empty")
        prios = self.hot.priority[: len(self.hot)].detach().cpu().numpy() ** alpha
        prios = prios / prios.sum()
        idx = self._rng.choice(len(self.hot), size=batch_size, p=prios, replace=True)
        # Importance sampling weights (unnormalized-in-tier)
        weights = (len(self.hot) * prios[idx]) ** (-alpha)
        weights = weights / weights.max()
        return self.hot.gather(idx), idx, weights.astype(np.float32)

    def update_hot_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.hot.update_priorities(indices, priorities)

    # ------------------------------------------------------------ helpers

    def _transitions_to_tensors(self, trs: list[Transition]) -> dict[str, torch.Tensor]:
        obs = np.stack([tr.obs for tr in trs])
        next_obs = np.stack([tr.next_obs for tr in trs])
        action = np.array([tr.action for tr in trs], dtype=np.int64)
        reward = np.array([tr.reward for tr in trs], dtype=np.float32)
        done = np.array([float(tr.done) for tr in trs], dtype=np.float32)
        priority = np.array([tr.priority for tr in trs], dtype=np.float32)
        return {
            "obs": torch.from_numpy(obs).to(self._device),
            "next_obs": torch.from_numpy(next_obs).to(self._device),
            "action": torch.from_numpy(action).to(self._device),
            "reward": torch.from_numpy(reward).to(self._device),
            "done": torch.from_numpy(done).to(self._device),
            "priority": torch.from_numpy(priority).to(self._device),
        }

    @staticmethod
    def _concat(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if len(parts) == 1:
            return parts[0]
        out: dict[str, torch.Tensor] = {}
        for key in parts[0]:
            out[key] = torch.cat([p[key] for p in parts], dim=0)
        return out

    # ---------------------------------------------------------- persistence

    def flush_cold(self) -> None:
        """Force flush any pending cold-tier writes."""
        self.cold.flush()

    def stats(self) -> dict:
        return {
            "hot": {"size": len(self.hot), "capacity": self.hot.capacity},
            "warm": {"size": len(self.warm), "capacity": self.warm.capacity},
            "cold": {
                "size": len(self.cold),
                "capacity": self.cold.capacity,
                "shards": len(self.cold.known_shards()),
                "max_shards": self._budget.cold_max_shards,
            },
            "total": len(self),
            "total_capacity": self.capacity,
        }
