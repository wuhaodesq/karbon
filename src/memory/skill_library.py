"""Bounded Skill Library.

Stage 4's core deliverable. Stores reusable "skills" — each a compact LoRA-style
low-rank weight delta applied to a base policy network, plus metadata (usage
count, last-used timestamp, average reward when invoked).

Enforces:

- Axiom 1: fixed capacities at three tiers (GPU top-K, CPU cache, SSD archive).
- Axiom 2: eviction (LRU × usefulness × avg-reward composite) before insertion
  when full.
- Axiom 3: hierarchical storage.
- Axiom 6: full state_dict / load_state_dict round-trip.

Similarity-based merging: skills whose LoRA representations have cosine
similarity above a threshold are fused (weighted average of A, B and metadata
combined), preventing accidental duplication.

有界技能库：GPU top-K + CPU 缓存 + SSD 归档三层；LoRA 低秩压缩表示技能；
LRU × usefulness × avg-reward 综合淘汰；余弦相似度合并。
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# LoRA-style skill representation
# =====================================================================


@dataclass
class SkillWeights:
    """Low-rank weight delta ``ΔW = A · B`` for a single target linear layer.

    - ``A``: (d_out, rank)
    - ``B``: (rank, d_in)

    Product yields ΔW of shape (d_out, d_in) — a compact skill delta.

    LoRA 低秩参数：A ∈ R^{d_out×r}, B ∈ R^{r×d_in}。
    """

    A: torch.Tensor
    B: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.A.shape[1])

    @property
    def d_out(self) -> int:
        return int(self.A.shape[0])

    @property
    def d_in(self) -> int:
        return int(self.B.shape[1])

    def num_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def to(self, device: torch.device | str) -> "SkillWeights":
        return SkillWeights(A=self.A.to(device), B=self.B.to(device))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ΔW · x = A · (B · x). ``x`` last dim must be ``d_in``."""
        return (x @ self.B.T) @ self.A.T


@dataclass
class SkillEntry:
    """One skill in the library."""

    id: int
    weights: SkillWeights
    tag: str = ""                            # human-readable label
    usage_count: int = 0                     # how often invoked
    total_reward: float = 0.0                # sum of rewards when used
    last_used_ts: float = field(default_factory=time.time)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.usage_count)

    def record_use(self, reward: float) -> None:
        self.usage_count += 1
        self.total_reward += float(reward)
        self.last_used_ts = time.time()


# =====================================================================
# Bounded three-tier skill library
# =====================================================================


@dataclass
class SkillLibraryBudget:
    gpu_capacity: int          # top-K skills resident on GPU
    cpu_capacity: int          # additional cached on CPU
    ssd_max_shards: int = 8    # archive shard cap
    ssd_shard_size: int = 64   # skills per shard
    merge_similarity_threshold: float = 0.95  # cosine >= this ⇒ merge


class BoundedSkillLibrary:
    """Three-tier bounded skill library.

    - New skills enter the **GPU tier**.
    - When GPU tier is full, the lowest-scoring skill demotes to **CPU tier**.
    - When CPU tier is full, the lowest-scoring skill demotes to **SSD archive**.
    - SSD archive is shard-capped; oldest shard deleted when exceeded.

    Scoring for eviction (higher = keep, lower = evict):

    .. code-block:: text

        score(skill) = α · avg_reward
                     + β · log1p(usage_count)
                     - γ · (now - last_used_ts) / TIMESCALE

    Default weights: α=1.0, β=0.5, γ=0.1, TIMESCALE=3600s.

    Similarity merging: on ``add``, if any GPU-tier skill has cosine similarity
    ≥ ``merge_similarity_threshold`` with the new skill's flattened LoRA
    representation, they are fused (weighted by usage_count) instead of adding.

    综合评分 = α·平均 reward + β·log(使用次数) - γ·(距上次使用)/TIMESCALE
    """

    _TIMESCALE_SECONDS = 3600.0

    def __init__(
        self,
        budget: SkillLibraryBudget,
        skill_shape: tuple[int, int, int],  # (d_out, rank, d_in)
        device: torch.device | str = "cpu",
        archive_dir: Path | None = None,
        score_alpha: float = 1.0,
        score_beta: float = 0.5,
        score_gamma: float = 0.1,
    ) -> None:
        self._budget = budget
        self._skill_shape = tuple(skill_shape)
        self._device = torch.device(device)
        self._archive_dir = Path(archive_dir) if archive_dir else Path("data/skills")
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        self._gpu: list[SkillEntry] = []
        self._cpu: list[SkillEntry] = []
        self._known_shards: list[int] = self._discover_shards()

        self._next_id = self._recover_next_id()
        self._alpha = score_alpha
        self._beta = score_beta
        self._gamma = score_gamma

    # -------------------------------------------------- bounded protocol

    @property
    def capacity(self) -> int:
        return (
            self._budget.gpu_capacity
            + self._budget.cpu_capacity
            + self._budget.ssd_max_shards * self._budget.ssd_shard_size
        )

    def __len__(self) -> int:
        return (
            len(self._gpu)
            + len(self._cpu)
            + len(self._known_shards) * self._budget.ssd_shard_size
        )

    # ------------------------------------------------------- construction

    def new_skill(self, tag: str = "", scale: float = 0.01) -> SkillEntry:
        """Create a fresh randomly-initialized skill entry (not yet added)."""
        d_out, rank, d_in = self._skill_shape
        A = torch.randn(d_out, rank, device=self._device) * scale
        B = torch.randn(rank, d_in, device=self._device) * scale
        skill_id = self._next_id
        self._next_id += 1
        return SkillEntry(id=skill_id, weights=SkillWeights(A=A, B=B), tag=tag)

    # ------------------------------------------------------- scoring

    def _score(self, s: SkillEntry, now: float | None = None) -> float:
        if now is None:
            now = time.time()
        recency = (now - s.last_used_ts) / self._TIMESCALE_SECONDS
        return (
            self._alpha * s.avg_reward
            + self._beta * math.log1p(s.usage_count)
            - self._gamma * recency
        )

    # -------------------------------------------------- similarity + merge

    def _flatten(self, s: SkillEntry) -> torch.Tensor:
        """Flatten LoRA (A, B) into a single vector for cosine sim."""
        return torch.cat([s.weights.A.flatten(), s.weights.B.flatten()])

    def _find_merge_target(self, s: SkillEntry) -> SkillEntry | None:
        if self._budget.merge_similarity_threshold >= 1.0:
            return None
        if not self._gpu:
            return None
        flat = self._flatten(s).unsqueeze(0)
        others = torch.stack([self._flatten(o) for o in self._gpu], dim=0)
        cos = F.cosine_similarity(flat, others, dim=1)
        best_idx = int(cos.argmax().item())
        if float(cos[best_idx].item()) >= self._budget.merge_similarity_threshold:
            return self._gpu[best_idx]
        return None

    def retrieve(
        self, query: "SkillWeights | SkillEntry", min_similarity: float = 0.5
    ) -> SkillEntry | None:
        """Retrieve the most similar GPU-tier skill to ``query``.

        Used by the M2 skill-reuse loop: given a freshly distilled skill
        candidate (or a skill embedding derived from the current state), return
        the closest existing library skill if cosine similarity ≥
        ``min_similarity``. Returns ``None`` when nothing matches — caller then
        creates a new skill instead.

        检索最相似技能：给定查询（新技能或状态派生的技能表征），
        返回 GPU 层中最接近的技能（cosine ≥ 阈值），否则 None。
        """
        if not self._gpu:
            return None
        if isinstance(query, SkillEntry):
            q_flat = self._flatten(query)
        else:
            q_flat = torch.cat([query.A.flatten(), query.B.flatten()])
        q_flat = q_flat.unsqueeze(0)
        others = torch.stack([self._flatten(o) for o in self._gpu], dim=0)
        cos = F.cosine_similarity(q_flat, others, dim=1)
        best_idx = int(cos.argmax().item())
        best_sim = float(cos[best_idx].item())
        if best_sim >= min_similarity:
            return self._gpu[best_idx]
        return None

    def _merge(self, existing: SkillEntry, new: SkillEntry) -> None:
        """Fuse ``new`` into ``existing`` (weighted by usage counts).

        The merged skill retains ``existing.id``.
        """
        w_e = max(1, existing.usage_count)
        w_n = max(1, new.usage_count)
        total = w_e + w_n
        existing.weights = SkillWeights(
            A=(existing.weights.A * w_e + new.weights.A * w_n) / total,
            B=(existing.weights.B * w_e + new.weights.B * w_n) / total,
        )
        existing.usage_count += new.usage_count
        existing.total_reward += new.total_reward
        existing.last_used_ts = max(existing.last_used_ts, new.last_used_ts)

    # ------------------------------------------------------------ add

    def add(self, skill: SkillEntry) -> str:
        """Insert a skill.

        Returns one of: ``"added"``, ``"merged"``, ``"demoted"``, ``"archived"``.
        (The returned string describes what happened to the *incoming* skill's
        effective destination.)
        """
        # First, try merging into an existing similar skill.
        target = self._find_merge_target(skill)
        if target is not None:
            self._merge(target, skill)
            return "merged"

        # Otherwise, add to GPU tier — evict if full.
        if len(self._gpu) < self._budget.gpu_capacity:
            self._gpu.append(skill)
            return "added"

        # GPU full: demote lowest-scoring GPU skill to CPU tier.
        now = time.time()
        idx = min(range(len(self._gpu)), key=lambda i: self._score(self._gpu[i], now))
        demoted = self._gpu.pop(idx)
        self._demote_to_cpu(demoted)
        self._gpu.append(skill)
        return "demoted"

    def _demote_to_cpu(self, s: SkillEntry) -> None:
        s.weights = s.weights.to("cpu")
        if len(self._cpu) < self._budget.cpu_capacity:
            self._cpu.append(s)
            return
        # CPU full: demote lowest-scoring CPU skill to SSD
        now = time.time()
        idx = min(range(len(self._cpu)), key=lambda i: self._score(self._cpu[i], now))
        archived = self._cpu.pop(idx)
        self._archive(archived)
        self._cpu.append(s)

    # --------------------------------------------------------- archive

    def _shard_path(self, idx: int) -> Path:
        return self._archive_dir / f"skills_{idx:08d}.pkl"

    def _discover_shards(self) -> list[int]:
        idxs: list[int] = []
        for p in sorted(self._archive_dir.glob("skills_*.pkl")):
            try:
                idxs.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
        return sorted(idxs)

    def _recover_next_id(self) -> int:
        max_id = 0
        for entry in self._gpu + self._cpu:
            max_id = max(max_id, entry.id)
        # Also scan shards for max id
        for shard_idx in self._known_shards:
            path = self._shard_path(shard_idx)
            try:
                with path.open("rb") as f:
                    entries = pickle.load(f)
                for e in entries:
                    max_id = max(max_id, e.id)
            except Exception:
                pass
        return max_id + 1

    def _archive(self, s: SkillEntry) -> None:
        """Append skill to the current partial shard; flush + trim if needed."""
        idx = (self._known_shards[-1] + 1) if self._known_shards else 0
        path = self._shard_path(idx)

        # Load existing partial shard if it's small enough to append into
        entries: list[SkillEntry] = []
        if path.exists():
            with path.open("rb") as f:
                entries = pickle.load(f)
            if len(entries) >= self._budget.ssd_shard_size:
                # Start a new shard
                idx = idx + 1
                path = self._shard_path(idx)
                entries = []
        entries.append(s)
        with path.open("wb") as f:
            pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
        if idx not in self._known_shards:
            self._known_shards.append(idx)
            self._known_shards.sort()
        # Trim oldest shards if over cap
        while len(self._known_shards) > self._budget.ssd_max_shards:
            oldest = self._known_shards.pop(0)
            try:
                self._shard_path(oldest).unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------ lookup

    def get(self, skill_id: int) -> SkillEntry | None:
        for e in self._gpu:
            if e.id == skill_id:
                return e
        for e in self._cpu:
            if e.id == skill_id:
                return e
        # Search shards (slow — meant for offline consolidation, not hot path)
        for shard_idx in self._known_shards:
            path = self._shard_path(shard_idx)
            with path.open("rb") as f:
                entries = pickle.load(f)
            for e in entries:
                if e.id == skill_id:
                    return e
        return None

    def top_k(self) -> list[SkillEntry]:
        """Return current GPU-tier skills (already the top-K)."""
        return list(self._gpu)

    def sample_for_injection(self) -> "SkillEntry | None":
        """Pick a GPU-tier skill to inject into the policy for an episode.

        Score-weighted sampling (higher score ⇒ more likely chosen) so the
        policy rehearses its most valuable skills, while still occasionally
        rehearsing weaker ones. Returns ``None`` if the GPU tier is empty.
        按评分加权抽样一个 GPU 层技能用于本 episode 注入（优先高价值技能，
        偶尔也练弱势技能）。GPU 层为空时返回 None。
        """
        if not self._gpu:
            return None
        scores = torch.tensor(
            [max(1e-3, self._score(s)) for s in self._gpu], dtype=torch.float32
        )
        probs = scores / scores.sum()
        idx = int(torch.multinomial(probs, 1).item())
        return self._gpu[idx]

    def iter_all_in_memory(self) -> Iterator[SkillEntry]:
        yield from self._gpu
        yield from self._cpu

    # -------------------------------------------------------- statistics

    def stats(self) -> dict:
        return {
            "gpu": {"size": len(self._gpu), "capacity": self._budget.gpu_capacity},
            "cpu": {"size": len(self._cpu), "capacity": self._budget.cpu_capacity},
            "ssd": {
                "shards": len(self._known_shards),
                "max_shards": self._budget.ssd_max_shards,
            },
            "total": len(self),
            "total_capacity": self.capacity,
            "next_id": self._next_id,
        }

    # ------------------------------------------------------- serialization

    def state_dict(self) -> dict:
        return {
            "gpu": [self._entry_to_dict(e) for e in self._gpu],
            "cpu": [self._entry_to_dict(e) for e in self._cpu],
            "known_shards": list(self._known_shards),
            "next_id": self._next_id,
            "skill_shape": self._skill_shape,
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state["skill_shape"]) != self._skill_shape:
            raise ValueError("Incompatible skill_shape")
        self._gpu = [self._entry_from_dict(d, self._device) for d in state["gpu"]]
        self._cpu = [self._entry_from_dict(d, torch.device("cpu")) for d in state["cpu"]]
        self._known_shards = list(state["known_shards"])
        self._next_id = int(state["next_id"])

    @staticmethod
    def _entry_to_dict(e: SkillEntry) -> dict:
        return {
            "id": e.id,
            "A": e.weights.A.detach().cpu().numpy(),
            "B": e.weights.B.detach().cpu().numpy(),
            "tag": e.tag,
            "usage_count": e.usage_count,
            "total_reward": e.total_reward,
            "last_used_ts": e.last_used_ts,
        }

    @staticmethod
    def _entry_from_dict(d: dict, device: torch.device) -> SkillEntry:
        A = torch.from_numpy(d["A"]).to(device)
        B = torch.from_numpy(d["B"]).to(device)
        return SkillEntry(
            id=int(d["id"]),
            weights=SkillWeights(A=A, B=B),
            tag=str(d["tag"]),
            usage_count=int(d["usage_count"]),
            total_reward=float(d["total_reward"]),
            last_used_ts=float(d["last_used_ts"]),
        )
