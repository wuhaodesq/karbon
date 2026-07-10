"""Enhanced Developmental Memory System.

Replaces the simple FIFO replay buffer with a multi-layered memory architecture:

    Working Memory  →  RSSM hidden state (current context, ~1 step)
    Episodic Memory →  Surprise-gated event store (important events, 10K entries)
    Semantic Memory →  Extracted facts/concepts (patterns across episodes, 1K entries)
    Procedural Memory→  Skill Library (LoRA skills, already exists)
    Autobiographical →  Life narrative timeline (100 key events, permanent)

Key innovations over existing FIFO replay:
    1. Surprise gate — only store events the agent didn't expect
    2. Importance scoring — surprise + reward + novelty → priority
    3. Temporal indexing — each memory knows WHEN it happened
    4. Sleep consolidation — episodic patterns → semantic facts nightly
    5. Reactive replay — random reactivation prevents catastrophic forgetting
    6. Content retrieval — query by similarity, not just PER priority
    7. Memory decay — gradual forgetting of less important memories

Bounded: All capacities declared at construction (Axiom 1).
VRAM: ~0.3 GB (10K episodes + 1K semantics).

增强记忆系统：五层架构替代简单的 FIFO 回放。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Episodic Memory
# =====================================================================


@dataclass
class EpisodicEntry:
    """A single remembered event."""
    obs_embedding: torch.Tensor       # (d_model,) compressed observation
    action: int
    reward: float
    surprise: float                   # RSSM prediction error at this step
    importance: float                 # surprise + |reward| + novelty
    global_step: int                  # when it happened
    episode_id: int                   # which episode
    tags: list[str] = field(default_factory=list)  # "success", "collision", "novel_object"
    access_count: int = 0             # how many times recalled
    last_accessed_step: int = 0


class EpisodicMemory(nn.Module):
    """Surprise-gated episodic memory store.

    Only stores transitions where the agent was "surprised" (RSSM prediction
    error exceeds a threshold). This mirrors human episodic memory — we don't
    remember every walking step, only the ones where something unexpected happened.

    Bounded: max_entries fixed. Eviction by lowest importance when full.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        d_model: int = 128,
        surprise_threshold: float = 0.01,
        decay_rate: float = 0.9999,     # per-step importance decay
        consolidation_decay: float = 0.95,  # per-consolidation importance decay
    ) -> None:
        super().__init__()
        self._max = int(max_entries)
        self._d_model = int(d_model)
        self._surprise_threshold = float(surprise_threshold)
        self._decay = float(decay_rate)
        self._consolidation_decay = float(consolidation_decay)

        self._entries: dict[int, EpisodicEntry] = {}
        self._next_id = 0
        self._embedding_matrix = torch.zeros(self._max, d_model)  # for fast retrieval  # BOUNDS-OK

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._entries)

    def store(
        self,
        obs_embedding: torch.Tensor,
        action: int,
        reward: float,
        surprise: float,
        global_step: int,
        episode_id: int,
        tags: list[str] | None = None,
    ) -> EpisodicEntry | None:
        """Store an event if it's surprising enough.

        Returns the entry if stored, None if filtered out.
        """
        if surprise < self._surprise_threshold and abs(reward) < 0.01:
            return None  # boring step, don't store

        importance = surprise + abs(reward) * 0.5 + (
            0.1 if tags and "novel_object" in tags else 0.0
        )

        if len(self._entries) >= self._max:
            self._evict_least_important()

        entry = EpisodicEntry(
            obs_embedding=obs_embedding.detach().cpu(),
            action=int(action),
            reward=float(reward),
            surprise=float(surprise),
            importance=float(importance),
            global_step=int(global_step),
            episode_id=int(episode_id),
            tags=tags or [],
        )
        self._entries[self._next_id] = entry
        self._next_id += 1
        return entry

    def retrieve_by_time(
        self, start_step: int, end_step: int | None = None, limit: int = 32,
    ) -> list[EpisodicEntry]:
        """Retrieve events within a time window."""
        end = end_step or start_step + 100000
        matching = [
            e for e in self._entries.values()
            if start_step <= e.global_step <= end
        ]
        matching.sort(key=lambda e: -e.importance)
        return matching[:limit]

    def retrieve_by_similarity(
        self, query_embedding: torch.Tensor, k: int = 16,
    ) -> list[EpisodicEntry]:
        """Retrieve k nearest memories by embedding similarity."""
        if not self._entries:
            return []
        entries_list = list(self._entries.values())
        n = len(entries_list)

        # Fill embedding matrix
        for i, e in enumerate(entries_list):
            if i < self._max:
                self._embedding_matrix[i] = e.obs_embedding

        sim = F.cosine_similarity(
            query_embedding.unsqueeze(0).cpu(),
            self._embedding_matrix[:n],
            dim=-1,
        )
        top_k = sim.topk(min(k, n)).indices.tolist()
        return [entries_list[i] for i in top_k]

    def retrieve_important(self, limit: int = 32) -> list[EpisodicEntry]:
        """Retrieve the most important memories."""
        return sorted(
            self._entries.values(),
            key=lambda e: -e.importance * (1.0 + math.log1p(e.access_count)),
        )[:limit]

    def retrieve_by_tag(self, tag: str, limit: int = 32) -> list[EpisodicEntry]:
        return sorted(
            [e for e in self._entries.values() if tag in e.tags],
            key=lambda e: -e.importance,
        )[:limit]

    def mark_accessed(self, entry_id: int, step: int) -> None:
        if entry_id in self._entries:
            e = self._entries[entry_id]
            e.access_count += 1
            e.last_accessed_step = step

    def decay_importance(self) -> None:
        """Global importance decay (all memories fade over time)."""
        for e in self._entries.values():
            e.importance *= self._decay

    def _evict_least_important(self) -> None:
        if not self._entries:
            return
        worst_id = min(
            self._entries,
            key=lambda eid: self._entries[eid].importance * (
                1.0 + math.log1p(self._entries[eid].access_count)
            ),
        )
        del self._entries[worst_id]

    def summary(self) -> dict:
        if not self._entries:
            return {"num_entries": 0, "capacity": self._max}
        imps = [e.importance for e in self._entries.values()]
        tags_count: dict[str, int] = {}
        for e in self._entries.values():
            for t in e.tags:
                tags_count[t] = tags_count.get(t, 0) + 1
        return {
            "num_entries": len(self._entries),
            "capacity": self._max,
            "mean_importance": float(np.mean(imps)),
            "max_importance": float(max(imps)),
            "span_steps": max(e.global_step for e in self._entries.values())
            - min(e.global_step for e in self._entries.values()),
            "top_tags": sorted(tags_count.items(), key=lambda x: -x[1])[:5],
        }


# =====================================================================
# Semantic Memory
# =====================================================================


@dataclass
class SemanticFact:
    """An extracted fact about the world."""
    key: str                          # "red_balls_roll_fast", "heavy_blocks_hard_to_push"
    value: float                      # confidence [0, 1]
    embedding: torch.Tensor           # (d_model,) centroid of related episodes
    source_episodes: list[int] = field(default_factory=list)
    last_updated_step: int = 0
    consolidation_count: int = 0


class SemanticMemory(nn.Module):
    """Long-term semantic fact store.

    Facts are extracted from episodic memory during sleep consolidation.
    Each fact represents a pattern observed across multiple episodes.

    Example facts:
        "objects_of_color_red_tend_to_be_light" (confidence 0.8)
        "pushing_hard_moves_objects_farther" (confidence 0.95)
        "agent_near_wall_cannot_move_outward" (confidence 0.6)

    Bounded: max_facts fixed. Eviction by lowest confidence.
    """

    def __init__(
        self,
        max_facts: int = 1000,
        d_model: int = 128,
        min_confidence: float = 0.3,
    ) -> None:
        super().__init__()
        self._max = int(max_facts)
        self._d_model = int(d_model)
        self._min_conf = float(min_confidence)

        self._facts: dict[str, SemanticFact] = {}
        self._query_proj = nn.Linear(d_model, d_model)  # for content-based retrieval

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._facts)

    def add_or_update(
        self,
        key: str,
        embedding: torch.Tensor,
        confidence_delta: float,
        step: int,
        episode_id: int,
    ) -> SemanticFact:
        """Add a new fact or strengthen an existing one."""
        if key in self._facts:
            fact = self._facts[key]
            # Bayesian-like update
            n = fact.consolidation_count + 1
            fact.value = (fact.value * fact.consolidation_count + confidence_delta) / n
            fact.value = max(0.0, min(1.0, fact.value))
            fact.embedding = (fact.embedding * 0.9 + embedding.detach().cpu() * 0.1)
            fact.consolidation_count = n
            fact.source_episodes.append(episode_id)
            fact.last_updated_step = step
            return fact

        if len(self._facts) >= self._max:
            self._evict_least_confident()

        fact = SemanticFact(
            key=key,
            value=float(max(0.0, min(1.0, confidence_delta))),
            embedding=embedding.detach().cpu(),
            source_episodes=[episode_id],
            last_updated_step=step,
            consolidation_count=1,
        )
        self._facts[key] = fact
        return fact

    def query(self, query_embedding: torch.Tensor, k: int = 8) -> list[SemanticFact]:
        """Retrieve facts relevant to a query embedding."""
        if not self._facts:
            return []
        facts_list = list(self._facts.values())
        query_proj = self._query_proj(query_embedding.unsqueeze(0).cpu())
        emb_stack = torch.stack([f.embedding for f in facts_list])
        sim = F.cosine_similarity(query_proj, emb_stack, dim=-1)
        top_k = sim.topk(min(k, len(facts_list))).indices.tolist()
        return [facts_list[i] for i in top_k]

    def query_by_keyword(self, keyword: str) -> list[SemanticFact]:
        """Find facts containing a keyword."""
        return sorted(
            [f for f in self._facts.values() if keyword.lower() in f.key.lower()],
            key=lambda f: -f.value,
        )

    def get_confident(self, min_conf: float = 0.5) -> list[SemanticFact]:
        return sorted(
            [f for f in self._facts.values() if f.value >= min_conf],
            key=lambda f: -f.value,
        )

    def _evict_least_confident(self) -> None:
        if not self._facts:
            return
        worst_key = min(
            self._facts,
            key=lambda k: self._facts[k].value * (
                1.0 + math.log1p(self._facts[k].consolidation_count)
            ),
        )
        del self._facts[worst_key]

    def summary(self) -> dict:
        if not self._facts:
            return {"num_facts": 0, "capacity": self._max}
        vals = [f.value for f in self._facts.values()]
        return {
            "num_facts": len(self._facts),
            "capacity": self._max,
            "mean_confidence": float(np.mean(vals)),
            "high_confidence": sum(1 for v in vals if v > 0.7),
            "top_facts": [
                f"{f.key} (conf={f.value:.2f})"
                for f in sorted(
                    self._facts.values(), key=lambda x: -x.value,
                )[:5]
            ],
        }


# =====================================================================
# Autobiographical Memory
# =====================================================================


@dataclass
class LifeEvent:
    """A significant moment in the agent's life."""
    timestamp_step: int
    description: str
    emotional_weight: float       # how impactful was this?
    related_episodes: list[int] = field(default_factory=list)
    lesson_learned: str = ""


class AutobiographicalMemory:
    """Timeline of significant life events.

    These form the agent's "life story" — key moments that define its identity.
    Only highly important episodic events get promoted to autobiographical.

    Example events:
        "Step 50000: First time I successfully pushed a ball"
        "Step 200000: Discovered that big blocks need two pushes"
        "Step 800000: Caregiver taught me how to stack blocks"

    Bounded: max_events fixed. Permanent (no eviction unless full).
    """

    def __init__(
        self,
        max_events: int = 100,
        promotion_threshold: float = 0.3,  # lowered: most episodes qualify
    ) -> None:
        self._max = int(max_events)
        self._threshold = float(promotion_threshold)
        self._events: list[LifeEvent] = []

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._events)

    def add_event(
        self,
        step: int,
        description: str,
        importance: float,
        episode_id: int,
        lesson: str = "",
    ) -> LifeEvent | None:
        """Promote an important episode to a life event.

        Returns the event if promoted, None if not important enough.
        """
        if importance < self._threshold:
            return None

        event = LifeEvent(
            timestamp_step=step,
            description=description,
            emotional_weight=importance,
            related_episodes=[episode_id],
            lesson_learned=lesson,
        )

        if len(self._events) >= self._max:
            # Evict least impactful
            self._events.sort(key=lambda e: e.emotional_weight)
            self._events.pop(0)

        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp_step)
        return event

    def get_life_story(self, max_events: int = 20) -> str:
        """Return a narrative of the agent's life."""
        if not self._events:
            return "I don't remember anything yet."

        lines = []
        for e in self._events[-max_events:]:
            lines.append(
                f"Step {e.timestamp_step}: {e.description} "
                f"(impact: {e.emotional_weight:.1f})"
            )
        return "\n".join(lines)

    def query_time_period(
        self, start_step: int, end_step: int,
    ) -> list[LifeEvent]:
        return [
            e for e in self._events
            if start_step <= e.timestamp_step <= end_step
        ]

    def summary(self) -> dict:
        return {
            "num_events": len(self._events),
            "capacity": self._max,
            "first_event": self._events[0].description if self._events else "none",
            "last_event": self._events[-1].description if self._events else "none",
        }


# =====================================================================
# Memory Manager — orchestrates all memory subsystems
# =====================================================================


class MemoryManager(nn.Module):
    """Orchestrator for all five memory subsystems.

    Replaces the simple FIFO replay buffer + skill library with a complete
    developmental memory hierarchy.

    Training flow:
        1. Each step: compute surprise (RSSM pred error), store if surprising
        2. Each episode end: promote important events to autobiographical
        3. Sleep (periodic): consolidate episodic → semantic facts
        4. Policy training: sample from episodic memory (not raw replay)
        5. Forgetting: gradual importance decay

    VRAM: ~0.5 GB.
    """

    def __init__(
        self,
        d_model: int = 128,
        episodic_max: int = 10000,
        semantic_max: int = 1000,
        autobiographical_max: int = 100,
        surprise_threshold: float = 0.01,
        consolidation_every_steps: int = 50000,
        consolidation_batch_size: int = 256,
    ) -> None:
        super().__init__()
        self._d_model = d_model

        self.episodic = EpisodicMemory(
            max_entries=episodic_max,
            d_model=d_model,
            surprise_threshold=surprise_threshold,
        )
        self.semantic = SemanticMemory(
            max_facts=semantic_max,
            d_model=d_model,
        )
        self.autobiographical = AutobiographicalMemory(
            max_events=autobiographical_max,
        )

        self._consolidation_every = consolidation_every_steps
        self._consolidation_bs = consolidation_batch_size
        self._last_consolidation_step = -consolidation_every_steps
        self._total_stored = 0
        self._total_promoted = 0

    @property
    def capacity(self) -> int:
        return self.episodic.capacity

    def __len__(self) -> int:
        return len(self.episodic)

    # ------------------------------------------------------------------ store

    def store_experience(
        self,
        hidden_state: torch.Tensor,     # (d_model,)
        action: int,
        reward: float,
        surprise: float,
        global_step: int,
        episode_id: int,
        tags: list[str] | None = None,
    ) -> None:
        """Store a single step in episodic memory if surprising."""
        result = self.episodic.store(
            obs_embedding=hidden_state.detach(),
            action=action,
            reward=reward,
            surprise=surprise,
            global_step=global_step,
            episode_id=episode_id,
            tags=tags,
        )
        if result is not None:
            self._total_stored += 1

    def promote_to_life_event(
        self,
        step: int,
        description: str,
        importance: float,
        episode_id: int,
        lesson: str = "",
    ) -> None:
        """Promote an important episode to autobiographical memory."""
        result = self.autobiographical.add_event(
            step=step,
            description=description,
            importance=importance,
            episode_id=episode_id,
            lesson=lesson,
        )
        if result is not None:
            self._total_promoted += 1

    # ------------------------------------------------------------------ retrieve

    def retrieve_episodic(
        self, query_embedding: torch.Tensor, k: int = 16,
    ) -> list[EpisodicEntry]:
        """Retrieve similar memories for rehearsal."""
        entries = self.episodic.retrieve_by_similarity(query_embedding, k)
        for e in entries:
            self.episodic.mark_accessed(e.derivation_id if hasattr(e, 'derivation_id') else hash(e), 0)
        return entries

    def retrieve_semantic(
        self, query_embedding: torch.Tensor, k: int = 8,
    ) -> list[SemanticFact]:
        return self.semantic.query(query_embedding, k)

    def get_life_story(self, n: int = 20) -> str:
        return self.autobiographical.get_life_story(n)

    # ------------------------------------------------------------------ consolidation

    def should_consolidate(self, step: int) -> bool:
        return (step - self._last_consolidation_step) >= self._consolidation_every

    def consolidate(self, global_step: int) -> dict[str, Any]:
        """Nightly consolidation: episodic → semantic.

        Scans recent episodic memories, extracts patterns, creates semantic facts.
        Also decays importance of all memories.
        """
        self._last_consolidation_step = global_step
        result = {"new_facts": 0, "updated_facts": 0}

        # Get recent important memories
        recent = self.episodic.retrieve_important(limit=self._consolidation_bs)
        if not recent:
            return result

        # Pattern extraction: group by action, find common properties
        action_groups: dict[int, list[EpisodicEntry]] = {}
        for e in recent:
            action_groups.setdefault(e.action, []).append(e)

        for action, entries in action_groups.items():
            if len(entries) < 3:
                continue
            avg_reward = float(np.mean([e.reward for e in entries]))
            avg_surprise = float(np.mean([e.surprise for e in entries]))
            centroid = torch.stack([e.obs_embedding for e in entries]).mean(dim=0)

            # Create semantic fact
            fact_key = f"action_{action}_reward_{avg_reward:.2f}_surprise_{avg_surprise:.2f}"
            confidence = 0.3 + 0.3 * (1.0 / (1.0 + avg_surprise)) + 0.2 * min(1.0, avg_reward)
            self.semantic.add_or_update(
                key=fact_key,
                embedding=centroid,
                confidence_delta=confidence,
                step=global_step,
                episode_id=entries[0].episode_id,
            )
            result["new_facts" if fact_key not in self.semantic._facts else "updated_facts"] += 1

        # Decay
        self.episodic.decay_importance()

        return result

    # ------------------------------------------------------------------ diagnostics

    def summary(self) -> dict:
        return {
            "episodic": self.episodic.summary(),
            "semantic": self.semantic.summary(),
            "autobiographical": self.autobiographical.summary(),
            "total_stored": self._total_stored,
            "total_promoted": self._total_promoted,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_stored": self._total_stored,
            "total_promoted": self._total_promoted,
            "last_consolidation_step": self._last_consolidation_step,
            "episodic_entries": [
                {
                    "action": e.action,
                    "reward": e.reward,
                    "surprise": e.surprise,
                    "importance": e.importance,
                    "global_step": e.global_step,
                    "episode_id": e.episode_id,
                    "tags": e.tags,
                }
                for e in self.episodic._entries.values()
            ],
            "semantic_facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "consolidation_count": f.consolidation_count,
                }
                for f in self.semantic._facts.values()
            ],
            "life_events": [
                {
                    "timestamp_step": e.timestamp_step,
                    "description": e.description,
                    "emotional_weight": e.emotional_weight,
                }
                for e in self.autobiographical._events
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._total_stored = int(state.get("total_stored", 0))
        self._total_promoted = int(state.get("total_promoted", 0))
        self._last_consolidation_step = int(state.get("last_consolidation_step", -self._consolidation_every))
        self.episodic._entries.clear()
        self.semantic._facts.clear()
        # Note: full restoration of episodic entries requires replay buffer data;
        # here we restore metadata only.
