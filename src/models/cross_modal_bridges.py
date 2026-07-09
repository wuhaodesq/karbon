"""Cross-Modal Bridges — Language ↔ Touch, Planning, Memory.

After Phase 9 (language fusion via frozen Qwen-7B), the agent needs to
connect language to ALL its sensory and cognitive modalities, not just vision.

Three bridges:
1. Language ↔ Touch: Qwen understands "slippery", "hard", "heavy"
   by mapping touch/force embeddings to language space.
2. Language ↔ Planning: Qwen participates in multi-step planning
   by converting plan steps to/from natural language.
3. Language ↔ Memory: Qwen helps summarize and query episodic memories
   stored in the bounded replay buffer.

Each bridge is a small (2-layer MLP, ~50K params each) that projects
between modality-specific embedding spaces and the LLM's embedding space.

跨模态桥梁：语言 ↔ 触觉/规划/记忆。三个小型投影 MLP。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Language ↔ Touch Bridge
# =====================================================================


class TouchBridge(nn.Module):
    """Maps proprioceptive/touch signals to language space and vice versa.

    The agent learns to associate touch experiences with language:
    - High contact force → "heavy" / "pushing hard"
    - Low friction → "slippery"
    - Multiple simultaneous contacts → "jammed" / "stuck"

    Architecture: dual MLP (touch→lang, lang→touch) with shared trunk.
    """

    def __init__(self, touch_dim: int = 6, lang_dim: int = 3584, hidden: int = 128) -> None:
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(touch_dim + lang_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # Touch → Language
        self.touch_proj_in = nn.Linear(touch_dim, lang_dim)
        self.to_lang = nn.Linear(hidden, lang_dim)
        # Language → Touch
        self.lang_proj_in = nn.Linear(lang_dim, touch_dim)
        self.to_touch = nn.Linear(hidden, touch_dim)

    def touch_to_lang(self, proprio: torch.Tensor) -> torch.Tensor:
        """(B, 6) proprio → (B, lang_dim) language embedding."""
        tp = self.touch_proj_in(proprio)
        combined = torch.cat([proprio, torch.zeros_like(tp)], dim=-1)
        return self.to_lang(self.trunk(combined))

    def lang_to_touch(self, lang_emb: torch.Tensor) -> torch.Tensor:
        """(B, lang_dim) → (B, 6) predicted touch."""
        lp = self.lang_proj_in(lang_emb)
        combined = torch.cat([torch.zeros(lp.shape[0], 6, device=lang_emb.device), lang_emb], dim=-1)
        return self.to_touch(self.trunk(combined))

    def forward(
        self, proprio: torch.Tensor | None, lang_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Bidirectional projection. At least one input must be provided."""
        if proprio is not None:
            return self.touch_to_lang(proprio)
        if lang_emb is not None:
            return self.lang_to_touch(lang_emb)
        raise ValueError("At least one input required")


# =====================================================================
# Language ↔ Planning Bridge
# =====================================================================


class PlanningBridge(nn.Module):
    """Converts hierarchical plan steps to/from natural language.

    Plan step representation: (d_model,) vector from HierarchicalPolicy sub-goal head.
    Language: Qwen's embedding for step descriptions like "go to the red ball".

    This allows:
    - LLM → plan: "first get the key, then open the door" → two sub-goal vectors
    - Plan → LLM: sub-goal vector → "I'm now moving toward the key"
    """

    def __init__(self, plan_dim: int = 128, lang_dim: int = 3584, hidden: int = 128) -> None:
        super().__init__()
        self.plan_to_lang = nn.Sequential(
            nn.Linear(plan_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, lang_dim),
        )
        self.lang_to_plan = nn.Sequential(
            nn.Linear(lang_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, plan_dim),
        )

    def encode_plan(self, sub_goal: torch.Tensor) -> torch.Tensor:
        """(B, plan_dim) → (B, lang_dim)."""
        return self.plan_to_lang(sub_goal)

    def decode_plan(self, lang_emb: torch.Tensor) -> torch.Tensor:
        """(B, lang_dim) → (B, plan_dim) sub-goal prediction."""
        return self.lang_to_plan(lang_emb)


# =====================================================================
# Language ↔ Memory Bridge
# =====================================================================


class MemoryBridge(nn.Module):
    """Helps Qwen summarize and query episodic memories.

    Takes replay buffer entries (observation, action, reward) and projects
    them to language space for the LLM to summarize. Also converts LLM
    queries ("what happened yesterday near the red block?") into memory
    retrieval keys for the replay buffer's PER sampling.
    """

    def __init__(
        self,
        obs_dim: int = 64 * 64 * 3,
        lang_dim: int = 3584,
        mem_key_dim: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        # Memory → Language (summarize)
        self.mem_encoder = nn.Sequential(
            nn.Linear(obs_dim + 1 + lang_dim, hidden),  # obs + reward + prev_lang
            nn.GELU(),
            nn.Linear(hidden, lang_dim),
        )
        # Language → Memory Key (query)
        self.query_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, mem_key_dim),
        )

    def summarize(self, obs_flat: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim) + (B, 1) → (B, lang_dim) memory summary embedding."""
        combined = torch.cat([
            obs_flat,
            reward.unsqueeze(-1),
            torch.zeros(obs_flat.shape[0], self.query_encoder[0].in_features - obs_flat.shape[1] - 1,
                       device=obs_flat.device),
        ], dim=-1)
        return self.mem_encoder(combined)

    def query_to_key(self, query_emb: torch.Tensor) -> torch.Tensor:
        """(B, lang_dim) → (B, mem_key_dim) retrieval key."""
        return self.query_encoder(query_emb)


# =====================================================================
# CrossModalManager — orchestrates all bridges
# =====================================================================


@dataclass
class CrossModalState:
    touch_lang: torch.Tensor | None = None
    plan_lang: torch.Tensor | None = None
    memory_summary: torch.Tensor | None = None
    plan_text: str = ""
    memory_text: str = ""


class CrossModalManager(nn.Module):
    """Orchestrates the three cross-modal bridges.

    At each step, the agent can:
    1. Read proprioception → convert to touch language → feed to LLM
    2. Get current sub-goal → convert to plan language → LLM explains plan
    3. Retrieve memory → summarize → LLM reflects on past experience

    All three embeddings are concatenated before being fed to the Qwen LLM
    as additional context tokens.
    """

    def __init__(
        self,
        touch_dim: int = 6,
        plan_dim: int = 128,
        obs_dim: int = 64 * 64 * 3,
        lang_dim: int = 3584,
    ) -> None:
        super().__init__()
        self.touch_bridge = TouchBridge(touch_dim=touch_dim, lang_dim=lang_dim)
        self.plan_bridge = PlanningBridge(plan_dim=plan_dim, lang_dim=lang_dim)
        self.memory_bridge = MemoryBridge(obs_dim=obs_dim, lang_dim=lang_dim)

    def process_step(
        self,
        proprio: torch.Tensor,
        sub_goal: torch.Tensor | None,
        obs_flat: torch.Tensor,
        reward: float,
    ) -> CrossModalState:
        """Process one step through all three bridges."""
        state = CrossModalState()

        with torch.no_grad():
            # Touch
            if proprio is not None:
                state.touch_lang = self.touch_bridge.touch_to_lang(proprio.unsqueeze(0))

            # Planning
            if sub_goal is not None:
                state.plan_lang = self.plan_bridge.encode_plan(sub_goal.unsqueeze(0))

            # Memory
            if obs_flat is not None:
                reward_t = torch.tensor([[reward]], device=obs_flat.device)
                state.memory_summary = self.memory_bridge.summarize(
                    obs_flat.unsqueeze(0), reward_t,
                )

        return state

    def to_context_embedding(self, state: CrossModalState) -> torch.Tensor | None:
        """Combine all bridge outputs into a single context vector for LLM."""
        parts = []
        if state.touch_lang is not None:
            parts.append(state.touch_lang.squeeze(0))
        if state.plan_lang is not None:
            parts.append(state.plan_lang.squeeze(0))
        if state.memory_summary is not None:
            parts.append(state.memory_summary.squeeze(0))
        if parts:
            return torch.stack(parts).mean(dim=0)  # (lang_dim,)
        return None
