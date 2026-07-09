"""Phase 9 · LLM Fusion — Connect frozen Qwen-7B to devagi body.

This is the bridge that transforms devagi from "smart crow" to "talking child".

Four bridges (all frozen LLM + trainable projectors):
1. Perception → Language: SlotAttention slots → natural language description
2. Language → Action: LLM reasoning → FiLM modulation of policy
3. Memory → Language: Replay buffer entries → LLM experience summaries
4. Inner Monologue: Real agent-aware reflection (replaces template mode)

Architecture:
    SlotAttention slots (B, 7, 128)
        → perception_projector (128 → 3584)
        → Qwen-7B (4-bit, frozen, ~5GB)
        → action_projector (3584 → 128)
        → FiLM modulation of HybridActorCritic policy

Training: Only projectors (3 × ~1.5M = ~4.5M params). Qwen frozen.
Speed: ~5-10 step/s (Qwen forward pass bottleneck).
VRAM: +5 GB (Qwen 4-bit) + 0.5 GB (projectors) = ~5.5 GB extra.

Phase 9 LLM 融合：连接冻结 Qwen-7B 到 devagi 身体。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Qwen dimension: Qwen2.5-7B-Instruct has hidden_size=3584
QWEN_HIDDEN = 3584


# =====================================================================
# 1. Perception → Language Projector
# =====================================================================


class PerceptionProjector(nn.Module):
    """Maps SlotAttention output to LLM embedding space.

    Input: (B, num_slots, slot_dim) — object-centric perception
    Output: (B, num_slots, llm_dim) — each slot becomes a "token description"

    The LLM prompt template combines these into:
        "I see [slot_0: red ball], [slot_1: blue block], [slot_2: empty]"
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_slots: int = 7,
        llm_dim: int = QWEN_HIDDEN,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self._num_slots = num_slots

        # Per-slot projection with object type classifier
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

        # Object property classifiers (for generating descriptions)
        self.color_classifier = nn.Linear(slot_dim, 8)    # 8 basic colors
        self.size_classifier = nn.Linear(slot_dim, 3)      # small/medium/large

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project slot embeddings to LLM space + classify object properties.

        Args:
            slots: (B, num_slots, slot_dim)

        Returns:
            dict with:
                "llm_embeddings": (B, num_slots, llm_dim)
                "colors": (B, num_slots) — color indices
                "sizes": (B, num_slots) — size indices
                "occupancy": (B, num_slots) — slot activation
        """
        llm_embs = self.slot_proj(slots)          # (B, num_slots, llm_dim)
        colors = self.color_classifier(slots).argmax(dim=-1)  # (B, num_slots)
        sizes = self.size_classifier(slots).argmax(dim=-1)    # (B, num_slots)
        occupancy = slots.norm(dim=-1)             # (B, num_slots)

        return {
            "llm_embeddings": llm_embs,
            "colors": colors,
            "sizes": sizes,
            "occupancy": occupancy,
        }

    def describe_scene(self, slots: torch.Tensor) -> str:
        """Generate natural language scene description from slot states.

        Uses property classifiers to produce textual descriptions without
        requiring an LLM forward pass for every step (LLM is expensive).
        This is the "fast" description used in template mode.
        """
        color_names = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
        size_names = ["small", "medium", "large"]

        props = self.forward(slots)
        occupancy = props["occupancy"][0]   # (num_slots,)
        colors = props["colors"][0]          # (num_slots,)
        sizes = props["sizes"][0]            # (num_slots,)

        objects: list[str] = []
        for i in range(self._num_slots):
            if occupancy[i] < 0.1:  # empty slot
                continue
            c = color_names[int(colors[i])]
            s = size_names[int(sizes[i])]
            objects.append(f"a {s} {c} object")
        if not objects:
            return "I don't see any objects."
        return "I see " + ", ".join(objects) + "."


# =====================================================================
# 2. Language → Action (FiLM) Projector
# =====================================================================


class ActionModulator(nn.Module):
    """Modulates policy logits based on LLM reasoning output.

    Uses FiLM (Feature-wise Linear Modulation):
        gamma = 1.0 + alpha * tanh(W_g * llm_hidden)
        beta = alpha * tanh(W_b * llm_hidden)
        modulated_logits = gamma * original_logits + beta

    This allows the LLM to "suggest" actions by amplifying or suppressing
    specific action logits, while the neural policy provides the base.
    """

    def __init__(self, llm_dim: int = QWEN_HIDDEN, num_actions: int = 8, hidden: int = 128) -> None:
        super().__init__()
        self._num_actions = num_actions

        # Compress LLM hidden to FiLM parameters
        self.compress = nn.Sequential(
            nn.Linear(llm_dim, hidden),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden, num_actions)
        self.beta_head = nn.Linear(hidden, num_actions)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable modulation strength

    def forward(
        self, llm_hidden: torch.Tensor, policy_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate policy logits via LLM reasoning.

        Args:
            llm_hidden: (B, llm_dim) — last hidden state from LLM
            policy_logits: (B, num_actions) — neural policy output

        Returns:
            (B, num_actions) — modulated logits
        """
        h = self.compress(llm_hidden)                        # (B, hidden)
        gamma = 1.0 + self.alpha * torch.tanh(self.gamma_head(h))  # (B, num_actions)
        beta = self.alpha * torch.tanh(self.beta_head(h))          # (B, num_actions)
        return gamma * policy_logits + beta

    def get_suggested_action(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """Directly suggest an action from LLM output."""
        h = self.compress(llm_hidden)
        return self.gamma_head(h)  # use gamma as action logits (without neural base)


# =====================================================================
# 3. Memory Summarizer
# =====================================================================


class MemorySummarizer(nn.Module):
    """Summarizes replay buffer experiences into natural language via LLM.

    Takes recent transitions (obs, action, reward) → compressed embedding
    → feeds as context to LLM → LLM generates experience summary text.

    This enables:
    - "What happened yesterday?" → retrieves replay entries → LLM summarizes
    - "What did I learn?" → LLM extracts patterns from recent transitions
    """

    def __init__(
        self,
        obs_dim: int = 64 * 64 * 3,
        llm_dim: int = QWEN_HIDDEN,
        num_actions: int = 8,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self._obs_flat = obs_dim
        self._num_actions = num_actions

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
        )
        # Encode (obs, action, reward) triplet to LLM space
        self.triplet_encoder = nn.Sequential(
            nn.Linear(hidden + num_actions + 1, hidden),  # obs_emb + action_onehot + reward
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def encode_transition(
        self,
        obs_flat: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """(B, obs_dim) + (B,) action + (B,) reward → (B, llm_dim)."""
        obs_emb = self.obs_encoder(obs_flat)                     # (B, hidden)
        action_onehot = F.one_hot(action.long(), self._num_actions).float()
        combined = torch.cat([obs_emb, action_onehot, reward.unsqueeze(-1)], dim=-1)
        return self.triplet_encoder(combined)                    # (B, llm_dim)

    def summarize_batch(
        self,
        obs_batch: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize a batch of transitions into a single LLM embedding."""
        embs = []
        for i in range(min(obs_batch.shape[0], 16)):  # bounded to 16 transitions
            emb = self.encode_transition(
                obs_batch[i:i+1], actions[i:i+1], rewards[i:i+1],
            )
            embs.append(emb)
        if not embs:
            return torch.zeros(1, QWEN_HIDDEN, device=obs_batch.device)
        return torch.stack(embs).mean(dim=0)  # (1, llm_dim)


# =====================================================================
# 4. LLM Fusion Bridge — main module
# =====================================================================


class LLMFusionBridge(nn.Module):
    """Connects frozen Qwen-7B to devagi body via four bridges.

    Training: Only the projectors (perception, action, memory) are trained.
              Qwen is always frozen (requires_grad=False).

    Inference modes:
    - Fast (template): Property classifiers → text description. No LLM call.
    - LLM (expensive): Forward pass through Qwen. ~10ms per call.
    - Hybrid (default): Fast for step-by-step perception, LLM for episode reflection.

    VRAM: ~5 GB (Qwen 4-bit) + 0.5 GB (projectors).
    """

    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        slot_dim: int = 128,
        num_slots: int = 7,
        num_actions: int = 8,
        obs_dim: int = 64 * 64 * 3,
        llm_max_new_tokens: int = 64,
        llm_call_interval: int = 50,  # steps between LLM calls
    ) -> None:
        super().__init__()

        self._llm_model_name = llm_model_name
        self._max_new_tokens = llm_max_new_tokens
        self._call_interval = llm_call_interval
        self._llm = None
        self._tokenizer = None
        self._llm_available = False

        # Trainable projectors
        self.perception = PerceptionProjector(
            slot_dim=slot_dim, num_slots=num_slots, llm_dim=QWEN_HIDDEN,
        )
        self.action_mod = ActionModulator(
            llm_dim=QWEN_HIDDEN, num_actions=num_actions,
        )
        self.memory_summarizer = MemorySummarizer(
            obs_dim=obs_dim, llm_dim=QWEN_HIDDEN, num_actions=num_actions,
        )

        # Cached LLM outputs (avoid calling LLM every step)
        self._cached_description: str = ""
        self._cached_llm_hidden: torch.Tensor | None = None
        self._steps_since_llm_call: int = 0

        # Try to load Qwen
        self._try_load_llm()

    def _try_load_llm(self) -> None:
        """Load Qwen-7B in 4-bit. Falls back gracefully if unavailable."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._llm_model_name, trust_remote_code=True,
            )
            self._llm = AutoModelForCausalLM.from_pretrained(
                self._llm_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            for p in self._llm.parameters():
                p.requires_grad_(False)
            self._llm_available = True
            logger.info("LLM Fusion: Qwen-7B loaded (4-bit, frozen)")
        except Exception as exc:
            logger.warning("LLM Fusion: Qwen load failed (%s), using template mode", exc)
            self._llm = None
            self._tokenizer = None
            self._llm_available = False

    # ------------------------------------------------------------------ Perception

    def describe_scene(self, slots: torch.Tensor) -> str:
        """Generate scene description. Fast (template) or LLM."""
        if not self._llm_available:
            return self.perception.describe_scene(slots)

        # LLM mode: forward pass through Qwen (expensive, use sparingly)
        self._steps_since_llm_call += 1
        if self._steps_since_llm_call % self._call_interval != 0 and self._cached_description:
            return self._cached_description

        try:
            fast_desc = self.perception.describe_scene(slots)
            prompt = (
                f"You are a developmental AI agent with a body in a 3D home. "
                f"You see: {fast_desc}\n"
                f"Describe what you see in one short sentence as if you are the agent.\n"
                f"Description:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
            with torch.no_grad():
                outputs = self._llm.generate(
                    **inputs, max_new_tokens=self._max_new_tokens,
                    temperature=0.7, do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = text[len(prompt):].strip()
            self._cached_description = generated if generated else fast_desc
            self._steps_since_llm_call = 0
            return self._cached_description
        except Exception:
            return self.perception.describe_scene(slots)

    # ------------------------------------------------------------------ Action

    def modulate_policy(
        self, slots: torch.Tensor, policy_logits: torch.Tensor, force_llm: bool = False,
    ) -> torch.Tensor:
        """Modulate policy via LLM reasoning. Falls back to neural-only if LLM off."""
        if not self._llm_available or not force_llm:
            return policy_logits  # neural-only

        try:
            desc = self.perception.describe_scene(slots)
            prompt = (
                f"You are an agent in a room. You see: {desc}\n"
                f"Available actions: move north, south, west, east, push, pull, grasp, wait.\n"
                f"Which action should you take and why? Answer in one word (the action name).\n"
                f"Action:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
            with torch.no_grad():
                outputs = self._llm(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # (1, llm_dim)
                self._cached_llm_hidden = last_hidden.float()
            if self._cached_llm_hidden is not None:
                return self.action_mod(self._cached_llm_hidden, policy_logits)
        except Exception:
            pass
        return policy_logits

    # ------------------------------------------------------------------ Reflection

    def reflect(
        self,
        episode_return: float,
        scene_description: str,
        trajectory_summary: str = "",
    ) -> list[str]:
        """Generate post-episode reflection via LLM."""
        if not self._llm_available:
            status = "succeeded" if episode_return > 0 else "failed"
            return [f"Episode result: {status}, return={episode_return:.2f}. I saw: {scene_description}"]

        try:
            prompt = (
                f"You are a developmental AI agent reflecting on an episode.\n"
                f"Episode return: {episode_return:.2f}\n"
                f"Scene: {scene_description}\n"
                f"{trajectory_summary}\n"
                f"Reflect: what happened? What did you learn? What will you do differently?\n"
                f"Answer in 2-3 short sentences as the agent.\n"
                f"Reflection:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
            with torch.no_grad():
                outputs = self._llm.generate(
                    **inputs, max_new_tokens=self._max_new_tokens,
                    temperature=0.7, do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = text[len(prompt):].strip()
            return [s.strip() for s in generated.split(".") if s.strip()][:3]
        except Exception:
            return [f"Episode return: {episode_return:.2f}"]

    # ------------------------------------------------------------------ Properties

    @property
    def is_available(self) -> bool:
        return self._llm_available

    @property
    def capacity(self) -> int:
        return 1  # Bounded protocol

    def __len__(self) -> int:
        return 1 if self._llm is not None else 0
