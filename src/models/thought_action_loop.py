"""Thought-Action Loop: inner dialogue → language → action closed loop.

Connects the agent's "thoughts" (from InnerDialogue) to its actions via:

    hidden state → SelfModel → reflection → InnerDialogue → thought text
                                                                ↓
                                          CLIP text encoder → language embedding
                                                                ↓
                                          FiLM: vision × language → fused features
                                                                ↓
                                          Hybrid backbone → action

This creates a CLOSED LOOP where the agent's thoughts actually influence
its behavior, not just logged text.

Every ``think_every_steps`` steps:
1. Read the agent's hidden state.
2. SelfModel assesses confidence/familiarity/progress.
3. ReflectionLoop generates a structured reflection.
4. InnerDialogue generates a natural-language thought.
5. CLIP encodes the thought → language embedding.
6. The language embedding is cached and used to FiLM-modulate vision features
   for the next ``think_every_steps`` steps.

Between thought steps, the cached language embedding is reused (efficiency).

Bounded: the thought is a single string, language embedding is (d_model,).
No accumulation. Axiom 1 satisfied.

思维闭环：让智能体的"想法"真正影响它的动作。
每 N 步思考一次，想法通过 CLIP 编码 → FiLM 调制视觉 → 改变动作。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .metacognition import (
    EpisodeReflection,
    InnerDialogue,
    ReflectionLoop,
    SelfModel,
)

logger = logging.getLogger(__name__)


class ThoughtActionLoop(nn.Module):
    """Closed loop: thought → language embedding → action modulation.

    This module sits BETWEEN the vision encoder and the Hybrid backbone.
    Every N steps, it generates a thought and converts it into a FiLM
    modulation vector. Between thought steps, the modulation is cached.

    Usage in a policy network:

        feats = vision_encoder(obs)           # (B, d_model)
        feats = thought_loop.modulate(feats)  # FiLM with cached thought
        seq = feats.unsqueeze(1)
        out = backbone(seq)

    The thought loop is passive (no gradient through the dialogue generation)
    but the FiLM projection IS trainable.

    Bounded: cached language embedding is (d_model,). One thought string.
    No growth over time.
    """

    def __init__(
        self,
        d_model: int = 384,
        self_model: SelfModel | None = None,
        reflection_loop: ReflectionLoop | None = None,
        inner_dialogue: InnerDialogue | None = None,
        language_encoder: Any | None = None,
        think_every_steps: int = 50,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._think_every = max(1, int(think_every_steps))
        self._step_count = 0

        # Components (all optional — graceful degradation)
        self.self_model = self_model
        self.reflection_loop = reflection_loop
        self.inner_dialogue = inner_dialogue or InnerDialogue(mode="template")
        self.language_encoder = language_encoder

        # Cached language embedding (FiLM modulation source)
        # Initialize to zeros → FiLM starts as identity (no modulation)
        self.register_buffer(
            "_cached_lang_embedding",
            torch.zeros(d_model),
            persistent=False,
        )
        self._has_active_thought = False

        # Trainable FiLM projection (projects language embedding → modulation)
        # If no language encoder is available, this still works as a no-op
        # because the cached embedding stays at zeros.
        self.film_projection = nn.Linear(d_model, d_model)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def has_active_thought(self) -> bool:
        """True if a non-trivial thought is currently cached."""
        return self._has_active_thought

    def modulate(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """Apply the cached thought's modulation to vision features.

        This is called every forward pass. The modulation is updated
        every ``think_every_steps`` by :meth:`maybe_think`.

        Args:
            vision_feats: (B, d_model) from the vision encoder.

        Returns:
            (B, d_model) modulated features.
        """
        if not self._has_active_thought:
            return vision_feats  # no thought → no modulation

        # Apply FiLM: scale + shift based on the cached thought embedding
        lang = self._cached_lang_embedding.unsqueeze(0).expand(vision_feats.shape[0], -1)
        gamma = 1.0 + 0.1 * torch.tanh(self.film_projection(lang))  # scale around 1
        beta = 0.1 * torch.tanh(self.film_projection(lang))         # small shift
        return gamma * vision_feats + beta

    def maybe_think(
        self,
        hidden_state: torch.Tensor,
        episode_return: float = 0.0,
        episode_done: bool = False,
    ) -> str | None:
        """Generate a thought if it's time. Returns the thought text or None.

        This should be called every step. It only generates a new thought
        every ``think_every_steps`` steps.

        Args:
            hidden_state: (d_model,) or (1, d_model) — the agent's current
                hidden state from the Hybrid backbone.
            episode_return: the current episode's return (for reflection).
            episode_done: whether the episode just ended.

        Returns:
            The thought text if a new thought was generated, else None.
        """
        self._step_count += 1
        if self._step_count % self._think_every != 0 and not episode_done:
            return None

        # --- 1. Self-assessment ---
        assessment = None
        if self.self_model is not None:
            h = hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state
            with torch.no_grad():
                assessment = self.self_model.assess(h.mean(dim=0))

        # --- 2. Reflection (if episode ended or periodic) ---
        reflection = None
        if self.reflection_loop is not None:
            if episode_done:
                reflection = self.reflection_loop.end_episode(episode_return)
            else:
                # Record step for ongoing episode
                self.reflection_loop.record_step(
                    hidden_state, action=0, reward=0.0, done=False,
                )

        # --- 3. Generate thought text ---
        thought_text = self._generate_thought(assessment, reflection)

        # --- 4. Encode thought → language embedding ---
        if self.language_encoder is not None and thought_text:
            try:
                with torch.no_grad():
                    lang_emb = self.language_encoder.encode_text(thought_text)
                    # Cache the embedding
                    if lang_emb.dim() == 2:
                        lang_emb = lang_emb.mean(dim=0)
                    self._cached_lang_embedding.copy_(lang_emb.to(self._cached_lang_embedding.device))
                    self._has_active_thought = True
            except Exception as exc:
                logger.debug("Thought encoding failed: %s", exc)

        return thought_text

    def _generate_thought(
        self,
        assessment: Any | None,
        reflection: EpisodeReflection | None,
    ) -> str:
        """Generate a natural-language thought from assessment + reflection."""
        if reflection is not None:
            lessons = self.inner_dialogue.generate(reflection)
            return " ".join(lessons[:2])  # Keep it short for encoding

        if assessment is not None:
            # Generate a brief situational thought
            parts = []
            if assessment.confidence < 0.3:
                parts.append("I am uncertain. I should explore carefully.")
            elif assessment.confidence > 0.9:
                parts.append("I am confident. I will proceed directly.")
            if assessment.familiarity < 0.3:
                parts.append("This is unfamiliar territory.")
            if assessment.progress > 0.6:
                parts.append("I am making good progress.")
            return " ".join(parts) if parts else "Continuing."

        return "Continuing."

    def get_cached_thought_embedding(self) -> torch.Tensor:
        """Return the current cached language embedding (d_model,)."""
        return self._cached_lang_embedding.clone()

    def reset(self) -> None:
        """Clear the cached thought (e.g., at episode boundaries)."""
        self._cached_lang_embedding.zero_()
        self._has_active_thought = False
        self._step_count = 0
