"""Narrative Loop Controller — close the memory->narrative->policy loop.

Stage 19 core module. Orchestrates the self-narrative cycle:

    experience -> AutobiographicalMemory -> IdentityNarrative
        -> InnerDialogue lessons -> (optional FiLM modulation)
        -> kanren predict_action -> symbol bias on action logits
        -> behavior -> experience (next cycle)

Unlike the Stage 18 status quo (IdentityNarrative logged every 50K steps,
InnerDialogue lessons never used, kanren rules never consumed), this module
makes the narrative genuinely *affect* action selection via:

1. ``episode_end_hook``: stores life events, periodically generates the
   self-narrative, and refreshes the symbol action bias.
2. ``get_symbol_bias``: returns a (num_actions,) logit bias derived from
   the kanren backend's highest-confidence rule match. Injected by
   :class:`HierarchicalActorCritic` via a callback (see train.py).
3. ``step_hook``: delegates to ThoughtActionLoop for FiLM modulation
   (optional; requires a language encoder).

Bounded (Axiom 1):
- Cached narrative: one string.
- Cached traits: one 5-float dict.
- Symbol bias: one (num_actions,) tensor.
- No growth over time. All state reset-able.

有界: 叙事字符串 + 5 维 trait + (num_actions,) 偏置, 无增长。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NarrativeLoopController(nn.Module):
    """Closed-loop self-narrative controller for Stage 19.

    All components are optional — graceful degradation when any is missing
    (Axiom: no hard dependencies). The module is a thin orchestrator over
    existing bounded modules.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_actions: int = 12,
        autobiographical: Any | None = None,
        identity_narrative: Any | None = None,
        symbol_backend: Any | None = None,
        thought_loop: Any | None = None,
        language_encoder: Any | None = None,
        narrative_every_episodes: int = 10,
        symbol_bias_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self._num_actions = int(num_actions)
        self._every = max(1, int(narrative_every_episodes))
        self._symbol_bias_weight = float(symbol_bias_weight)

        # Optional components
        self.autobiographical = autobiographical
        self.identity_narrative = identity_narrative
        self.symbol_backend = symbol_backend
        self.thought_loop = thought_loop
        self.language_encoder = language_encoder

        # Cached narrative state (bounded)
        self._last_narrative: str = ""
        self._last_traits: dict[str, float] = {}
        self._narrative_count: int = 0
        self._episode_count: int = 0

        # Cached symbol bias (num_actions,) or None — plain attribute,
        # fresh tensor on every update (no inplace ops, AGENTS.md §12)
        self._symbol_bias: torch.Tensor | None = None

    # ------------------------------------------------------------ boundedness

    @property
    def capacity(self) -> int:
        return 1

    def __len__(self) -> int:
        return 1

    @property
    def has_active_narrative(self) -> bool:
        return bool(self._last_narrative)

    @property
    def has_symbol_bias(self) -> bool:
        return self._symbol_bias is not None

    # ---------------------------------------------------------------- hooks

    def step_hook(
        self,
        hidden_state: torch.Tensor,
        episode_return: float = 0.0,
        episode_done: bool = False,
    ) -> str | None:
        """Call every training step.

        Delegates to ThoughtActionLoop for FiLM modulation (optional).
        Returns the thought text if a new thought was generated.
        """
        if self.thought_loop is not None:
            try:
                return self.thought_loop.maybe_think(
                    hidden_state, episode_return, episode_done,
                )
            except Exception as _e:
                logger.warning("[narrative] step_hook failed: %s", _e)
        return None

    def episode_end_hook(
        self,
        step: int,
        ep_ret: float,
        ep_id: int,
        description: str = "",
        lesson: str = "",
    ) -> None:
        """Call at episode end: store life event + periodic narrative + bias."""
        self._episode_count += 1

        # 1. Store significant episode in autobiographical memory
        if self.autobiographical is not None and ep_ret > 0:
            try:
                self.autobiographical.add_event(
                    step=step,
                    description=description or f"Episode {ep_id}: return={ep_ret:.2f}",
                    importance=float(ep_ret),
                    episode_id=ep_id,
                    lesson=lesson,
                )
            except Exception as _e:
                logger.warning("[narrative] add_event failed: %s", _e)

        # 2. Periodic identity narrative
        if (self.identity_narrative is not None
                and self.autobiographical is not None
                and self._episode_count % self._every == 0):
            try:
                events = self.autobiographical._events
                min_events = getattr(self.identity_narrative, "_min_events", 20)
                if len(events) >= min_events:
                    self._generate_narrative(events)
            except Exception as _e:
                logger.warning("[narrative] narrative generation failed: %s", _e)

        # 3. Refresh symbol bias from kanren rules (every episode end)
        self._update_symbol_bias()

    # -------------------------------------------------------------- internals

    def _generate_narrative(self, events: list[Any]) -> None:
        """Run IdentityNarrative and cache the result (+ optional FiLM)."""
        out = self.identity_narrative(events)
        self._last_traits = dict(out.get("traits", {}))
        self._last_narrative = str(out.get("narrative", ""))
        self._narrative_count += 1
        logger.info("[narrative] #%d: %s (events=%d, traits=%s)",
                    self._narrative_count, self._last_narrative[:80],
                    len(events),
                    {k: round(v, 2) for k, v in self._last_traits.items()})

        # Optional: encode narrative -> FiLM modulation via ThoughtActionLoop
        if self.language_encoder is not None and self.thought_loop is not None:
            try:
                with torch.no_grad():
                    lang_emb = self.language_encoder.encode_text(self._last_narrative)
                    if lang_emb.dim() == 2:
                        lang_emb = lang_emb.mean(dim=0)
                    tl = self.thought_loop
                    tl._cached_lang_embedding = lang_emb.to(tl._cached_lang_embedding.device)
                    tl._has_active_thought = True
            except Exception as _e:
                logger.debug("[narrative] FiLM encoding failed: %s", _e)

    def _update_symbol_bias(self) -> None:
        """Query the kanren backend for the best rule match -> action bias.

        The bias is a one-hot (num_actions,) tensor scaled by
        ``symbol_bias_weight * confidence``. Detached (kanren is not
        differentiable); PPO gradients are unaffected (learning-back is
        via feedback(), not end-to-end).
        """
        self._symbol_bias = None
        if self.symbol_backend is None:
            return
        try:
            best_action = -1
            best_conf = 0.0
            for rule in self.symbol_backend._rules_db:
                if rule["then"][0] != "action":
                    continue
                result = self.symbol_backend.predict_action(rule["if"])
                if result.answers:
                    # predict_action answers are ("action", int)
                    a = result.answers[0][1]
                    if isinstance(a, int) and a >= 0 and result.confidence > best_conf:
                        best_action = a
                        best_conf = result.confidence
            if best_action >= 0:
                bias = torch.zeros(self._num_actions, dtype=torch.float32)
                bias[best_action] = self._symbol_bias_weight * min(1.0, best_conf)
                self._symbol_bias = bias  # fresh tensor (no inplace)
        except Exception as _e:
            logger.warning("[narrative] symbol bias update failed: %s", _e)

    def get_symbol_bias(self) -> torch.Tensor | None:
        """Return the (num_actions,) logit bias or None (no bias)."""
        return self._symbol_bias

    # ---------------------------------------------------------------- state

    def reset_episode_state(self) -> None:
        """Reset per-episode counters (not the narrative itself)."""
        self._episode_count = 0

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["_last_narrative"] = self._last_narrative
        sd["_last_traits"] = self._last_traits
        sd["_narrative_count"] = self._narrative_count
        if self._symbol_bias is not None:
            sd["_symbol_bias"] = self._symbol_bias
        return sd

    def load_state_dict(self, state_dict, strict=True):
        sd = dict(state_dict)
        self._last_narrative = sd.pop("_last_narrative", "")
        self._last_traits = sd.pop("_last_traits", {})
        self._narrative_count = int(sd.pop("_narrative_count", 0))
        sb = sd.pop("_symbol_bias", None)
        self._symbol_bias = sb.detach() if sb is not None else None
        return super().load_state_dict(sd, strict=strict)
