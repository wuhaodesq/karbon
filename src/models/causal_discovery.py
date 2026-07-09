"""Causal Discovery via Counterfactual Intervention.

Extends the RSSM world model and CounterfactualImagination with:
1. Intervention: "What if I had done X instead?"
2. Counterfactual comparison: "Did doing X cause Y?"
3. Causal graph construction: "A → B" edges from repeated interventions.

This upgrades the agent from "I can predict what happens" (RSSM) to
"I understand WHY it happens" (causal reasoning).

Architecture:
    RSSM world model
    → imagine_step(action_A) → predicted_state_A
    → imagine_step(action_B) → predicted_state_B
    → diff = ||state_A - state_B||
    → if diff is large, action choice CAUSES a meaningful change
    → record causal edge: "action A causes state change of magnitude diff"

因果发现：通过反事实干预从世界模型提取因果关系。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    source: str           # "action_3" or "object_2_moved"
    target: str           # "object_1_moved" or "reward_increased"
    strength: float = 0.0 # [0, 1], updated via EMA
    sample_count: int = 0
    last_updated_step: int = 0


@dataclass
class CausalGraph:
    edges: dict[tuple[str, str], CausalEdge] = field(default_factory=dict)

    def record_cause(
        self,
        source: str,
        target: str,
        strength_delta: float,
        step: int,
        ema_alpha: float = 0.1,
    ) -> CausalEdge:
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = CausalEdge(source=source, target=target)
        edge = self.edges[key]
        edge.strength = (1 - ema_alpha) * edge.strength + ema_alpha * strength_delta
        edge.strength = max(0.0, min(1.0, edge.strength))
        edge.sample_count += 1
        edge.last_updated_step = step
        return edge

    def get_causes(self, target: str, min_strength: float = 0.3) -> list[CausalEdge]:
        return sorted(
            [e for (s, t), e in self.edges.items() if t == target and e.strength >= min_strength],
            key=lambda e: -e.strength,
        )

    def get_effects(self, source: str, min_strength: float = 0.3) -> list[CausalEdge]:
        return sorted(
            [e for (s, t), e in self.edges.items() if s == source and e.strength >= min_strength],
            key=lambda e: -e.strength,
        )

    def summary(self) -> dict:
        return {
            "num_edges": len(self.edges),
            "mean_strength": sum(e.strength for e in self.edges.values()) / max(1, len(self.edges)),
            "top_edges": [
                f"{e.source} → {e.target} ({e.strength:.2f})"
                for e in sorted(self.edges.values(), key=lambda x: -x.strength)[:5]
            ],
        }


class CausalDiscovery:
    """Discovers causal relationships via counterfactual intervention.

    1. Observes actual action→outcome transition
    2. Imagines counterfactual: "what if I had done a different action?"
    3. Compares predicted outcomes → computes causal effect size
    4. Records causal edges in a graph

    Bounded: max_edges limits the causal graph size (Axiom 1).
    """

    def __init__(
        self,
        num_actions: int = 8,
        max_edges: int = 256,
        min_intervention_effect: float = 0.01,
        ema_alpha: float = 0.1,
    ) -> None:
        self._num_actions = num_actions
        self._max_edges = max_edges
        self._min_effect = min_intervention_effect
        self._ema_alpha = ema_alpha
        self._graph = CausalGraph()
        self._intervention_count = 0

    @property
    def capacity(self) -> int:
        return self._max_edges

    def __len__(self) -> int:
        return len(self._graph.edges)

    def intervene(
        self,
        world_model: Any,  # RSSM
        initial_state: Any,  # RSSMState
        actual_action: int,
        slot_states: torch.Tensor,  # (num_slots, slot_dim)
        step: int,
    ) -> dict[str, float]:
        """Perform counterfactual interventions and record causal effects.

        Args:
            world_model: RSSM instance with imagine_step and decode.
            initial_state: RSSMState before the action.
            actual_action: the action actually taken.
            slot_states: current SlotAttention output (for object-level causation).
            step: global step count.

        Returns:
            dict mapping "action_X_effect" → effect magnitude.
        """
        effects: dict[str, float] = {}
        if world_model is None or self._intervention_count >= self._max_edges:
            return effects

        # Baseline: imagine the actual action
        actual_onehot = F.one_hot(
            torch.tensor([actual_action]), self._num_actions,
        ).float().to(slot_states.device)
        state_actual, _ = world_model.imagine_step(initial_state, actual_onehot)
        pred_actual = world_model.decode(state_actual)

        # Counterfactual: imagine each alternative action
        for alt_action in range(self._num_actions):
            if alt_action == actual_action:
                continue
            if self._intervention_count >= self._max_edges:
                break

            alt_onehot = F.one_hot(
                torch.tensor([alt_action]), self._num_actions,
            ).float().to(slot_states.device)
            state_alt, _ = world_model.imagine_step(initial_state, alt_onehot)
            pred_alt = world_model.decode(state_alt)

            # Effect size: how different would the world be?
            effect = float(F.mse_loss(pred_alt, pred_actual).item())
            effects[f"action_{alt_action}_effect"] = effect

            if effect > self._min_effect:
                self._graph.record_cause(
                    source=f"action_{actual_action}",
                    target=f"world_state",
                    strength_delta=min(1.0, effect * 10.0),
                    step=step,
                    ema_alpha=self._ema_alpha,
                )

            # Object-level causation: which slot changed most?
            if slot_states.dim() == 2 and slot_states.shape[0] > 0:
                actual_slots_norm = slot_states.norm(dim=-1)
                alt_slots = _reconstruct_slots(
                    world_model, state_alt, slot_states.shape[0], slot_states.shape[1],
                )
                if alt_slots is not None:
                    alt_slots_norm = alt_slots.norm(dim=-1)
                    slot_deltas = (actual_slots_norm - alt_slots_norm).abs()
                    best_slot = int(slot_deltas.argmax().item())
                    delta = float(slot_deltas[best_slot].item())
                    if delta > self._min_effect:
                        self._graph.record_cause(
                            source=f"action_{actual_action}",
                            target=f"object_{best_slot}_changed",
                            strength_delta=min(1.0, delta),
                            step=step,
                            ema_alpha=self._ema_alpha,
                        )

            self._intervention_count += 1

        return effects

    def query_why(self, target: str) -> list[str]:
        """Return explanations: what causes this target?"""
        edges = self._graph.get_causes(target)
        return [f"{e.source} → {e.target} (strength={e.strength:.2f})" for e in edges]

    def query_what_if(self, source: str) -> list[str]:
        """Return predictions: what would happen if this source activates?"""
        edges = self._graph.get_effects(source)
        return [f"{e.source} → {e.target} (strength={e.strength:.2f})" for e in edges]

    def summary(self) -> dict:
        return self._graph.summary()

    def state_dict(self) -> dict:
        return {
            "intervention_count": self._intervention_count,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "strength": e.strength,
                    "sample_count": e.sample_count,
                    "last_updated_step": e.last_updated_step,
                }
                for e in self._graph.edges.values()
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._intervention_count = int(state["intervention_count"])
        self._graph.edges.clear()
        for e_dict in state["edges"]:
            self._graph.edges[(e_dict["source"], e_dict["target"])] = CausalEdge(
                source=e_dict["source"],
                target=e_dict["target"],
                strength=e_dict["strength"],
                sample_count=e_dict["sample_count"],
                last_updated_step=e_dict["last_updated_step"],
            )


def _reconstruct_slots(
    wm: Any, state: Any, num_slots: int, slot_dim: int,
) -> torch.Tensor | None:
    """Attempt to reconstruct slot-level features from world model state."""
    try:
        decoded = wm.decode(state)  # (1, obs_dim)
        # Truncate/pad to match slot structure
        flat = decoded.reshape(-1)
        needed = num_slots * slot_dim
        if flat.shape[0] >= needed:
            return flat[:needed].reshape(num_slots, slot_dim)
        padded = torch.cat([flat, torch.zeros(needed - flat.shape[0], device=flat.device)])
        return padded.reshape(num_slots, slot_dim)
    except Exception:
        return None
