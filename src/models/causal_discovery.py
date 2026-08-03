"""Causal Discovery via Counterfactual Intervention.

Extends the RSSM world model and CounterfactualImagination with:
1. Intervention: "What if I had done X instead?"
2. Counterfactual comparison: "Did doing X cause Y?"
3. Causal graph construction: "A → B" edges from repeated interventions.

Two modes:
- ``wm`` (default, legacy): counterfactual interventions imagined inside the
  RSSM world model. Requires usable latent dynamics; on low-information
  MiniGrid renders the RSSM never learns to encode (kl_raw ~ 0.001, verified
  over 84K steps), so this mode stays at effect=0.
- ``experience`` (Stage 14 active): intervention statistics over REAL
  transitions (observational do(a) approximation). Works without a trained
  world model; the stochastic exploration policy provides the (s, a) mix.

This upgrades the agent from "I can predict what happens" (RSSM) to
"I understand WHY it happens" (causal reasoning).

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
class ExperienceTransition:
    """One real (s, a, s', r) transition from the actual training trajectory.

    Used by the "experience" mode of causal discovery: instead of imagining
    counterfactuals with the world model (which is useless while the RSSM has
    no usable latent dynamics on low-information MiniGrid renders), we do
    intervention statistics over real experience, exploiting the fact that a
    stochastic exploration policy visits many (state, action) pairs.
    """

    state: torch.Tensor      # (D,) state embedding
    action: int
    next_state: torch.Tensor  # (D,)
    reward: float
    step: int


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

    def get_causes(self, target: str, min_strength: float = 0.02) -> list[CausalEdge]:
        return sorted(
            [e for (s, t), e in self.edges.items() if t == target and e.strength >= min_strength],
            key=lambda e: -e.strength,
        )

    def get_effects(self, source: str, min_strength: float = 0.02) -> list[CausalEdge]:
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
        mode: str = "experience",
        buffer_capacity: int = 4096,
    ) -> None:
        self._num_actions = num_actions
        self._max_edges = max_edges
        self._min_effect = min_intervention_effect
        self._ema_alpha = ema_alpha
        self._mode = mode
        # Bounded real-experience buffer (Axiom 1): capacity declared at
        # construction, oldest transitions evicted automatically.
        self._buffer: deque[ExperienceTransition] = deque(maxlen=buffer_capacity)
        self._buffer_capacity = buffer_capacity
        self._graph = CausalGraph()
        self._intervention_count = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def capacity(self) -> int:
        return self._max_edges

    def __len__(self) -> int:
        return len(self._graph.edges)

    def observe(
        self,
        state_embed: torch.Tensor,
        action: int,
        next_embed: torch.Tensor,
        reward: float,
        step: int,
    ) -> None:
        """Feed one real (s, a, s', r) transition into the experience buffer.

        Only active in "experience" mode. Detached copies are stored so the
        buffer never holds autograd graph references (Stage 11 lessons).
        """
        if self._mode != "experience":
            return
        self._buffer.append(
            ExperienceTransition(
                state=state_embed.detach().float(),
                action=int(action),
                next_state=next_embed.detach().float(),
                reward=float(reward),
                step=int(step),
            )
        )

    def intervene_from_experience(self, step: int) -> dict[str, float]:
        """Counterfactual-style statistics over real transitions.

        For each action a, the average transition magnitude
        ``E[||s' - s|| | a]`` is compared against the global baseline
        ``E[||s' - s||]``. An action whose effect stands out from the
        baseline is recorded as a causal cause of world change. This is the
        observational equivalent of ``do(a)`` under the stochastic
        exploration policy: every action is taken from a wide mix of states.

        Returns dict mapping "action_X_effect" -> effect magnitude.
        """
        effects: dict[str, float] = {}
        if len(self._buffer) < 64:  # too few samples for stable statistics
            return effects

        per_action: dict[int, list[float]] = {}
        per_action_r: dict[int, list[float]] = {}
        all_mags: list[float] = []
        for tr in self._buffer:
            mag = float((tr.next_state - tr.state).pow(2).mean().sqrt().item())
            per_action.setdefault(tr.action, []).append(mag)
            per_action_r.setdefault(tr.action, []).append(tr.reward)
            all_mags.append(mag)
        baseline = sum(all_mags) / max(1, len(all_mags))
        all_r = [r for lst in per_action_r.values() for r in lst]
        baseline_r = sum(all_r) / max(1, len(all_r))

        for a in range(self._num_actions):
            mags = per_action.get(a)
            if not mags or len(mags) < 8:
                continue
            mean_mag = sum(mags) / len(mags)
            # Relative effect vs the global baseline: stable across embedding
            # scales. A 5% above-baseline transition magnitude is a causal
            # signature of the action on the world state.
            rel = (mean_mag - baseline) / max(baseline, 1e-6)
            effects[f"action_{a}_effect"] = rel
            if rel > 0.05:
                self._graph.record_cause(
                    source=f"action_{a}",
                    target="world_state",
                    strength_delta=min(1.0, rel),
                    step=step,
                    ema_alpha=self._ema_alpha,
                )
            r_effect = sum(per_action_r[a]) / len(per_action_r[a]) - baseline_r
            if r_effect > self._min_effect * 2.0:
                self._graph.record_cause(
                    source=f"action_{a}",
                    target="reward_increased",
                    strength_delta=min(1.0, r_effect * 100.0),
                    step=step,
                    ema_alpha=self._ema_alpha,
                )
            elif r_effect < -self._min_effect * 2.0:
                self._graph.record_cause(
                    source=f"action_{a}",
                    target="reward_decreased",
                    strength_delta=min(1.0, -r_effect * 100.0),
                    step=step,
                    ema_alpha=self._ema_alpha,
                )

        self._intervention_count += 1
        self._trim_graph()
        return effects

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
        if world_model is None:
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
            self._trim_graph()

        return effects

    def _trim_graph(self) -> None:
        """Keep the causal graph bounded (Axiom 1): evict weakest edges when
        at capacity.  Interventions run forever; the *stored* graph is capped."""
        if len(self._graph.edges) <= self._max_edges:
            return
        worst_key = min(
            self._graph.edges,
            key=lambda k: (self._graph.edges[k].strength, self._graph.edges[k].sample_count),
        )
        del self._graph.edges[worst_key]

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
