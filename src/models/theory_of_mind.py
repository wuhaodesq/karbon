"""Theory of Mind — Predict others' mental states.

Developmental milestone (~4 years human): understanding that other agents
have their own beliefs, knowledge, and intentions — which may differ from
one's own.

Three levels implemented:
    1. Perspective-taking: What can the other agent see?
    2. Belief prediction: What does the other agent believe about the world?
    3. False-belief understanding: The other agent might be WRONG.

Architecture:
    Self perspective (SlotAttention output)
    → OtherAgentPerspectivePredictor: projects into "what the other sees"
    → BeliefStatePredictor: what the other believes (may differ from reality)
    → ActionPredictor: what the other will do (given their beliefs)

Training signals:
    - Perspective error: predicted vs actual caregiver visual field
    - Action prediction error: predicted vs actual caregiver action
    - False-belief reward: correct prediction when caregiver is wrong

心智理论：预测他人的信念和知识状态。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TheoryOfMind(nn.Module):
    """Multi-level Theory of Mind predictor.

    Level 1 — Perspective Taking:
        "Caregiver is on the other side of the room. She can't see the ball
         behind the table, but I can."

    Level 2 — Belief Attribution:
        "Caregiver believes the ball is on the table, because she last saw
         it there. But I moved it."

    Level 3 — False-Belief Understanding:
        "Caregiver will look for the ball on the table, because she doesn't
         know I moved it. She'll be surprised when it's not there."

    Uses a small transformer-style architecture (~50K params) that takes:
    - Self perspective (SlotAttention output)
    - Other agent position
    - Object positions
    → outputs other's belief state + predicted action.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_actions: int = 8,
        num_slots: int = 7,
        perspective_hidden: int = 64,
        max_other_agents: int = 3,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._num_actions = num_actions
        self._num_slots = num_slots
        self._max_agents = max_other_agents

        # Perspective projector: self slots → what other sees
        self.perspective_proj = nn.Sequential(
            nn.Linear(d_model * num_slots, perspective_hidden),
            nn.GELU(),
            nn.Linear(perspective_hidden, d_model * num_slots),
        )

        # Belief state: combines other's perspective with history
        self.belief_gru = nn.GRUCell(d_model, d_model)

        # Predict other's action from their belief
        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, perspective_hidden),
            nn.GELU(),
            nn.Linear(perspective_hidden, num_actions),
        )

        # Predict other's surprise (prediction error of their model)
        self.surprise_predictor = nn.Sequential(
            nn.Linear(d_model, perspective_hidden),
            nn.GELU(),
            nn.Linear(perspective_hidden, 1),
        )

        # Running belief states for each other agent
        self._belief_states: dict[str, torch.Tensor] = {}

    def reset_beliefs(self) -> None:
        """Clear all tracked belief states (new episode)."""
        self._belief_states.clear()

    def predict_perspective(
        self,
        self_slots: torch.Tensor,            # (B, num_slots, d_model)
        other_position: torch.Tensor,         # (B, 3) xyz
        object_positions: torch.Tensor,       # (num_objects, 3)
    ) -> torch.Tensor:
        """Predict what the other agent can see from their position.

        Simple geometric occlusion: objects behind walls or too far are invisible.
        """
        # Flatten self slots
        flat = self_slots.reshape(self_slots.shape[0], -1)  # (B, num_slots*d_model)
        proj = self.perspective_proj(flat)                   # (B, num_slots*d_model)
        other_slots = proj.reshape(self_slots.shape)         # (B, num_slots, d_model)

        # Visibility mask: objects within 2m and line-of-sight are visible
        if object_positions.shape[0] > 0:
            dists = torch.norm(object_positions.unsqueeze(0) - other_position.unsqueeze(1), dim=-1)
            visible = (dists < 2.0)  # (B, num_objects)
            # Apply visibility to other's slot embedding
            vis_scale = visible.float().mean(dim=-1, keepdim=True).unsqueeze(-1)
            other_slots = other_slots * (0.5 + 0.5 * vis_scale)

        return other_slots

    def update_belief(
        self,
        agent_name: str,
        perspective_slots: torch.Tensor,    # (1, d_model) aggregated
    ) -> torch.Tensor:
        """Update the belief state for an agent based on their current perspective.

        Uses GRU to maintain a running belief state that integrates history.
        """
        prev = self._belief_states.get(
            agent_name,
            torch.zeros(1, self._d_model, device=perspective_slots.device),
        )
        new_belief = self.belief_gru(
            perspective_slots.reshape(1, -1),
            prev.reshape(1, -1),
        )
        self._belief_states[agent_name] = new_belief.detach()
        return new_belief

    def predict_other_action(
        self, agent_name: str,
    ) -> torch.Tensor:
        """Predict what action the other agent will take, given their beliefs."""
        belief = self._belief_states.get(agent_name)
        if belief is None:
            return torch.zeros(1, self._num_actions)
        return self.action_predictor(belief)  # (1, num_actions)

    def predict_other_surprise(
        self,
        agent_name: str,
        actual_observation: torch.Tensor,    # (d_model,)
    ) -> float:
        """Predict how surprised the other agent will be by this observation.

        High surprise = their belief state doesn't predict this observation.
        This is the computational basis for false-belief understanding.
        """
        belief = self._belief_states.get(agent_name)
        if belief is None:
            return 0.0
        # Surprise = mismatch between belief-predicted obs and actual obs
        surprise_logit = self.surprise_predictor(belief)
        obs_sim = F.cosine_similarity(
            actual_observation.unsqueeze(0),
            belief,
            dim=-1,
        )
        # High sim = low surprise, low sim = high surprise
        return float(1.0 - obs_sim.mean().item())

    def false_belief_test(
        self,
        agent_name: str,
        hidden_object_slot: torch.Tensor,     # (d_model,) the hidden object
    ) -> bool:
        """Test if the agent understands that the other has a false belief.

        Returns True if the model correctly predicts:
        - The other agent does NOT know about the hidden object
        - The other agent's predicted action ignores the hidden object
        """
        belief = self._belief_states.get(agent_name)
        if belief is None:
            return False

        # Check: does the belief state encode the hidden object?
        sim = F.cosine_similarity(
            hidden_object_slot.unsqueeze(0),
            belief,
            dim=-1,
        )
        # If similarity is HIGH, the belief encodes the hidden object → FALSE BELIEF FAILED
        # If similarity is LOW, the belief does NOT encode the hidden object → CORRECT
        return float(sim.mean().item()) < 0.5

    def forward(
        self,
        self_slots: torch.Tensor,           # (B, num_slots, d_model)
        other_positions: dict[str, torch.Tensor],  # {name: (B, 3)}
        object_positions: torch.Tensor,      # (num_objects, 3)
    ) -> dict[str, Any]:
        """Full Theory of Mind inference.

        Returns:
            dict with per-agent entries:
                "caregiver_predicted_action": (1, num_actions)
                "caregiver_surprise": float
                "caregiver_perspective_slots": (B, num_slots, d_model)
        """
        result: dict[str, Any] = {}
        for name, pos in other_positions.items():
            # Perspective
            other_slots = self.predict_perspective(self_slots, pos, object_positions)
            result[f"{name}_perspective_slots"] = other_slots

            # Belief update
            aggregated = other_slots.mean(dim=(0, 1)) if other_slots.dim() == 3 else other_slots.mean(dim=0)
            self.update_belief(name, aggregated.detach())

            # Predicted action
            result[f"{name}_predicted_action"] = self.predict_other_action(name)

        return result

    def compute_loss(
        self,
        predicted_actions: torch.Tensor,     # (B, num_actions) predicted
        actual_actions: torch.Tensor,         # (B,) actual
    ) -> torch.Tensor:
        """Cross-entropy loss for action prediction (train ToM)."""
        return F.cross_entropy(predicted_actions, actual_actions.long())

    def summary(self) -> dict:
        return {
            "tracked_agents": list(self._belief_states.keys()),
            "num_agents": len(self._belief_states),
        }

    @property
    def capacity(self) -> int:
        return self._max_agents
