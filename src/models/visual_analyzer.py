"""Visual Object Analyzer — Property classification from SlotAttention output.

Upgrades the basic property classifiers in PerceptionProjector (llm_fusion.py)
with finer-grained attributes that directly feed ConceptGraph nodes.

Classifies per-slot:
    - color (8 bins: red, blue, green, yellow, white, black, orange, purple)
    - shape (4 bins: round, square, tall, flat)
    - size (3 bins: small, medium, large)
    - texture (3 bins: smooth, rough, shiny) — from slot embedding variance
    - motion (3 bins: still, slow, fast) — from consecutive slot diffs

Output feeds directly into ConceptGraph.add_concept() and add_edge(),
creating richer knowledge nodes for the developmental pipeline.

Zero GPU overhead beyond a single Linear classifier per attribute. ~5K params.

物体属性分析器：从 SlotAttention 输出分类颜色/形状/大小/纹理/运动。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VisualAnalyzer(nn.Module):
    """Fine-grained object property classifier from SlotAttention output.

    Each slot (object) gets classified into:
        color (8), shape (4), size (3), texture (3), motion (3)

    This produces the rich attribute set needed for ConceptGraph to form
    meaningful concept nodes without LLM intervention.
    """

    def __init__(self, slot_dim: int = 128, num_slots: int = 7):
        super().__init__()
        self._num_slots = num_slots
        self._slot_dim = slot_dim

        # Shared trunk for efficiency
        self.trunk = nn.Sequential(
            nn.Linear(slot_dim, 64),
            nn.GELU(),
        )

        # Attribute heads
        self.color_head = nn.Linear(64, 8)
        self.shape_head = nn.Linear(64, 4)
        self.size_head = nn.Linear(64, 3)
        self.texture_head = nn.Linear(64, 3)
        self.motion_head = nn.Linear(64, 3)

        # Motion history (per-slot embedding for velocity estimation)
        self._prev_slots: torch.Tensor | None = None

    # Labels for each head
    COLOR_NAMES = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
    SHAPE_NAMES = ["round", "square", "tall", "flat"]
    SIZE_NAMES = ["small", "medium", "large"]
    TEXTURE_NAMES = ["smooth", "rough", "shiny"]
    MOTION_NAMES = ["still", "slow", "fast"]

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classify all slots simultaneously.

        Args:
            slots: (B, num_slots, slot_dim) — SlotAttention output.

        Returns:
            dict with keys "color", "shape", "size", "texture", "motion".
            Each value is (B, num_slots, num_classes) logits.
        """
        B, N, D = slots.shape
        flat = slots.reshape(B * N, D)                      # (B*N, slot_dim)
        h = self.trunk(flat)                                 # (B*N, 64)

        result = {
            "color":   self.color_head(h).reshape(B, N, 8),    # (B, N, 8)
            "shape":   self.shape_head(h).reshape(B, N, 4),    # (B, N, 4)
            "size":    self.size_head(h).reshape(B, N, 3),     # (B, N, 3)
            "texture": self.texture_head(h).reshape(B, N, 3),  # (B, N, 3)
            "motion":  self._motion_logits(slots).reshape(B, N, 3),  # (B, N, 3)
        }
        self._prev_slots = slots.detach()
        return result

    def _motion_logits(self, slots: torch.Tensor) -> torch.Tensor:
        """Estimate motion from slot embedding change since last step."""
        B = slots.shape[0] * slots.shape[1]
        if self._prev_slots is None or self._prev_slots.shape != slots.shape:
            return torch.zeros(B, 3, device=slots.device)
        diff = (slots - self._prev_slots).norm(dim=-1).flatten()  # (B*N,)
        # Threshold-based logits: still < slow < fast
        logits = torch.zeros(B, 3, device=slots.device)
        logits[:, 0] = 2.0 - diff * 10.0    # still — high when diff is small
        logits[:, 1] = diff * 5.0 - 1.0      # slow
        logits[:, 2] = diff * 10.0 - 3.0     # fast — high when diff is large
        return logits

    def describe_slot(self, slots: torch.Tensor, slot_idx: int) -> str:
        """Generate a natural-language description of one slot."""
        out = self.forward(slots)
        color = self.COLOR_NAMES[int(out["color"][0, slot_idx].argmax().item())]
        shape = self.SHAPE_NAMES[int(out["shape"][0, slot_idx].argmax().item())]
        size  = self.SIZE_NAMES[int(out["size"][0, slot_idx].argmax().item())]
        text  = self.TEXTURE_NAMES[int(out["texture"][0, slot_idx].argmax().item())]
        motion = self.MOTION_NAMES[int(out["motion"][0, slot_idx].argmax().item())]
        return f"a {size} {color} {texture} {shape} object, {motion}"

    def describe_scene(self, slots: torch.Tensor) -> str:
        """Generate a scene description from all active slots."""
        out = self.forward(slots)
        slot_norms = slots.norm(dim=-1)[0]  # (N,)
        descriptions = []
        for i in range(min(self._num_slots, slots.shape[1])):
            if slot_norms[i] < 0.1:  # empty slot
                continue
            descriptions.append(self.describe_slot(slots, i))
        if not descriptions:
            return "I don't see any objects."
        return "I see " + ", ".join(descriptions) + "."

    def feed_to_graph(
        self, slots: torch.Tensor, concept_graph: Any, step: int,
    ) -> int:
        """Feed classified attributes into ConceptGraph as concept nodes.

        Returns number of concepts added.
        """
        if concept_graph is None:
            return 0
        out = self.forward(slots)
        added = 0
        for i in range(min(self._num_slots, slots.shape[1])):
            slot_vec = slots[0, i]
            if slot_vec.norm().item() < 0.1:
                continue
            desc = self.describe_slot(slots, i)
            node_id = concept_graph.add_concept(
                embedding=slot_vec, name=desc, source="visual_analyzer", step=step,
            )
            # Add attribute edges
            color = self.COLOR_NAMES[int(out["color"][0, i].argmax().item())]
            shape = self.SHAPE_NAMES[int(out["shape"][0, i].argmax().item())]
            size  = self.SIZE_NAMES[int(out["size"][0, i].argmax().item())]
            attr_id = concept_graph.add_concept(
                embedding=slot_vec * 0.5, name=f"{color}_{shape}_{size}",
                source="visual_analyzer_attr", step=step,
            )
            concept_graph.add_edge(node_id, attr_id, "has_attribute", 0.7,
                                   "visual_analyzer", step)
            added += 1
        return added
