"""Number Sense Module — Cardinality and Arithmetic from Slot Attention.

Maps the output of SlotAttention (n slot vectors representing n objects) to
a numerical understanding: "how many?", "which is more?", "add two piles".

This is NOT an LLM-based number sense. It's a minimal MLP that learns
cardinality from the structure of Slot Attention output — specifically from
slot occupancy (which slots have meaningful content vs. which are empty).

Architecture:
    SlotAttention output: (B, num_slots, d_model)
    → slot_occupancy: soft attention mass per slot
    → NumberSense MLP: (B, num_slots) → (B, 1) cardinality prediction
    → trained with MSE against ground-truth count

This gives the agent the developmental milestone of "I see 3 things" without
any language model. It's the same ability as a 3-year-old counting objects.

数字感模块：从 Slot Attention 的槽位占有率预测物体数量。
不需要 LLM，只有一个小 MLP。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberSense(nn.Module):
    """Predicts how many objects are present from Slot Attention slot states.

    Two complementary signals:
    1. Slot occupancy: variance of each slot's embedding (empty slots = near-zero variance)
    2. Slot distinctness: cosine distance between slot pairs (distinct objects = high distance)

    The MLP fuses these into a single cardinality prediction.

    Bounded: max_count is fixed. ~2K parameters.
    """

    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 128,
        max_count: int = 10,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self._num_slots = num_slots
        self._max_count = max_count

        # from occupancy (num_slots) + distinctness (num_slots choose 2) → hidden → count
        n_distinct_pairs = num_slots * (num_slots - 1) // 2
        input_dim = num_slots + n_distinct_pairs

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_count + 1),  # class 0..10
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Predict cardinality from slot vectors.

        Args:
            slots: (B, num_slots, slot_dim) — SlotAttention output.

        Returns:
            (B, max_count+1) — logits for count 0..max_count.
        """
        B = slots.shape[0]

        # Occupancy: how "active" is each slot? Measure embedding norm variance.
        slot_norms = slots.norm(dim=-1)                                    # (B, num_slots)
        slot_occupancy = slot_norms / (slot_norms.sum(dim=-1, keepdim=True) + 1e-8)

        # Distinctness: cosine distance between all slot pairs
        slots_normed = F.normalize(slots, dim=-1)                          # (B, num_slots, slot_dim)
        sim = slots_normed @ slots_normed.transpose(-1, -2)                # (B, num_slots, num_slots)
        # Take upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(self._num_slots, self._num_slots, device=slots.device), diagonal=1)
        distinctness = (1.0 - sim) * mask                                  # (B, num_slots, num_slots)
        # Flatten upper triangle
        upper = distinctness[:, mask.bool()]                               # (B, n_pairs)

        # Combine signals
        features = torch.cat([slot_occupancy, upper], dim=-1)              # (B, input_dim)
        return self.net(features)                                          # (B, max_count+1)

    def predict_count(self, slots: torch.Tensor) -> torch.Tensor:
        """Return integer count predictions."""
        logits = self.forward(slots)
        return logits.argmax(dim=-1)  # (B,) integer

    def compare(self, slots_a: torch.Tensor, slots_b: torch.Tensor) -> str:
        """Which has more objects? Returns 'a', 'b', or 'equal'."""
        count_a = self.predict_count(slots_a).item()
        count_b = self.predict_count(slots_b).item()
        if count_a > count_b:
            return "a"
        elif count_b > count_a:
            return "b"
        return "equal"

    def loss(self, slots: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss against ground-truth counts."""
        logits = self.forward(slots)
        return F.cross_entropy(logits, true_counts.long())
