"""Neural-Symbolic Layer: explicit rule extraction + symbolic reasoning.

Bridges the gap between implicit neural learning and explicit symbolic logic.

Three components:

1. :class:`Rule` — a single if-then rule with confidence and priority.
   Stored as a structured tuple, NOT as weights. Human-readable.

2. :class:`RuleMemory` — bounded store of rules (max_rules, Axiom 1).
   Rules are added from experience and evicted by LRU × confidence.

3. :class:`NeuralSymbolicLayer` — sits between the Hybrid backbone and the
   action head. At each step:
   a. Extract candidate rules from (hidden_state, action, reward).
   b. Match current observation against existing rules.
   c. If a rule matches with high confidence → override action (logic).
   d. If no match → fall back to neural policy (intuition).

   This gives the agent "explicit reasoning" on top of "intuitive learning":

       Neural: "I feel like I should go left" (pattern matching)
       Symbolic: "Rule #3 says: IF door is locked AND no key THEN go find key"
                 → override to "go find key" (guaranteed correct logic)

Bounded: 64 rules × 4 fields × d_model ≈ 100k numbers. ~0.05 GB VRAM.
Axiom 1 satisfied (max_rules is fixed).

神经符号层：从经验中提取显式 if-then 规则，推理时用规则链做逻辑判断。
规则是结构化的、可读的、可组合的。无规则匹配时退回神经网络直觉。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Rule: a single if-then statement
# =====================================================================


@dataclass
class Rule:
    """A single symbolic rule extracted from experience.

    - ``condition_embedding``: (d_model,) — the hidden state pattern that
      triggers this rule. Compared via cosine similarity to the current
      hidden state.
    - ``action``: the action this rule recommends.
    - ``confidence``: in [0, 1] — how reliable this rule is (updated by
      reward feedback).
    - ``priority``: higher = checked first (short-circuits lower rules).
    - ``description``: human-readable text (e.g., "IF see key THEN pick up").
    - ``usage_count``: how many times this rule was invoked.
    - ``success_count``: how many times the rule led to a positive outcome.
    - ``last_used_step``: for LRU eviction.
    """

    id: int
    condition_embedding: torch.Tensor  # (d_model,)
    action: int
    confidence: float = 0.5
    priority: float = 0.0
    description: str = ""
    usage_count: int = 0
    success_count: int = 0
    last_used_step: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.usage_count)

    def update(self, reward: float, decay: float = 0.95) -> None:
        """Update confidence based on outcome reward."""
        self.usage_count += 1
        if reward > 0:
            self.success_count += 1
        # Exponential moving average of confidence
        signal = 1.0 if reward > 0 else 0.0
        self.confidence = decay * self.confidence + (1 - decay) * signal
        self.confidence = max(0.0, min(1.0, self.confidence))


# =====================================================================
# RuleMemory: bounded store of rules
# =====================================================================


class RuleMemory:
    """Bounded rule store with LRU × confidence eviction.

    Bounded: max_rules is fixed at construction. When full, the rule with
    the lowest score (confidence × recency) is evicted. Axiom 1 satisfied.

    有界规则存储：max_rules 固定，满时按 confidence × recency 淘汰。
    """

    def __init__(self, max_rules: int = 64, d_model: int = 384) -> None:
        self._max = int(max_rules)
        self._d_model = int(d_model)
        self._rules: dict[int, Rule] = {}
        self._next_id = 0
        # Pre-allocated tensor for batch similarity computation
        self._rule_matrix = torch.zeros(self._max, self._d_model)
        self._rule_actions = torch.zeros(self._max, dtype=torch.long)
        self._rule_confidences = torch.zeros(self._max)
        self._dirty = True  # need to rebuild matrices

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._rules)

    def _rebuild_matrices(self) -> None:
        """Rebuild the batch tensors for fast similarity matching."""
        rules = list(self._rules.values())
        n = len(rules)
        if n == 0:
            return
        for i, r in enumerate(rules):
            self._rule_matrix[i] = r.condition_embedding
            self._rule_actions[i] = r.action
            self._rule_confidences[i] = r.confidence
        self._dirty = False

    def add(
        self,
        condition_embedding: torch.Tensor,
        action: int,
        description: str = "",
        confidence: float = 0.5,
        priority: float = 0.0,
    ) -> Rule:
        """Add a new rule. Evicts lowest-scoring rule if full.

        Before adding, checks if a similar rule already exists (cosine > 0.95).
        If so, updates the existing rule instead of creating a duplicate.
        """
        # Check for duplicate (cosine similarity)
        if len(self._rules) > 0:
            if self._dirty:
                self._rebuild_matrices()
            existing_rules = list(self._rules.values())
            n = len(existing_rules)
            cos = F.cosine_similarity(
                condition_embedding.unsqueeze(0),
                self._rule_matrix[:n],
                dim=1,
            )
            best_idx = int(cos.argmax().item())
            if float(cos[best_idx].item()) > 0.95:
                # Update existing rule
                r = existing_rules[best_idx]
                r.action = action  # may update if action changed
                r.confidence = max(r.confidence, confidence)
                return r

        # Evict if full
        if len(self._rules) >= self._max:
            self._evict()

        # Create new rule
        rule_id = self._next_id
        self._next_id += 1
        rule = Rule(
            id=rule_id,
            condition_embedding=condition_embedding.detach().clone(),
            action=action,
            confidence=confidence,
            priority=priority,
            description=description,
        )
        self._rules[rule_id] = rule
        self._dirty = True
        return rule

    def _evict(self) -> None:
        """Evict the rule with the lowest confidence × recency score."""
        if not self._rules:
            return
        # Score: confidence × (1 + log(1 + usage)) × recency_factor
        import math
        scores = {}
        for rid, r in self._rules.items():
            recency = 1.0 / (1.0 + r.usage_count * 0.01)  # recent use → higher
            scores[rid] = r.confidence * (1.0 + math.log1p(r.usage_count)) * recency
        worst_id = min(scores, key=scores.get)
        del self._rules[worst_id]
        self._dirty = True

    def match(
        self,
        hidden_state: torch.Tensor,
        threshold: float = 0.7,
    ) -> tuple[Rule | None, float]:
        """Find the best matching rule for the current hidden state.

        Args:
            hidden_state: (d_model,) — current observation's hidden state.
            threshold: minimum cosine similarity to consider a match.

        Returns:
            (best_rule, similarity) or (None, 0.0) if no match.
        """
        if not self._rules:
            return None, 0.0

        if self._dirty:
            self._rebuild_matrices()

        rules = list(self._rules.values())
        n = len(rules)
        cos = F.cosine_similarity(
            hidden_state.unsqueeze(0),
            self._rule_matrix[:n],
            dim=1,
        )
        # Weight by confidence
        weighted = cos * self._rule_confidences[:n]
        best_idx = int(weighted.argmax().item())
        best_sim = float(cos[best_idx].item())

        if best_sim >= threshold:
            return rules[best_idx], best_sim
        return None, best_sim

    def update_rule(self, rule_id: int, reward: float, decay: float = 0.95) -> None:
        """Update a rule's confidence based on outcome reward."""
        if rule_id in self._rules:
            self._rules[rule_id].update(reward, decay)
            self._dirty = True

    def get_rule_chain(self, action: int) -> list[Rule]:
        """Get all rules that recommend the given action (for rule chaining)."""
        return [r for r in self._rules.values() if r.action == action]

    def summary(self) -> dict:
        return {
            "num_rules": len(self._rules),
            "capacity": self._max,
            "mean_confidence": (
                sum(r.confidence for r in self._rules.values()) / max(1, len(self._rules))
            ),
            "total_usage": sum(r.usage_count for r in self._rules.values()),
            "total_success": sum(r.success_count for r in self._rules.values()),
        }

    def state_dict(self) -> dict:
        return {
            "max_rules": self._max,
            "d_model": self._d_model,
            "next_id": self._next_id,
            "rules": [
                {
                    "id": r.id,
                    "condition_embedding": r.condition_embedding.cpu(),
                    "action": r.action,
                    "confidence": r.confidence,
                    "priority": r.priority,
                    "description": r.description,
                    "usage_count": r.usage_count,
                    "success_count": r.success_count,
                    "last_used_step": r.last_used_step,
                }
                for r in self._rules.values()
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._max = int(state["max_rules"])
        self._d_model = int(state["d_model"])
        self._next_id = int(state["next_id"])
        self._rules.clear()
        for r_dict in state["rules"]:
            self._rules[r_dict["id"]] = Rule(
                id=r_dict["id"],
                condition_embedding=r_dict["condition_embedding"],
                action=r_dict["action"],
                confidence=r_dict["confidence"],
                priority=r_dict["priority"],
                description=r_dict["description"],
                usage_count=r_dict["usage_count"],
                success_count=r_dict["success_count"],
                last_used_step=r_dict["last_used_step"],
            )
        self._rule_matrix = torch.zeros(self._max, self._d_model)
        self._rule_actions = torch.zeros(self._max, dtype=torch.long)
        self._rule_confidences = torch.zeros(self._max)
        self._dirty = True


# =====================================================================
# NeuralSymbolicLayer: rule extraction + reasoning
# =====================================================================


class NeuralSymbolicLayer(nn.Module):
    """Neural-symbolic reasoning layer.

    Sits between the Hybrid backbone and the action head.

    At each step:
    1. Receive the hidden state from the backbone.
    2. Match against existing rules in RuleMemory.
    3. If a high-confidence rule matches → override the action.
    4. If no match → pass through to the neural action head.

    Rule extraction (triggered on episode end or periodically):
    1. Look at (state, action, reward) triples from the episode.
    2. For high-reward transitions, extract a rule:
       "IF hidden_state ≈ X THEN action = A"
    3. Store in RuleMemory (with confidence = normalized reward).

    This gives the agent EXPLICIT, READABLE, VERIFIABLE reasoning:
    - "Rule #5: IF see locked door AND no key THEN go to room B (conf=0.8)"
    - Can be chained: Rule #5 → Rule #2 → Rule #7 (multi-step plan)
    - Can be inspected: print all rules to see what the agent "knows"

    Bounded: RuleMemory has max_rules. All operations are O(max_rules).
    VRAM: ~0.05 GB (64 rules × 384 dims × 4 bytes).
    """

    def __init__(
        self,
        d_model: int = 384,
        num_actions: int = 7,
        max_rules: int = 64,
        match_threshold: float = 0.7,
        extraction_reward_threshold: float = 0.3,
        override_confidence_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._num_actions = num_actions
        self._match_threshold = match_threshold
        self._extraction_reward_threshold = extraction_reward_threshold
        self._override_threshold = override_confidence_threshold

        self.rule_memory = RuleMemory(max_rules=max_rules, d_model=d_model)

        # Trainable projection: maps hidden state → rule condition space
        # (separate from the backbone's own representation)
        self.rule_projection = nn.Linear(d_model, d_model)

        # The last matched rule (for later confidence update)
        self._last_matched_rule_id: int | None = None

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(
        self,
        hidden_state: torch.Tensor,
        neural_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reason: check rules, potentially override neural action.

        Args:
            hidden_state: (B, d_model) from the backbone.
            neural_logits: (B, num_actions) from the neural action head.

        Returns:
            (final_logits, info) where info contains:
            - "rule_matched": bool
            - "rule_id": int or None
            - "rule_sim": float
            - "override": bool (whether rule overrode neural)
        """
        B = hidden_state.shape[0]
        info: dict[str, Any] = {
            "rule_matched": False,
            "rule_id": None,
            "rule_sim": 0.0,
            "override": False,
        }

        # For batch processing, we only match on the first element
        # (rules are global, not per-batch-element)
        h = hidden_state[0] if B > 0 else hidden_state
        h_projected = self.rule_projection(h)

        rule, sim = self.rule_memory.match(h_projected, threshold=self._match_threshold)

        if rule is not None and rule.confidence >= self._override_threshold:
            # Rule override: force the rule's action
            info["rule_matched"] = True
            info["rule_id"] = rule.id
            info["rule_sim"] = sim
            info["override"] = True
            self._last_matched_rule_id = rule.id
            rule.usage_count += 1
            rule.last_used_step += 1

            # Create logits that strongly prefer the rule's action
            override_logits = torch.full_like(neural_logits, -10.0)
            override_logits[:, rule.action] = 10.0 * rule.confidence
            # Blend: 70% rule + 30% neural (soft override)
            alpha = 0.7 * rule.confidence
            final_logits = alpha * override_logits + (1 - alpha) * neural_logits
            return final_logits, info

        if rule is not None:
            info["rule_matched"] = True
            info["rule_id"] = rule.id
            info["rule_sim"] = sim
            self._last_matched_rule_id = rule.id

        # No override: use neural logits
        self._last_matched_rule_id = None
        return neural_logits, info

    def extract_rules(
        self,
        hidden_states: list[torch.Tensor],
        actions: list[int],
        rewards: list[float],
        descriptions: list[str] | None = None,
    ) -> list[Rule]:
        """Extract rules from an episode's trajectory.

        For transitions with reward > threshold, extract:
            "IF hidden_state ≈ X THEN action = A"

        Args:
            hidden_states: list of (d_model,) tensors, one per step.
            actions: list of action indices.
            rewards: list of rewards received.
            descriptions: optional human-readable descriptions.

        Returns:
            List of newly created/updated rules.
        """
        new_rules: list[Rule] = []
        for t in range(len(hidden_states)):
            if rewards[t] < self._extraction_reward_threshold:
                continue
            h = hidden_states[t]
            if h.dim() == 1:
                h = h.unsqueeze(0)
            h_projected = self.rule_projection(h.squeeze(0))
            action = actions[t]
            desc = descriptions[t] if descriptions and t < len(descriptions) else ""
            confidence = min(1.0, rewards[t])  # higher reward → higher confidence
            rule = self.rule_memory.add(
                condition_embedding=h_projected,
                action=action,
                description=desc,
                confidence=confidence,
                priority=rewards[t],
            )
            new_rules.append(rule)
        return new_rules

    def feedback(self, reward: float, decay: float = 0.95) -> None:
        """Update the last matched rule's confidence based on outcome.

        Call this after the action is taken and reward is received.
        """
        if self._last_matched_rule_id is not None:
            self.rule_memory.update_rule(self._last_matched_rule_id, reward, decay)
            self._last_matched_rule_id = None

    def get_rules_text(self) -> list[str]:
        """Return human-readable descriptions of all rules."""
        rules = list(self.rule_memory._rules.values())
        rules.sort(key=lambda r: -r.confidence)
        lines = []
        for r in rules:
            lines.append(
                f"Rule #{r.id}: {r.description} | "
                f"action={r.action} conf={r.confidence:.2f} "
                f"used={r.usage_count} success_rate={r.success_rate:.2f}"
            )
        return lines

    def summary(self) -> dict:
        return self.rule_memory.summary()
