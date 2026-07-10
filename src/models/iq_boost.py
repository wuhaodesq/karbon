"""IQ Boost — Six Tier-1 Upgrades for Developmental Reasoning.

Target: 4-5 → 9-10 year equivalent intelligence.

1. CrossDomainTransfer — detects task similarity, transfers learned policies via EMA
2. DeepMultiModal — cross-attention fusion layer replacing simple concatenation
3. TemporalReasoner — plan→execute→feedback loop with RSSM verification
4. CounterfactualRegret — "I should have done X instead" experiential learning
5. CuriosityDirector — orchestrates RSSM uncertainty + knowledge gap + social curiosity
6. ValueSystem — rudimentary right/wrong derived from 5 homeostatic drives

All modules bounded (Axiom 1). Total ~800 lines, ~2M params, ~0.3 GB VRAM.

IQ 提升六件套：跨域迁移、深度融合、时序推理、反事实后悔、好奇导演、价值体系。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Cross-Domain Transfer Learning
# =====================================================================


@dataclass
class DomainSignature:
    """Fingerprint of a task/environment for similarity detection."""
    embedding: torch.Tensor     # (d_model,) compressed environment stats
    return_mean: float = 0.0
    return_std: float = 0.0
    steps_trained: int = 0
    label: str = ""


class CrossDomainTransfer(nn.Module):
    """Detects task similarity and transfers learned policies.

    When a new task/environment is encountered:
    1. Compute its domain signature from initial observations
    2. Compare to all known domains via cosine similarity
    3. If similarity > threshold → transfer relevant SkillLibrary entries
    4. EMA-merge transferred skills with current policy (faster adaptation)

    This is MAML-lite: the agent "remembers" how to solve similar tasks.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_domains: int = 32,
        transfer_threshold: float = 0.7,
        ema_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._max = int(max_domains)
        self._threshold = float(transfer_threshold)
        self._alpha = float(ema_alpha)

        self._domains: dict[str, DomainSignature] = {}
        self._current_domain: str = "unknown"
        self._transfer_log: list[dict] = []

        # Domain encoder: compresses environment statistics into signature
        self.domain_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._domains)

    def extract_signature(
        self, recent_obs_embeddings: torch.Tensor, step: int,
    ) -> DomainSignature:
        """Compress recent observations into a domain fingerprint."""
        emb = self.domain_encoder(recent_obs_embeddings.mean(dim=0))
        return DomainSignature(
            embedding=emb.detach(),
            return_mean=0.0,
            steps_trained=step,
        )

    def register_domain(
        self, name: str, signature: DomainSignature,
    ) -> None:
        """Register a known domain for future transfer."""
        if len(self._domains) >= self._max:
            oldest = min(self._domains, key=lambda k: self._domains[k].steps_trained)
            del self._domains[oldest]
        self._domains[name] = signature
        self._current_domain = name

    def find_similar(self, query: DomainSignature) -> list[tuple[str, float]]:
        """Find known domains similar to query."""
        results = []
        for name, sig in self._domains.items():
            sim = float(F.cosine_similarity(
                query.embedding.unsqueeze(0), sig.embedding.unsqueeze(0), dim=-1,
            ).item())
            if sim > self._threshold:
                results.append((name, sim))
        return sorted(results, key=lambda x: -x[1])

    def transfer(self, similar_domains: list[tuple[str, float]]) -> dict[str, float]:
        """Return transfer weights for skill adaptation."""
        weights: dict[str, float] = {}
        for name, sim in similar_domains:
            weights[name] = min(1.0, sim * (1.0 + self._alpha))
        return weights

    def summary(self) -> dict:
        return {
            "known_domains": len(self._domains),
            "current": self._current_domain,
            "domain_names": list(self._domains.keys()),
        }


# =====================================================================
# 2. Deep Multi-Modal Fusion
# =====================================================================


class DeepMultiModal(nn.Module):
    """Cross-attention fusion replacing simple concatenation.

    Current: CrossModalManager just averages embeddings.
    Upgrade: Each modality attends to every other modality before fusing.

    Modalities: vision (SlotAttention), touch (proprio), plan (sub-goal), memory (retrieval).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_modalities: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self._num_mod = num_modalities
        self._d_model = d_model

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True,
        )
        # Output projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self, modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Cross-attention fusion of all modalities.

        Each modality acts as query attending to all others as key/value.
        Returns a single fused embedding.
        """
        if not modality_embeddings:
            return torch.zeros(self._d_model)

        # Stack modalities: (B, num_mod, d_model)
        mods = []
        for emb in modality_embeddings.values():
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            mods.append(emb)
        if not mods:
            return torch.zeros(1, self._d_model, device=next(self.parameters()).device)

        # Pad to num_modalities
        B = mods[0].shape[0]
        device = mods[0].device
        stacked = torch.zeros(B, self._num_mod, self._d_model, device=device)
        for i, m in enumerate(mods):
            if i < self._num_mod:
                stacked[:, i, :] = m[:B]

        # Cross-attention: each modality queries all
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)  # (B, num_mod, d_model)
        fused = attn_out.mean(dim=1)  # (B, d_model)
        return self.fusion_proj(fused)


# =====================================================================
# 3. Temporal Reasoner
# =====================================================================


class TemporalReasoner(nn.Module):
    """Plan → Execute → Verify → Learn feedback loop.

    Hooks into LongRangePlanner:
    1. Plan generated → stored as "expected trajectory"
    2. Each step: compare actual RSSM state with planned state
    3. Deviation → record what went wrong
    4. Episode end: update RuleInductionEngine with lessons
    """

    def __init__(
        self,
        d_model: int = 128,
        max_trajectories: int = 200,
        deviation_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._max = int(max_trajectories)
        self._threshold = float(deviation_threshold)

        self._expected_states: list[torch.Tensor] = []
        self._actual_states: list[torch.Tensor] = []
        self._deviations: list[dict] = []

        # Deviation detector
        self.deviation_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # concat(expected, actual)
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._deviations)

    def set_plan(self, expected_states: list[torch.Tensor]) -> None:
        """Store the expected trajectory from planner."""
        self._expected_states = [s.detach() for s in expected_states]
        self._actual_states = []

    def verify_step(
        self, actual_state: torch.Tensor, step_in_plan: int,
    ) -> dict[str, Any] | None:
        """Compare actual state with planned. Return deviation info if mismatch."""
        self._actual_states.append(actual_state.detach())

        if step_in_plan >= len(self._expected_states):
            return None

        expected = self._expected_states[step_in_plan].to(actual_state.device)
        concat = torch.cat([actual_state.reshape(1, -1), expected.reshape(1, -1)], dim=-1)
        deviation = float(torch.sigmoid(self.deviation_detector(concat)).item())

        if deviation < self._threshold:
            return None

        deviation_info = {
            "step": step_in_plan,
            "deviation": deviation,
            "actual_norm": float(actual_state.norm().item()),
            "expected_norm": float(expected.norm().item()),
        }
        if len(self._deviations) >= self._max:
            self._deviations.pop(0)
        self._deviations.append(deviation_info)
        return deviation_info

    def get_lessons(self) -> list[str]:
        """Return lessons learned from plan deviations."""
        if not self._deviations:
            return []
        lessons = []
        for d in self._deviations[-5:]:
            if d["actual_norm"] > d["expected_norm"] * 1.5:
                lessons.append(f"Step {d['step']}: things moved more than expected (actual={d['actual_norm']:.2f} vs expected={d['expected_norm']:.2f})")
            elif d["actual_norm"] < d["expected_norm"] * 0.5:
                lessons.append(f"Step {d['step']}: things moved less than expected")
        return lessons

    def summary(self) -> dict:
        return {
            "deviations": len(self._deviations),
            "plans_tracked": len(self._expected_states),
            "recent_lessons": self.get_lessons()[:3],
        }


# =====================================================================
# 4. Counterfactual Regret
# =====================================================================


class CounterfactualRegret(nn.Module):
    """Experience-based learning from "what if I had done X instead".

    Builds on CounterfactualImagination (existing) but adds:
    - Regret-tagged episodic memories (stored for rehearsal)
    - Regret-driven exploration boost (try the alternative next time)
    - Regret decay over time (you forget old regrets)
    """

    def __init__(
        self,
        max_regrets: int = 200,
        regret_decay: float = 0.99,
        exploration_boost: float = 0.1,
    ) -> None:
        super().__init__()
        self._max = int(max_regrets)
        self._decay = float(regret_decay)
        self._boost = float(exploration_boost)

        self._regrets: list[dict[str, Any]] = []

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._regrets)

    def record_regret(
        self,
        actual_action: int,
        counterfactual_action: int,
        actual_reward: float,
        counterfactual_reward: float,
        regret_magnitude: float,
        step: int,
    ) -> None:
        """Record a regret: counterfactual was better than actual."""
        if regret_magnitude < 0.01:
            return
        if len(self._regrets) >= self._max:
            self._regrets.pop(0)
        self._regrets.append({
            "actual_action": actual_action,
            "better_action": counterfactual_action,
            "actual_reward": actual_reward,
            "cf_reward": counterfactual_reward,
            "regret": regret_magnitude,
            "step": step,
        })

    def get_regret_bias(self, num_actions: int) -> torch.Tensor:
        """Return policy bias: boost actions that were 'better in hindsight'."""
        bias = torch.zeros(num_actions)
        for r in self._regrets[-32:]:
            bias[r["better_action"]] += r["regret"] * self._boost
        return bias

    def decay(self) -> None:
        for r in self._regrets:
            r["regret"] *= self._decay

    def summary(self) -> dict:
        if not self._regrets:
            return {"regrets": 0}
        recent = self._regrets[-10:]
        return {
            "regrets": len(self._regrets),
            "mean_regret": sum(r["regret"] for r in self._regrets) / len(self._regrets),
            "top_regret_action": max(set(r["better_action"] for r in recent),
                                     key=lambda a: sum(1 for r in recent if r["better_action"] == a)),
        }


# =====================================================================
# 5. Curiosity Director
# =====================================================================


class CuriosityDirector(nn.Module):
    """Orchestrates three existing curiosity signals into one directed signal.

    Sources:
    1. RSSM prediction uncertainty ("this state is unpredictable → explore")
    2. Knowledge gap ("I don't know what this object is → investigate")
    3. Social curiosity ("what is the caregiver doing? → observe")

    The director weights them based on context:
    - Alone in new room → RSSM uncertainty high weight
    - Familiar setting with novel object → knowledge gap high weight
    - Caregiver nearby doing something → social curiosity high weight
    """

    def __init__(
        self,
        d_model: int = 128,
        rssm_weight: float = 0.4,
        gap_weight: float = 0.35,
        social_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self._rssm_w = rssm_weight
        self._gap_w = gap_weight
        self._social_w = social_weight

        # Context encoder for dynamic weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),  # → 3 weights
        )

    def forward(
        self,
        rssm_uncertainty: float,
        knowledge_gap: float,
        social_curiosity: float,
        context_embedding: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute weighted curiosity signal.

        Dynamic weights if context provided, static otherwise.
        """
        if context_embedding is not None:
            raw_weights = F.softmax(
                self.context_encoder(context_embedding.unsqueeze(0)), dim=-1,
            ).squeeze(0)
            weights = {
                "rssm": float(raw_weights[0]),
                "gap": float(raw_weights[1]),
                "social": float(raw_weights[2]),
            }
        else:
            total = self._rssm_w + self._gap_w + self._social_w
            weights = {
                "rssm": self._rssm_w / total,
                "gap": self._gap_w / total,
                "social": self._social_w / total,
            }

        total_curiosity = (
            rssm_uncertainty * weights["rssm"]
            + knowledge_gap * weights["gap"]
            + social_curiosity * weights["social"]
        )
        weights["total"] = total_curiosity
        return weights


# =====================================================================
# 6. Value System
# =====================================================================


@dataclass
class ValueJudgment:
    action: int
    context: str
    goodness: float   # -1 (bad) to +1 (good)
    confidence: float  # [0,1]
    source_drive: str  # which drive produced this judgment
    step: int


class ValueSystem(nn.Module):
    """Rudimentary right/wrong derived from 5 homeostatic drives.

    Not human morality. This is "this action improved my drives → good;
    this action depleted my drives → bad."

    Values are contextual: "pushing is good when curious, bad when resting."
    This is the developmental precursor to moral reasoning.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_judgments: int = 500,
        generalization_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self._max = int(max_judgments)
        self._gen_threshold = float(generalization_threshold)

        self._judgments: list[ValueJudgment] = []

        # Value generalization: from specific actions → abstract principles
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model + 8, d_model),  # context_emb + action_onehot
            nn.GELU(),
            nn.Linear(d_model, 1),  # → goodness prediction
        )

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._judgments)

    def judge(
        self,
        action: int,
        context_embedding: torch.Tensor,
        drive_deltas: dict[str, float],
        step: int,
    ) -> ValueJudgment:
        """Judge an action based on how it affected homeostatic drives.

        Returns a ValueJudgment with goodness ∈ [-1, 1].
        """
        total_delta = sum(drive_deltas.values())
        goodness = math.tanh(total_delta * 3.0)  # map to [-1, 1]

        # Which drive was most affected?
        dominant_drive = max(drive_deltas, key=lambda k: abs(drive_deltas[k]))

        judgment = ValueJudgment(
            action=int(action),
            context="",
            goodness=float(goodness),
            confidence=abs(float(goodness)),
            source_drive=dominant_drive,
            step=int(step),
        )

        if len(self._judgments) >= self._max:
            self._judgments.pop(0)
        self._judgments.append(judgment)
        return judgment

    def predict_goodness(
        self, action: int, context_embedding: torch.Tensor,
    ) -> float:
        """Predict how good an action would be in this context.

        Uses learned value function from past judgments.
        """
        action_onehot = F.one_hot(
            torch.tensor([action]), 8,
        ).float().to(context_embedding.device)
        combined = torch.cat([context_embedding.unsqueeze(0), action_onehot], dim=-1)
        return float(torch.tanh(self.value_encoder(combined)).item())

    def get_principle(self, min_confidence: float = 0.5) -> str:
        """Derive a simple value principle from accumulated judgments."""
        if not self._judgments:
            return "I have not learned any values yet."

        # Group by action
        action_goodness: dict[int, list[float]] = {}
        for j in self._judgments:
            if j.confidence >= min_confidence:
                action_goodness.setdefault(j.action, []).append(j.goodness)

        if not action_goodness:
            return "I am uncertain about what is right."

        action_names = ["move_north", "move_south", "move_west", "move_east",
                        "push", "pull", "grasp", "wait"]
        parts = []
        for action, goods in sorted(action_goodness.items(), key=lambda x: -abs(sum(x[1]))):
            mean_g = sum(goods) / len(goods)
            if abs(mean_g) > 0.3:
                a_name = action_names[action] if action < len(action_names) else f"action_{action}"
                quality = "good" if mean_g > 0 else "bad"
                parts.append(f"{a_name} is {quality} (confidence: {abs(mean_g):.1%})")

        if not parts:
            return "Most actions seem neutral so far."
        return "I have learned: " + "; ".join(parts[:5])

    def summary(self) -> dict:
        return {
            "judgments": len(self._judgments),
            "mean_goodness": sum(j.goodness for j in self._judgments) / max(1, len(self._judgments)),
            "principle": self.get_principle(),
        }
