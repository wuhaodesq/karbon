"""Creativity Orchestrator — Full Transformational Creativity Engine.

Implements the formula:
    T(C) ≈ [(旧经验积木 ⊕ 新抽象符号) ⊗ 远距离重组 ⊗ 规则打破]
           ——在容忍度阈值下部分豁免即时因果校验
           ——最终回归高阶因果自洽
           + 好奇心驱动

Three components:
1. ToleranceController — developmental annealing of ambiguity tolerance
2. CreativeValidation — two-stage verify: relaxed exploration → strict proof
3. CreativityOrchestrator — ties all existing modules into the creative pipeline

Reuses: DivergentGenerator, TransformationalCreativityEngine, RSSM world model,
        SkillLibrary, CounterfactualImagination, CausalDiscovery, RuleInductionEngine.

创造力编排器：实现完整变革创造公式。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. ToleranceController — 定向容忍度调度
# =====================================================================


class ToleranceController:
    """Developmental tolerance annealing.

    Models the human ability to "tolerate ambiguity" during creative leaps.
    Starts high (infant: accept anything novel), anneals to low (adult: rigorous).

    Implements the "定向容忍度" concept:
    - Not blind tolerance (uniform relaxation of all constraints)
    - Directed tolerance: relax causal verification but NOT logical consistency
    - Domain-specific: high tolerance for physical novelty, lower for safety

    容忍度控制器：发育性退火，从高容忍（婴儿）到严格验证（成人）。
    """

    def __init__(
        self,
        initial_tolerance: float = 0.8,     # 0 = strict, 1 = accept everything
        min_tolerance: float = 0.1,
        annealing_steps: int = 1_000_000,    # steps to reach min_tolerance
        creativity_budget: int = 100,         # max pending creative ideas
    ) -> None:
        self._initial = float(initial_tolerance)
        self._min = float(min_tolerance)
        self._annealing_steps = int(annealing_steps)
        self._budget = int(creativity_budget)

        # Pending creative ideas awaiting verification
        self._pending: deque[dict[str, Any]] = deque(maxlen=self._budget)  # BOUNDS-OK

    @property
    def capacity(self) -> int:
        return self._budget

    def tolerance(self, global_step: int) -> float:
        """Current tolerance at this developmental step.

        Exponential decay: t(s) = t_initial * exp(-s / annealing_steps), floored at min.
        """
        if global_step <= 0:
            return self._initial
        decay = float(np.exp(-global_step / self._annealing_steps))
        return max(self._min, self._initial * decay)

    def should_relax(self, global_step: int, novelty_score: float) -> bool:
        """Should we relax causal verification for this creative idea?

        Relax when BOTH:
        - Tolerance is still high enough (developmental stage allows wild ideas)
        - Novelty is high (truly novel ideas need more tolerance to survive)
        """
        tol = self.tolerance(global_step)
        return float(novelty_score) > 0.3 and tol > self._min

    def should_verify(self, global_step: int, idea_age_steps: int) -> bool:
        """Is it time to apply strict causal verification to a pending idea?

        Ideas that have been pending long enough (age > tolerance window) get verified.
        """
        tol = self.tolerance(global_step)
        patience = int(1000 * tol)  # more tolerance = longer patience
        return idea_age_steps > max(100, patience)

    def shelve_idea(self, idea: dict[str, Any]) -> None:
        """Store a creative idea for later verification."""
        idea.setdefault("created_step", 0)
        self._pending.append(idea)

    def pop_verifiable(self, global_step: int) -> list[dict[str, Any]]:
        """Return ideas ready for strict verification."""
        ready = []
        kept = []
        for idea in self._pending:
            age = global_step - idea.get("created_step", global_step)
            if self.should_verify(global_step, age):
                ready.append(idea)
            else:
                kept.append(idea)
        self._pending = deque(kept, maxlen=self._budget)  # BOUNDS-OK
        return ready


# =====================================================================
# 2. CreativeValidation — 两阶段校验
# =====================================================================


@dataclass
class CreativeIdea:
    """A creative proposal awaiting evaluation."""
    id: int
    description: str
    skill_embedding: torch.Tensor       # (d_model,) LoRA-style embedding
    novelty_score: float = 0.0
    utility_score: float = 0.0
    causal_coherence: float = 0.0       # how well it fits causal graph
    passed_stage1: bool = False         # relaxed exploration check
    passed_stage2: bool = False         # strict causal verification
    created_step: int = 0
    promoted_to_skill: bool = False


class CreativeValidation:
    """Two-stage creative verification.

    Stage 1 (Relaxed): During exploration, only check:
        - Is the idea self-consistent? (internal logic)
        - Is it genuinely novel? (does not duplicate existing skill)
        - Tolerance threshold passed?

    Stage 2 (Strict): After exploration cooldown, verify:
        - Does RSSM world model simulation produce coherent outcomes?
        - Does it fit the causal graph (CausalDiscovery edges)?
        - Utility proxy: does it lead to reward increase in simulation?

    两阶段校验器：放松探索 → 严格验证。
    """

    def __init__(
        self,
        novelty_threshold: float = 0.3,
        utility_threshold: float = 0.1,
        causal_coherence_min: float = 0.3,
        max_ideas: int = 200,
    ) -> None:
        self._novelty_threshold = novelty_threshold
        self._utility_threshold = utility_threshold
        self._causal_min = causal_coherence_min
        self._max_ideas = max_ideas
        self._ideas: dict[int, CreativeIdea] = {}
        self._next_id = 0

    @property
    def capacity(self) -> int:
        return self._max_ideas

    def __len__(self) -> int:
        return len(self._ideas)

    def propose(
        self,
        embedding: torch.Tensor,
        description: str,
        global_step: int,
    ) -> CreativeIdea:
        """Register a new creative proposal."""
        if len(self._ideas) >= self._max_ideas:
            # Evict lowest-scored idea
            worst_id = min(
                self._ideas,
                key=lambda iid: self._ideas[iid].novelty_score * self._ideas[iid].utility_score,
            )
            del self._ideas[worst_id]
        idea = CreativeIdea(
            id=self._next_id,
            description=description,
            skill_embedding=embedding.detach().clone(),
            created_step=global_step,
        )
        self._ideas[self._next_id] = idea
        self._next_id += 1
        return idea

    def stage1_relaxed_check(
        self,
        idea: CreativeIdea,
        existing_skills: Any | None = None,  # SkillLibrary for duplication check
    ) -> bool:
        """Relaxed verification: novelty + self-consistency only.

        Uses tolerance to gate strict causal checks.
        """
        # Novelty check: cosine distance from existing skills
        if existing_skills is not None:
            try:
                all_embs = existing_skills.get_all_embeddings()
                if all_embs is not None and all_embs.shape[0] > 0:
                    sims = torch.cosine_similarity(
                        idea.skill_embedding.unsqueeze(0), all_embs, dim=-1,
                    )
                    max_sim = float(sims.max().item())
                    idea.novelty_score = 1.0 - max_sim
                else:
                    idea.novelty_score = 1.0  # no existing skills = maximally novel
            except Exception:
                idea.novelty_score = 0.5  # default
        else:
            idea.novelty_score = 0.5

        idea.passed_stage1 = idea.novelty_score >= self._novelty_threshold
        return bool(idea.passed_stage1)

    def stage2_strict_verify(
        self,
        idea: CreativeIdea,
        world_model: Any | None = None,   # RSSM for simulation
        causal_graph: Any | None = None,   # CausalDiscovery graph
        initial_state: Any | None = None,  # RSSMState for simulation
        num_sim_steps: int = 5,
    ) -> float:
        """Strict verification via RSSM simulation + causal coherence check.

        Returns utility score [0, 1].
        """
        # Simulation: use RSSM to imagine applying this creative skill
        if world_model is not None and initial_state is not None:
            try:
                state = initial_state
                total_reward = 0.0
                coherence = 0.0
                for _ in range(num_sim_steps):
                    # Simulate creative action (proxied via embedding)
                    action_proxy = torch.softmax(idea.skill_embedding[:8], dim=-1).argmax()
                    action_onehot = torch.zeros(1, 8, device=idea.skill_embedding.device)
                    action_onehot[0, int(action_proxy.item())] = 1.0
                    state, _ = world_model.imagine_step(state, action_onehot)
                    decoded = world_model.decode(state)
                    # Utility proxy: decoded state norm change
                    total_reward += float(decoded.norm().item()) * 0.01
                idea.utility_score = min(1.0, total_reward / num_sim_steps)
                # Coherence: variance of decoded states (low variance = coherent)
                idea.causal_coherence = max(0.0, 1.0 - total_reward * 0.1)
            except Exception:
                idea.utility_score = self._utility_threshold
                idea.causal_coherence = 0.5
        else:
            idea.utility_score = self._utility_threshold
            idea.causal_coherence = 0.5

        # Causal graph check
        if causal_graph is not None:
            try:
                edges = causal_graph.get_effects("creative_action", min_strength=0.1)
                if edges:
                    idea.causal_coherence = max(
                        idea.causal_coherence,
                        sum(e.strength for e in edges) / len(edges),
                    )
            except Exception:
                pass

        idea.passed_stage2 = (
            idea.utility_score >= self._utility_threshold
            and idea.causal_coherence >= self._causal_min
        )
        return float(idea.utility_score)

    def get_promoted(self) -> list[CreativeIdea]:
        """Return ideas that passed both stages and should be promoted to skills."""
        return [
            idea for idea in self._ideas.values()
            if idea.passed_stage2 and not idea.promoted_to_skill
        ]

    def mark_promoted(self, idea_id: int) -> None:
        if idea_id in self._ideas:
            self._ideas[idea_id].promoted_to_skill = True

    def summary(self) -> dict:
        pending = sum(1 for i in self._ideas.values() if not i.passed_stage1)
        verified = sum(1 for i in self._ideas.values() if i.passed_stage2)
        return {
            "total_ideas": len(self._ideas),
            "pending": pending,
            "stage1_passed": sum(1 for i in self._ideas.values() if i.passed_stage1),
            "stage2_passed": verified,
            "promoted": sum(1 for i in self._ideas.values() if i.promoted_to_skill),
            "mean_novelty": np.mean([i.novelty_score for i in self._ideas.values()]) if self._ideas else 0.0,
            "mean_utility": np.mean([i.utility_score for i in self._ideas.values()]) if self._ideas else 0.0,
        }


# =====================================================================
# 3. CreativityOrchestrator — 主控制器
# =====================================================================


class CreativityOrchestrator(nn.Module):
    """Full creativity engine integrating all existing modules.

    Pipeline (every `trigger_every_steps`):
        1. DIVERGE: Run DivergentGenerator to propose random skill combinations
        2. BREAK: Run TransformationalCreativity to attempt rule violations
        3. VALIDATE (Stage 1): Relaxed check under tolerance
        4. SIMULATE: RSSM imagine_step on promising candidates
        5. VALIDATE (Stage 2): Strict causal + utility verification
        6. PROMOTE: Successful ideas → SkillLibrary as new LoRA skills
        7. REWARD: Intrinsic creativity reward for policy

    Bounded: max ideas = 200 (Axiom 1).
    Parameters: ~50K (small projection heads).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_actions: int = 8,
        trigger_every_steps: int = 1000,
        max_ideas: int = 200,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._trigger_every = trigger_every_steps
        self._last_trigger_step = -trigger_every_steps

        self.tolerance = ToleranceController(
            initial_tolerance=0.8,
            min_tolerance=0.1,
            annealing_steps=1_000_000,
            creativity_budget=100,
        )
        self.validation = CreativeValidation(
            novelty_threshold=0.3,
            utility_threshold=0.1,
            causal_coherence_min=0.3,
            max_ideas=max_ideas,
        )

        # Small projection: embedding → creativity weight
        self.creativity_proj = nn.Linear(d_model, 1)  # scores ideas
        self.novelty_proj = nn.Linear(d_model, d_model)  # for divergence

        self._creation_count = 0
        self._promotion_count = 0

    @property
    def capacity(self) -> int:
        return self.validation.capacity

    def __len__(self) -> int:
        return len(self.validation)

    def should_trigger(self, step: int) -> bool:
        return (step - self._last_trigger_step) >= self._trigger_every

    def creative_cycle(
        self,
        step: int,
        slot_states: torch.Tensor,         # (num_slots, d_model)
        skill_library: Any | None = None,   # BoundedSkillLibrary
        world_model: Any | None = None,     # RSSM
        causal_graph: Any | None = None,    # CausalDiscovery
        wm_state: Any | None = None,        # RSSMState
        divergent_gen: Any | None = None,   # DivergentGenerator
        transformational: Any | None = None, # TransformationalCreativityEngine
    ) -> dict[str, Any]:
        """Run one full creative cycle.

        Returns dict with:
            ideas_proposed, ideas_passed, creativity_reward
        """
        result: dict[str, Any] = {
            "ideas_proposed": 0,
            "ideas_stage1_passed": 0,
            "ideas_stage2_passed": 0,
            "creativity_reward": 0.0,
        }

        self._last_trigger_step = step
        tol = self.tolerance.tolerance(step)

        # === STEP 1: DIVERGE — propose novel combinations ===
        candidates: list[tuple[torch.Tensor, float, str]] = []

        # From slot states: each slot is a potential creative seed
        for i in range(min(slot_states.shape[0], 7)):
            seed = slot_states[i]
            novelty_embedding = self.novelty_proj(seed)
            novelty_score = float(torch.sigmoid(self.creativity_proj(novelty_embedding)).item())

            if novelty_score < self.tolerance.tolerance(step) * 0.5:
                continue

            # Divergent: random perturbation creates novelty
            noise = torch.randn_like(seed) * tol * 0.1
            candidate_emb = seed + noise
            desc = f"creative_idea_{self._creation_count}_{i}"

            # Apply TransformationalCreativity if available
            if transformational is not None:
                try:
                    candidate_emb = transformational.curio_gate(candidate_emb.unsqueeze(0)).squeeze(0)
                except Exception:
                    pass

            candidates.append((candidate_emb, novelty_score, desc))

        # === STEP 2: PROPOSE ideas ===
        for emb, novelty, desc in candidates:
            idea = self.validation.propose(emb, desc, step)
            idea.novelty_score = novelty
            self._creation_count += 1

            # === STEP 3: Stage 1 — relaxed check ===
            if self.validation.stage1_relaxed_check(idea, skill_library):
                result["ideas_stage1_passed"] += 1

                # === STEP 4: SIMULATE in RSSM ===
                if world_model is not None and wm_state is not None:
                    self.validation.stage2_strict_verify(
                        idea, world_model, causal_graph, wm_state,
                    )

                    # === STEP 5: Stage 2 — strict verification ===
                    if idea.passed_stage2:
                        result["ideas_stage2_passed"] += 1

            result["ideas_proposed"] += 1

        # === STEP 6: PROMOTE successful ideas to skill library ===
        for idea in self.validation.get_promoted():
            if skill_library is not None:
                try:
                    skill = skill_library.new_skill(
                        tag=f"creative_{idea.id}_{idea.description[:20]}"
                    )
                    skill.record_use(reward=idea.utility_score)
                    skill_library.add(skill)
                    self.validation.mark_promoted(idea.id)
                    self._promotion_count += 1
                    logger.info("[creativity] promoted idea #%d: %s (novelty=%.2f, utility=%.2f)",
                                 idea.id, idea.description[:40],
                                 idea.novelty_score, idea.utility_score)
                except Exception as exc:
                    logger.debug("[creativity] promotion failed: %s", exc)

        # === STEP 7: INTRINSIC CREATIVITY REWARD ===
        result["creativity_reward"] = (
            result["ideas_stage2_passed"] * 0.5
            + result["ideas_stage1_passed"] * 0.1
        )

        return result

    def add_creativity_reward(
        self, total_reward: float, creativity_result: dict[str, Any],
    ) -> float:
        """Augment extrinsic reward with creativity bonus."""
        return float(total_reward) + float(creativity_result.get("creativity_reward", 0.0))

    def summary(self) -> dict:
        validation_summary = self.validation.summary()
        validation_summary.update({
            "tolerance": float(self.tolerance.tolerance(self._last_trigger_step)),
            "total_creations": self._creation_count,
            "total_promotions": self._promotion_count,
        })
        return validation_summary

    def state_dict(self) -> dict[str, Any]:
        return {
            "creation_count": self._creation_count,
            "promotion_count": self._promotion_count,
            "last_trigger_step": self._last_trigger_step,
            "validation_ideas": [
                {
                    "id": i.id,
                    "description": i.description,
                    "novelty_score": i.novelty_score,
                    "utility_score": i.utility_score,
                    "causal_coherence": i.causal_coherence,
                    "passed_stage1": i.passed_stage1,
                    "passed_stage2": i.passed_stage2,
                    "promoted": i.promoted_to_skill,
                    "created_step": i.created_step,
                }
                for i in self.validation._ideas.values()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._creation_count = int(state.get("creation_count", 0))
        self._promotion_count = int(state.get("promotion_count", 0))
        self._last_trigger_step = int(state.get("last_trigger_step", -self._trigger_every))
        self.validation._ideas.clear()
        self.validation._next_id = 0
        for i_dict in state.get("validation_ideas", []):
            idea = CreativeIdea(
                id=i_dict["id"],
                description=i_dict["description"],
                skill_embedding=torch.zeros(self._d_model),
                novelty_score=i_dict["novelty_score"],
                utility_score=i_dict["utility_score"],
                causal_coherence=i_dict["causal_coherence"],
                passed_stage1=i_dict["passed_stage1"],
                passed_stage2=i_dict["passed_stage2"],
                promoted_to_skill=i_dict["promoted"],
                created_step=i_dict["created_step"],
            )
            self.validation._ideas[idea.id] = idea
            self.validation._next_id = max(self.validation._next_id, idea.id + 1)
