"""Metacognition, self-reflection, and inner dialogue.

Three add-on modules that give the agent self-awareness capabilities:

1. :class:`SelfModel` — reads the agent's own hidden state and outputs
   confidence / familiarity / progress estimates. This is **metacognition**:
   "how sure am I?", "have I seen this before?", "am I improving?"

2. :class:`ReflectionLoop` — after each episode, replays the trajectory
   through the SelfModel and generates a structured self-assessment.
   This is **self-reflection**: "what went well? what went wrong? what
   should I do differently?"

3. :class:`InnerDialogue` — uses a small language model (optional) to
   generate a natural-language "inner monologue" from the reflection.
   This is **inner dialogue**: "I tried to open the door but I didn't have
   the key. Next time I should look for the key first."

All three modules are **bounded** (fixed-size state, Axiom 1) and
**optional** (graceful degradation when components are missing).

三个附加层：元认知（SelfModel）、自我反思（ReflectionLoop）、内心独白（InnerDialogue）。
都是可选的、有界的、可叠加的。
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
# 1. SelfModel — Metacognition
# =====================================================================


@dataclass
class SelfAssessment:
    """The agent's self-evaluation for a single timestep.

    All values are in [0, 1] (sigmoid outputs).

    - ``confidence``: how sure the agent is about its current action.
    - ``familiarity``: how much the current state resembles past experiences.
    - ``progress``: whether the agent thinks it's improving.
    - ``uncertainty``: 1 - confidence (convenience field).
    """

    confidence: float
    familiarity: float
    progress: float

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence": self.confidence,
            "familiarity": self.familiarity,
            "progress": self.progress,
            "uncertainty": self.uncertainty,
        }

    def __repr__(self) -> str:
        return (
            f"SelfAssessment(conf={self.confidence:.2f}, "
            f"fam={self.familiarity:.2f}, prog={self.progress:.2f})"
        )


class SelfModel(nn.Module):
    """Reads the agent's own hidden state → outputs self-assessment.

    Architecture: 3 independent linear heads on top of the agent's
    backbone output (d_model → 1 each). Trained via auxiliary losses:

    - confidence: supervised on |action_log_prob - optimal| (how close to greedy)
    - familiarity: supervised on coverage_tracker (has this state been seen?)
    - progress: supervised on LP delta (is return improving?)

    Or trained end-to-end via PPO (the self-assessment modulates exploration).

    Bounded: 3 × d_model params ≈ 1k. No state accumulation (Axiom 1).

    元认知模块：读自己的 hidden state，输出"我有多确信/熟悉/在进步"。
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # Three heads
        self.confidence_head = nn.Linear(hidden, 1)
        self.familiarity_head = nn.Linear(hidden, 1)
        self.progress_head = nn.Linear(hidden, 1)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read the agent's own hidden state.

        Args:
            hidden_state: (B, d_model) — the output of the Hybrid backbone
                (after squeeze from seq dimension).

        Returns:
            dict with keys "confidence", "familiarity", "progress".
            Each is (B, 1) in [0, 1] via sigmoid.
        """
        h = self.trunk(hidden_state)
        return {
            "confidence": torch.sigmoid(self.confidence_head(h)),
            "familiarity": torch.sigmoid(self.familiarity_head(h)),
            "progress": torch.sigmoid(self.progress_head(h)),
        }

    def assess(self, hidden_state: torch.Tensor) -> SelfAssessment:
        """Convenience: single-example assessment → SelfAssessment dataclass."""
        out = self.forward(hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state)
        return SelfAssessment(
            confidence=float(out["confidence"].squeeze(-1).mean().item()),
            familiarity=float(out["familiarity"].squeeze(-1).mean().item()),
            progress=float(out["progress"].squeeze(-1).mean().item()),
        )

    def auxiliary_loss(
        self,
        hidden_states: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Supervised auxiliary loss for training the SelfModel.

        Args:
            hidden_states: (B, d_model)
            targets: dict with "confidence", "familiarity", "progress"
                each (B,) or (B, 1) in [0, 1].
        """
        preds = self.forward(hidden_states)
        loss = 0.0
        for key in ("confidence", "familiarity", "progress"):
            if key in targets:
                target = targets[key]
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                loss = loss + F.binary_cross_entropy(
                    preds[key].clamp(1e-6, 1 - 1e-6), target
                )
        return loss


# =====================================================================
# 2. ReflectionLoop — Self-reflection after each episode
# =====================================================================


@dataclass
class EpisodeReflection:
    """Structured self-assessment of a completed episode.

    - ``episode_return``: the actual return achieved.
    - ``mean_confidence``: average self-confidence during the episode.
    - ``mean_familiarity``: average familiarity (low = explored new states).
    - ``success``: whether the episode achieved a positive return.
    - ``lessons``: list of natural-language lessons (if InnerDialogue is active).
    - ``adjustments``: dict of suggested parameter adjustments.
    """

    episode_return: float
    mean_confidence: float
    mean_familiarity: float
    mean_progress: float
    success: bool
    lessons: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_return": self.episode_return,
            "mean_confidence": self.mean_confidence,
            "mean_familiarity": self.mean_familiarity,
            "mean_progress": self.mean_progress,
            "success": self.success,
            "lessons": self.lessons,
            "adjustments": self.adjustments,
        }

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        lessons_str = "; ".join(self.lessons) if self.lessons else "none"
        return (
            f"EpisodeReflection[{status}] ret={self.episode_return:.3f} "
            f"conf={self.mean_confidence:.2f} fam={self.mean_familiarity:.2f} "
            f"prog={self.mean_progress:.2f} lessons=[{lessons_str}]"
        )


class ReflectionLoop:
    """Post-episode self-reflection.

    After each episode, the agent:
    1. Collects self-assessments from the SelfModel across the trajectory.
    2. Aggregates them into a structured reflection.
    3. (Optional) Generates natural-language lessons via InnerDialogue.
    4. (Optional) Suggests parameter adjustments (e.g., increase exploration
       when familiarity is low).

    Bounded: uses a fixed-length ring buffer of recent reflections
    (max_reflections, Axiom 1). No unbounded growth.

    自我反思循环：每个 episode 后评估自己的表现，生成结构化反思 + 改进建议。
    """

    def __init__(
        self,
        self_model: SelfModel,
        max_reflections: int = 256,
        reflection_every_episodes: int = 10,
    ) -> None:
        self.self_model = self_model
        self._max = int(max_reflections)
        self._reflections: deque[EpisodeReflection] = deque(maxlen=self._max)  # BOUNDS-OK: maxlen bounded
        self._every = int(reflection_every_episodes)
        self._episode_count = 0
        self._trajectory: list[dict[str, torch.Tensor]] = []

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._reflections)

    def record_step(
        self,
        hidden_state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Record one step's data for the current episode."""
        self._trajectory.append({
            "hidden": hidden_state.detach().cpu(),
            "action": action,
            "reward": reward,
            "done": done,
        })

    def end_episode(self, episode_return: float) -> EpisodeReflection | None:
        """Called at episode end. Returns a Reflection if it's time to reflect."""
        self._episode_count += 1

        # Always compute a brief reflection
        if not self._trajectory:
            r = EpisodeReflection(
                episode_return=episode_return,
                mean_confidence=0.5, mean_familiarity=0.5, mean_progress=0.5,
                success=episode_return > 0,
            )
            self._reflections.append(r)
            self._trajectory.clear()
            return r if self._episode_count % self._every == 0 else None

        # Stack hidden states and get self-assessments
        hiddens = torch.stack([t["hidden"] for t in self._trajectory])
        with torch.no_grad():
            assessments = self.self_model.forward(hiddens)

        conf = float(assessments["confidence"].mean().item())
        fam = float(assessments["familiarity"].mean().item())
        prog = float(assessments["progress"].mean().item())

        # Generate adjustments
        adjustments: dict[str, float] = {}
        if fam < 0.3:
            adjustments["exploration_epsilon_boost"] = 0.05  # explore more
        if conf < 0.3 and episode_return < 0:
            adjustments["learning_rate_boost"] = 1.5  # learn faster from failure
        if conf > 0.9 and episode_return > 0.5:
            adjustments["exploration_epsilon_decay"] = 0.95  # exploit more

        reflection = EpisodeReflection(
            episode_return=episode_return,
            mean_confidence=conf,
            mean_familiarity=fam,
            mean_progress=prog,
            success=episode_return > 0,
            adjustments=adjustments,
        )

        self._reflections.append(reflection)
        self._trajectory.clear()

        if self._episode_count % self._every == 0:
            return reflection
        return None

    def recent_summary(self, n: int = 10) -> dict:
        """Summarize the last N reflections."""
        recent = list(self._reflections)[-n:]
        if not recent:
            return {"n": 0}
        return {
            "n": len(recent),
            "mean_return": sum(r.episode_return for r in recent) / len(recent),
            "mean_confidence": sum(r.mean_confidence for r in recent) / len(recent),
            "mean_familiarity": sum(r.mean_familiarity for r in recent) / len(recent),
            "success_rate": sum(1 for r in recent if r.success) / len(recent),
        }

    def state_dict(self) -> dict:
        return {
            "episode_count": self._episode_count,
            "max": self._max,
            "reflections": [r.to_dict() for r in self._reflections],
        }

    def load_state_dict(self, state: dict) -> None:
        self._episode_count = int(state["episode_count"])
        self._max = int(state["max"])
        self._reflections.clear()
        for r_dict in state["reflections"]:
            self._reflections.append(EpisodeReflection(
                episode_return=r_dict["episode_return"],
                mean_confidence=r_dict["mean_confidence"],
                mean_familiarity=r_dict["mean_familiarity"],
                mean_progress=r_dict["mean_progress"],
                success=r_dict["success"],
                lessons=r_dict.get("lessons", []),
                adjustments=r_dict.get("adjustments", {}),
            ))


# =====================================================================
# 3. InnerDialogue — Natural-language inner monologue
# =====================================================================


class InnerDialogue:
    """Generates a natural-language "inner monologue" from reflections.

    Uses a template-based generator (no LLM required) OR an optional small
    language model for richer dialogue.

    Two modes:
    - ``mode="template"``: rule-based natural language from reflection fields.
      Zero VRAM, zero dependencies, always available.
    - ``mode="llm"``: uses a small local LLM (e.g., Qwen-7B) for richer
      dialogue. Requires ~6 GB VRAM + transformers library.

    Bounded: generates text on demand, no state accumulation (Axiom 1).

    内心独白：把反思转成自然语言。模板模式零依赖，LLM 模式需要小模型。
    """

    def __init__(
        self,
        mode: str = "template",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 128,
    ) -> None:
        self._mode = mode
        self._llm_model_name = llm_model_name
        self._max_new_tokens = max_new_tokens
        self._llm = None
        self._tokenizer = None

        if mode == "llm":
            self._try_load_llm(llm_model_name)

    def _try_load_llm(self, model_name: str) -> None:
        """Try to load a small LLM for richer dialogue."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("InnerDialogue: LLM loaded: %s", model_name)
        except Exception as exc:
            logger.warning(
                "InnerDialogue: LLM load failed (%s), falling back to template mode", exc
            )
            self._mode = "template"
            self._llm = None
            self._tokenizer = None

    @property
    def mode(self) -> str:
        return self._mode

    def generate(self, reflection: EpisodeReflection) -> list[str]:
        """Generate natural-language lessons from a reflection.

        Returns a list of lesson strings. In template mode, these are
        rule-based. In LLM mode, they are generated by the language model.
        """
        if self._mode == "llm" and self._llm is not None and self._tokenizer is not None:
            return self._generate_llm(reflection)
        return self._generate_template(reflection)

    def _generate_template(self, r: EpisodeReflection) -> list[str]:
        """Rule-based natural language from reflection fields."""
        lessons: list[str] = []

        status = "succeeded" if r.success else "failed"
        lessons.append(f"I {status} this episode with return {r.episode_return:.3f}.")

        if r.mean_confidence < 0.3:
            lessons.append("I was very uncertain about my actions. I should explore more to build confidence.")
        elif r.mean_confidence > 0.9 and not r.success:
            lessons.append("I was overconfident but failed. I need to reconsider my strategy.")
        elif r.mean_confidence > 0.8 and r.success:
            lessons.append("I felt confident and succeeded. This task is becoming familiar.")

        if r.mean_familiarity < 0.3:
            lessons.append("I encountered many unfamiliar states. This is a good learning opportunity.")
        elif r.mean_familiarity > 0.8:
            lessons.append("I've seen most of these states before. I should consider moving to a harder task.")

        if r.mean_progress > 0.6:
            lessons.append("I'm making good progress on this task.")
        elif r.mean_progress < 0.2 and r.success:
            lessons.append("I've plateaued on this task. Time to try something new.")

        if "exploration_epsilon_boost" in r.adjustments:
            lessons.append("Adjusting: increasing exploration to discover new strategies.")
        if "learning_rate_boost" in r.adjustments:
            lessons.append("Adjusting: increasing learning rate to recover from failure faster.")
        if "exploration_epsilon_decay" in r.adjustments:
            lessons.append("Adjusting: reducing exploration to exploit my confident policy.")

        return lessons

    def _generate_llm(self, r: EpisodeReflection) -> list[str]:
        """Use a small LLM to generate richer inner dialogue."""
        prompt = self._build_prompt(r)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the generated part (after the prompt)
        generated = text[len(prompt):].strip()
        # Split into sentences as lessons
        lessons = [s.strip() for s in generated.split(".") if s.strip()]
        return lessons[:5]  # limit to 5 lessons

    def _build_prompt(self, r: EpisodeReflection) -> str:
        return (
            f"You are an AI agent reflecting on a completed episode.\n"
            f"Episode result: {'success' if r.success else 'failure'}\n"
            f"Return: {r.episode_return:.3f}\n"
            f"Average confidence: {r.mean_confidence:.2f}\n"
            f"Average familiarity: {r.mean_familiarity:.2f}\n"
            f"Average progress: {r.mean_progress:.2f}\n"
            f"What did you learn? What should you do differently next time?\n"
        )
