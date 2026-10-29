"""Language Generation: Qwen2.5-7B 4-bit for speech and explanation.

Gives the agent the ability to SPEAK — not just understand language (CLIP)
but generate natural-language descriptions, explanations, and dialogue.

Three capabilities:

1. :meth:`describe` — "What do you see?"
   Takes vision features + rules → generates a description of the current scene.

2. :meth:`explain` — "Why did you do that?"
   Takes the matched rule + action → generates a natural-language explanation.

3. :meth:`dialogue` — Free-form conversation.
   Takes a user question + agent state → generates a response.

The LLM is **frozen** (4-bit quantized, ~5 GB VRAM). It doesn't train.
The agent's "reasoning" is done by the neural-symbolic layer — the LLM
just translates that reasoning into human-readable language.

    Neural-Symbolic: "Rule #5: IF see key THEN pick up (conf=0.9)"
         ↓
    Language Gen: "I see a key on the floor. Based on my experience,
                   picking it up is the right move — I'm 90% confident."

Bounded: LLM is frozen → constant VRAM. No growing state. Axiom 1 satisfied.

语言生成模块：用 Qwen2.5-7B 4-bit 让智能体能说话。
LLM 冻结，不做推理——只把神经符号层的推理结果翻译成人话。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# LanguageGenerator: wraps a quantized LLM for text generation
# =====================================================================


class LanguageGenerator(nn.Module):
    """Frozen 4-bit LLM for language generation.

    Loads Qwen2.5-7B (or similar) in 4-bit NF4 quantization. The model
    is frozen — no gradients flow through it. Used purely for inference
    (text generation).

    Falls back to a template-based generator if:
    - transformers library not installed
    - bitsandbytes not installed (4-bit quantization)
    - model download fails (no internet)
    - GPU VRAM insufficient

    In fallback mode, generates simple template-based text. Less natural
    but always works.

    Bounded: model is frozen → constant VRAM (~5 GB for 7B 4-bit).
    No state accumulation between calls. Axiom 1 satisfied.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        fallback_temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._fallback_temp = fallback_temperature
        self._mode = "template"  # default fallback
        self._llm = None
        self._tokenizer = None
        self._device = device

        self._try_load_llm(model_name, load_in_4bit)

    def _try_load_llm(self, model_name: str, load_in_4bit: bool) -> None:
        """Attempt to load the quantized LLM. Fall back to template on failure."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # 4-bit quantization config
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._llm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=self._device,
                    trust_remote_code=True,
                )
            else:
                self._llm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=self._device,
                    trust_remote_code=True,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Freeze
            for p in self._llm.parameters():
                p.requires_grad_(False)
            self._llm.eval()

            self._mode = "llm"
            n_params = sum(p.numel() for p in self._llm.parameters())
            logger.info(
                "LanguageGenerator: %s loaded (4-bit=%s, params=%dB, mode=llm)",
                model_name, load_in_4bit, n_params // 10**9,
            )

        except ImportError as exc:
            logger.warning(
                "LanguageGenerator: transformers/bitsandbytes not installed (%s). "
                "Falling back to template mode.", exc
            )
            self._mode = "template"

        except Exception as exc:
            logger.warning(
                "LanguageGenerator: LLM load failed (%s). Falling back to template mode.",
                exc
            )
            self._mode = "template"

    @property
    def mode(self) -> str:
        """Current mode: 'llm' or 'template'."""
        return self._mode

    @property
    def is_llm_active(self) -> bool:
        return self._mode == "llm" and self._llm is not None

    # ---------------------------------------------------------- generation

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt.

        In LLM mode: uses the quantized model with chat template.
        In template mode: returns the prompt as-is (no real generation).

        Args:
            prompt: the user input / task description.
            system: optional system prompt (e.g., "You are a helpful AI agent...")

        Returns:
            Generated text string.
        """
        if self.is_llm_active:
            return self._generate_llm(prompt, system)
        return self._generate_template(prompt, system)

    def _generate_llm(self, prompt: str, system: str | None = None) -> str:
        """Generate using the quantized LLM."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._llm.device)

        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._fallback_temp,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                top_p=0.9,
            )

        # Decode only the generated part (after the prompt)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_template(self, prompt: str, system: str | None = None) -> str:
        """Fallback: simple template-based response."""
        return f"[template mode] {prompt}"

    # ---------------------------------------------------------- high-level API

    def describe(
        self,
        agent_state: dict[str, Any],
        grounded: bool = True,
    ) -> str:
        """"What do you see?" - describe the current situation.

        grounded=True (default) implements GROUNDED SPEECH: the factual
        skeleton is built from the agent's structured state (perception),
        the frozen LLM may only rephrase it, and every output is
        fact-checked; on any numeric/kind violation we return the skeleton
        verbatim. Rationale (Stage 23 M3a): the raw frozen LLM measured at
        faithfulness 0.125 / hallucination 0.458 on the strict battery -
        it invents counts. Grounding must come from the agent, not the LLM.

        Args:
            agent_state: dict with keys like:
                - "confidence"/"familiarity"/"progress": float
                - "active_rules": list[str]
                - "current_task": str
                - "episode_return": float
                - "num_objects": int, "kinds_seen": str,
                  "occluded_now": int (observation facts, optional)
            grounded: use the fact-checked skeleton path when facts present.
        """
        system = (
            "You are an embodied AI agent in a 3D world. "
            "Describe the scene in 1-2 sentences using ONLY the facts listed "
            "below. State the total number of objects, how many are hidden "
            "behind walls, and the object kinds exactly as given. "
            "Do not invent numbers or object kinds. Be concise and natural."
        )

        n = agent_state.get("num_objects")
        kinds_str = str(agent_state.get("kinds_seen", "")).strip()
        occ = int(agent_state.get("occluded_now", 0) or 0)
        if grounded and n is not None:
            n = int(n)
            skeleton = (
                f"I see {n} objects in total: {kinds_str}. "
                f"{occ} of them are hidden behind walls."
            )
            if not self.is_llm_active:
                return skeleton
            prompt = (
                f"FACTS (keep exactly): total objects = {n}; "
                f"kinds = {kinds_str}; hidden behind walls now = {occ}.\n"
                f"Skeleton: {skeleton}\n"
                "Rewrite as ONE natural sentence. Do not change any numbers, "
                "object counts, or object kinds."
            )
            text = self.generate(prompt, system=system).strip()
            if text and self._facts_consistent(text, n, occ, kinds_str):
                return text
            return skeleton
        rules_str = "; ".join(agent_state.get("active_rules", [])) or "no active rules"
        obs = ""
        if agent_state.get("num_objects") is not None:
            obs = (
                f"- Objects in view: {agent_state['num_objects']}"
                f" ({agent_state.get('kinds_seen', 'unknown kinds')})\n"
                f"- Objects currently hidden behind walls:"
                f" {agent_state.get('occluded_now', 0)}\n"
            )
        prompt = (
            f"Current state:\n"
            f"- Task: {agent_state.get('current_task', 'unknown')}\n"
            f"- Confidence: {agent_state.get('confidence', 0.5):.0%}\n"
            f"- Familiarity: {agent_state.get('familiarity', 0.5):.0%}\n"
            f"- Progress: {agent_state.get('progress', 0.5):.0%}\n"
            f"- Active rules: {rules_str}\n"
            f"- Episode return so far: {agent_state.get('episode_return', 0.0):.3f}\n"
            + obs + "\n"
            f"What do you see and what are you doing?"
        )
        return self.generate(prompt, system=system)

    _KIND_ALIASES = {
        "sphere": ("sphere", "ball", "round object"),
        "box": ("box", "block", "cube", "crate"),
        "cylinder": ("cylinder", "can", "bottle", "tube"),
        "capsule": ("capsule", "pill"),
        "key": ("key",),
        "door": ("door",),
        "wall": ("wall", "wall_block"),
        "occluder": ("occluder", "wall", "barrier"),
    }

    def _facts_consistent(self, text: str, n: int, occ: int,
                          kinds_str: str) -> bool:
        """Light fact-check: exact total, exact hidden count, no new kinds."""
        import re

        low = text.lower()
        totals = [int(x) for x in re.findall(
            r"(\d{1,3})\s+(?:objects?|things?|items?)\b", low)]
        if totals != [n]:
            return False
        hiddens = [int(x) for x in re.findall(
            r"(\d{1,3})\s+(?:of\s+(?:them|which)\s+)?"
            r"(?:other\s+|more\s+)?(?:objects?\s+)?"
            r"(?:are\s+|is\s+)?(?:hidden|occluded|behind)", low)]
        if occ > 0:
            if not hiddens or set(hiddens) != {occ}:
                return False
        elif any(h > 0 for h in hiddens):
            return False
        allowed = {k.strip().lower() for k in kinds_str.split(",") if k.strip()}
        for kind, words in self._KIND_ALIASES.items():
            present = any(w in low for w in words)
            if present and kind not in allowed:
                return False
        for kind in allowed:
            if not any(w in low for w in self._KIND_ALIASES.get(kind, (kind,))):
                return False
        return True

    def explain(
        self,
        action: int,
        action_name: str,
        rule_description: str | None,
        confidence: float,
    ) -> str:
        """"Why did you do that?" — explain a decision.

        Args:
            action: the action index taken.
            action_name: human-readable action name.
            rule_description: the matched rule (if any), or None.
            confidence: the agent's confidence (0-1).
        """
        system = (
            "You are an AI agent explaining your decision. "
            "Explain in 1-2 sentences why you chose this action. "
            "Be honest about your confidence level."
        )
        if rule_description:
            prompt = (
                f"I chose to '{action_name}' (action {action}).\n"
                f"My reasoning: {rule_description}\n"
                f"My confidence: {confidence:.0%}\n\n"
                f"Why did I choose this action?"
            )
        else:
            prompt = (
                f"I chose to '{action_name}' (action {action}).\n"
                f"I used my neural network's intuition (no explicit rule matched).\n"
                f"My confidence: {confidence:.0%}\n\n"
                f"Why did I choose this action?"
            )
        return self.generate(prompt, system=system)

    def answer(
        self,
        question: str,
        agent_state: dict[str, Any],
    ) -> str:
        """Free-form Q&A — answer a user's question about the agent's state.

        Args:
            question: the user's question (e.g., "What are you doing?")
            agent_state: dict with agent's current state.
        """
        system = (
            "You are an AI agent. Answer the user's question honestly and concisely "
            "based on your current state. You can see the gridworld, have learned "
            "rules, and are navigating toward goals."
        )
        rules_str = "; ".join(agent_state.get("active_rules", [])) or "none"
        prompt = (
            f"User question: {question}\n\n"
            f"My current state:\n"
            f"- Task: {agent_state.get('current_task', 'unknown')}\n"
            f"- Confidence: {agent_state.get('confidence', 0.5):.0%}\n"
            f"- Familiarity: {agent_state.get('familiarity', 0.5):.0%}\n"
            f"- Progress: {agent_state.get('progress', 0.5):.0%}\n"
            f"- Known rules: {rules_str}\n"
            f"- Skills: {agent_state.get('num_skills', 0)} skills in library\n"
            f"- Episode return: {agent_state.get('episode_return', 0.0):.3f}\n"
        )
        return self.generate(prompt, system=system)

    def inner_monologue(
        self,
        reflection_text: str,
        confidence: float,
        familiarity: float,
    ) -> str:
        """Generate a natural inner monologue from a reflection.

        This replaces the template-based InnerDialogue with LLM-generated text.

        Args:
            reflection_text: structured reflection (from ReflectionLoop).
            confidence: self-confidence (0-1).
            familiarity: state familiarity (0-1).
        """
        system = (
            "You are an AI agent thinking to yourself. "
            "Generate a brief inner monologue (1-2 sentences) reflecting on your "
            "current situation. Be natural and honest."
        )
        prompt = (
            f"Reflection: {reflection_text}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Familiarity: {familiarity:.0%}\n\n"
            f"What am I thinking?"
        )
        return self.generate(prompt, system=system)

    # ---------------------------------------------------------- diagnostics

    def summary(self) -> dict:
        return {
            "mode": self._mode,
            "model": self._model_name if self.is_llm_active else "template (fallback)",
            "max_new_tokens": self._max_new_tokens,
        }
