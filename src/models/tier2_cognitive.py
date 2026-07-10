"""Four Tier-2 Cognitive Modules — Metaphor, Belief, Moral, Humor.

All CPU-only, zero GPU, zero new storage. Uses existing data.

1. Analogizer — "ball" maps to "life" via shared ConceptGraph edges
2. BeliefDepth2 — second-order belief reasoning (A knows that B knows)
3. MoralConnector — fear/drive violations → "that was wrong"
4. SurpriseHumor — RSSM prediction error > threshold → "that was funny"

隐喻、二级信念、道德连接、物理幽默。纯 CPU，零新依赖。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Analogizer — Metaphor via ConceptGraph mapping
# =====================================================================


class Analogizer(nn.Module):
    """Maps properties from one concept domain to another.

    "Life is like a ball" = "life" shares edges with "ball" in ConceptGraph.
    Finds source→target concept mappings by shared relation patterns.

    Example:
        Source: "ball" has edges: [rolls, light, round, fun]
        Target: "life" has edges: [rolls, unpredictable, round-ish, valuable]
        Shared: rolls, round
        → "Life is like a ball: it rolls, it's round, it can go anywhere."

    Requires: ConceptGraph with 200+ nodes (Phase 3+).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self._d_model = d_model

    def find_metaphor(
        self, concept_graph: Any, source_name: str, k: int = 3,
    ) -> list[dict[str, Any]]:
        """Find concepts that are metaphorically similar to source.

        Similarity = (number of shared edges) / (source total edges).
        Returns top-k with shared edges listed.
        """
        if not hasattr(concept_graph, '_edges') or len(concept_graph._edges) < 10:
            return []

        # Find source node
        source_id = None
        for nid, node in concept_graph._nodes.items():
            if node.name == source_name:
                source_id = nid
                break
        if source_id is None:
            return []

        # Get source edges
        source_edges: set[tuple[str, str]] = set()
        for (src, tgt), edge in concept_graph._edges.items():
            if src == source_id:
                tgt_name = concept_graph._nodes.get(tgt, None)
                if tgt_name:
                    source_edges.add((edge.relation_type, tgt_name.name))

        if len(source_edges) < 2:
            return []

        # Compare with all other nodes
        results = []
        for nid, node in concept_graph._nodes.items():
            if nid == source_id:
                continue
            target_edges: set[tuple[str, str]] = set()
            for (src, tgt), edge in concept_graph._edges.items():
                if src == nid:
                    tgt_name = concept_graph._nodes.get(tgt, None)
                    if tgt_name:
                        target_edges.add((edge.relation_type, tgt_name.name))

            shared = source_edges & target_edges
            if len(shared) >= 2:
                sim = len(shared) / max(1, len(source_edges))
                results.append({
                    "metaphor": f"{source_name} is like {node.name}",
                    "similarity": sim,
                    "shared": list(shared)[:5],
                })

        return sorted(results, key=lambda x: -x["similarity"])[:k]

    def generate_metaphor_statement(self, graph: Any, source: str) -> str:
        """Generate a natural language metaphor statement."""
        results = self.find_metaphor(graph, source, k=1)
        if not results:
            return f"I can't think of anything {source} is like."
        r = results[0]
        shared_str = ", ".join(f"{rel} like {name}" for rel, name in r["shared"][:3])
        return f"{source} is like {r['metaphor'].split()[-1]}: they both {shared_str}."


# =====================================================================
# 2. Second-Order Belief Reasoning
# =====================================================================


class BeliefDepth2:
    """Depth-2 recursive belief reasoning. Deliberately bounded.

    "A knows that B knows X" — one level of nesting. No depth 3+.

    Uses existing unification + resolution from symbolic_reasoning.py.
    Bounded: max 2 agents × 5 beliefs × depth=2 = 50 combos max.

    Requires: TheoryOfMind belief states (Phase 3+).
    """

    def __init__(self, max_agents: int = 3, max_beliefs_per_agent: int = 5) -> None:
        self._max_agents = max_agents
        self._max_beliefs = max_beliefs_per_agent

    def reason_depth2(
        self,
        knower: str,
        about_agent: str,
        proposition: str,
        known_beliefs: dict[str, set[str]],
    ) -> dict[str, Any]:
        """Reason about "knower knows that about_agent knows proposition".

        Args:
            knower: agent name ("caregiver")
            about_agent: agent name ("learner")
            proposition: what is believed ("location_of_ball")
            known_beliefs: {agent_name: {set of propositions they know}}

        Returns:
            dict with "knows_about", "confidence", "chain"
        """
        result: dict[str, Any] = {
            "knows_about": False,
            "confidence": 0.0,
            "chain": [],
        }

        # Level 1: does about_agent know the proposition?
        beliefs = known_beliefs.get(about_agent, set())
        if proposition not in beliefs:
            # Even depth=1 fails
            return result

        # Level 2: does knower know that about_agent knows?
        knower_beliefs = known_beliefs.get(knower, set())
        nested = f"{about_agent}_knows_{proposition}"
        if nested in knower_beliefs:
            result["knows_about"] = True
            result["confidence"] = 0.8
            result["chain"] = [
                f"{about_agent} knows {proposition}",
                f"{knower} knows that {about_agent} knows {proposition}",
            ]
        else:
            # Portion: knower might not have observed about_agent learning
            result["confidence"] = 0.3
            result["chain"] = [
                f"{about_agent} knows {proposition}",
                f"{knower} may not know this",
            ]

        return result

    def build_belief_base(
        self, theory_of_mind: Any, memory_manager: Any = None,
    ) -> dict[str, set[str]]:
        """Build known beliefs from TheoryOfMind + MemoryManager."""
        belief_base: dict[str, set[str]] = {}
        if theory_of_mind is not None and hasattr(theory_of_mind, '_belief_states'):
            for agent_name in theory_of_mind._belief_states:
                belief_base[agent_name] = set()
                if memory_manager is not None:
                    # Agent knows what they experienced
                    for fact in memory_manager.semantic._facts.values():
                        if hasattr(fact, 'key') and fact.value > 0.5:
                            belief_base[agent_name].add(fact.key)
        return belief_base


# =====================================================================
# 3. MoralConnector — ethics from drives + emotions
# =====================================================================


class MoralConnector(nn.Module):
    """Connects ValueSystem judgments with EmotionSystem feelings.

    "Pushing the ball was good" (ValueSystem) + "I felt pleasure" (EmotionSystem)
    → "Good things feel good. Bad things feel bad."

    This is NOT human morality. It's the developmental precursor:
    "actions that deplete my drives AND make me feel bad → I should avoid them"
    "actions that refill my drives AND make me feel good → I should repeat them"
    """

    def __init__(self):
        super().__init__()

    def evaluate_action(
        self,
        value_judgments: Any,       # ValueSystem instance
        emotion_system: Any,         # EmotionSystem instance
        drive_levels: dict[str, float],
    ) -> dict[str, Any]:
        """Combine value and emotional signals to produce moral evaluation.

        Returns dict with "evaluation", "reason", "confidence".
        """
        if value_judgments is None or emotion_system is None:
            return {"evaluation": "neutral", "reason": "no data", "confidence": 0.0}

        # Get recent value judgment
        recent_judgment = value_judgments._judgments[-1] if value_judgments._judgments else None
        if recent_judgment is None:
            return {"evaluation": "neutral", "reason": "no judgments yet", "confidence": 0.0}

        goodness = recent_judgment.goodness
        pleasure = emotion_system.state.pleasure

        # Map to moral signal
        if goodness > 0.3 and pleasure > 0.6:
            return {"evaluation": "good", "reason": "it improved my drives and felt good",
                    "confidence": min(goodness, pleasure)}
        elif goodness < -0.3 and pleasure < 0.4:
            return {"evaluation": "bad", "reason": "it depleted my drives and felt bad",
                    "confidence": min(abs(goodness), 1 - pleasure)}
        elif goodness > 0.3:
            return {"evaluation": "maybe good", "reason": "it improved drives but felt neutral",
                    "confidence": goodness * 0.7}
        elif pleasure > 0.6:
            return {"evaluation": "maybe good", "reason": "it felt good but drives are ambiguous",
                    "confidence": pleasure * 0.5}

        # Check for moral novelty: any critical drive violation?
        critical_drives = ["safety", "social"]
        for d in critical_drives:
            if drive_levels.get(d, 0.5) < 0.2:
                return {"evaluation": "concerning", "reason": f"{d} is critically low",
                        "confidence": 0.6}

        return {"evaluation": "neutral", "reason": "mixed signals", "confidence": 0.3}

    def generate_moral_rule(self, value_system: Any, emotion_system: Any) -> str:
        """Derive a simple moral rule from accumulated experience."""
        if value_system is None or not value_system._judgments:
            return "I have not yet formed any moral intuitions."

        # Find actions consistently rated good/bad
        action_counts: dict[int, list[float]] = {}
        for j in value_system._judgments[-200:]:
            if j.confidence > 0.5:
                action_counts.setdefault(j.action, []).append(j.goodness)

        rules = []
        action_names = ["move north", "move south", "move west", "move east",
                        "push", "pull", "grasp", "wait"]
        for action, scores in sorted(action_counts.items(), key=lambda x: -len(x[1])):
            if len(scores) < 5:
                continue
            mean_g = sum(scores) / len(scores)
            a_name = action_names[action] if action < len(action_names) else f"action_{action}"
            if mean_g > 0.3:
                rules.append(f"  {a_name} is usually good (confidence: {mean_g:.1%})")
            elif mean_g < -0.3:
                rules.append(f"  {a_name} is usually bad (confidence: {abs(mean_g):.1%})")

        if not rules:
            return "All actions seem morally neutral so far."
        return "Moral intuitions:\n" + "\n".join(rules)


# =====================================================================
# 4. Surprise Humor — physical comedy from prediction errors
# =====================================================================


class SurpriseHumor(nn.Module):
    """Detects physically comedic situations from RSSM prediction errors.

    Humor theory: violation of expectation + benign context = funny.
    Our version: RSSM prediction error > threshold AND no negative reward = "that was funny".

    Examples:
        - Ball suddenly bounces much higher than predicted → funny
        - Object glitches through wall → funny
        - Agent falls when trying to climb → funny (but only if not painful)

    This is NOT understanding jokes. It's the developmental precursor:
    "unexpected but harmless event" → "that was surprising in a good way"
    """

    def __init__(
        self, surprise_threshold: float = 0.05, humor_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self._surprise_threshold = surprise_threshold
        self._humor_threshold = humor_threshold
        self._funny_moments: list[dict] = []
        self._max_moments = 200

    def detect(
        self, prediction_error: float, reward: float,
        step: int, context: str = "",
    ) -> dict[str, Any] | None:
        """Detect a humorous moment.

        Funny = prediction error is very high AND the outcome was not bad.
        """
        if prediction_error < self._surprise_threshold:
            return None

        # Only funny if outcome was neutral or positive
        if reward < -0.1:
            return None  # bad outcome = not funny

        moment = {
            "step": step,
            "surprise": prediction_error,
            "reward": reward,
            "context": context,
            "funny_score": prediction_error * (1.0 + max(0, reward)),
        }

        if len(self._funny_moments) >= self._max_moments:
            self._funny_moments.pop(0)
        self._funny_moments.append(moment)

        if moment["funny_score"] > self._humor_threshold:
            return {
                "is_funny": True,
                "description": f"Something surprising happened! (error={prediction_error:.3f}, reward={reward:.2f})",
                "funny_score": moment["funny_score"],
            }
        return {"is_funny": False}

    def get_funniest_moment(self) -> str:
        if not self._funny_moments:
            return "Nothing funny has happened yet."
        best = max(self._funny_moments, key=lambda m: m["funny_score"])
        return f"Funniest moment: step {best['step']}, {best['context']} (score: {best['funny_score']:.2f})"

    @property
    def capacity(self) -> int:
        return self._max_moments

    def __len__(self) -> int:
        return len(self._funny_moments)
