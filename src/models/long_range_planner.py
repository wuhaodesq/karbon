"""Long-Range Planning — MCTS + RSSM World Model.

Combines Monte Carlo Tree Search with the RSSM world model to plan
multi-step action sequences in imagination before executing them.

Unlike simple PPO (which optimizes one-step actions), this enables:
    "First get the key, then move to the door, then open it, then get the reward"

Architecture:
    1. Node = RSSMState + action_history
    2. Selection: UCB1 (Upper Confidence Bound) to balance explore/exploit
    3. Expansion: RSSM.imagine_step to create child nodes
    4. Simulation: roll out imagination trajectory to estimate value
    5. Backpropagation: update Q-values up the tree

Bounded: max_nodes, max_depth, max_sim_steps all fixed (Axiom 1).

长程规划：MCTS + RSSM 世界模型在想象中搜索多步动作序列。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PlanNode:
    """A node in the MCTS planning tree."""
    state: Any                           # RSSMState
    action: int | None = None            # action that led to this node
    parent: PlanNode | None = None
    children: dict[int, PlanNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0                  # policy prior (from PPO)
    depth: int = 0

    @property
    def q_value(self) -> float:
        return self.total_value / max(1, self.visits)

    def ucb(self, exploration_constant: float = 1.414) -> float:
        """UCB1 for selection."""
        if self.visits == 0 or self.parent is None:
            return float("inf")
        exploit = self.q_value
        explore = exploration_constant * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )
        return exploit + explore


class LongRangePlanner(nn.Module):
    """MCTS-based long-range planner using RSSM world model.

    Plans k steps ahead by imagining action sequences in the world model.
    Returns the best action sequence found.

    Used every N steps (or on demand) to replan. Between plans,
    the agent executes the planned action sequence greedily.

    VRAM: ~0.1 GB (tree nodes stored in CPU).
    """

    def __init__(
        self,
        num_actions: int = 8,
        max_depth: int = 10,          # max planning horizon
        max_nodes: int = 500,         # total nodes per search
        num_simulations: int = 50,    # MCTS iterations
        exploration_constant: float = 1.414,
        temperature: float = 0.5,     # softmax temperature for action selection
    ) -> None:
        super().__init__()
        self._num_actions = num_actions
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._num_sims = num_simulations
        self._c = exploration_constant
        self._temperature = temperature

        self._current_plan: list[int] = []
        self._plan_step_idx: int = 0

    @property
    def capacity(self) -> int:
        return self._max_nodes

    def __len__(self) -> int:
        return len(self._current_plan)

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        root_state: Any,             # RSSMState
        world_model: Any,            # RSSM with imagine_step
        policy_net: Any = None,      # HybridActorCritic for priors
        obs: torch.Tensor | None = None,  # for policy prior
    ) -> list[int]:
        """Run MCTS from root_state. Returns planned action sequence.

        Args:
            root_state: RSSMState from current observation.
            world_model: RSSM instance.
            policy_net: optional neural policy for action priors.
            obs: optional observation tensor for policy prior.

        Returns:
            List of action indices (length ≤ max_depth).
        """
        root = PlanNode(state=root_state, depth=0)
        node_count = 1

        # Get policy priors
        if policy_net is not None and obs is not None:
            with torch.no_grad():
                logits, _ = policy_net(obs)
                priors = F.softmax(logits / self._temperature, dim=-1).squeeze(0).cpu().numpy()
        else:
            priors = np.ones(self._num_actions) / self._num_actions

        for _ in range(self._num_sims):
            if node_count >= self._max_nodes:
                break

            # Selection
            node = root
            while node.children and node.depth < self._max_depth:
                best_child = None
                best_ucb = -float("inf")
                for child in node.children.values():
                    ucb_val = child.ucb(self._c)
                    if ucb_val > best_ucb:
                        best_ucb = ucb_val
                        best_child = child
                if best_child is None:
                    break
                node = best_child

            # Expansion
            if node.depth < self._max_depth and node_count < self._max_nodes:
                for action in range(self._num_actions):
                    action_onehot = F.one_hot(
                        torch.tensor([action]), self._num_actions,
                    ).float().to(
                        node.state.z.device if hasattr(node.state, 'z') else
                        torch.device("cpu")
                    )
                    try:
                        new_state, _ = world_model.imagine_step(node.state, action_onehot)
                        child = PlanNode(
                            state=new_state,
                            action=action,
                            parent=node,
                            depth=node.depth + 1,
                            prior=float(priors[action]),
                        )
                        node.children[action] = child
                        node_count += 1
                    except Exception:
                        continue

            # Simulation (rollout to estimate value)
            if node.children:
                # Value = weighted average of child Q-values + priors
                for child in node.children.values():
                    leaf_value = self._rollout_value(
                        child.state, world_model, depth=min(3, self._max_depth - child.depth),
                    )
                    # Backpropagation
                    self._backpropagate(child, leaf_value)

        # Extract best action sequence from root
        plan = self._extract_plan(root)
        self._current_plan = plan
        self._plan_step_idx = 0
        return plan

    def _rollout_value(
        self, state: Any, world_model: Any, depth: int,
    ) -> float:
        """Roll out a random trajectory from state to estimate value."""
        total = 0.0
        current = state
        for _ in range(depth):
            action = np.random.randint(0, self._num_actions)
            action_onehot = F.one_hot(
                torch.tensor([action]), self._num_actions,
            ).float().to(
                current.z.device if hasattr(current, 'z') else torch.device("cpu")
            )
            try:
                current, _ = world_model.imagine_step(current, action_onehot)
                # Value proxy: decoded state norm (more activity = higher value)
                decoded = world_model.decode(current)
                total += float(decoded.norm().item()) * 0.01
            except Exception:
                break
        return total

    def _backpropagate(self, node: PlanNode, value: float) -> None:
        """Propagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent  # BOUNDS-OK: guaranteed to terminate at root

    def _extract_plan(self, root: PlanNode) -> list[int]:
        """Extract the best action sequence from root by choosing highest-Q children."""
        plan: list[int] = []
        node = root
        visited = set()
        while node.children and len(plan) < self._max_depth:
            best_action = None
            best_q = -float("inf")
            for action, child in node.children.items():
                if child.visits > 0 and child.q_value > best_q:
                    best_q = child.q_value
                    best_action = action
            if best_action is None or best_action in visited:
                break
            visited.add(best_action)
            plan.append(best_action)
            node = node.children[best_action]
        return plan

    # ------------------------------------------------------------------ execute

    def get_next_action(self) -> int | None:
        """Return the next action in the current plan."""
        if self._plan_step_idx >= len(self._current_plan):
            return None
        action = self._current_plan[self._plan_step_idx]
        self._plan_step_idx += 1
        return action

    def has_plan(self) -> bool:
        return self._plan_step_idx < len(self._current_plan)

    # ------------------------------------------------------------------ diagnostics

    def summary(self) -> dict:
        return {
            "plan_length": len(self._current_plan),
            "plan_remaining": len(self._current_plan) - self._plan_step_idx,
            "current_plan": self._current_plan[:10] if self._current_plan else [],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_plan": self._current_plan,
            "plan_step_idx": self._plan_step_idx,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._current_plan = state.get("current_plan", [])
        self._plan_step_idx = int(state.get("plan_step_idx", 0))
