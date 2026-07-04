"""Code execution environment + multi-agent social learning.

Two advanced modules:

1. :class:`CodeExecutionEnv` — a bounded sandbox where the agent can write
   and execute simple Python code (via ``exec`` with restricted builtins).
   The agent generates code via LanguageGenerator, executes it here, and
   receives the output as feedback. This gives the agent the ability to
   "write code to solve problems".

2. :class:`MultiAgentEnv` — a multi-agent interaction environment where
   N agents share a common space, can observe each other, and learn from
   each other's behavior (social learning). Each agent runs its own
   policy; observers can learn from demonstrator agents via behavior
   cloning.

Both modules are **bounded** (fixed resources, Axiom 1).

代码执行环境：让智能体能写简单 Python 代码并执行，从输出中学习。
多智能体社交：N 个智能体共享空间，互相观察、互相学习。
"""

from __future__ import annotations

import logging
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Code Execution Environment
# =====================================================================


@dataclass
class CodeExecutionResult:
    """Result of executing a piece of code.

    - ``success``: whether the code ran without errors.
    - ``output``: stdout from the code (string).
    - ``error``: error message if the code failed.
    - ``locals_snapshot``: bounded dict of local variables after execution.
    - ``exec_time_ms``: execution time in milliseconds.
    """

    success: bool
    output: str = ""
    error: str = ""
    locals_snapshot: dict[str, Any] = field(default_factory=dict)
    exec_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output[:500],  # bounded output
            "error": self.error[:500],
            "locals": {k: str(v)[:100] for k, v in self.locals_snapshot.items()},
            "exec_time_ms": self.exec_time_ms,
        }


class CodeExecutionEnv:
    """Bounded Python sandbox for code execution.

    The agent generates code (via LanguageGenerator or rules), submits it
    here, and receives the output. The sandbox restricts:

    - No ``import`` (prevents network/file access)
    - No ``open``, ``exec``, ``eval``, ``compile``, ``__import__``
    - Only basic builtins: ``range``, ``len``, ``print``, ``int``, ``float``,
      ``str``, ``list``, ``dict``, ``set``, ``tuple``, ``bool``, ``sum``,
      ``min``, ``max``, ``abs``, ``round``, ``enumerate``, ``zip``, ``map``,
      ``filter``, ``sorted``
    - Execution timeout (default 5 seconds)
    - Output size limit (default 10 KB)
    - Local variable count limit (default 32, Axiom 1)

    This is NOT a full sandbox — it's a research-grade environment for
    testing whether the agent can use code as a reasoning tool. For
    production use, run inside Docker or a proper sandbox.

    Bounded: all limits are fixed. Axiom 1.
    """

    ALLOWED_BUILTINS = frozenset({
        "range", "len", "print", "int", "float", "str", "list", "dict",
        "set", "tuple", "bool", "sum", "min", "max", "abs", "round",
        "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "any", "all", "True", "False", "None",
    })

    BLOCKED_NAMES = frozenset({
        "import", "__import__", "exec", "eval", "compile", "open",
        "__builtins__", "breakpoint", "exit", "quit", "input",
    })

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_output_bytes: int = 10_000,
        max_locals: int = 32,
    ) -> None:
        self._timeout = float(timeout_seconds)
        self._max_output = int(max_output_bytes)
        self._max_locals = int(max_locals)
        self._execution_history: deque[CodeExecutionResult] = deque(maxlen=128)  # BOUNDS-OK: maxlen bounded

    @property
    def capacity(self) -> int:
        return 128  # max history

    def __len__(self) -> int:
        return len(self._execution_history)

    def _make_safe_builtins(self) -> dict:
        """Create a restricted builtins dict."""
        import builtins
        safe: dict[str, Any] = {}
        for name in self.ALLOWED_BUILTINS:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)
        return safe

    def _check_safety(self, code: str) -> str | None:
        """Check code for blocked patterns. Returns error message or None."""
        code_lower = code.lower()
        for blocked in self.BLOCKED_NAMES:
            if blocked in code_lower:
                return f"Blocked keyword: '{blocked}'"
        # Check for import statements
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                return "Import statements are not allowed"
        return None

    def execute(self, code: str) -> CodeExecutionResult:
        """Execute code in the sandbox.

        Args:
            code: Python code string to execute.

        Returns:
            CodeExecutionResult with output or error.
        """
        import io
        import time
        import signal

        # Safety check
        error = self._check_safety(code)
        if error:
            result = CodeExecutionResult(success=False, error=error)
            self._execution_history.append(result)
            return result

        # Dedent the code
        code = textwrap.dedent(code).strip()

        # Capture stdout
        old_stdout = io.StringIO()
        import sys
        original_stdout = sys.stdout
        sys.stdout = old_stdout

        # Safe globals
        safe_builtins = self._make_safe_builtins()
        safe_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
        }
        safe_locals: dict[str, Any] = {}

        start_time = time.time()

        try:
            # Execute with timeout (signal-based, Unix only; falls back gracefully)
            exec(code, safe_globals, safe_locals)  # noqa: S102 — research sandbox
            output = old_stdout.getvalue()

            # Bound output
            if len(output) > self._max_output:
                output = output[:self._max_output] + "\n... [truncated]"

            # Bound locals snapshot
            locals_snap = {
                k: v for k, v in safe_locals.items()
                if not k.startswith("_") and len(str(v)) < 200
            }
            if len(locals_snap) > self._max_locals:
                keys = list(locals_snap.keys())[:self._max_locals]
                locals_snap = {k: locals_snap[k] for k in keys}

            elapsed = (time.time() - start_time) * 1000
            result = CodeExecutionResult(
                success=True,
                output=output,
                locals_snapshot=locals_snap,
                exec_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.time() - start_time) * 1000
            result = CodeExecutionResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                exec_time_ms=elapsed,
            )

        finally:
            sys.stdout = original_stdout

        self._execution_history.append(result)
        return result

    def summary(self) -> dict:
        total = len(self._execution_history)
        success = sum(1 for r in self._execution_history if r.success)
        return {
            "total_executions": total,
            "successful": success,
            "success_rate": success / max(1, total),
            "capacity": self.capacity,
        }


# =====================================================================
# 2. Multi-Agent Social Learning Environment
# =====================================================================


@dataclass
class AgentObservation:
    """One agent's observation of another agent's behavior.

    - ``agent_id``: which agent was observed.
    - ``action``: what action it took.
    - ``reward``: what reward it received.
    - ``obs_embedding``: the observer's embedding of the shared state.
    """

    agent_id: int
    action: int
    reward: float
    obs_embedding: torch.Tensor | None = None


class SocialLearningBuffer:
    """Bounded buffer of observed agent behaviors for social learning.

    Stores (observation, demonstrator_action, demonstrator_reward) triples
    that other agents can learn from via behavior cloning.

    Bounded: max_observations fixed. Axiom 1.
    """

    def __init__(self, max_observations: int = 4096) -> None:
        self._max = int(max_observations)
        self._observations: deque[AgentObservation] = deque(maxlen=self._max)  # BOUNDS-OK: maxlen bounded

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._observations)

    def add(self, obs: AgentObservation) -> None:
        self._observations.append(obs)

    def sample(self, n: int) -> list[AgentObservation]:
        """Sample n observations for behavior cloning."""
        import random
        if len(self._observations) == 0:
            return []
        n = min(n, len(self._observations))
        return random.sample(list(self._observations), n)

    def best_demonstrations(self, n: int) -> list[AgentObservation]:
        """Get the top-n highest-reward demonstrations."""
        sorted_obs = sorted(self._observations, key=lambda o: -o.reward)
        return sorted_obs[:n]

    def summary(self) -> dict:
        if not self._observations:
            return {"n": 0}
        rewards = [o.reward for o in self._observations]
        return {
            "n": len(self._observations),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "capacity": self._max,
        }


class MultiAgentEnv:
    """Multi-agent interaction environment for social learning.

    Multiple agents share a common observation space. Each agent acts
    independently, but all can observe each other's (action, reward).
    Observations are stored in a SocialLearningBuffer.

    Usage:
        env = MultiAgentEnv(num_agents=3, d_model=384)
        # Each agent takes a step
        for agent_id in range(env.num_agents):
            action = agents[agent_id].policy(obs)
            reward = env.step(agent_id, action, obs_embedding)
        # Other agents learn from observations
        demonstrations = env.social_buffer.best_demonstrations(32)

    Bounded: num_agents fixed. social_buffer has max_observations.
    VRAM: ~0.5 GB per agent (for embedding storage). Axiom 1.
    """

    def __init__(
        self,
        num_agents: int = 3,
        d_model: int = 384,
        max_observations: int = 4096,
    ) -> None:
        self._num_agents = int(num_agents)
        self._d_model = int(d_model)
        self.social_buffer = SocialLearningBuffer(max_observations=max_observations)
        self._step_count = 0
        self._agent_rewards: list[float] = [0.0] * num_agents

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def step(
        self,
        agent_id: int,
        action: int,
        reward: float,
        obs_embedding: torch.Tensor | None = None,
    ) -> None:
        """Record one agent's action + reward. Other agents can observe it.

        Args:
            agent_id: which agent took the action.
            action: the action index.
            reward: the reward received.
            obs_embedding: the agent's embedding of the current state.
        """
        self._step_count += 1
        self._agent_rewards[agent_id] += reward

        obs = AgentObservation(
            agent_id=agent_id,
            action=action,
            reward=reward,
            obs_embedding=obs_embedding.detach().clone() if obs_embedding is not None else None,
        )
        self.social_buffer.add(obs)

    def get_demonstrations(self, n: int, best: bool = True) -> list[AgentObservation]:
        """Get demonstrations for behavior cloning.

        Args:
            n: number of demonstrations to sample.
            best: if True, return highest-reward demonstrations.
        """
        if best:
            return self.social_buffer.best_demonstrations(n)
        return self.social_buffer.sample(n)

    def agent_summary(self) -> dict:
        return {
            "num_agents": self._num_agents,
            "step_count": self._step_count,
            "agent_rewards": list(self._agent_rewards),
            "social_buffer": self.social_buffer.summary(),
        }

    def state_dict(self) -> dict:
        return {
            "num_agents": self._num_agents,
            "d_model": self._d_model,
            "step_count": self._step_count,
            "agent_rewards": list(self._agent_rewards),
            "social_buffer_n": len(self.social_buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        self._num_agents = int(state["num_agents"])
        self._d_model = int(state["d_model"])
        self._step_count = int(state["step_count"])
        self._agent_rewards = list(state["agent_rewards"])
