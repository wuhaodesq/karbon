"""Tests for code execution environment + multi-agent social learning."""

from __future__ import annotations

import pytest
import torch

from src.models.code_and_social import (
    AgentObservation,
    CodeExecutionEnv,
    CodeExecutionResult,
    MultiAgentEnv,
    SocialLearningBuffer,
)


# =====================================================================
# CodeExecutionEnv
# =====================================================================


def test_code_exec_simple_print():
    env = CodeExecutionEnv()
    result = env.execute("print('hello world')")
    assert result.success is True
    assert "hello world" in result.output


def test_code_exec_math():
    env = CodeExecutionEnv()
    result = env.execute("x = 3 + 4\nprint(x)")
    assert result.success is True
    assert "7" in result.output
    assert "x" in result.locals_snapshot


def test_code_exec_error():
    env = CodeExecutionEnv()
    result = env.execute("x = 1 / 0")
    assert result.success is False
    assert "ZeroDivisionError" in result.error


def test_code_exec_blocks_import():
    env = CodeExecutionEnv()
    result = env.execute("import os\nos.listdir('/')")
    assert result.success is False
    assert "Import" in result.error or "import" in result.error.lower()


def test_code_exec_blocks_open():
    env = CodeExecutionEnv()
    result = env.execute("f = open('/etc/passwd')")
    assert result.success is False
    assert "Blocked" in result.error or "open" in result.error.lower()


def test_code_exec_blocks_exec():
    env = CodeExecutionEnv()
    result = env.execute("exec('print(1)')")
    assert result.success is False


def test_code_exec_locals_bounded():
    """Local variable snapshot should be bounded (Axiom 1)."""
    env = CodeExecutionEnv(max_locals=5)
    result = env.execute("a=1; b=2; c=3; d=4; e=5; f=6; g=7")
    assert result.success is True
    assert len(result.locals_snapshot) <= 5


def test_code_exec_output_truncated():
    """Large output should be truncated (Axiom 1)."""
    env = CodeExecutionEnv(max_output_bytes=100)
    result = env.execute("print('x' * 1000)")
    assert result.success is True
    assert len(result.output) <= 200  # truncated + message


def test_code_exec_history_bounded():
    """Execution history should be bounded (Axiom 1)."""
    env = CodeExecutionEnv()
    for i in range(200):
        env.execute(f"print({i})")
    assert len(env) <= env.capacity


def test_code_exec_summary():
    env = CodeExecutionEnv()
    env.execute("print(1)")
    env.execute("print(2)")
    env.execute("x = 1/0")
    s = env.summary()
    assert s["total_executions"] == 3
    assert s["successful"] == 2
    assert 0 < s["success_rate"] < 1


def test_code_exec_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    env = CodeExecutionEnv()
    assert isinstance(env, BoundedComponent)


def test_code_exec_loop():
    """Agent can write a loop to compute something."""
    env = CodeExecutionEnv()
    result = env.execute("total = 0\nfor i in range(10):\n    total += i\nprint(total)")
    assert result.success is True
    assert "45" in result.output


def test_code_exec_list_comprehension():
    env = CodeExecutionEnv()
    result = env.execute("squares = [x*x for x in range(5)]\nprint(squares)")
    assert result.success is True
    assert "[0, 1, 4, 9, 16]" in result.output


# =====================================================================
# SocialLearningBuffer
# =====================================================================


def test_social_buffer_add_and_sample():
    buf = SocialLearningBuffer(max_observations=16)
    for i in range(10):
        buf.add(AgentObservation(
            agent_id=0, action=i, reward=float(i),
        ))
    assert len(buf) == 10
    sample = buf.sample(5)
    assert len(sample) == 5


def test_social_buffer_capacity_bounded():
    buf = SocialLearningBuffer(max_observations=8)
    for i in range(20):
        buf.add(AgentObservation(agent_id=0, action=0, reward=0.0))
    assert len(buf) <= 8  # Axiom 1


def test_social_buffer_best_demonstrations():
    buf = SocialLearningBuffer(max_observations=16)
    for i in range(10):
        buf.add(AgentObservation(agent_id=0, action=i, reward=float(i)))
    best = buf.best_demonstrations(3)
    assert len(best) == 3
    assert best[0].reward >= best[1].reward >= best[2].reward
    assert best[0].reward == 9.0


def test_social_buffer_empty_sample():
    buf = SocialLearningBuffer(max_observations=16)
    assert buf.sample(5) == []


def test_social_buffer_summary():
    buf = SocialLearningBuffer(max_observations=16)
    for i in range(5):
        buf.add(AgentObservation(agent_id=0, action=0, reward=0.5))
    s = buf.summary()
    assert s["n"] == 5
    assert s["mean_reward"] == 0.5
    assert s["max_reward"] == 0.5


def test_social_buffer_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    buf = SocialLearningBuffer(max_observations=8)
    assert isinstance(buf, BoundedComponent)


# =====================================================================
# MultiAgentEnv
# =====================================================================


def test_multi_agent_env_init():
    env = MultiAgentEnv(num_agents=3, d_model=64)
    assert env.num_agents == 3


def test_multi_agent_env_step():
    env = MultiAgentEnv(num_agents=2, d_model=64)
    env.step(agent_id=0, action=2, reward=0.5)
    env.step(agent_id=1, action=3, reward=0.8)
    s = env.agent_summary()
    assert s["step_count"] == 2
    assert s["agent_rewards"] == [0.5, 0.8]


def test_multi_agent_env_demonstrations():
    """Agents can learn from each other's demonstrations."""
    env = MultiAgentEnv(num_agents=3, d_model=64)
    # Agent 0 does well
    for _ in range(10):
        env.step(0, action=2, reward=0.9)
    # Agent 1 does poorly
    for _ in range(10):
        env.step(1, action=0, reward=0.1)
    # Agent 2 can learn from agent 0's demonstrations
    demos = env.get_demonstrations(5, best=True)
    assert len(demos) == 5
    assert all(d.reward == 0.9 for d in demos)


def test_multi_agent_env_bounded():
    """Social buffer should be bounded (Axiom 1)."""
    env = MultiAgentEnv(num_agents=2, max_observations=16)
    for _ in range(100):
        env.step(0, action=0, reward=0.1)
    assert len(env.social_buffer) <= 16


def test_multi_agent_env_state_dict():
    env = MultiAgentEnv(num_agents=3, d_model=64)
    env.step(0, action=1, reward=0.5)
    state = env.state_dict()
    env2 = MultiAgentEnv(num_agents=3, d_model=64)
    env2.load_state_dict(state)
    assert env2.num_agents == 3
    assert env2._step_count == 1
    assert env2._agent_rewards[0] == 0.5


def test_multi_agent_env_with_embeddings():
    """Agents can share observation embeddings."""
    env = MultiAgentEnv(num_agents=2, d_model=64)
    emb = torch.randn(64)
    env.step(0, action=2, reward=0.5, obs_embedding=emb)
    demos = env.get_demonstrations(1, best=True)
    assert demos[0].obs_embedding is not None
    assert demos[0].obs_embedding.shape == (64,)
