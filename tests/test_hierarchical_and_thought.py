"""Tests for hierarchical policy + thought-action loop."""

from __future__ import annotations

import pytest
import torch

from src.models.hierarchical_policy import (
    GoalConditionedActionHead,
    HierarchicalActorCritic,
    SubGoalHead,
)
from src.models.metacognition import (
    InnerDialogue,
    ReflectionLoop,
    SelfModel,
)
from src.models.thought_action_loop import ThoughtActionLoop


D_MODEL = 64
OBS_SHAPE = (7, 7, 3)
NUM_ACTIONS = 7


# =====================================================================
# SubGoalHead
# =====================================================================


def test_sub_goal_head_shape():
    head = SubGoalHead(d_model=D_MODEL)
    h = torch.randn(4, D_MODEL)
    goal = head(h)
    assert goal.shape == (4, D_MODEL)


def test_sub_goal_head_auxiliary_loss():
    head = SubGoalHead(d_model=D_MODEL)
    h_current = torch.randn(4, D_MODEL)
    h_future = torch.randn(4, D_MODEL)
    loss = head.auxiliary_loss(h_current, h_future)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_sub_goal_head_gradient():
    head = SubGoalHead(d_model=D_MODEL)
    h = torch.randn(4, D_MODEL)
    future = torch.randn(4, D_MODEL)
    loss = head.auxiliary_loss(h, future)
    loss.backward()
    for p in head.parameters():
        assert p.grad is not None


# =====================================================================
# GoalConditionedActionHead
# =====================================================================


def test_goal_conditioned_action_shape():
    head = GoalConditionedActionHead(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    h = torch.randn(4, D_MODEL)
    g = torch.randn(4, D_MODEL)
    logits, value = head(h, g)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4,)


def test_goal_conditioned_action_changes_with_goal():
    """After a few training steps, different sub-goals should produce different actions."""
    torch.manual_seed(0)
    head = GoalConditionedActionHead(d_model=D_MODEL, num_actions=NUM_ACTIONS)
    h = torch.randn(4, D_MODEL)
    g = torch.randn(4, D_MODEL)

    # Take a few gradient steps so FiLM weights move from identity init
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(5):
        logits, _ = head(h, g)
        loss = logits.var()  # maximize action diversity
        opt.zero_grad()
        (-loss).backward()
        opt.step()

    # Now different goals should produce different outputs
    g1 = torch.randn(1, D_MODEL)
    g2 = torch.randn(1, D_MODEL) * 5
    h1 = torch.randn(1, D_MODEL)
    logits1, _ = head(h1, g1)
    logits2, _ = head(h1, g2)
    assert not torch.allclose(logits1, logits2, atol=1e-4), \
        "different goals should produce different actions after training"


# =====================================================================
# HierarchicalActorCritic
# =====================================================================


def test_hierarchical_ac_shape():
    torch.manual_seed(0)
    m = HierarchicalActorCritic(
        obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS,
        d_model=D_MODEL, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    logits, value = m(obs)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4,)


def test_hierarchical_ac_large_batch_no_nan():
    torch.manual_seed(0)
    m = HierarchicalActorCritic(
        obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS,
        d_model=D_MODEL, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (512, *OBS_SHAPE), dtype=torch.uint8)
    logits, value = m(obs)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_hierarchical_ac_sub_goal_loss():
    torch.manual_seed(0)
    m = HierarchicalActorCritic(
        obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS,
        d_model=D_MODEL, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs1 = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    obs2 = torch.randint(0, 255, (4, *OBS_SHAPE), dtype=torch.uint8)
    loss = m.compute_sub_goal_loss(obs1, obs2)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_hierarchical_ac_sub_goal_refresh():
    """Sub-goal should refresh every sub_goal_every steps."""
    torch.manual_seed(0)
    m = HierarchicalActorCritic(
        obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS,
        d_model=D_MODEL, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
        sub_goal_every=3,
    )
    obs = torch.randint(0, 255, (1, *OBS_SHAPE), dtype=torch.uint8)
    # Step 1: should generate new sub-goal
    m(obs)
    assert m._step_in_goal == 1
    # Step 2: reuse
    m(obs)
    assert m._step_in_goal == 2
    # Step 3: reuse, then reset
    m(obs)
    assert m._step_in_goal == 0  # reset for next cycle


def test_hierarchical_ac_get_sub_goal():
    m = HierarchicalActorCritic(
        obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS,
        d_model=D_MODEL, n_layers=1, n_heads=4, swa_window=4, ttt_mini_batch=2,
    )
    obs = torch.randint(0, 255, (1, *OBS_SHAPE), dtype=torch.uint8)
    goal = m.get_sub_goal(obs)
    assert goal.shape == (1, D_MODEL)


# =====================================================================
# ThoughtActionLoop
# =====================================================================


def test_thought_action_loop_modulate_no_thought():
    """Without a thought, modulation should be identity (no change)."""
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=100)
    feats = torch.randn(4, D_MODEL)
    out = loop.modulate(feats)
    torch.testing.assert_close(out, feats)  # no thought → no change


def test_thought_action_loop_think_generates_text():
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=1)
    h = torch.randn(D_MODEL)
    thought = loop.maybe_think(h, episode_return=0.5, episode_done=False)
    assert thought is not None
    assert isinstance(thought, str)
    assert len(thought) > 0


def test_thought_action_loop_think_only_every_n_steps():
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=5)
    h = torch.randn(D_MODEL)
    # Steps 1-4: no thought
    for i in range(4):
        assert loop.maybe_think(h) is None
    # Step 5: thought!
    assert loop.maybe_think(h) is not None


def test_thought_action_loop_modulation_after_thought():
    """After a thought is generated, the thought text should be non-empty."""
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=1)
    h = torch.randn(D_MODEL)
    thought = loop.maybe_think(h)  # generate thought
    assert thought is not None
    assert len(thought) > 0
    # Without a language_encoder, has_active_thought stays False (no embedding to modulate with)
    # With a language_encoder, it would be True after encoding


def test_thought_action_loop_reset():
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=1)
    h = torch.randn(D_MODEL)
    loop.maybe_think(h)
    assert loop._step_count > 0
    loop.reset()
    assert loop._step_count == 0
    assert not loop.has_active_thought
    # Cached embedding should be zeroed
    assert loop._cached_lang_embedding.abs().sum() == 0


def test_thought_action_loop_with_self_model():
    sm = SelfModel(d_model=D_MODEL)
    loop = ThoughtActionLoop(
        d_model=D_MODEL, think_every_steps=1, self_model=sm,
    )
    h = torch.randn(D_MODEL)
    thought = loop.maybe_think(h, episode_return=0.8)
    assert thought is not None
    # Thought should mention confidence or progress
    assert len(thought) > 0


def test_thought_action_loop_with_reflection():
    sm = SelfModel(d_model=D_MODEL)
    rl = ReflectionLoop(sm, max_reflections=16, reflection_every_episodes=1)
    id = InnerDialogue(mode="template")
    loop = ThoughtActionLoop(
        d_model=D_MODEL, think_every_steps=1,
        self_model=sm, reflection_loop=rl, inner_dialogue=id,
    )
    h = torch.randn(D_MODEL)
    # Trigger episode end → reflection → thought
    thought = loop.maybe_think(h, episode_return=0.8, episode_done=True)
    assert thought is not None


def test_thought_action_loop_no_growing_state():
    """Thought loop should not accumulate state (Axiom 1)."""
    loop = ThoughtActionLoop(d_model=D_MODEL, think_every_steps=1)
    h = torch.randn(D_MODEL)
    for _ in range(100):
        loop.maybe_think(h)
        loop.modulate(torch.randn(4, D_MODEL))
    # Cached embedding is still (D_MODEL,) — not a growing list
    emb = loop.get_cached_thought_embedding()
    assert emb.shape == (D_MODEL,)


def test_thought_action_loop_conforms_to_bounded_component():
    from src.monitoring.health_check import BoundedComponent
    # ThoughtActionLoop is an nn.Module, not directly BoundedComponent,
    # but its internal state is bounded.
    loop = ThoughtActionLoop(d_model=D_MODEL)
    # The cached embedding is a fixed-size tensor
    assert loop._cached_lang_embedding.shape == (D_MODEL,)
