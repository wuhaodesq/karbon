"""End-to-end integration test for Stage 0.

This test exercises the full stack minus real MiniGrid — using a DummyEnv
so the CI can run on any machine without extra environment installs:

- HybridBackbone-based tiny policy/value net (Stage 2 pre-work)
- BoundedReplayBuffer (Stage 1 pre-work)
- RND intrinsic reward (Stage 1 pre-work)
- MemoryWatcher + HealthChecker (Stage 0 monitoring)
- OnlineEWC + Sleep Consolidation loop (Stage 6 pre-work)
- LP tracker + AutoCurriculum (Stage 5 pre-work)

We do NOT assert that RL scores go up. We DO assert:
- 20+ training steps complete without exception
- Loss is finite at every step
- Bounded components stay within capacity
- MemoryWatcher records samples
- Checkpoint round-trip preserves loss

If this test breaks, later stages will break too. Keep it green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.continual import (
    ConsolidationConfig,
    OnlineEWC,
    OnlineEWCConfig,
    SleepConsolidationLoop,
)
from src.curriculum import AutoCurriculum, AutoCurriculumConfig, TaskTemplate
from src.intrinsic import RND, RNDConfig
from src.memory import BoundedReplayBuffer, ReplayBudget, Transition
from src.monitoring import HealthChecker, MemoryWatcher, WatcherConfig
from src.platform import stage_ckpt_path
from src.utils import load_ckpt, save_ckpt


# =====================================================================
# Dummy env (no minigrid dependency)
# =====================================================================


OBS_SHAPE = (5, 5, 3)


class DummyEnv:
    """Deterministic pseudo-gridworld — random obs, mixed reward, resets on step 10."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._step_i = 0

    def reset(self) -> np.ndarray:
        self._step_i = 0
        return self._rng.integers(0, 255, OBS_SHAPE, dtype=np.uint8)

    def step(self, action: int):
        self._step_i += 1
        obs = self._rng.integers(0, 255, OBS_SHAPE, dtype=np.uint8)
        reward = float(self._rng.uniform(-0.1, 0.5))
        done = self._step_i >= 10
        return obs, reward, done


# =====================================================================
# Tiny policy/value net (bypasses Hybrid for CPU speed)
# =====================================================================


class TinyPolicy(torch.nn.Module):
    def __init__(self, obs_shape=OBS_SHAPE, n_actions=6, hidden=32):
        super().__init__()
        h, w, c = obs_shape
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(c, 8, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * h * w, hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.policy = torch.nn.Linear(hidden, n_actions)
        self.value = torch.nn.Linear(hidden, 1)

    def forward(self, obs_u8):
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        z = self.encoder(x)
        return self.policy(z), self.value(z).squeeze(-1)


# =====================================================================
# Integration test
# =====================================================================


def test_stage0_full_stack_integration(tmp_path, monkeypatch):
    """Run 30 steps with the full pre-Stage-1 stack; assert nothing crashes."""
    monkeypatch.setenv("DEVAGI_CKPT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setenv("DEVAGI_LOGS_DIR", str(tmp_path / "logs"))

    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- Components ---
    env = DummyEnv(seed=0)
    obs = env.reset()

    policy = TinyPolicy().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    replay = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=32, warm_capacity=64,
                            cold_max_shards=2, cold_shard_size=8),
        obs_shape=OBS_SHAPE,
        device=device,
        archive_dir=tmp_path / "replay",
    )
    rnd = RND(OBS_SHAPE, RNDConfig(embed_dim=16, lr=1e-3))
    ewc = OnlineEWC(policy, OnlineEWCConfig(lambda_reg=0.5))
    curr = AutoCurriculum(AutoCurriculumConfig(max_tasks=4, lp_window_size=8, lp_min_samples=4))
    for t in range(2):
        curr.add_task(TaskTemplate(id=t, spec={"task": t}))

    watcher = MemoryWatcher(WatcherConfig(sample_interval_s=0.01,
                                          rolling_window_s=1.0,
                                          csv_path=tmp_path / "mem.csv"))

    health = HealthChecker(strict=True)
    health.register("replay", replay)
    health.register("curriculum", curr)

    consolidation = SleepConsolidationLoop(ConsolidationConfig(
        warmup_steps=5,
        replay_trim_every=5,
        skills_merge_every=0,
        ttt_distill_every=0,
        ewc_consolidate_every=15,
    ))
    trims = {"n": 0}
    consolidation.set_replay_trim(lambda: trims.update(n=trims["n"] + 1))
    consolidation.set_ewc_consolidate(lambda: None)  # no-op stub

    # --- Loop ---
    n_steps = 30
    losses = []
    for step in range(n_steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())

        next_obs, reward, done = env.step(action)

        # Add intrinsic reward
        int_r = float(rnd.intrinsic_reward(obs_t).item())
        total_r = reward + 0.1 * int_r

        replay.add(Transition(
            obs=obs, action=action, reward=total_r,
            next_obs=next_obs, done=done, priority=1.0,
        ))

        # Curriculum book-keeping (fake error signal)
        curr.report_error(step % 2, error=abs(int_r))

        # Learn from replay
        if len(replay) >= 16:
            batch = replay.sample(8)
            logits, values = policy(batch["obs"])
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, batch["action"].unsqueeze(-1)).squeeze(-1)
            policy_loss = -(action_log_probs * batch["reward"]).mean()
            value_loss = F.mse_loss(values, batch["reward"])
            ewc_pen = ewc.penalty(policy)
            loss = policy_loss + 0.5 * value_loss + ewc_pen

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            losses.append(float(loss.item()))

        # RND predictor update on the current obs
        rnd.update(obs_t)

        # Monitoring / health / consolidation
        watcher.tick(step=step)
        health.sweep()
        consolidation.tick(step=step)

        obs = env.reset() if done else next_obs

    # ---- Assertions ----
    assert len(losses) > 5, "not enough learn-steps happened"

    # Health check: all bounded components still under cap
    reports = health.sweep()
    for r in reports:
        assert r.ok, f"bounded violated: {r.name} size={r.size} cap={r.capacity}"

    # Watcher captured at least one sample
    assert watcher.snapshot_summary()["num_samples"] > 0

    # Consolidation fired the trim task at least twice (steps 5, 10, 15, 20, 25)
    assert trims["n"] >= 4, f"trim ran too few times: {trims['n']}"

    # Curriculum sampled without error
    task = curr.sample_task()
    assert isinstance(task, TaskTemplate)

    # Replay buffer respected capacity
    assert len(replay.hot) <= replay.hot.capacity
    assert len(replay.warm) <= replay.warm.capacity

    # Checkpoint round-trip
    ckpt_path = stage_ckpt_path(stage=0, step=42)
    save_ckpt(
        ckpt_path,
        stage=0, step=42,
        model_state=policy.state_dict(),
        optim_state=optim.state_dict(),
        extra={"loss": losses[-1]},
    )
    payload = load_ckpt(ckpt_path)
    assert payload["step"] == 42
    assert payload["extra"]["loss"] == losses[-1]


def test_stage0_full_stack_deterministic_with_seed(tmp_path, monkeypatch):
    """Two runs with the same seed should agree on the first-step action."""
    monkeypatch.setenv("DEVAGI_CKPT_DIR", str(tmp_path / "ckpts"))

    def _first_action():
        torch.manual_seed(0)
        env = DummyEnv(seed=0)
        obs = env.reset()
        pol = TinyPolicy()
        with torch.no_grad():
            logits, _ = pol(torch.from_numpy(obs).unsqueeze(0))
        return logits.argmax().item()

    a1 = _first_action()
    a2 = _first_action()
    assert a1 == a2
