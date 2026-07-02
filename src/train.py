"""Stage 0 baseline trainer.

Minimal PPO on MiniGrid. Not tuned for performance — this exists to validate:

- Preset system (local_smoke / cloud_24g / home_64g)
- Platform abstraction (device/paths/memory_probe)
- Monitoring stack (memory_watcher, longevity_test, health_check)
- Checkpoint schema
- End-to-end wiring so subsequent stages can plug in

用于验证 preset / 平台层 / 监控栈 / checkpoint schema 的最简 PPO baseline。

Usage:
    python -m src.train --stage 0 --preset local_smoke --smoke-only
    python -m src.train --stage 0 --preset cloud_24g
    python -m src.train --stage 0 --preset cloud_24g --resume checkpoints/ckpt_stage0_000010000.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs import MiniGridWrapper
from src.monitoring import HealthChecker, MemoryWatcher, WatcherConfig
from src.platform import get_device, get_device_info, stage_ckpt_path
from src.utils import (
    load_ckpt,
    load_config,
    make_run_id,
    open_stage_log_dir,
    save_ckpt,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Model
# =====================================================================


class ActorCritic(nn.Module):
    """Tiny CNN actor-critic for MiniGrid image obs.

    Not the point of Stage 0 — just enough to close the training loop.
    """

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, hidden: int = 64) -> None:
        super().__init__()
        h, w, c = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        flat_dim = 32 * h * w
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs_u8: (B, H, W, C) uint8 → (B, C, H, W) float
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        z = self.trunk(self.encoder(x))
        return self.policy_head(z), self.value_head(z).squeeze(-1)


# =====================================================================
# Rollout buffer (bounded, capacity-declared) — implements BoundedComponent
# =====================================================================


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class RolloutBuffer:
    """Fixed-capacity on-policy rollout buffer.

    有界容量（Axiom 1）：构造时声明 capacity，任何时候 ``len(self) <= capacity``。
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        self._capacity = int(capacity)
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._ptr = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._ptr

    def clear(self) -> None:
        self._ptr = 0

    def full(self) -> bool:
        return self._ptr >= self._capacity

    def add(
        self,
        obs: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        if self._ptr >= self._capacity:
            raise IndexError("RolloutBuffer full (Axiom 1: no unbounded growth)")
        i = self._ptr
        self.obs[i] = torch.from_numpy(obs)
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self._ptr += 1

    def as_batch(self) -> TransitionBatch:
        return TransitionBatch(
            obs=self.obs[: self._ptr],
            actions=self.actions[: self._ptr],
            logprobs=self.logprobs[: self._ptr],
            values=self.values[: self._ptr],
            rewards=self.rewards[: self._ptr],
            dones=self.dones[: self._ptr],
        )


# =====================================================================
# GAE
# =====================================================================


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns)."""
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1].item()
        next_non_terminal = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_value * next_non_terminal - values[t].item()
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# =====================================================================
# Trainer
# =====================================================================


@dataclass
class TrainState:
    step: int
    episode: int


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(obs).unsqueeze(0).to(device)


def train(config: dict[str, Any], smoke_only: bool, resume: Path | None) -> int:
    setup_logging("INFO")
    device_info = get_device_info(config.get("device_preferred"))
    device = get_device(config.get("device_preferred"))
    logger.info("Preset: %s  device: %s (%s)", config.get("preset"), device_info.kind, device_info.name)

    set_seed(42)

    # --- Env
    env_cfg = config["env"]
    env = MiniGridWrapper(
        env_id=env_cfg["id"],
        seed=42,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        auto_reset=True,
    )
    obs = env.reset()
    obs_shape = env.observation_shape
    num_actions = env.action_space_n
    logger.info("Env: %s  obs_shape=%s  actions=%d", env_cfg["id"], obs_shape, num_actions)

    # --- Model
    hidden = int(config["model"]["hidden_size"])
    model = ActorCritic(obs_shape, num_actions, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # --- Bounded rollout buffer (declared capacity)
    rollout_capacity = 128 if smoke_only else 512
    buffer = RolloutBuffer(rollout_capacity, obs_shape, device=device)

    # --- Health check
    health = HealthChecker(strict=True)
    health.register("rollout_buffer", buffer)

    # --- Monitoring
    train_cfg = config["train"]
    monitor_cfg = config.get("monitor", {})
    total_steps = int(train_cfg.get("total_steps", 200))
    if smoke_only:
        total_steps = min(total_steps, 200)
    run_id = make_run_id()
    log_dir = open_stage_log_dir(config.get("stage", 0), run_id)
    watcher = MemoryWatcher(
        WatcherConfig(
            sample_interval_s=float(monitor_cfg.get("sample_interval_s", 5.0)),
            slope_alarm_gb_per_hour=float(monitor_cfg.get("slope_alarm_gb_per_hour", 0.2)),
            empty_cache_every_steps=int(monitor_cfg.get("empty_cache_every_steps", 10_000)),
            csv_path=log_dir / "memory.csv",
        )
    )
    logger.info("Logs → %s", log_dir)

    state = TrainState(step=0, episode=0)

    if resume is not None:
        payload = load_ckpt(resume)
        model.load_state_dict(payload["model_state"])
        if payload.get("optim_state"):
            optimizer.load_state_dict(payload["optim_state"])
        state.step = int(payload.get("step", 0))
        logger.info("Resumed from %s at step %d", resume, state.step)

    # --- PPO hyperparams
    ppo_clip = float(train_cfg.get("ppo_clip", 0.2))
    ppo_epochs = int(train_cfg.get("ppo_epochs", 4))
    entropy_coef = float(train_cfg.get("entropy_coef", 0.01))
    value_coef = float(train_cfg.get("value_coef", 0.5))
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    log_every = int(train_cfg.get("log_every_steps", 50))
    ckpt_every = int(train_cfg.get("ckpt_every_steps", 200))

    t0 = time.time()
    logger.info(
        "Starting Stage %s baseline: total_steps=%d smoke=%s",
        config.get("stage", 0),
        total_steps,
        smoke_only,
    )

    # ---- Main loop
    while state.step < total_steps:
        buffer.clear()

        # Collect a rollout of exactly `rollout_capacity` steps
        while not buffer.full():
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            step_out = env.step(int(action.item()))
            buffer.add(
                obs=obs,
                action=int(action.item()),
                logprob=float(logprob.item()),
                value=float(value.item()),
                reward=step_out.reward,
                done=step_out.terminated or step_out.truncated,
            )
            obs = step_out.obs
            state.step += 1
            watcher.tick(step=state.step)

            if state.step >= total_steps:
                break

        batch = buffer.as_batch()
        # Estimate last value for GAE
        with torch.no_grad():
            _, last_value_t = model(_obs_to_tensor(obs, device))
        advantages, returns = compute_gae(
            batch.rewards.cpu(),
            batch.values.cpu(),
            batch.dones.cpu(),
            float(last_value_t.item()),
            gamma,
            gae_lambda,
        )
        advantages = advantages.to(device)
        returns = returns.to(device)
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(ppo_epochs):
            logits, values = model(batch.obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(batch.actions)
            ratio = (new_logprobs - batch.logprobs).exp()
            unclipped = ratio * adv_norm
            clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv_norm
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # Health sweep after each rollout+update cycle
        health.sweep()

        if state.step % log_every < rollout_capacity:
            summary = env.summary()
            mem = watcher.snapshot_summary()
            logger.info(
                "step=%d ep=%d mean_ret=%.3f loss=%.4f mem_used=%.2fGB slope=%s",
                state.step,
                summary["episodes"],
                summary["mean_return"],
                float(loss.item()),
                (mem.get("used_bytes", 0) or 0) / 1024**3,
                mem.get("slope_gb_per_hour"),
            )

        if state.step % ckpt_every < rollout_capacity:
            save_ckpt(
                stage_ckpt_path(config.get("stage", 0), state.step),
                stage=config.get("stage", 0),
                step=state.step,
                model_state=model.state_dict(),
                optim_state=optimizer.state_dict(),
                extra={"preset": config.get("preset"), "run_id": run_id},
            )

    elapsed = time.time() - t0
    final_summary = env.summary()
    mem_summary = watcher.snapshot_summary()
    logger.info(
        "Training finished. steps=%d episodes=%d mean_ret=%.3f elapsed=%.1fs",
        state.step,
        final_summary["episodes"],
        final_summary["mean_return"],
        elapsed,
    )
    logger.info("Memory summary: %s", mem_summary)

    env.close()
    return 0


# =====================================================================
# CLI
# =====================================================================


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="devagi trainer (Stage 0 baseline)")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--preset", type=str, default=None,
                    help="Override preset. Env DEVAGI_PRESET used if unset.")
    ap.add_argument("--config", type=str, default=None,
                    help="Stage config filename under configs/ (default: stage{N}_baseline.yaml)")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Cap training to a tiny smoke workload (<5 min).")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return ap.parse_args()


def main() -> int:
    import os

    args = _parse_args()
    preset = args.preset or os.environ.get("DEVAGI_PRESET") or "local_smoke"
    stage_cfg = args.config or f"stage{args.stage}_baseline.yaml"
    cfg = load_config(stage_cfg, preset)
    cfg.setdefault("stage", args.stage)
    resume = Path(args.resume) if args.resume else None
    return train(cfg, smoke_only=args.smoke_only, resume=resume)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
