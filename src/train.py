"""devagi trainer.

Stage 0 baseline: PPO on MiniGrid to validate the skeleton.
Stage 1 (this file also): PPO + RND intrinsic reward + Bounded 3-tier Replay.

Backward-compat: Stage 0 configs skip Stage 1 blocks; behaviour is identical
to the earlier Stage-0-only trainer.

Bounded design axioms enforced throughout — see DESIGN_PRINCIPLES.md.

Usage:
    # Stage 0
    python -m src.train --stage 0 --preset local_smoke --smoke-only
    python -m src.train --stage 0 --preset cloud_5090

    # Stage 1 (adds RND + Bounded Replay)
    python -m src.train --stage 1 --preset cloud_5090

    # Resume from any ckpt
    python -m src.train --stage 1 --preset cloud_5090 \
        --resume checkpoints/ckpt_stage0_003000000.pt
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
from src.intrinsic import RND, RNDConfig
from src.memory import BoundedReplayBuffer, ReplayBudget, Transition
from src.monitoring import HealthChecker, MemoryWatcher, WatcherConfig
from src.platform import data_dir, get_device, get_device_info, stage_ckpt_path
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

    Same architecture used from Stage 0 onward. Stage 2 will swap in the
    Hybrid backbone (TTT-Linear + SWA + FFN) via a config flag.
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
# Rollout buffer (on-policy, Stage 0+)
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
# Coverage tracker (Stage 1) — bounded FP-hash bucket count
# =====================================================================


class BoundedCoverage:
    """Track state-visitation entropy using a fixed number of hash buckets.

    - `capacity` = num_buckets (fixed at construction; Axiom 1).
    - `__len__` = number of buckets *visited so far* (grows toward capacity).
    - `coverage_ratio` = |visited| / capacity.

    This is intentionally an approximation — a bucket collision inflates the
    apparent overlap between distinct states. For a 5×5 MiniGrid with ~50
    unique obs, 4096 buckets makes collisions negligible.

    有界状态覆盖率跟踪：固定 hash buckets 数（Axiom 1），
    只记 bucket 是否被访问过。approx entropy = visited / total_buckets。
    """

    def __init__(self, num_buckets: int = 4096) -> None:
        self._capacity = int(num_buckets)
        if self._capacity <= 0:
            raise ValueError("num_buckets must be positive")
        # Bit-vector: 1 bit per bucket → fixed memory (Axiom 1).
        self._seen = np.zeros(self._capacity, dtype=bool)
        self._n_visits = 0
        self._n_unique = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._n_unique

    def touch(self, obs: np.ndarray) -> None:
        """Register one visit. Constant-time; no growth."""
        # Fast hash: xor-bytes reduction. Numpy view + rolling xor is cheap.
        h = int(hash(obs.tobytes())) & (self._capacity - 1)
        if not self._seen[h]:
            self._seen[h] = True
            self._n_unique += 1
        self._n_visits += 1

    def coverage_ratio(self) -> float:
        return self._n_unique / self._capacity

    def summary(self) -> dict:
        return {
            "visits": self._n_visits,
            "unique_buckets": self._n_unique,
            "capacity": self._capacity,
            "coverage_ratio": self.coverage_ratio(),
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self._capacity,
            "seen": self._seen.copy(),
            "n_visits": self._n_visits,
            "n_unique": self._n_unique,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self._capacity:
            raise ValueError("coverage capacity mismatch")
        self._seen = np.asarray(state["seen"], dtype=bool).copy()
        self._n_visits = int(state["n_visits"])
        self._n_unique = int(state["n_unique"])


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
    stage = int(config.get("stage", 0))
    logger.info("Preset: %s  device: %s (%s)  stage: %d",
                config.get("preset"), device_info.kind, device_info.name, stage)

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
    log_dir = open_stage_log_dir(stage, run_id)
    watcher = MemoryWatcher(
        WatcherConfig(
            sample_interval_s=float(monitor_cfg.get("sample_interval_s", 5.0)),
            slope_alarm_gb_per_hour=float(monitor_cfg.get("slope_alarm_gb_per_hour", 0.2)),
            empty_cache_every_steps=int(monitor_cfg.get("empty_cache_every_steps", 10_000)),
            csv_path=log_dir / "memory.csv",
            warmup_seconds=float(monitor_cfg.get("warmup_seconds", 300.0)),
        )
    )
    logger.info("Logs → %s", log_dir)

    # --- Stage 1: intrinsic motivation + coverage + replay ---
    intrinsic_cfg = config.get("intrinsic")
    replay_cfg = config.get("replay")
    coverage_cfg = config.get("coverage")

    rnd: RND | None = None
    replay: BoundedReplayBuffer | None = None
    coverage: BoundedCoverage | None = None

    if stage >= 1 and intrinsic_cfg:
        rnd = RND(
            obs_shape,
            RNDConfig(
                embed_dim=int(intrinsic_cfg.get("embed_dim", 128)),
                lr=float(intrinsic_cfg.get("lr", 1e-4)),
                reward_clip=float(intrinsic_cfg.get("reward_clip", 5.0)),
            ),
        ).to(device)
        logger.info("RND enabled (embed_dim=%d, reward_coef=%s)",
                    rnd.config.embed_dim, intrinsic_cfg.get("reward_coef", 0.1))

    if stage >= 1 and replay_cfg:
        replay = BoundedReplayBuffer(
            budget=ReplayBudget(
                hot_capacity=int(replay_cfg.get("hot_capacity", 4096)),
                warm_capacity=int(replay_cfg.get("warm_capacity", 32768)),
                cold_max_shards=int(replay_cfg.get("cold_max_shards", 8)),
                cold_shard_size=int(replay_cfg.get("cold_shard_size", 4096)),
            ),
            obs_shape=obs_shape,
            device=device,
            archive_dir=data_dir() / str(replay_cfg.get("archive_subdir", "replay")),
        )
        health.register("bounded_replay", replay)
        logger.info("BoundedReplayBuffer enabled (capacity=%d)", replay.capacity)

    if stage >= 1 and coverage_cfg:
        coverage = BoundedCoverage(num_buckets=int(coverage_cfg.get("num_buckets", 4096)))
        health.register("coverage", coverage)
        logger.info("BoundedCoverage enabled (buckets=%d)", coverage.capacity)

    state = TrainState(step=0, episode=0)

    if resume is not None:
        payload = load_ckpt(resume)
        try:
            model.load_state_dict(payload["model_state"])
        except RuntimeError as exc:
            # Cross-stage resume with different model shape: warn + skip.
            logger.warning("Model state mismatch on resume (%s); starting model fresh.", exc)
        if payload.get("optim_state"):
            try:
                optimizer.load_state_dict(payload["optim_state"])
            except (ValueError, RuntimeError) as exc:
                logger.warning("Optimizer state mismatch on resume (%s); starting fresh.", exc)
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

    # Stage 1 knobs
    intrinsic_coef = float(intrinsic_cfg.get("reward_coef", 0.1)) if intrinsic_cfg else 0.0
    rnd_update_every = int(intrinsic_cfg.get("update_every_steps", 1)) if intrinsic_cfg else 0
    coverage_log_every = int(coverage_cfg.get("log_every_steps", 5000)) if coverage_cfg else 0
    replay_sample_every = int(replay_cfg.get("sample_every_steps", 4)) if replay_cfg else 0
    replay_min_size = int(replay_cfg.get("min_size_to_sample", 1024)) if replay_cfg else 0
    replay_batch_size = int(replay_cfg.get("batch_size_offpolicy", 128)) if replay_cfg else 0
    per_alpha = float(replay_cfg.get("per_alpha", 0.6)) if replay_cfg else 0.6

    t0 = time.time()
    logger.info(
        "Starting Stage %s: total_steps=%d smoke=%s intrinsic=%s replay=%s coverage=%s",
        stage,
        total_steps,
        smoke_only,
        rnd is not None,
        replay is not None,
        coverage is not None,
    )

    last_coverage_log_step = 0

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
            extrinsic_r = step_out.reward
            total_r = extrinsic_r

            # --- Stage 1: intrinsic reward from RND ---
            if rnd is not None:
                with torch.no_grad():
                    int_r = float(rnd.intrinsic_reward(obs_t).item())
                total_r = extrinsic_r + intrinsic_coef * int_r

            buffer.add(
                obs=obs,
                action=int(action.item()),
                logprob=float(logprob.item()),
                value=float(value.item()),
                reward=total_r,
                done=step_out.terminated or step_out.truncated,
            )

            # --- Stage 1: coverage tracking ---
            if coverage is not None:
                coverage.touch(obs)

            # --- Stage 1: push transition to bounded replay ---
            if replay is not None:
                replay.add(Transition(
                    obs=obs,
                    action=int(action.item()),
                    reward=total_r,
                    next_obs=step_out.obs,
                    done=step_out.terminated or step_out.truncated,
                    priority=1.0 + abs(int_r) if rnd is not None else 1.0,
                ))

            # --- Stage 1: RND predictor SGD ---
            if rnd is not None and rnd_update_every > 0 and state.step % rnd_update_every == 0:
                rnd.update(obs_t)

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

        # --- Stage 1: off-policy value refresh from replay (small, extra grad) ---
        if (
            replay is not None
            and len(replay) >= replay_min_size
            and state.step % replay_sample_every < rollout_capacity
        ):
            try:
                sample, indices, weights = replay.sample_prioritized(
                    replay_batch_size, alpha=per_alpha
                )
                _, offp_values = model(sample["obs"])
                # TD target with intrinsic-augmented reward + bootstrapped next value
                with torch.no_grad():
                    _, next_v = model(sample["next_obs"])
                    td_target = sample["reward"] + gamma * next_v * (1.0 - sample["done"])
                w = torch.as_tensor(weights, dtype=torch.float32, device=device)
                td_loss = (w * (offp_values - td_target).pow(2)).mean()
                optimizer.zero_grad(set_to_none=True)
                td_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                # Update priorities to |TD error|
                new_prios = (offp_values - td_target).detach().abs().cpu().numpy() + 1e-6
                replay.update_hot_priorities(indices, new_prios)
            except (ValueError, IndexError) as exc:
                # e.g., hot tier still tiny; just skip this cycle
                logger.debug("replay sample skipped: %s", exc)

        # Health sweep after each rollout+update cycle
        health.sweep()

        if state.step % log_every < rollout_capacity:
            summary = env.summary()
            mem = watcher.snapshot_summary()
            extras: list[str] = []
            if coverage is not None:
                extras.append(f"cov={coverage.coverage_ratio() * 100:.1f}%")
            if replay is not None:
                extras.append(f"replay={len(replay)}/{replay.capacity}")
            logger.info(
                "step=%d ep=%d mean_ret=%.3f loss=%.4f mem_used=%.2fGB slope=%s %s",
                state.step,
                summary["episodes"],
                summary["mean_return"],
                float(loss.item()),
                (mem.get("used_bytes", 0) or 0) / 1024**3,
                mem.get("slope_gb_per_hour"),
                " ".join(extras),
            )

        # Periodic coverage snapshot (Stage 1)
        if (
            coverage is not None
            and coverage_log_every > 0
            and state.step - last_coverage_log_step >= coverage_log_every
        ):
            logger.info("coverage summary @ step=%d: %s", state.step, coverage.summary())
            last_coverage_log_step = state.step

        if state.step % ckpt_every < rollout_capacity:
            extra: dict[str, Any] = {"preset": config.get("preset"), "run_id": run_id}
            if rnd is not None:
                extra["rnd_state"] = rnd.rnd_state_dict()
            if coverage is not None:
                extra["coverage_state"] = coverage.state_dict()
            # NB: replay state not serialized here (too big for on-policy ckpts;
            # rely on data disk to persist replay across restarts).
            save_ckpt(
                stage_ckpt_path(stage, state.step),
                stage=stage,
                step=state.step,
                model_state=model.state_dict(),
                optim_state=optimizer.state_dict(),
                extra=extra,
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
    if coverage is not None:
        logger.info("Coverage final: %s", coverage.summary())
    if replay is not None:
        logger.info("Replay final: %s", replay.stats())

    env.close()
    return 0


# =====================================================================
# CLI
# =====================================================================


_DEFAULT_STAGE_CONFIGS = {
    0: "stage0_baseline.yaml",
    1: "stage1_curiosity.yaml",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="devagi trainer")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--preset", type=str, default=None,
                    help="Override preset. Env DEVAGI_PRESET used if unset.")
    ap.add_argument("--config", type=str, default=None,
                    help="Stage config filename under configs/ (default: pick by stage)")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Cap training to a tiny smoke workload (<5 min).")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return ap.parse_args()


def main() -> int:
    import os

    args = _parse_args()
    preset = args.preset or os.environ.get("DEVAGI_PRESET") or "local_smoke"
    # Pick default config filename by stage
    if args.config:
        stage_cfg = args.config
    else:
        stage_cfg = _DEFAULT_STAGE_CONFIGS.get(args.stage, f"stage{args.stage}_baseline.yaml")
    cfg = load_config(stage_cfg, preset)
    cfg.setdefault("stage", args.stage)
    resume = Path(args.resume) if args.resume else None
    return train(cfg, smoke_only=args.smoke_only, resume=resume)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
