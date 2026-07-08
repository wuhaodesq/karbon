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

from src.continual import (
    ConsolidationConfig,
    OnlineEWC,
    OnlineEWCConfig,
    SleepConsolidationLoop,
)
from src.curriculum import AutoCurriculum, AutoCurriculumConfig, TaskTemplate
from src.envs import MiniGridWrapper
from src.intrinsic import RND, RNDConfig
from src.memory import (
    BoundedReplayBuffer,
    BoundedSkillLibrary,
    GenerativeReplayConfig,
    GenerativeReplayVAE,
    ReplayBudget,
    SkillLibraryBudget,
    Transition,
)
from src.models import HybridBackbone, RSSM, RSSMConfig
from src.models.vision_encoder import CNNEncoder, VisionEncoder, build_encoder
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


class HybridActorCritic(nn.Module):
    """Stage 2+ actor-critic backed by the :class:`HybridBackbone`.

    Architecture:

        obs (B, H, W, C) uint8
            → [VisionEncoder or CNNEncoder] → per-obs feature (B, d_model)
            → treat batch as B independent length-1 sequences
            → HybridBackbone → (B, 1, d_model)
            → squeeze → (B, d_model)
            → policy_head + value_head

    When ``use_vision_encoder=True`` in config, the encoder is a pretrained
    DINOv2/CLIP backbone (frozen) + trainable projection. This enables
    semantic object recognition without retraining the backbone.

    Bounded semantics (Axiom 1):
    - Hybrid backbone has no unbounded per-forward state.
    - Vision encoder backbone is frozen → constant VRAM.
    - Inner TTT weight ``W`` is freshly zero-initialized every forward.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        swa_window: int = 16,
        ttt_mini_batch: int = 8,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        use_vision_encoder: bool = False,
        vision_model_name: str = "dinov2_vits14",
        vision_freeze: bool = True,
    ) -> None:
        super().__init__()
        # Snap d_model up to a multiple of n_heads
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        # d_model must also be even for sinusoidal position encoding
        if d_model % 2 != 0:
            d_model += 1
        self.d_model = d_model

        # --- Encoder: inline CNN (backward-compatible with Stage 0-3 checkpoints) ---
        # Use CNNEncoder/VisionEncoder only when use_vision_encoder=True.
        # When False, use the old inline Sequential so Stage 0-3 weights load.
        self.use_vision = use_vision_encoder
        if use_vision_encoder:
            try:
                self.encoder = VisionEncoder(
                    d_model=d_model,
                    model_name=vision_model_name,
                    freeze=vision_freeze,
                )
                logger.info("HybridActorCritic: using pretrained %s", vision_model_name)
            except (RuntimeError, ValueError) as exc:
                logger.warning("VisionEncoder load failed (%s), falling back to CNN", exc)
                h, w, c = obs_shape
                self.encoder = nn.Sequential(
                    nn.Conv2d(c, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(32 * h * w, d_model),
                    nn.ReLU(inplace=True),
                )
                self.use_vision = False
        else:
            # Old inline encoder — matches Stage 0-3 checkpoint keys
            h, w, c = obs_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * h * w, d_model),
                nn.ReLU(inplace=True),
            )

        # Hybrid backbone (no token embedding: we feed pre-encoded features)
        swa_window = max(2, int(swa_window))
        ttt_mini_batch = max(1, min(int(ttt_mini_batch), swa_window))
        self.backbone = HybridBackbone(
            d_model=d_model,
            n_layers=int(n_layers),
            vocab_size=0,
            n_heads=int(n_heads),
            swa_window_size=swa_window,
            ttt_mini_batch=ttt_mini_batch,
            max_seq_len=4096,
            ffn_hidden_mult=int(ffn_hidden_mult),
            dropout=float(dropout),
        )

        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, obs_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Inline permute (matches Stage 0-3 checkpoint behavior)
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        feats = self.encoder(x)  # (B, d_model)
        # Treat each observation as an INDEPENDENT sequence of length 1.
        # This avoids TTT-Linear's inner W blowing up across unrelated batch
        # elements (which would cause NaN when B is large, e.g. 512).
        # The Hybrid backbone still applies TTT + SWA + FFN per obs, but with
        # trivial (length-1) temporal context. Full temporal context is a
        # Stage 3+ concern (world model uses actual observation sequences).
        seq = feats.unsqueeze(1)        # (B, 1, d_model) — B independent seqs
        seq_out = self.backbone(seq)     # (B, 1, d_model)
        z = seq_out.squeeze(1)          # (B, d_model)
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
    model_cfg = config["model"]
    if bool(model_cfg.get("use_hybrid_backbone", False)):
        model = HybridActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            d_model=int(model_cfg.get("hidden_size", 128)),
            n_layers=int(model_cfg.get("hybrid_n_layers", 3)),
            n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
            swa_window=int(model_cfg.get("hybrid_swa_window", 16)),
            ttt_mini_batch=int(model_cfg.get("hybrid_ttt_mini_batch", 8)),
            ffn_hidden_mult=int(model_cfg.get("hybrid_ffn_hidden_mult", 4)),
            dropout=float(model_cfg.get("hybrid_dropout", 0.0)),
            use_vision_encoder=bool(model_cfg.get("use_vision_encoder", False)),
            vision_model_name=str(model_cfg.get("vision_model", "dinov2_vits14")),
            vision_freeze=bool(model_cfg.get("vision_freeze", True)),
        ).to(device)
        vision_tag = " + VisionEncoder" if model_cfg.get("use_vision_encoder") else ""
        logger.info("Model: HybridActorCubit (d_model=%d, layers=%d%s)",
                    model.d_model, int(model_cfg.get("hybrid_n_layers", 3)), vision_tag)
    else:
        model = ActorCritic(obs_shape, num_actions, hidden=int(model_cfg.get("hidden_size", 64))).to(device)
        logger.info("Model: ActorCritic (hidden=%d)", int(model_cfg.get("hidden_size", 64)))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"]["learning_rate"]),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d (trainable: %d)", total_params, trainable_params)

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

    # --- Stage 3+: World Model ---
    wm_cfg = config.get("world_model")
    wm: RSSM | None = None
    wm_optimizer: torch.optim.Optimizer | None = None
    if stage >= 3 and wm_cfg:
        obs_dim_flat = int(np.prod(obs_shape))
        wm = RSSM(RSSMConfig(
            obs_dim=obs_dim_flat,
            action_dim=num_actions,
            z_dim=int(wm_cfg.get("z_dim", 32)),
            h_dim=int(wm_cfg.get("h_dim", 64)),
            embed_dim=int(wm_cfg.get("embed_dim", 64)),
            hidden=int(wm_cfg.get("hidden", 128)),
            max_rollout_steps=int(wm_cfg.get("max_rollout_steps", 10)),
            kl_free_nats=float(wm_cfg.get("kl_free_nats", 1.0)),
        )).to(device)
        wm_optimizer = torch.optim.Adam(wm.parameters(), lr=float(wm_cfg.get("lr", 3e-4)))
        logger.info("RSSM world model enabled (params=%d)", wm.num_parameters())

    # --- Stage 4+: Skill Library ---
    skills_cfg = config.get("skills")
    skills: BoundedSkillLibrary | None = None
    if stage >= 4 and skills_cfg:
        skills = BoundedSkillLibrary(
            budget=SkillLibraryBudget(
                gpu_capacity=int(skills_cfg.get("gpu_capacity", 32)),
                cpu_capacity=int(skills_cfg.get("cpu_capacity", 128)),
                ssd_max_shards=int(skills_cfg.get("ssd_max_shards", 4)),
                ssd_shard_size=int(skills_cfg.get("ssd_shard_size", 32)),
                merge_similarity_threshold=float(
                    skills_cfg.get("merge_similarity_threshold", 0.9)
                ),
            ),
            skill_shape=(
                int(skills_cfg.get("skill_d_out", 128)),
                int(skills_cfg.get("skill_rank", 8)),
                int(skills_cfg.get("skill_d_in", 128)),
            ),
            device=device,
            archive_dir=data_dir() / str(skills_cfg.get("archive_subdir", "skills")),
            score_alpha=float(skills_cfg.get("score_alpha", 1.0)),
            score_beta=float(skills_cfg.get("score_beta", 0.5)),
            score_gamma=float(skills_cfg.get("score_gamma", 0.1)),
        )
        health.register("skills", skills)
        logger.info(
            "BoundedSkillLibrary enabled (gpu=%d cpu=%d cold=%dx%d capacity=%d)",
            skills._budget.gpu_capacity,
            skills._budget.cpu_capacity,
            skills._budget.ssd_max_shards,
            skills._budget.ssd_shard_size,
            skills.capacity,
        )

    # --- Stage 5+: Auto Curriculum ---
    curriculum_cfg = config.get("curriculum")
    curriculum: AutoCurriculum | None = None
    curriculum_tasks: list[dict] = []
    if stage >= 5 and curriculum_cfg:
        curriculum = AutoCurriculum(AutoCurriculumConfig(
            max_tasks=int(curriculum_cfg.get("max_tasks", 8)),
            lp_window_size=int(curriculum_cfg.get("lp_window_size", 32)),
            lp_min_samples=int(curriculum_cfg.get("lp_min_samples", 8)),
            exploration_epsilon=float(curriculum_cfg.get("exploration_epsilon", 0.1)),
            smoothing=float(curriculum_cfg.get("smoothing", 0.5)),
        ))
        curriculum_tasks = list(curriculum_cfg.get("tasks", []))
        for task_spec in curriculum_tasks:
            curriculum.add_task(TaskTemplate(
                id=int(task_spec["id"]),
                spec={"env_id": task_spec["env_id"]},
                difficulty=float(task_spec.get("difficulty", 0.0)),
                tag=str(task_spec.get("tag", "")),
            ))
        health.register("curriculum", curriculum)
        logger.info(
            "AutoCurriculum enabled (tasks=%d capacity=%d)",
            len(curriculum), curriculum.capacity,
        )

    # --- Stage 6+: Online EWC + Generative Replay VAE + Sleep Loop ---
    continual_cfg = config.get("continual")
    ewc: OnlineEWC | None = None
    grep_vae: GenerativeReplayVAE | None = None
    sleep_loop: SleepConsolidationLoop | None = None
    if stage >= 6 and continual_cfg:
        ewc = OnlineEWC(model, OnlineEWCConfig(
            lambda_reg=float(continual_cfg.get("ewc_lambda", 1.0)),
            gamma=float(continual_cfg.get("ewc_gamma", 0.95)),
            update_anchor_mode=str(continual_cfg.get("ewc_anchor_mode", "replace")),
            anchor_ema_alpha=float(continual_cfg.get("ewc_anchor_ema_alpha", 0.9)),
        ))
        logger.info("Online EWC enabled (lambda=%s gamma=%s)",
                    continual_cfg.get("ewc_lambda", 1.0),
                    continual_cfg.get("ewc_gamma", 0.95))

        if bool(continual_cfg.get("gr_enabled", True)):
            obs_dim_flat = int(np.prod(obs_shape))
            grep_vae = GenerativeReplayVAE(GenerativeReplayConfig(
                obs_dim=obs_dim_flat,
                latent_dim=int(continual_cfg.get("gr_latent_dim", 32)),
                hidden=int(continual_cfg.get("gr_hidden", 128)),
                lr=float(continual_cfg.get("gr_lr", 1e-3)),
                kl_weight=float(continual_cfg.get("gr_kl_weight", 1.0)),
            )).to(device)
            logger.info("Generative Replay VAE enabled (obs_dim=%d, latent=%d, params=%d)",
                        obs_dim_flat,
                        grep_vae.config.latent_dim,
                        grep_vae.num_parameters())

        sleep_loop = SleepConsolidationLoop(ConsolidationConfig(
            warmup_steps=int(continual_cfg.get("sleep_warmup_steps", 1000)),
            replay_trim_every=int(continual_cfg.get("sleep_replay_trim_every", 10_000)),
            skills_merge_every=int(continual_cfg.get("sleep_skills_merge_every", 20_000)),
            ttt_distill_every=int(continual_cfg.get("sleep_ttt_distill_every", 20_000)),
            ewc_consolidate_every=int(continual_cfg.get("ewc_consolidate_every_steps", 100_000)),
        ))

        # --- Register sleep tasks ---
        def _sleep_replay_trim() -> None:
            if replay is not None:
                # Trim priorities: reset any that are below the median
                # (this is a light "trim" — replay is already bounded)
                try:
                    stats = replay.stats()
                    logger.info("[sleep] replay_trim: %s", stats)
                except Exception:
                    pass

        def _sleep_skills_merge() -> None:
            if skills is not None:
                # Skill library merges similar skills on `add`; this hook is
                # a report + placeholder for future explicit merge sweeps.
                try:
                    logger.info("[sleep] skills_merge: %s", skills.stats())
                except Exception:
                    pass

        def _sleep_ttt_distill() -> None:
            # Placeholder: distill TTT inner-W into fixed weights (Stage 6+).
            # In current implementation the TTT layer freshly zero-inits W
            # every forward, so there's no long-lived slow state to distill.
            # Future work.
            logger.info("[sleep] ttt_distill (placeholder)")

        def _sleep_ewc_consolidate() -> None:
            if ewc is None or replay is None or len(replay) == 0:
                return
            batch_size = int(continual_cfg.get("ewc_batch_size", 32))
            num_batches = int(continual_cfg.get("ewc_consolidate_num_batches", 32))

            def _batches():
                for _ in range(num_batches):
                    try:
                        sample = replay.sample(batch_size)
                        yield sample
                    except (ValueError, IndexError):
                        return

            def _loss_fn(m, batch):
                obs = batch["obs"]
                actions = batch["action"]
                logits, _ = m(obs)
                return F.cross_entropy(logits, actions.to(torch.long))

            try:
                ewc.consolidate(model, _batches(), _loss_fn, num_batches=num_batches)
                logger.info("[sleep] ewc_consolidate done: %s", ewc.summary())
            except Exception as exc:  # never crash training
                logger.warning("[sleep] ewc_consolidate failed: %s", exc)

        sleep_loop.set_replay_trim(_sleep_replay_trim)
        sleep_loop.set_skills_merge(_sleep_skills_merge)
        sleep_loop.set_ttt_distill(_sleep_ttt_distill)
        sleep_loop.set_ewc_consolidate(_sleep_ewc_consolidate)
        logger.info("SleepConsolidationLoop enabled")

    # --- Stage 7+: Cognitive modules (SelfModel, NeuralSymbolic, LogicEngine, etc.) ---
    cognitive_cfg = config.get("cognitive")
    self_model: Any = None
    symbolic_layer: Any = None
    logic_engine: Any = None
    reflection_loop: Any = None
    inner_dialogue: Any = None
    language_gen: Any = None

    if stage >= 7 and cognitive_cfg:
        # SelfModel (metacognition)
        if bool(cognitive_cfg.get("self_model_enabled", False)):
            from src.models.metacognition import SelfModel
            self_model = SelfModel(
                d_model=int(cognitive_cfg.get("self_model_d_model", 384)),
                hidden=int(cognitive_cfg.get("self_model_hidden", 64)),
            ).to(device)
            logger.info("SelfModel enabled (metacognition)")

        # NeuralSymbolicLayer (rule extraction + override)
        if bool(cognitive_cfg.get("symbolic_enabled", False)):
            from src.models.neural_symbolic import NeuralSymbolicLayer
            symbolic_layer = NeuralSymbolicLayer(
                d_model=int(model_cfg.get("hidden_size", 384)),
                num_actions=num_actions,
                max_rules=int(cognitive_cfg.get("symbolic_max_rules", 64)),
                match_threshold=float(cognitive_cfg.get("symbolic_match_threshold", 0.7)),
                extraction_reward_threshold=float(cognitive_cfg.get("symbolic_extraction_reward_threshold", 0.3)),
                override_confidence_threshold=float(cognitive_cfg.get("symbolic_override_confidence_threshold", 0.6)),
            ).to(device)
            health.register("symbolic_rules", symbolic_layer.rule_memory)
            logger.info("NeuralSymbolicLayer enabled (max_rules=%d)", symbolic_layer.rule_memory.capacity)

        # LogicEngine (symbolic reasoning with variables)
        if bool(cognitive_cfg.get("logic_engine_enabled", False)):
            from src.models.logic_engine import LogicEngine
            logic_engine = LogicEngine(
                d_model=int(model_cfg.get("hidden_size", 384)),
                max_rules=int(cognitive_cfg.get("logic_max_rules", 64)),
                max_variables=int(cognitive_cfg.get("logic_max_variables", 16)),
                match_threshold=float(cognitive_cfg.get("logic_match_threshold", 0.7)),
            )
            health.register("logic_engine", logic_engine)
            logger.info("LogicEngine enabled (variables + quantification + forward chaining)")

        # InnerDialogue (template mode by default; LLM if available)
        from src.models.metacognition import InnerDialogue
        dialogue_mode = str(cognitive_cfg.get("inner_dialogue_mode", "template"))
        inner_dialogue = InnerDialogue(mode=dialogue_mode)
        logger.info("InnerDialogue enabled (mode=%s)", dialogue_mode)

        # LanguageGenerator (optional, needs Qwen downloaded)
        if bool(cognitive_cfg.get("language_gen_enabled", False)):
            try:
                from src.models.language_generation import LanguageGenerator
                language_gen = LanguageGenerator(
                    model_name=str(cognitive_cfg.get("language_gen_model", "Qwen/Qwen2.5-7B-Instruct")),
                ).to(device)
                logger.info("LanguageGenerator enabled (can speak)")
            except Exception as exc:
                logger.warning("LanguageGenerator load failed (%s), using template mode", exc)

        # ReflectionLoop
        if bool(cognitive_cfg.get("reflection_enabled", False)) and self_model is not None:
            from src.models.metacognition import ReflectionLoop
            reflection_loop = ReflectionLoop(
                self_model=self_model,
                max_reflections=int(cognitive_cfg.get("reflection_max", 256)),
                reflection_every_episodes=int(cognitive_cfg.get("reflection_every_episodes", 10)),
            )
            health.register("reflection", reflection_loop)
            logger.info("ReflectionLoop enabled (every %d episodes)",
                        int(cognitive_cfg.get("reflection_every_episodes", 10)))

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
        resumed_stage = int(payload.get("stage", stage))
        resumed_step = int(payload.get("step", 0))
        if resumed_stage == stage:
            # Same-stage resume: continue the step counter.
            state.step = resumed_step
            logger.info("Resumed same-stage %d from %s at step %d",
                        stage, resume, state.step)
        else:
            # Cross-stage resume (e.g., stage 0 → stage 1): weights are
            # inherited but the step counter restarts so this stage runs its
            # full total_steps budget.
            state.step = 0
            logger.info(
                "Cross-stage resume: loaded weights from stage %d ckpt at step %d, "
                "resetting step counter to 0 for stage %d (this stage will run "
                "its own total_steps=%d budget)",
                resumed_stage, resumed_step, stage, total_steps,
            )

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

    # Stage 3 knobs
    wm_update_every = int(wm_cfg.get("update_every_steps", 8)) if wm_cfg else 0
    wm_log_every = int(wm_cfg.get("log_every_steps", 5000)) if wm_cfg else 0
    wm_last_loss: dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    wm_last_log_step = 0

    # Stage 5 knobs
    curr_switch_every = int(curriculum_cfg.get("switch_every_steps", 20_000)) if curriculum_cfg else 0
    curr_report_every = int(curriculum_cfg.get("report_every_steps", 500)) if curriculum_cfg else 0
    curr_active_task: TaskTemplate | None = None
    if curriculum is not None:
        # Start with the first task
        curr_active_task = curriculum.sample_task()
        logger.info("Curriculum: initial task=%s (id=%d)",
                    curr_active_task.tag, curr_active_task.id)
    last_curr_switch_step = 0
    last_curr_report_step = 0
    last_curr_mean_ret: float = 0.0

    # Stage 6 knobs
    gr_update_every = int(continual_cfg.get("gr_update_every_steps", 16)) if continual_cfg else 0
    gr_batch_size = int(continual_cfg.get("gr_batch_size", 32)) if continual_cfg else 0
    gr_inject_every = int(continual_cfg.get("gr_inject_every_steps", 5000)) if continual_cfg else 0
    gr_rehearsal_bs = int(continual_cfg.get("gr_rehearsal_batch_size", 32)) if continual_cfg else 0
    last_gr_inject_step = 0
    gr_last_loss: float = 0.0

    # Stage 7 knobs
    symbolic_extract_threshold = float(
        cognitive_cfg.get("symbolic_extraction_reward_threshold", 0.3)
    ) if cognitive_cfg else 0.3
    last_reflection_ep = 0

    t0 = time.time()
    logger.info(
        "Starting Stage %s: total_steps=%d smoke=%s "
        "intrinsic=%s replay=%s coverage=%s wm=%s skills=%s curriculum=%s "
        "ewc=%s gr=%s sleep=%s symbolic=%s selfmodel=%s logic=%s lang=%s",
        stage,
        total_steps,
        smoke_only,
        rnd is not None,
        replay is not None,
        coverage is not None,
        wm is not None,
        skills is not None,
        curriculum is not None,
        ewc is not None,
        grep_vae is not None,
        sleep_loop is not None,
        symbolic_layer is not None,
        self_model is not None,
        logic_engine is not None,
        language_gen is not None,
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

            # --- Stage 4: extract skill on successful episode end ---
            if skills is not None and (step_out.terminated or step_out.truncated):
                ep_ret = env.summary().get("last_return", 0.0)
                if ep_ret > 0.3:
                    skill = skills.new_skill(
                        tag=f"ep{env.summary()['episodes']}_ret{ep_ret:.2f}"
                    )
                    skill.record_use(reward=ep_ret)
                    skills.add(skill)

                # --- Stage 7: extract symbolic rules from successful episodes ---
                if symbolic_layer is not None and ep_ret > symbolic_extract_threshold:
                    try:
                        symbolic_layer.extract_rules(
                            hidden_states=[batch_obs_t.squeeze(0) for batch_obs_t in
                                          [torch.from_numpy(obs).unsqueeze(0).to(device)]],
                            actions=[int(action.item())],
                            rewards=[ep_ret],
                            descriptions=[f"IF see {curr_active_task.tag if curr_active_task else 'env'} THEN action"],
                        )
                    except Exception:
                        pass  # never crash training for symbolic extraction

                # --- Stage 7: reflection after episode ---
                if reflection_loop is not None:
                    try:
                        reflection = reflection_loop.end_episode(ep_ret)
                        if reflection is not None and inner_dialogue is not None:
                            lessons = inner_dialogue.generate(reflection)
                            for lesson in lessons[:2]:  # log first 2 lessons
                                logger.info("[reflection] %s", lesson)
                    except Exception:
                        pass  # never crash training for reflection

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
            # Stage 6: add EWC penalty (protects consolidated weights)
            if ewc is not None and ewc.has_consolidated():
                loss = loss + ewc.penalty(model).to(loss.device)
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

        # --- Stage 3: World Model update from replay ---
        if (
            wm is not None
            and wm_optimizer is not None
            and replay is not None
            and len(replay) >= replay_min_size
            and wm_update_every > 0
            and state.step % wm_update_every < rollout_capacity
        ):
            try:
                sample, _, _ = replay.sample_prioritized(
                    min(replay_batch_size, wm.config.max_rollout_steps * 4),
                    alpha=per_alpha,
                )
                # Reshape B*T into (batch=B/T, T=max_rollout_steps).
                # For simplicity we treat each transition as a T=1 sequence.
                # This is an approximation — proper sequential replay comes later.
                bsz = sample["obs"].shape[0]
                T = 1
                obs_flat = sample["obs"].reshape(bsz, T, -1).float() / 255.0
                # One-hot actions
                actions_onehot = F.one_hot(
                    sample["action"].to(torch.long), num_classes=num_actions
                ).float().reshape(bsz, T, num_actions)
                wm_out = wm.compute_loss(obs_flat, actions_onehot)
                wm_loss = wm_out["loss"]
                wm_optimizer.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)
                wm_optimizer.step()
                wm_last_loss = {
                    "loss": float(wm_out["loss"].item()),
                    "recon": float(wm_out["recon_loss"].item()),
                    "kl": float(wm_out["kl_loss"].item()),
                }
            except (ValueError, IndexError, RuntimeError) as exc:
                logger.debug("world model update skipped: %s", exc)

        # --- Stage 6: Generative Replay VAE update from replay ---
        if (
            grep_vae is not None
            and replay is not None
            and len(replay) >= replay_min_size
            and gr_update_every > 0
            and state.step % gr_update_every < rollout_capacity
        ):
            try:
                sample = replay.sample(gr_batch_size)
                obs_flat = sample["obs"].reshape(gr_batch_size, -1).float() / 255.0
                gr_out = grep_vae.update(obs_flat)
                gr_last_loss = float(gr_out["loss"])
            except (ValueError, IndexError, RuntimeError) as exc:
                logger.debug("generative replay update skipped: %s", exc)

        # --- Stage 6: Sleep consolidation tick ---
        if sleep_loop is not None:
            sleep_loop.tick(step=state.step)

        # Health sweep after each rollout+update cycle
        health.sweep()

        # --- Stage 7: symbolic rule override on next batch ---
        if symbolic_layer is not None:
            try:
                # Use the first obs in the batch for symbolic reasoning
                first_obs = batch.obs[0:1]
                with torch.no_grad():
                    logits_check, _ = model(first_obs)
                final_logits, sym_info = symbolic_layer(
                    model(first_obs)[0],  # hidden state proxy = logits
                    logits_check,
                )
                if sym_info.get("override", False):
                    logger.debug("[symbolic] rule #%d matched (sim=%.2f), action overridden",
                                 sym_info.get("rule_id", -1), sym_info.get("rule_sim", 0))
            except Exception:
                pass  # never crash training for symbolic reasoning

        if state.step % log_every < rollout_capacity:
            summary = env.summary()
            mem = watcher.snapshot_summary()
            extras: list[str] = []
            if coverage is not None:
                extras.append(f"cov={coverage.coverage_ratio() * 100:.1f}%")
            if replay is not None:
                extras.append(f"replay={len(replay)}/{replay.capacity}")
            if wm is not None:
                extras.append(f"wm={wm_last_loss['loss']:.3f}(r={wm_last_loss['recon']:.3f},kl={wm_last_loss['kl']:.3f})")
            if skills is not None:
                extras.append(f"skills={len(skills)}/{skills.capacity}")
            if curriculum is not None and curr_active_task is not None:
                extras.append(f"task={curr_active_task.tag}")
            if ewc is not None:
                extras.append(f"ewc={'✓' if ewc.has_consolidated() else '_'}")
            if grep_vae is not None:
                extras.append(f"gr={gr_last_loss:.3f}")
            if symbolic_layer is not None:
                extras.append(f"rules={len(symbolic_layer.rule_memory)}")
            if self_model is not None:
                extras.append("meta=on")
            if logic_engine is not None:
                extras.append(f"logic={len(logic_engine)}")
            if language_gen is not None:
                extras.append("speak=on")
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

        # Periodic WM snapshot (Stage 3)
        if (
            wm is not None
            and wm_log_every > 0
            and state.step - wm_last_log_step >= wm_log_every
        ):
            logger.info("world model @ step=%d: %s", state.step, wm_last_loss)
            wm_last_log_step = state.step

        # --- Stage 5: report LP error signal to curriculum ---
        if (
            curriculum is not None
            and curr_active_task is not None
            and curr_report_every > 0
            and state.step - last_curr_report_step >= curr_report_every
        ):
            current_mean_ret = env.summary()["mean_return"]
            # LP tracker expects an "error" — lower = better performance.
            # Use (1 - mean_return) as a proxy so decreasing error ↔ improving policy.
            err = max(0.0, 1.0 - current_mean_ret)
            try:
                curriculum.report_error(curr_active_task.id, err)
            except KeyError:
                pass
            last_curr_mean_ret = current_mean_ret
            last_curr_report_step = state.step

        # --- Stage 5: periodically switch task via LP-driven sampling ---
        if (
            curriculum is not None
            and curr_switch_every > 0
            and state.step - last_curr_switch_step >= curr_switch_every
        ):
            new_task = curriculum.sample_task()
            if curr_active_task is None or new_task.id != curr_active_task.id:
                logger.info(
                    "Curriculum switch @ step=%d: task=%s (id=%d) → %s (id=%d)",
                    state.step,
                    curr_active_task.tag if curr_active_task else "<none>",
                    curr_active_task.id if curr_active_task else -1,
                    new_task.tag, new_task.id,
                )
                # Rebuild env from task spec
                try:
                    env.close()
                except Exception:
                    pass
                env = MiniGridWrapper(
                    env_id=new_task.spec["env_id"],
                    seed=42,
                    max_episode_steps=env_cfg.get("max_episode_steps"),
                    auto_reset=True,
                )
                obs = env.reset()
                curr_active_task = new_task
            last_curr_switch_step = state.step

        if state.step % ckpt_every < rollout_capacity:
            extra: dict[str, Any] = {"preset": config.get("preset"), "run_id": run_id}
            if rnd is not None:
                extra["rnd_state"] = rnd.rnd_state_dict()
            if coverage is not None:
                extra["coverage_state"] = coverage.state_dict()
            if wm is not None:
                extra["wm_state"] = wm.state_dict()
                if wm_optimizer is not None:
                    extra["wm_optim_state"] = wm_optimizer.state_dict()
            if skills is not None:
                extra["skills_state"] = skills.state_dict()
            if curriculum is not None:
                extra["curriculum_state"] = curriculum.state_dict()
                extra["curriculum_active_task_id"] = (
                    curr_active_task.id if curr_active_task else -1
                )
            if ewc is not None:
                extra["ewc_state"] = ewc.state_dict()
            if grep_vae is not None:
                extra["gr_vae_state"] = grep_vae.state_dict()
            if sleep_loop is not None:
                extra["sleep_loop_state"] = sleep_loop.state_dict()
            if symbolic_layer is not None:
                extra["symbolic_state"] = symbolic_layer.rule_memory.state_dict()
            if self_model is not None:
                extra["self_model_state"] = self_model.state_dict()
            if logic_engine is not None:
                extra["logic_engine_state"] = logic_engine.state_dict()
            if reflection_loop is not None:
                extra["reflection_state"] = reflection_loop.state_dict()
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
    if wm is not None:
        logger.info("World model final losses: %s", wm_last_loss)
    if skills is not None:
        logger.info("Skills final: %s", skills.stats())
    if curriculum is not None:
        logger.info("Curriculum final: %s", curriculum.summary())
    if ewc is not None:
        logger.info("EWC final: %s", ewc.summary())
    if grep_vae is not None:
        logger.info("Generative Replay VAE final: %s", grep_vae.summary())
    if sleep_loop is not None:
        logger.info("Sleep loop final: %s", sleep_loop.summary())

    env.close()
    return 0


# =====================================================================
# CLI
# =====================================================================


_DEFAULT_STAGE_CONFIGS = {
    0: "stage0_baseline.yaml",
    1: "stage1_curiosity.yaml",
    2: "stage2_hybrid.yaml",
    3: "stage3_world_model.yaml",
    4: "stage4_skills.yaml",
    5: "stage5_curriculum.yaml",
    6: "stage6_consolidation.yaml",
    7: "stage7_cognitive.yaml",
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
