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
import math
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
from src.intrinsic import (
    ExplorationBonus,
    IntentionConfig,
    IntentionCuriosity,
    KnowledgeGapConfig,
    KnowledgeGapDetector,
    RND,
    RNDConfig,
    SocialCuriosity,
    SocialCuriosityConfig,
)
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
from src.sensory import AudioEncoder, AudioEncoderConfig
from src.training import ImaginationConfig, ImaginationTrainer
from src.models.number_sense import NumberSense
from src.models.rule_induction import RuleInductionEngine
from src.models.causal_discovery import CausalDiscovery
from src.models.cross_modal_bridges import CrossModalManager
from src.models.creativity_orchestrator import CreativityOrchestrator
from src.models.llm_fusion import LLMFusionBridge
from src.models.developmental_memory import MemoryManager
from src.models.theory_of_mind import TheoryOfMind
from src.models.homeostatic_drives import HomeostaticDrives
from src.models.long_range_planner import LongRangePlanner
from src.models.emotion_system import EmotionSystem
from src.models.concept_graph import ConceptGraph
from src.models.metacognition_v2 import SelfReflectionValidator, ConceptClusterer
from src.models.iq_boost import (
    CrossDomainTransfer, DeepMultiModal, TemporalReasoner,
    CounterfactualRegret, CuriosityDirector, ValueSystem,
)
from src.models.abstract_reasoning import MicroPrologMath, IdentityNarrative
from src.models.tier2_cognitive import Analogizer, BeliefDepth2, MoralConnector, SurpriseHumor
from src.models.neuro_symbolic_bridge import Causal2Prolog, Number2Math, SchemaDetector
from src.models.program_synthesis import ProgramSynthesizer, ActiveExperimenter, TemporalAbstractor
from src.models.counterfactual_planner import CounterfactualPlanner
from src.models.marginal_gains import CompositionalTester, LearningProgressTracker
from src.models.visual_analyzer import VisualAnalyzer
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
            nn.AdaptiveAvgPool2d((8, 8)),   # P1: down-sample -> fixed spatial dim
            nn.Flatten(),
        )
        flat_dim = 32 * 8 * 8  # was 32*h*w (e.g. 2M @256px -> GB-scale Linear)
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, obs_u8: torch.Tensor, return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        z = self.trunk(self.encoder(x))
        if return_hidden:
            return self.policy_head(z), self.value_head(z).squeeze(-1), z
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
        use_slot_attention: bool = False,
        slot_num_slots: int = 7,
        slot_dim: int = 128,
        slot_num_iterations: int = 3,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        if d_model % 2 != 0:
            d_model += 1
        self.d_model = d_model

        self.use_vision = use_vision_encoder
        self.use_slots = use_slot_attention
        if use_slot_attention:
            from src.models.slot_attention import SlotAttention
            self.encoder = SlotAttention(
                d_model=d_model,
                num_slots=slot_num_slots,
                slot_dim=slot_dim,
                num_iterations=slot_num_iterations,
            )
            logger.info("HybridActorCritic: using SlotAttention (num_slots=%d)", slot_num_slots)
        elif use_vision_encoder:
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
                    nn.AdaptiveAvgPool2d((8, 8)),   # P1: fixed spatial dim
                    nn.Flatten(),
                    nn.Linear(32 * 8 * 8, d_model),
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
                nn.AdaptiveAvgPool2d((8, 8)),   # P1: fixed spatial dim
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, d_model),
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
        self.obs_shape = tuple(obs_shape)
        self._last_slots = None  # set per forward; avoids a 2nd encoder call in the rollout loop

    def forward(
        self, obs_u8: torch.Tensor, return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_slots:
            seq = self.encoder(obs_u8)  # (B, num_slots, d_model)
        elif self.use_vision:
            feats = self.encoder(obs_u8)
            seq = feats.unsqueeze(1)
        else:
            x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
            feats = self.encoder(x)
            seq = feats.unsqueeze(1)
        seq_out = self.backbone(seq)
        self._last_slots = seq
        z = seq_out.mean(dim=1) if self.use_slots else seq_out.squeeze(1)
        if return_hidden:
            return self.policy_head(z), self.value_head(z).squeeze(-1), z
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

    Vectorization (Stage-2): stores ``(capacity, n_envs, *obs_shape)``
    transitions — each collected timestep holds ``n_envs`` independent
    env-steps. Capacity is in *timesteps* (T), so a rollout holds
    ``T * n_envs`` total transitions. ``as_batch`` flattens to
    ``(T*n_envs, *obs_shape)`` so the PPO update is unchanged.
    """

    def __init__(
        self, capacity: int, obs_shape: tuple[int, ...], device: torch.device,
        n_envs: int = 1,
    ) -> None:
        self._capacity = int(capacity)
        self.n_envs = int(n_envs)
        self.obs = torch.zeros(
            (capacity, self.n_envs, *obs_shape), dtype=torch.uint8, device=device
        )
        self.actions = torch.zeros((capacity, self.n_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((capacity, self.n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity, self.n_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, self.n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, self.n_envs), dtype=torch.float32, device=device)
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
        action: np.ndarray,
        logprob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add ``n_envs`` transitions for one collected timestep.

        Shapes: ``obs`` (n_envs, *obs_shape) uint8; the rest (n_envs,).
        """
        if self._ptr >= self._capacity:
            raise IndexError("RolloutBuffer full (Axiom 1: no unbounded growth)")
        i = self._ptr
        self.obs[i] = torch.from_numpy(np.asarray(obs))
        self.actions[i] = torch.as_tensor(np.asarray(action)).to(self.actions.dtype)
        self.logprobs[i] = torch.as_tensor(np.asarray(logprob)).to(self.logprobs.dtype)
        self.values[i] = torch.as_tensor(np.asarray(value)).to(self.values.dtype)
        self.rewards[i] = torch.as_tensor(np.asarray(reward)).to(self.rewards.dtype)
        self.dones[i] = torch.as_tensor(np.asarray(done)).to(self.dones.dtype)
        self._ptr += 1

    def as_batch(self) -> TransitionBatch:
        T = self._ptr
        N = self.n_envs
        obs_shape = self.obs.shape[2:]
        return TransitionBatch(
            obs=self.obs[:T].reshape(T * N, *obs_shape),
            actions=self.actions[:T].reshape(T * N),
            logprobs=self.logprobs[:T].reshape(T * N),
            values=self.values[:T].reshape(T * N),
            rewards=self.rewards[:T].reshape(T * N),
            dones=self.dones[:T].reshape(T * N),
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


def compute_gae_vec(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE over N envs. Inputs ``(T, N)``; returns ``(T, N)``.

    Computes GAE independently per env column (no cross-env
    bootstrapping) and is pure-tensor (no ``.item()`` syncs),
    which also removes a per-step CUDA-sync bottleneck.
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]          # (N,)
        next_non_terminal = 1.0 - dones[t].float()                          # (N,)
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]  # (N,)
        gae = delta + gamma * lam * next_non_terminal * gae                    # (N,)
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# =====================================================================
# Trainer
# =====================================================================


class ReturnNormalizer:
    """EMA-based reward/return normalization (PopArt-lite).

    Prevents dead-policy lock when rewards are small and uniform:
    - EMA tracks mean/std of returns
    - Normalizes returns before computing value loss
    - Denormalizes predicted values before GAE advantage computation
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean: float = 0.0
        self.var: float = 1.0

    def update(self, returns: torch.Tensor) -> None:
        self.mean = (1 - self.alpha) * self.mean + self.alpha * float(returns.mean().item())
        self.var = (1 - self.alpha) * self.var + self.alpha * float(returns.var().item())

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return (returns - self.mean) / (self.var ** 0.5 + 1e-8)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.var ** 0.5 + 1e-8) + self.mean


def _normalize_advantages(
    advantages: torch.Tensor, zero_var_eps: float = 1e-7
) -> torch.Tensor:
    """Standardize advantages to ~N(0, 1) with a zero-variance guard.

    When advantages have (near-)zero variance — e.g., the 3D-deadlock case
    where every episode earns essentially the same tiny reward — dividing by
    ``std + 1e-8`` can amplify float noise into a spurious huge gradient.
    The guard falls back to raw centered advantages (~0) in that case, so the
    policy update becomes a safe no-op instead of a noise-driven one.

    Note: this does NOT create learning signal where none exists; if rewards
    are truly constant across the batch, the policy cannot improve from this
    batch. The upstream fix is reward variance (intrinsic curiosity that does
    not decay to zero, or reward-scale normalization before GAE).
    """
    if advantages.numel() < 2:
        # torch.std of a <2-element tensor is NaN and warns; skip standardization.
        return advantages - advantages.mean()
    std = advantages.std()
    std_val = float(std.item())
    # `std_val` is NaN when the batch has <2 elements (torch semantics); treat
    # that as the zero-variance case (fall back to raw centered advantages).
    if not math.isfinite(std_val) or std_val < zero_var_eps:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


@dataclass
class TrainState:
    step: int
    episode: int


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(obs))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def _ckpt_layer_count(path) -> int:
    """Infer the number of backbone blocks in a checkpoint's model_state.

    Used to build the model with the SAME layer count as a resume checkpoint,
    so a 3-layer checkpoint does not fail to load into a 2-layer model (which
    would silently reinitialize the model randomly). Returns 0 if unknown.
    """
    try:
        import torch as _torch
        _ck = _torch.load(path, map_location="cpu")
        _ms = _ck.get("model_state") if isinstance(_ck, dict) else None
        if _ms is None:
            return 0
        _n = 0
        for _k in _ms.keys():
            if _k.startswith("backbone.blocks."):
                _parts = _k.split(".")
                if len(_parts) > 2 and _parts[2].isdigit():
                    _n = max(_n, int(_parts[2]) + 1)
        return _n
    except Exception:  # noqa: BLE001
        return 0


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
    env_id = str(env_cfg.get("id", "MiniGrid-Empty-5x5-v0"))
    n_envs = 1
    if env_id == "PhysicsSandbox":
        from src.envs.physics_sandbox import PhysicsSandbox
        env = PhysicsSandbox(
            num_objects=int(env_cfg.get("num_objects", 3)),
            seed=42,
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
            render_size=int(env_cfg.get("render_size", 64)),
            gravity=float(env_cfg.get("gravity", -9.8)),
            action_force=float(env_cfg.get("action_force", 50.0)),
        )
        logger.info("Env: PhysicsSandbox (2D physics, %d objects)", env_cfg.get("num_objects", 3))
    elif env_id == "SocialTeacher":
        from src.envs.social_teacher import SocialTeacherWrapper
        env = SocialTeacherWrapper(
            num_objects=int(env_cfg.get("num_objects", 3)),
            seed=42,
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
            render_size=int(env_cfg.get("render_size", 64)),
        )
        logger.info("Env: SocialTeacherWrapper (2D physics + teacher, %d objects)", env_cfg.get("num_objects", 3))
    elif env_id == "ThreeDWorld":
        from src.envs.three_d_world import ThreeDWorld
        from src.envs.vec_three_d_world import VecThreeDWorld
        _kw = dict(
            num_objects=int(env_cfg.get("num_objects", 100)),
            max_episode_steps=env_cfg.get("max_episode_steps", 500),
            render_size=int(env_cfg.get("render_size", 256)),
            action_force=float(env_cfg.get("action_force", 2.0)),
            developmental_age=float(env_cfg.get("developmental_age", 0.0)),
        )
        n_envs = int(env_cfg.get("num_envs", 1))
        if n_envs > 1:
            env = VecThreeDWorld(n_envs=n_envs, **_kw)
            logger.info("Env: VecThreeDWorld x%d (3D home, %d objects)", n_envs, _kw["num_objects"])
        else:
            env = ThreeDWorld(seed=42, **_kw)
            logger.info("Env: ThreeDWorld (3D home, %d objects)", _kw["num_objects"])
    elif env_id == "ExtendedThreeDWorld":
        from src.envs.extended_3d_world import ExtendedThreeDWorld
        env = ExtendedThreeDWorld(
            num_objects=int(env_cfg.get("num_objects", 500)),
            num_siblings=int(env_cfg.get("num_siblings", 2)),
            seed=42,
            max_episode_steps=env_cfg.get("max_episode_steps", 500),
            render_size=int(env_cfg.get("render_size", 256)),
            developmental_age=float(env_cfg.get("developmental_age", 0.0)),
        )
        logger.info("Env: ExtendedThreeDWorld (4 rooms + %d siblings)", env_cfg.get("num_siblings", 2))
    else:
        env = MiniGridWrapper(
            env_id=env_id,
            seed=42,
            max_episode_steps=env_cfg.get("max_episode_steps"),
            auto_reset=True,
        )
        logger.info("Env: %s  obs_shape=%s  actions=%d", env_cfg["id"], obs_shape, num_actions)

    obs = env.reset()
    obs_shape = env.observation_shape
    num_actions = env.action_space_n
    if int(env_cfg.get("num_envs", 1)) > 1 and n_envs == 1:
        logger.warning("n_envs>1 only supported for ThreeDWorld; falling back to 1")

    # --- Model
    model_cfg = config["model"]
    # On resume, build the model with the SAME number of backbone blocks as the
    # checkpoint. Otherwise a 3-layer checkpoint loaded into a 2-layer model
    # raises a state_dict size mismatch, the model is reinitialized RANDOMLY,
    # and the grower's subsequent 2->3 growth silently trains from scratch
    # (this produced the "spurious growth + crashed mean_return" symptom).
    model_n_layers = int(model_cfg.get("hybrid_n_layers", 3))
    grower_initial = int(model_cfg.get("hybrid_n_layers", 2))
    if resume is not None:
        _n = _ckpt_layer_count(resume)
        if _n > 0:
            model_n_layers = _n
            grower_initial = _n
            logger.info(
                "Resume ckpt has %d layers; building model+grower to match "
                "(prevents random reinit on layer-count mismatch).", _n)
    if bool(model_cfg.get("use_hybrid_backbone", False)):
        model = HybridActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            d_model=int(model_cfg.get("hidden_size", 128)),
            n_layers=model_n_layers,
            n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
            swa_window=int(model_cfg.get("hybrid_swa_window", 16)),
            ttt_mini_batch=int(model_cfg.get("hybrid_ttt_mini_batch", 8)),
            ffn_hidden_mult=int(model_cfg.get("hybrid_ffn_hidden_mult", 4)),
            dropout=float(model_cfg.get("hybrid_dropout", 0.0)),
            use_vision_encoder=bool(model_cfg.get("use_vision_encoder", False)),
            vision_model_name=str(model_cfg.get("vision_model", "dinov2_vits14")),
            vision_freeze=bool(model_cfg.get("vision_freeze", True)),
            use_slot_attention=bool(model_cfg.get("use_slot_attention", False)),
            slot_num_slots=int(model_cfg.get("slot_num_slots", 7)),
            slot_dim=int(model_cfg.get("slot_dim", 128)),
            slot_num_iterations=int(model_cfg.get("slot_num_iterations", 3)),
        ).to(device)
        tag_parts = []
        if model_cfg.get("use_slot_attention"):
            tag_parts.append(f"SlotAttention ({model_cfg.get('slot_num_slots', 7)} slots)")
        if model_cfg.get("use_vision_encoder"):
            tag_parts.append("VisionEncoder")
        vision_tag = " + " + " + ".join(tag_parts) if tag_parts else ""
        logger.info("Model: HybridActorCubit (d_model=%d, layers=%d%s)",
                    model.d_model, model_n_layers, vision_tag)
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
    buffer = RolloutBuffer(rollout_capacity, obs_shape, device=device, n_envs=n_envs)

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
    curiosity_cfg = config.get("curiosity")
    replay_cfg = config.get("replay")
    coverage_cfg = config.get("coverage")

    rnd: RND | None = None
    replay: BoundedReplayBuffer | None = None
    coverage: BoundedCoverage | None = None
    # Phase 0+: curiosity from RSSM uncertainty instead of RND
    curiosity_mode = str(curiosity_cfg.get("mode", "none")) if curiosity_cfg else "none"
    curiosity_coef = float(curiosity_cfg.get("coef", 0.5)) if curiosity_cfg else 0.0

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

    if curiosity_mode == "rssm_uncertainty":
        logger.info("Curiosity: RSSM uncertainty (coef=%.2f)", curiosity_coef)
    elif curiosity_mode == "rnd" and rnd is not None:
        logger.info("Curiosity: RND (coef=%.2f)", curiosity_coef)

    # Count-based exploration bonus: a *state-dependent* floor on the
    # reward signal so 3D cannot deadlock when env reward is sparse
    # (the value head fits a constant but cannot predict visit-count-based
    # novelty -> advantages keep a residual -> policy keeps exploring).
    # Top-level key (NOT under `intrinsic:`) so enabling it does not
    # also switch on RND / change the curiosity mode.
    expl_bonus: ExplorationBonus | None = None
    eb_cfg = config.get("exploration_bonus")
    if eb_cfg and eb_cfg.get("enabled", False):
        expl_bonus = ExplorationBonus(
            obs_shape,
            capacity=int(eb_cfg.get("capacity", 1 << 16)),
            coef=float(eb_cfg.get("coef", 0.1)),
            grid=int(eb_cfg.get("grid", 8)),
        ).to(device)
        logger.info(
            "ExplorationBonus enabled (coef=%.3f, buckets=%d)",
            expl_bonus.coef, expl_bonus.capacity,
        )

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
    if stage >= 3 and wm_cfg and bool(wm_cfg.get("enabled", True)):
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
            reward_loss_weight=float(wm_cfg.get("reward_loss_weight", 1.0)),
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
                temporal=True,
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
                use_trainable_unifier=True,
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

    # --- Stage 8+: Full cognitive activation ---
    advanced_cfg = config.get("advanced")
    language_cfg = config.get("language")
    hypothesis_tester: Any = None
    counterfactual: Any = None
    behavior_cloning: Any = None
    meta_learner: Any = None
    code_exec_env: Any = None
    divergent_gen: Any = None
    transformational: Any = None
    thought_action: Any = None
    model_grower: Any = None
    language_encoder: Any = None
    language_generator: Any = None

    if stage >= 8:
        # Hypothesis tester
        if advanced_cfg and bool(advanced_cfg.get("hypothesis_tester_enabled", False)):
            from src.models.advanced_cognition import HypothesisTester
            hypothesis_tester = HypothesisTester(
                d_model=int(model_cfg.get("hidden_size", 384)),
                num_actions=num_actions,
                max_hypotheses=int(advanced_cfg.get("hypothesis_max", 32)),
                probe_epsilon=float(advanced_cfg.get("hypothesis_probe_epsilon", 0.1)),
            ).to(device)
            health.register("hypotheses", hypothesis_tester)
            logger.info("HypothesisTester enabled (max=%d)", hypothesis_tester.capacity)

        # Counterfactual imagination
        if advanced_cfg and bool(advanced_cfg.get("counterfactual_enabled", False)):
            from src.models.advanced_cognition import CounterfactualImagination
            counterfactual = CounterfactualImagination(
                max_imagination_steps=int(advanced_cfg.get("counterfactual_max_steps", 5)),
            )
            logger.info("CounterfactualImagination enabled (max_steps=%d)", counterfactual.max_steps)

        # Behavior cloning
        if advanced_cfg and bool(advanced_cfg.get("behavior_cloning_enabled", False)):
            from src.models.advanced_cognition import BehaviorCloningHead
            behavior_cloning = BehaviorCloningHead(
                bc_coef=float(advanced_cfg.get("behavior_cloning_coef", 0.3)),
            )
            logger.info("BehaviorCloningHead enabled (coef=%.2f)", behavior_cloning.current_coef)

        # Meta learner
        if advanced_cfg and bool(advanced_cfg.get("meta_learner_enabled", False)):
            from src.models.advanced_cognition import MetaLearner
            meta_learner = MetaLearner(
                model=model,
                ema_decay=float(advanced_cfg.get("meta_ema_decay", 0.9)),
            )
            logger.info("MetaLearner enabled (ema_decay=%.2f)", meta_learner._ema_decay)

        # Code execution env
        if advanced_cfg and bool(advanced_cfg.get("code_execution_enabled", False)):
            from src.models.code_and_social import CodeExecutionEnv
            code_exec_env = CodeExecutionEnv(
                timeout_seconds=float(advanced_cfg.get("code_exec_timeout", 5.0)),
            )
            health.register("code_exec", code_exec_env)
            logger.info("CodeExecutionEnv enabled (sandbox)")

        # Divergent generator (combinational creativity)
        if cognitive_cfg and bool(cognitive_cfg.get("divergent_generator_enabled", False)):
            from src.models.divergent_generator import DivergentGenerator
            divergent_gen = DivergentGenerator(
                d_model=int(model_cfg.get("hidden_size", 384)),
            ).to(device)
            health.register("divergent", divergent_gen)
            logger.info("DivergentGenerator enabled (combinational creativity)")

        # Transformational creativity
        if cognitive_cfg and bool(cognitive_cfg.get("transformational_creativity_enabled", False)):
            from src.models.transformational_creativity import TransformationalCreativityEngine
            transformational = TransformationalCreativityEngine(
                d_model=int(model_cfg.get("hidden_size", 384)),
                max_transformations=int(cognitive_cfg.get("transformational_max", 32)),
                distance_threshold=float(cognitive_cfg.get("transformational_distance_threshold", 0.3)),
                curiosity_threshold=float(cognitive_cfg.get("transformational_curiosity_threshold", 0.3)),
            ).to(device)
            health.register("transformational", transformational)
            logger.info("TransformationalCreativityEngine enabled")

        # Thought-action loop
        if cognitive_cfg and bool(cognitive_cfg.get("thought_action_enabled", False)):
            from src.models.thought_action_loop import ThoughtActionLoop
            thought_action = ThoughtActionLoop(
                d_model=int(model_cfg.get("hidden_size", 384)),
                think_every_steps=int(cognitive_cfg.get("think_every_steps", 50)),
            ).to(device)
            logger.info("ThoughtActionLoop enabled (every %d steps)", thought_action._think_every)

        # Model grower
        if advanced_cfg and bool(advanced_cfg.get("model_grower_enabled", False)):
            from src.models.model_growth import GrowthConfig, ModelGrower
            model_grower = ModelGrower(GrowthConfig(
                initial_params=sum(p.numel() for p in model.parameters()),
                max_params=int(advanced_cfg.get("model_grower_max_params", 200_000_000)),
                grow_factor=float(advanced_cfg.get("model_grower_factor", 1.5)),
                min_steps_between_growths=int(advanced_cfg.get("model_grower_min_steps", 100_000)),
            ))
            logger.info("ModelGrower enabled (current=%dM, max=%dM)",
                        model_grower.current_params // 10**6,
                        model_grower.max_params // 10**6)

        # Language encoder (CLIP)
        if language_cfg and bool(language_cfg.get("enabled", False)):
            try:
                from src.models.language_encoder import LanguageEncoder
                language_encoder = LanguageEncoder(
                    d_model=int(language_cfg.get("d_model", 384)),
                ).to(device)
                logger.info("LanguageEncoder (CLIP) enabled")
            except Exception as exc:
                logger.warning("LanguageEncoder load failed (%s)", exc)

        # Language generator (Qwen-7B)
        if cognitive_cfg and bool(cognitive_cfg.get("language_gen_enabled", False)):
            try:
                from src.models.language_generation import LanguageGenerator
                language_generator = LanguageGenerator(
                    model_name=str(cognitive_cfg.get("language_gen_model", "Qwen/Qwen2.5-7B-Instruct")),
                ).to(device)
                logger.info("LanguageGenerator enabled (can speak)")
            except Exception as exc:
                logger.warning("LanguageGenerator load failed (%s), using template", exc)

    # --- Phase 0+: Developmental modules (7 new improvements) ---
    num_sense_cfg = config.get("number_sense")
    social_cfg = config.get("social_teacher")
    rule_ind_cfg = config.get("rule_induction")
    causal_cfg = config.get("causal_discovery")
    growth_cfg = config.get("model_growth")
    xmodal_cfg = config.get("cross_modal")

    number_sense: NumberSense | None = None
    num_sense_optimizer: torch.optim.Optimizer | None = None
    rule_engine: RuleInductionEngine | None = None
    causal_disc: CausalDiscovery | None = None
    model_grower_v2: Any = None
    xmodal_manager: CrossModalManager | None = None

    if num_sense_cfg and bool(num_sense_cfg.get("enabled", False)):
        slot_count = int(model_cfg.get("slot_num_slots", 7))
        slot_d = int(model_cfg.get("slot_dim", int(model_cfg.get("hidden_size", 128))))
        number_sense = NumberSense(
            num_slots=slot_count, slot_dim=slot_d,
            max_count=int(num_sense_cfg.get("max_count", 10)),
            hidden=int(num_sense_cfg.get("hidden", 32)),
        ).to(device)
        num_sense_optimizer = torch.optim.Adam(
            number_sense.parameters(),
            lr=float(num_sense_cfg.get("lr", 3e-4)),
        )
        logger.info("NumberSense enabled (max_count=%d)", num_sense_cfg.get("max_count", 10))

    if rule_ind_cfg and bool(rule_ind_cfg.get("enabled", False)):
        rule_engine = RuleInductionEngine(
            num_slots=int(model_cfg.get("slot_num_slots", 7)),
            max_rules=int(rule_ind_cfg.get("max_rules", 128)),
            max_chain_depth=int(rule_ind_cfg.get("max_chain_depth", 5)),
            min_confidence=float(rule_ind_cfg.get("min_confidence", 0.3)),
            induction_min_positive=int(rule_ind_cfg.get("induction_min_positive", 3)),
        )
        health.register("rule_engine", rule_engine)
        logger.info("RuleInductionEngine enabled (max_rules=%d)", rule_engine.capacity)

    if causal_cfg and bool(causal_cfg.get("enabled", False)):
        causal_disc = CausalDiscovery(
            num_actions=num_actions,
            max_edges=int(causal_cfg.get("max_edges", 256)),
            min_intervention_effect=float(causal_cfg.get("min_intervention_effect", 0.01)),
        )
        health.register("causal_edges", causal_disc)
        logger.info("CausalDiscovery enabled (max_edges=%d)", causal_disc.capacity)

    if growth_cfg and bool(growth_cfg.get("enabled", False)):
        from src.models.model_growth_v2 import ModelGrowerV2, GrowthConfigV2
        model_grower_v2 = ModelGrowerV2(
            d_model=int(model_cfg.get("hidden_size", 128)),
            n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
            config=GrowthConfigV2(
                initial_layers=grower_initial,
                max_layers=int(growth_cfg.get("max_layers", 20)),
                min_steps_between_growths=int(growth_cfg.get("min_steps_between_growths", 100_000)),
                grow_trigger_lp_threshold=float(growth_cfg.get("grow_trigger_lp_threshold", 0.05)),
                grow_trigger_coverage=float(growth_cfg.get("grow_trigger_coverage", 0.3)),
                distill_steps=int(growth_cfg.get("distill_steps", 100)),
                distill_lr=float(growth_cfg.get("distill_lr", 1e-3)),
            ),
        ).to(device)
        logger.info("ModelGrowerV2 enabled (max_layers=%d)", growth_cfg.get("max_layers", 20))

    if xmodal_cfg and bool(xmodal_cfg.get("enabled", False)):
        xmodal_manager = CrossModalManager(
            touch_dim=int(xmodal_cfg.get("touch_dim", 6)),
            plan_dim=int(xmodal_cfg.get("plan_dim", 128)),
            obs_dim=64 * 64 * 3,
            lang_dim=int(xmodal_cfg.get("lang_dim", 3584)),
        ).to(device)
        logger.info("CrossModalManager enabled")

    # --- Creativity orchestrator
    creativity_orch: CreativityOrchestrator | None = None
    creativity_cfg = config.get("creativity")
    if creativity_cfg and bool(creativity_cfg.get("enabled", False)):
        creativity_orch = CreativityOrchestrator(
            d_model=int(model_cfg.get("hidden_size", 128)),
            num_actions=num_actions,
            trigger_every_steps=int(creativity_cfg.get("trigger_every_steps", 1000)),
            max_ideas=int(creativity_cfg.get("max_ideas", 200)),
        ).to(device)
        health.register("creativity", creativity_orch)
        logger.info("CreativityOrchestrator enabled (trigger_every=%d)", creativity_orch._trigger_every)

    # --- Phase 9: LLM Fusion ---
    llm_fusion: LLMFusionBridge | None = None
    llm_cfg = config.get("llm_fusion")
    if llm_cfg and bool(llm_cfg.get("enabled", False)):
        try:
            obs_dim_flat = int(np.prod(obs_shape))
            llm_fusion = LLMFusionBridge(
                llm_model_name=str(llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")),
                slot_dim=int(model_cfg.get("slot_dim", int(model_cfg.get("hidden_size", 128)))),
                num_slots=int(model_cfg.get("slot_num_slots", 7)),
                num_actions=num_actions,
                obs_dim=obs_dim_flat,
                llm_max_new_tokens=int(llm_cfg.get("max_new_tokens", 64)),
                llm_call_interval=int(llm_cfg.get("call_interval_steps", 50)),
            ).to(device)
            logger.info("LLM Fusion: %s (available=%s)", llm_cfg.get("model"), llm_fusion.is_available)
        except Exception as exc:
            logger.warning("LLM Fusion load failed (%s)", exc)

    # --- Enhanced Memory System ---
    mem_cfg = config.get("developmental_memory")
    memory_manager: MemoryManager | None = None
    if mem_cfg and bool(mem_cfg.get("enabled", False)):
        memory_manager = MemoryManager(
            d_model=int(model_cfg.get("hidden_size", 128)),
            episodic_max=int(mem_cfg.get("episodic_max", 10000)),
            semantic_max=int(mem_cfg.get("semantic_max", 1000)),
            autobiographical_max=int(mem_cfg.get("autobiographical_max", 100)),
            surprise_threshold=float(mem_cfg.get("surprise_threshold", 0.01)),
            consolidation_every_steps=int(mem_cfg.get("consolidation_every_steps", 50000)),
        ).to(device)
        health.register("memory", memory_manager)
        logger.info("MemoryManager enabled (episodic=%d, semantic=%d)",
                     mem_cfg.get("episodic_max", 10000),
                     mem_cfg.get("semantic_max", 1000))

    # --- Theory of Mind ---
    tom_cfg = config.get("theory_of_mind")
    theory_of_mind: TheoryOfMind | None = None
    if tom_cfg and bool(tom_cfg.get("enabled", False)):
        theory_of_mind = TheoryOfMind(
            d_model=int(model_cfg.get("hidden_size", 128)),
            num_actions=num_actions,
            num_slots=int(model_cfg.get("slot_num_slots", 7)),
        ).to(device)
        logger.info("TheoryOfMind enabled")

    # --- Homeostatic Drives ---
    drives_cfg = config.get("homeostatic_drives")
    homeostatic_drives: HomeostaticDrives | None = None
    if drives_cfg and bool(drives_cfg.get("enabled", False)):
        homeostatic_drives = HomeostaticDrives(
            curiosity_decay=float(drives_cfg.get("curiosity_decay", 0.002)),
            competence_decay=float(drives_cfg.get("competence_decay", 0.001)),
            social_decay=float(drives_cfg.get("social_decay", 0.003)),
            safety_decay=float(drives_cfg.get("safety_decay", 0.001)),
            rest_decay=float(drives_cfg.get("rest_decay", 0.005)),
        ).to(device)
        health.register("drives", homeostatic_drives)
        logger.info("HomeostaticDrives enabled (5 drives)")

    # --- Long-Range Planner ---
    planner_cfg = config.get("long_range_planner")
    long_range_planner: LongRangePlanner | None = None
    if planner_cfg and bool(planner_cfg.get("enabled", False)):
        long_range_planner = LongRangePlanner(
            num_actions=num_actions,
            max_depth=int(planner_cfg.get("max_depth", 10)),
            max_nodes=int(planner_cfg.get("max_nodes", 500)),
            num_simulations=int(planner_cfg.get("num_simulations", 50)),
        ).to(device)
        logger.info("LongRangePlanner enabled (depth=%d)", planner_cfg.get("max_depth", 10))

    # --- Emotion System ---
    emotion_cfg = config.get("emotion_system")
    emotion_system: EmotionSystem | None = None
    if emotion_cfg and bool(emotion_cfg.get("enabled", False)):
        emotion_system = EmotionSystem(
            pleasure_decay=float(emotion_cfg.get("pleasure_decay", 0.01)),
            frustration_decay=float(emotion_cfg.get("frustration_decay", 0.02)),
            surprise_decay=float(emotion_cfg.get("surprise_decay", 0.05)),
            fear_decay=float(emotion_cfg.get("fear_decay", 0.03)),
        ).to(device)
        health.register("emotions", emotion_system)
        logger.info("EmotionSystem enabled")

    # --- Concept Graph ---
    graph_cfg = config.get("concept_graph")
    concept_graph: ConceptGraph | None = None
    if graph_cfg and bool(graph_cfg.get("enabled", False)):
        concept_graph = ConceptGraph(
            d_model=int(model_cfg.get("hidden_size", 128)),
            max_nodes=int(graph_cfg.get("max_nodes", 1000)),
            max_edges=int(graph_cfg.get("max_edges", 5000)),
        ).to(device)
        health.register("concept_graph", concept_graph)
        logger.info("ConceptGraph enabled (max_nodes=%d, max_edges=%d)",
                     graph_cfg.get("max_nodes", 1000), graph_cfg.get("max_edges", 5000))

    # --- Self-Reflection Validator + Concept Clusterer ---
    reflection_validator: SelfReflectionValidator | None = None
    concept_clusterer: ConceptClusterer | None = None
    meta_cfg = config.get("metacognition_v2")
    if meta_cfg and bool(meta_cfg.get("reflection_enabled", False)):
        reflection_validator = SelfReflectionValidator(
            max_records=int(meta_cfg.get("reflection_max", 500)),
        ).to(device)
        health.register("reflection_validator", reflection_validator)
        logger.info("SelfReflectionValidator enabled")
    if meta_cfg and bool(meta_cfg.get("clustering_enabled", False)):
        concept_clusterer = ConceptClusterer(
            min_cluster_size=int(meta_cfg.get("cluster_min_size", 3)),
            min_shared_edges=int(meta_cfg.get("cluster_min_edges", 3)),
            max_categories=int(meta_cfg.get("cluster_max_cats", 50)),
            cluster_every_steps=int(meta_cfg.get("cluster_every_steps", 5000)),
        ).to(device)
        health.register("concept_clusterer", concept_clusterer)
        logger.info("ConceptClusterer enabled (every %d steps)", meta_cfg.get("cluster_every_steps", 5000))

    # --- IQ Boost: 6 tier-1 upgrades ---
    iq_cfg = config.get("iq_boost")
    cross_domain: CrossDomainTransfer | None = None
    deep_fusion: DeepMultiModal | None = None
    temporal_reasoner: TemporalReasoner | None = None
    cf_regret: CounterfactualRegret | None = None
    curiosity_director: CuriosityDirector | None = None
    value_system: ValueSystem | None = None

    if iq_cfg and bool(iq_cfg.get("enabled", False)):
        cross_domain = CrossDomainTransfer(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        health.register("cross_domain", cross_domain)
        deep_fusion = DeepMultiModal(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        temporal_reasoner = TemporalReasoner(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        cf_regret = CounterfactualRegret().to(device)
        health.register("regrets", cf_regret)
        curiosity_director = CuriosityDirector(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        value_system = ValueSystem(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        health.register("values", value_system)
        logger.info("IQ Boost: 6 modules enabled (cross-domain, deep-fusion, temporal, "
                     "regret, curiosity-director, value-system)")

    # --- Abstract Reasoning: micro-math + identity narrative ---
    reason_cfg = config.get("abstract_reasoning")
    micro_math: MicroPrologMath | None = None
    identity_narrative: IdentityNarrative | None = None
    if reason_cfg and bool(reason_cfg.get("enabled", False)):
        micro_math = MicroPrologMath()
        identity_narrative = IdentityNarrative(
            d_model=int(model_cfg.get("hidden_size", 128)),
        ).to(device)
        logger.info("AbstractReasoning: MicroPrologMath + IdentityNarrative enabled")

    # --- Tier 2 Cognitive: metaphor, belief, moral, humor ---
    tier2_cfg = config.get("tier2_cognitive")
    analogizer: Analogizer | None = None
    belief_depth2: BeliefDepth2 | None = None
    moral_connector: MoralConnector | None = None
    surprise_humor: SurpriseHumor | None = None
    if tier2_cfg and bool(tier2_cfg.get("enabled", False)):
        analogizer = Analogizer(d_model=int(model_cfg.get("hidden_size", 128))).to(device)
        belief_depth2 = BeliefDepth2()
        moral_connector = MoralConnector().to(device)
        surprise_humor = SurpriseHumor().to(device)
        health.register("surprise_humor", surprise_humor)
        logger.info("Tier2 Cognitive: Analogizer + BeliefDepth2 + MoralConnector + SurpriseHumor")

    # --- Neuro-Symbolic Bridge ---
    bridge_cfg = config.get("neuro_symbolic_bridge")
    causal2prolog: Causal2Prolog | None = None
    number2math: Number2Math | None = None
    schema_detector: SchemaDetector | None = None
    if bridge_cfg and bool(bridge_cfg.get("enabled", False)):
        causal2prolog = Causal2Prolog()
        number2math = Number2Math()
        schema_detector = SchemaDetector()
        logger.info("NeuroSymbolicBridge: Causal2Prolog + Number2Math + SchemaDetector")

    # --- Program Synthesis + Active Experimentation + Temporal Abstraction ---
    synth_cfg = config.get("program_synthesis")
    program_synth: ProgramSynthesizer | None = None
    active_experimenter: ActiveExperimenter | None = None
    temporal_abstractor: TemporalAbstractor | None = None
    if synth_cfg and bool(synth_cfg.get("enabled", False)):
        program_synth = ProgramSynthesizer()
        active_experimenter = ActiveExperimenter(
            test_every_steps=int(synth_cfg.get("test_every_steps", 2000)),
        ).to(device)
        temporal_abstractor = TemporalAbstractor(
            max_patterns=int(synth_cfg.get("max_patterns", 100)),
            min_occurrences=int(synth_cfg.get("min_occurrences", 3)),
        )
        logger.info("ProgramSynthesis: Synthesizer + ActiveExperimenter + TemporalAbstractor")

    # --- Counterfactual Planning ---
    plan_cfg = config.get("counterfactual_planner")
    cf_planner: CounterfactualPlanner | None = None
    if plan_cfg and bool(plan_cfg.get("enabled", False)):
        cf_planner = CounterfactualPlanner(
            num_actions=num_actions,
            num_candidates=int(plan_cfg.get("num_candidates", 5)),
            max_imagine_steps=int(plan_cfg.get("max_imagine_steps", 8)),
        ).to(device)
        logger.info("CounterfactualPlanner enabled (candidates=%d)", plan_cfg.get("num_candidates", 5))

    # --- Marginal Gains ---
    gap_cfg = config.get("marginal_gains")
    compositional_test: CompositionalTester | None = None
    lp_tracker: LearningProgressTracker | None = None
    if gap_cfg and bool(gap_cfg.get("enabled", False)):
        compositional_test = CompositionalTester()
        lp_tracker = LearningProgressTracker().to(device)
        logger.info("MarginalGains: CompositionalTester + LP Tracker")

    # --- Visual Analyzer ---
    visual_analyzer: VisualAnalyzer | None = None
    if config.get("visual_analyzer", {}).get("enabled", False):
        visual_analyzer = VisualAnalyzer(
            slot_dim=int(model_cfg.get("slot_dim", int(model_cfg.get("hidden_size", 128)))),
            num_slots=int(model_cfg.get("slot_num_slots", 7)),
        ).to(device)
        logger.info("VisualAnalyzer enabled (color/shape/size/texture/motion)")

    # --- Phase 1+: Imagination Trainer (Dreamer-style) ---
    imagination_cfg = config.get("imagination")
    imagination_trainer: ImaginationTrainer | None = None
    imagination_last_loss: dict[str, float] = {}
    if imagination_cfg and bool(imagination_cfg.get("enabled", False)):
        imagination_trainer = ImaginationTrainer(
            config=ImaginationConfig(
                imagination_horizon=int(imagination_cfg.get("imagination_horizon", 8)),
                imagination_batch=int(imagination_cfg.get("imagination_batch", 32)),
                discount=float(imagination_cfg.get("discount", 0.99)),
                actor_entropy_scale=float(imagination_cfg.get("actor_entropy_scale", 0.01)),
                critic_loss_scale=float(imagination_cfg.get("critic_loss_scale", 0.5)),
                update_every_steps=int(imagination_cfg.get("update_every_steps", 16)),
                min_replay_size=int(imagination_cfg.get("min_replay_size", 256)),
                lambda_gae=float(imagination_cfg.get("lambda_gae", 0.95)),
                lr=float(imagination_cfg.get("lr", 3e-4)),
            ),
            device=device,
        )
        logger.info("ImaginationTrainer enabled (horizon=%d, batch=%d)",
                     imagination_cfg.get("imagination_horizon", 8),
                     imagination_cfg.get("imagination_batch", 32))

    # --- Phase 1+: Intention Achievement Curiosity ---
    intention_cfg = config.get("intention")
    intention_curiosity: IntentionCuriosity | None = None
    if intention_cfg and bool(intention_cfg.get("enabled", False)):
        intention_curiosity = IntentionCuriosity(
            IntentionConfig(
                error_mode=str(intention_cfg.get("error_mode", "kl")),
                coef=float(intention_cfg.get("coef", 0.5)),
                ema_decay=float(intention_cfg.get("ema_decay", 0.99)),
                reward_clip=float(intention_cfg.get("reward_clip", 5.0)),
                min_steps_before_active=int(intention_cfg.get("min_steps_before_active", 1000)),
            ),
        ).to(device)
        logger.info("IntentionCuriosity enabled (mode=%s, coef=%.2f)",
                     intention_cfg.get("error_mode", "kl"),
                     intention_cfg.get("coef", 0.5))

    # --- Phase 1+: Knowledge Gap Detector ---
    kgap_cfg = config.get("knowledge_gap")
    knowledge_gap: KnowledgeGapDetector | None = None
    if kgap_cfg and bool(kgap_cfg.get("enabled", False)):
        knowledge_gap = KnowledgeGapDetector(
            KnowledgeGapConfig(
                num_slots=int(model_cfg.get("slot_num_slots", 7)),
                ema_decay=float(kgap_cfg.get("ema_decay", 0.99)),
                gap_threshold=float(kgap_cfg.get("gap_threshold", 1.5)),
                boost_factor=float(kgap_cfg.get("boost_factor", 2.0)),
            ),
        ).to(device)
        logger.info("KnowledgeGapDetector enabled (num_slots=%d, boost=%.1f)",
                     kgap_cfg.get("num_slots", model_cfg.get("slot_num_slots", 7)),
                     kgap_cfg.get("boost_factor", 2.0))

    # --- Phase 3+: Social Curiosity ---
    social_cur_cfg = config.get("social_curiosity")
    social_curiosity: SocialCuriosity | None = None
    if social_cur_cfg and bool(social_cur_cfg.get("enabled", False)):
        obs_dim_flat = int(np.prod(obs_shape))
        social_curiosity = SocialCuriosity(
            obs_dim=obs_dim_flat,
            num_actions=num_actions,
            config=SocialCuriosityConfig(
                action_predictor_hidden=int(social_cur_cfg.get("action_predictor_hidden", 64)),
                social_coef=float(social_cur_cfg.get("social_coef", 0.3)),
                ema_decay=float(social_cur_cfg.get("ema_decay", 0.99)),
                reward_clip=float(social_cur_cfg.get("reward_clip", 5.0)),
            ),
        ).to(device)
        logger.info("SocialCuriosity enabled (coef=%.2f)", social_cur_cfg.get("social_coef", 0.3))

    # --- Phase 4+: Audio Encoder ---
    audio_cfg = config.get("audio")
    audio_encoder: AudioEncoder | None = None
    if audio_cfg and bool(audio_cfg.get("enabled", False)):
        try:
            audio_encoder = AudioEncoder(
                AudioEncoderConfig(
                    sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                    n_mels=int(audio_cfg.get("n_mels", 64)),
                    d_model=int(model_cfg.get("hidden_size", 128)),
                    hidden=int(audio_cfg.get("hidden", 64)),
                ),
                device=device,
            ).to(device)
            logger.info("AudioEncoder enabled (available=%s, params=%d)",
                         audio_encoder.is_available,
                         sum(p.numel() for p in audio_encoder.parameters()))
        except Exception as exc:
            logger.warning("AudioEncoder init failed (%s)", exc)

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
        # Restore the autonomous grower's state so its layer count and growth
        # bookkeeping stay in sync with the (resumed) model. Without this the
        # grower is recreated as `initial_layers` (2) while the model may already
        # be 3/4/… layers, causing a spurious no-op growth on the first check —
        # or, worse, calling `_create_larger_model(model, 3)` on a 4-layer model
        # and silently DROPPING a layer on the next resume.
        if model_grower_v2 is not None and payload.get("model_grower_v2_state"):
            try:
                model_grower_v2.load_state_dict(payload["model_grower_v2_state"])
                logger.info(
                    "Resumed ModelGrowerV2 state (layers=%d, growth_count=%d)",
                    len(model_grower_v2), model_grower_v2._growth_count,
                )
            except (ValueError, RuntimeError, KeyError) as exc:
                logger.warning(
                    "ModelGrowerV2 state mismatch on resume (%s); starting fresh.", exc)
            # Safety sync: grower's layer count MUST match the actual model.
            try:
                actual = int(model.backbone.n_layers)
                if actual != model_grower_v2._current_layers:
                    logger.warning(
                        "ModelGrowerV2 layer desync (grower=%d, model=%d); "
                        "syncing grower to model.",
                        model_grower_v2._current_layers, actual)
                    model_grower_v2._current_layers = actual
            except AttributeError:
                pass
        # TODO(Phase5+): restore extra states (RND, EWC Fisher, coverage, skills,
        # WM, imagination, symbolic, etc. — ~40 keys) for homogeneous resume.
        # Currently ~40 extra keys are written (lines ~2680-2740) but never read
        # back. This is safe for cross-stage resume (architecture changes → fresh
        # modules) but needed for long-running homogeneous Phases (Phase 5+).
        resumed_stage = int(payload.get("stage", stage))
        resumed_step = int(payload.get("step", 0))
        # Enforce a growth cooldown from the resumed step so the transient
        # resume spike / exploration-reset dip (inflated-or-collapsed first-step
        # mean_return) cannot immediately drive a 3->4 (or N->N+1) growth. The
        # natural plateau (rmax decay) + 1M cooldown then gate the next *real*
        # growth. NOTE: this must run after `resumed_step` is resolved (above),
        # not while `state.step` is still 0.
        if model_grower_v2 is not None:
            try:
                model_grower_v2._last_growth_step = max(
                    int(getattr(model_grower_v2, "_last_growth_step", -10**9)),
                    resumed_step,
                )
            except Exception:  # noqa: BLE001
                pass
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

    # P1: reward/return EMA normalization (PopArt-lite)
    # Prevents dead-policy lock when extrinsic rewards are small and uniform.
    reward_ema = ReturnNormalizer(alpha=0.01)
    ppo_minibatches = int(train_cfg.get("ppo_minibatches", 8))  # P2: mini-batch count

    # Stage 1 knobs
    intrinsic_coef = float(intrinsic_cfg.get("reward_coef", 0.1)) if intrinsic_cfg else curiosity_coef if curiosity_mode != "none" else 0.0
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
    # Per-episode trajectory buffers for rule extraction and LLM reflection
    rollout_hidden_states: list[torch.Tensor] = []
    rollout_actions: list[int] = []
    rollout_rewards: list[float] = []
    rollout_trajectory: list[dict] = []
    _collect_cognitive = (symbolic_layer is not None) or (reflection_loop is not None)

    # Phase 0 knobs
    num_sense_train_every = int(num_sense_cfg.get("train_every_episodes", 10)) if num_sense_cfg else 0
    rule_induce_every = int(rule_ind_cfg.get("induction_every_episodes", 20)) if rule_ind_cfg else 0
    causal_intervene_every = int(causal_cfg.get("intervene_every_steps", 500)) if causal_cfg else 0
    xmodal_train_every = int(xmodal_cfg.get("train_every_steps", 1000)) if xmodal_cfg else 0
    num_sense_last_train = 0
    rule_induce_last_ep = 0
    causal_last_intervene = 0
    xmodal_last_train = 0
    # Trajectory buffers for rule induction predicates
    episode_predicates: list[dict[str, bool]] = []
    episode_true_counts: list[int] = []

    # --- Phase 0 knobs (from training) ---
    imagination_update_every = int(imagination_cfg.get("update_every_steps", 16)) if imagination_cfg else 0
    knowledge_gap_update_every = int(kgap_cfg.get("update_every_steps", 50)) if kgap_cfg else 0
    knowledge_gap_last_update = 0
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
    _prof_env = _prof_model = _prof_cog = _prof_buf = _prof_total = 0.0
    _prof_n = 0
    while state.step < total_steps:
        buffer.clear()

        # Collect a rollout of exactly `rollout_capacity` steps
        while not buffer.full():
            t0 = time.perf_counter()
            obs_t = _obs_to_tensor(obs, device)  # (N,3,H,W) for vec; (1,3,H,W) single
            with torch.no_grad():
                t_model = time.perf_counter()
                if _collect_cognitive and n_envs == 1:
                    logits, value, hidden = model(obs_t, return_hidden=True)
                else:
                    logits, value = model(obs_t)  # value: (N,)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()              # (N,)
                logprob = dist.log_prob(action)     # (N,)
            t_model_end = time.perf_counter()

            # --- Phase 0: collect slot output for number sense + rule predicates
            # (single-env only; per-env predicate buffers are not maintained for vec) ---
            slot_states_for_step: torch.Tensor | None = None
            if (number_sense is not None or rule_engine is not None) and model.use_slots and n_envs == 1:
                with torch.no_grad():
                    slot_states_for_step = model._last_slots.squeeze(0)  # (num_slots, d_model)
                if rule_engine is not None:
                    preds = rule_engine.extract_predicates(slot_states_for_step.unsqueeze(0))
                    episode_predicates.append(preds)

            te = time.perf_counter()
            a_np = action.cpu().numpy()
            step_out = env.step(a_np if n_envs > 1 else int(a_np.item()))
            ta = time.perf_counter()
            done_arr = np.asarray(step_out.terminated) | np.asarray(step_out.truncated)
            extrinsic_r = np.asarray(step_out.reward, dtype=np.float32)  # (N,) or scalar
            int_r = np.zeros(n_envs, dtype=np.float32)
            t_cog = time.perf_counter()
            total_r = extrinsic_r.copy()  # (N,) per-env reward accumulator

            # --- Collect hidden state for cognitive modules (single-env only) ---
            if _collect_cognitive and n_envs == 1:
                rollout_hidden_states.append(
                    hidden.squeeze(0).detach().cpu()
                    if hidden.dim() == 2 else hidden.detach().cpu()
                )
                rollout_actions.append(int(action.item()))
                rollout_rewards.append(float(total_r))
                rollout_trajectory.append({
                    "action": int(action.item()),
                    "reward": float(total_r),
                })

            # --- Homeostatic drives: compute intrinsic motivation ---
            if homeostatic_drives is not None and n_envs == 1:
                try:
                    success = extrinsic_r > 0.5
                    danger = 0.0
                    if hasattr(env, '_agent'):
                        ax, ay = env._agent.x, env._agent.y
                        hw = getattr(env, '_hw', 1.0)
                        wall_dist = hw - max(abs(ax), abs(ay), 0.05)
                        danger = max(0.0, 1.0 - wall_dist)
                    movement = float(abs(getattr(env._agent, 'vx', 0)) + abs(getattr(env._agent, 'vy', 0)))
                    social_dist = 1.0  # default far
                    drive_rewards = homeostatic_drives.tick(
                        novelty=int_r if curiosity_mode != "none" else 0.0,
                        success=success,
                        caregiver_proximity=social_dist,
                        danger_level=danger,
                        movement_level=min(1.0, movement),
                    )
                    total_r += drive_rewards.get("total", 0.0) * 0.5
                except Exception:
                    pass

            # --- Emotion system: update from experience ---
            if emotion_system is not None and n_envs == 1:
                try:
                    emotion_system.update(
                        reward=float(extrinsic_r),
                        surprise=float(np.asarray(int_r).mean()) if curiosity_mode != "none" else 0.0,
                        danger_level=0.0,
                        success=extrinsic_r > 0.5,
                        episode_done=bool(step_out.terminated or step_out.truncated),
                    )
                except Exception:
                    pass
            if curiosity_mode == "rssm_uncertainty" and wm is not None:
                with torch.no_grad():
                    B = obs_t.shape[0]
                    obs_flat = obs_t.float().reshape(B, -1) / 255.0
                    wm_state = wm.initial_state(B, device)
                    dummy_action = F.one_hot(action, num_actions).float()  # (B, A)
                    wm_state, _ = wm.imagine_step(wm_state, dummy_action)
                    pred_obs = wm.decode(wm_state)
                    int_r = F.mse_loss(pred_obs, obs_flat, reduction="none").mean(dim=-1).detach().cpu().numpy()  # (B,)
                intrinsic_coef = curiosity_coef
                total_r = extrinsic_r + intrinsic_coef * int_r

            # --- Phase 1+: intention achievement curiosity ---
            if (intention_curiosity is not None and wm is not None
                    and intention_curiosity.is_active()):
                try:
                    with torch.no_grad():
                        B = obs_t.shape[0]
                        obs_flat = obs_t.float().reshape(B, -1) / 255.0
                        wm_state = wm.initial_state(B, device)
                        action_onehot = F.one_hot(action, num_actions).float()  # (B, A)
                        intent_r = intention_curiosity.intention_reward(
                            wm, wm_state, action_onehot, obs_flat,
                        )
                        intent_r = np.asarray(intent_r).reshape(B)
                        int_r = np.maximum(int_r, intent_r)
                        total_r = extrinsic_r + curiosity_coef * int_r
                    intention_curiosity.step()
                except Exception:
                    pass

            # --- Phase 1+: knowledge gap boost on curiosity (single-env only) ---
            if (knowledge_gap is not None and model.use_slots
                    and n_envs == 1 and float(int_r) > 0):
                try:
                    with torch.no_grad():
                        slot_out = model.encoder(obs_t).squeeze(0)
                        boost = knowledge_gap.get_gap_boost(slot_out.unsqueeze(0))
                        total_r += (boost.item() - 1.0) * curiosity_coef * int_r
                except Exception:
                    pass

            # --- Concept graph: bind cross-modal observations (single-env only) ---
            if concept_graph is not None and model.use_slots and n_envs == 1 and state.step % 100 == 0:
                try:
                    with torch.no_grad():
                        slot_out = model.encoder(obs_t).squeeze(0)
                        modality_data: dict[str, Any] = {
                            "slot": slot_out.mean(dim=0),
                        }
                        if xmodal_manager is not None:
                            prop = torch.tensor(getattr(step_out, 'proprio', [0.0]*12),
                                              device=device).float()
                            touch_emb = xmodal_manager.touch_bridge.touch_to_lang(prop.unsqueeze(0))
                            modality_data["touch"] = touch_emb.squeeze(0)
                        concept_graph.bind_cross_modal(modality_data, step=state.step)
                except Exception:
                    pass
            if memory_manager is not None and n_envs == 1:
                try:
                    surprise_val = int_r if curiosity_mode != "none" else 0.0
                    current_ep = env.summary().get("episodes", 0)
                    tags = []
                    if extrinsic_r > 0.5:
                        tags.append("high_reward")
                    if hasattr(model, 'use_slots') and model.use_slots:
                        with torch.no_grad():
                            slot_out = model.encoder(obs_t)
                            hidden_for_mem = slot_out.squeeze(0).mean(dim=0)
                    else:
                        hidden_for_mem = torch.randn(model.d_model, device=device)
                    memory_manager.store_experience(
                        hidden_state=hidden_for_mem,
                        action=int(action.item()),
                        reward=float(total_r),
                        surprise=float(np.asarray(surprise_val).mean()),
                        global_step=state.step,
                        episode_id=int(current_ep),
                        tags=tags,
                    )
                except Exception:
                    pass

            # --- Phase 0: creativity reward (single-env only) ---
            if creativity_orch is not None and creativity_orch.should_trigger(state.step) and n_envs == 1:
                try:
                    slot_input = model.encoder(obs_t).squeeze(0) if model.use_slots else None
                    wm_state = wm.initial_state(1, device) if wm is not None else None
                    creativity_result = creativity_orch.creative_cycle(
                        step=state.step,
                        slot_states=slot_input if slot_input is not None else torch.randn(7, model.d_model, device=device),
                        skill_library=skills,
                        world_model=wm,
                        causal_graph=causal_disc,
                        wm_state=wm_state,
                        divergent_gen=divergent_gen,
                        transformational=transformational,
                    )
                    total_r = creativity_orch.add_creativity_reward(total_r, creativity_result)
                except Exception:
                    pass

            # --- Phase 9: LLM scene description + policy modulation (single-env only) ---
            if llm_fusion is not None and llm_fusion.is_available and model.use_slots and n_envs == 1:
                with torch.no_grad():
                    slot_out = model.encoder(obs_t)
                    # Get scene description (cached, not called every step)
                    _scene_desc = llm_fusion.describe_scene(slot_out)
                    # Modulate policy with LLM reasoning
                    logits = llm_fusion.modulate_policy(slot_out, logits)
            elif rnd is not None and n_envs == 1:
                with torch.no_grad():
                    int_r = float(rnd.intrinsic_reward(obs_t).item())
                total_r = extrinsic_r + intrinsic_coef * int_r

            # --- Count-based exploration bonus: state-dependent floor on the
            # reward signal (prevents the 3D deadlock when env reward is
            # sparse). Its magnitude already carries `coef`, so it is added
            # directly. The term varies per state and with visitation history,
            # so the value head cannot fit it away -> advantages keep a
            # residual signal.
            if expl_bonus is not None:
                eb = expl_bonus.bonus(obs_t).reshape(-1).detach().cpu().numpy()  # (B,)
                total_r = total_r + eb
                expl_bonus.update(obs_t)

            t_cog_end = time.perf_counter()
            t_buf = time.perf_counter()
            buffer.add(
                obs=obs,
                action=a_np,
                logprob=logprob.cpu().numpy(),
                value=value.cpu().numpy(),
                reward=total_r,
                done=done_arr,
            )
            t_buf_end = time.perf_counter()

            # --- Stage 1: coverage tracking ---
            if coverage is not None:
                coverage.touch(obs)

            # --- Stage 1: push transition to bounded replay ---
            if replay is not None:
                if obs.ndim == 4:  # vectorized env
                    so = np.asarray(step_out.obs)
                    tr = total_r if isinstance(total_r, np.ndarray) else np.full(n_envs, float(total_r), dtype=np.float32)
                    for i in range(n_envs):
                        replay.add(Transition(
                            obs=obs[i],
                            action=int(a_np[i]),
                            reward=float(tr[i]),
                            next_obs=so[i],
                            done=bool(done_arr[i]),
                            priority=1.0 + abs(int_r[i]) if (rnd is not None or curiosity_mode != "none") else 1.0,
                        ))
                else:
                    total_r_s = float(np.asarray(total_r).reshape(-1)[0])
                    int_r_s = float(np.asarray(int_r).reshape(-1)[0])
                    replay.add(Transition(
                        obs=obs,
                        action=int(action.item()),
                        reward=total_r_s,
                        next_obs=step_out.obs,
                        done=bool(done_arr),
                        priority=1.0 + abs(int_r_s) if (rnd is not None or curiosity_mode != "none") else 1.0,
                    ))

            # --- Stage 1: RND predictor SGD ---
            if rnd is not None and rnd_update_every > 0 and state.step % rnd_update_every == 0 and n_envs == 1:
                rnd.update(obs_t)

            obs = step_out.obs
            state.step += n_envs

            # --- lightweight per-step profiler (cloud bottleneck diagnosis) ---
            tb = time.perf_counter()
            _prof_env += (ta - te)
            _prof_model += (t_model_end - t_model)
            _prof_cog += (t_cog_end - t_cog)
            _prof_buf += (t_buf_end - t_buf)
            _prof_total += (tb - t0)
            _prof_n += 1
            if _prof_n >= 1000:
                _n = max(_prof_n, 1)
                _pt = 1000.0 * _prof_total / _n
                _pe = 1000.0 * _prof_env / _n
                _pm = 1000.0 * _prof_model / _n
                _pc = 1000.0 * _prof_cog / _n
                _pb = 1000.0 * _prof_buf / _n
                _po = _pt - _pe - _pm - _pc - _pb
                logger.info(
                    "PROF per_step=%.1fms env=%.1f(%.0f%%) model=%.1f(%.0f%%) cog=%.1f(%.0f%%) buf=%.1f(%.0f%%) other=%.1f(%.0f%%)",
                    _pt,
                    _pe, 100.0 * _pe / max(_pt, 1e-9),
                    _pm, 100.0 * _pm / max(_pt, 1e-9),
                    _pc, 100.0 * _pc / max(_pt, 1e-9),
                    _pb, 100.0 * _pb / max(_pt, 1e-9),
                    _po, 100.0 * _po / max(_pt, 1e-9),
                )
                _prof_env = _prof_model = _prof_cog = _prof_buf = _prof_total = 0.0
                _prof_n = 0

            watcher.tick(step=state.step)

            # --- Stage 4: extract skill on successful episode end ---
            if skills is not None and n_envs == 1 and (step_out.terminated or step_out.truncated):
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
                        # Compute per-step advantages from rollout buffer
                        if len(rollout_hidden_states) > 0 and len(buffer.rewards) > 0:
                            n_rollout = min(len(rollout_hidden_states), len(buffer.rewards))
                            buf_rewards = buffer.rewards[:n_rollout].cpu().tolist()
                            buf_vals = buffer.values[:n_rollout].cpu().tolist()
                            # Simple advantage: reward - baseline for steps where baseline exists
                            with torch.no_grad():
                                # Use mean return as crude advantage proxy
                                mean_ret_val = ep_ret / max(1, n_rollout)
                                step_advantages = [r - mean_ret_val for r in buf_rewards]
                            symbolic_layer.extract_rules(
                                hidden_states=rollout_hidden_states[:n_rollout],
                                actions=rollout_actions[:n_rollout],
                                rewards=rollout_rewards[:n_rollout],
                                advantages=step_advantages,
                                descriptions=[f"IF see {curr_active_task.tag if curr_active_task else 'env'} THEN act"],
                            )
                        else:
                            symbolic_layer.extract_rules(
                                hidden_states=rollout_hidden_states or [torch.from_numpy(obs).to(device)],
                                actions=rollout_actions or [int(action.item())],
                                rewards=rollout_rewards or [float(ep_ret)],
                            )
                    except Exception:
                        pass

                # --- Stage 7: reflection after episode ---
                if reflection_loop is not None:
                    try:
                        # Record steps first
                        if rollout_trajectory:
                            for i, t_step in enumerate(rollout_trajectory[-32:]):
                                idx = max(0, len(rollout_hidden_states) - 32 + i) if rollout_hidden_states else 0
                                h = rollout_hidden_states[min(idx, len(rollout_hidden_states) - 1)] if rollout_hidden_states else torch.randn(384)
                                if i < len(rollout_hidden_states):
                                    h = rollout_hidden_states[max(0, len(rollout_hidden_states) - 32 + i)]
                                reflection_loop.record_step(
                                    hidden_state=h,
                                    action=int(t_step.get("action", 0)),
                                    reward=float(t_step.get("reward", 0)),
                                    done=True,
                                )
                        reflection = reflection_loop.end_episode(ep_ret)
                        if reflection is not None and inner_dialogue is not None:
                            lessons = inner_dialogue.generate(
                                reflection, trajectory_data=rollout_trajectory[-32:],
                            )
                            for lesson in lessons[:2]:
                                logger.info("[reflection] %s", lesson)
                    except Exception:
                        pass

                # Clear per-episode trajectory buffers
                rollout_hidden_states.clear()
                rollout_actions.clear()
                rollout_rewards.clear()
                rollout_trajectory.clear()

                # --- Temporal abstraction: extract patterns from episode ---
                if temporal_abstractor is not None:
                    try:
                        patterns = temporal_abstractor.extract_episode_patterns()
                        if patterns:
                            logger.info("[temporal] %s", temporal_abstractor.summary())
                    except Exception:
                        pass

                # --- Program synthesis: feed synthesized rules ---
                if program_synth is not None and rule_engine is not None:
                    try:
                        added = program_synth.feedback_to_rules(rule_engine)
                        if added > 0:
                            logger.info("[synthesis] %d new rules synthesized", added)
                    except Exception:
                        pass

                # --- Learning progress tracking ---
                if lp_tracker is not None:
                    try:
                        summary = env.summary()
                        lp_result = lp_tracker.update(
                            float(summary.get("mean_return", 0)), state.step,
                        )
                        if lp_result.get("curiosity_boost", 0) > 0:
                            intrinsic_coef = curiosity_coef + lp_result["curiosity_boost"]
                    except Exception:
                        pass

                # --- Compositional generalization test ---
                if (compositional_test is not None and concept_graph is not None
                        and state.step % 50000 < rollout_capacity):
                    try:
                        result = compositional_test.test(concept_graph)
                        if result["passed"]:
                            logger.info("[compositional] passed: %s", result.get("components", "?"))
                    except Exception:
                        pass

                # --- Phase 9: LLM reflection after episode ---
                if llm_fusion is not None and llm_fusion.is_available:
                    try:
                        scene = llm_fusion.describe_scene(
                            model.encoder(obs_t) if model.use_slots else obs_t
                        )
                        lessons = llm_fusion.reflect(ep_ret, scene)
                        for lesson in lessons[:2]:
                            logger.info("[llm_refl] %s", lesson)
                    except Exception:
                        pass

                # --- Enhanced memory: promote significant events to life story ---
                if memory_manager is not None and ep_ret > 0.5:
                    try:
                        scene_desc = llm_fusion.describe_scene(
                            model.encoder(obs_t) if model.use_slots and llm_fusion is not None else obs_t
                        ) if llm_fusion is not None and llm_fusion.is_available else "an episode"
                        task = curr_active_task.tag if curr_active_task else "sandbox"
                        description = f"Completed {task}: return={ep_ret:.2f}, scene={scene_desc[:60]}"
                        lesson = f"Learned to navigate {task}" if ep_ret > 0.7 else f"Explored {task}"
                        memory_manager.promote_to_life_event(
                            step=state.step,
                            description=description,
                            importance=float(ep_ret),
                            episode_id=int(env.summary().get("episodes", 0)),
                            lesson=lesson,
                        )
                    except Exception:
                        pass

                # --- Theory of Mind update ---
                if theory_of_mind is not None:
                    try:
                        theory_of_mind.reset_beliefs()
                    except Exception:
                        pass

                # --- IQ Boost: counterfactual regret recording ---
                if cf_regret is not None:
                    try:
                        cf_regret.record_regret(
                            actual_action=int(action.item()),
                            counterfactual_action=(int(action.item()) + 4) % 8,
                            actual_reward=float(ep_ret),
                            counterfactual_reward=float(ep_ret) * 0.5,
                            regret_magnitude=abs(float(ep_ret)) * 0.1,
                            step=state.step,
                        )
                        cf_regret.decay()
                    except Exception:
                        pass

                # --- IQ Boost: value judgment ---
                if value_system is not None and homeostatic_drives is not None:
                    try:
                        drive_levels = homeostatic_drives.drive_levels()
                        drive_deltas = {k: v - 0.5 for k, v in drive_levels.items()}
                        ctx = torch.zeros(int(model_cfg.get("hidden_size", 128)), device=device)
                        value_system.judge(
                            action=int(action.item()),
                            context_embedding=ctx,
                            drive_deltas=drive_deltas,
                            step=state.step,
                        )
                    except Exception:
                        pass
                if reflection_validator is not None and long_range_planner is not None:
                    try:
                        predicted_r = float(env.summary().get("mean_return", 0))
                        lesson = reflection_validator.reflect(
                            expected="plan_execution",
                            actual_reward=float(ep_ret),
                            predicted_reward=predicted_r,
                            step=state.step,
                            context=f"task_ep{env.summary().get('episodes',0)}",
                        )
                        if lesson:
                            logger.info("[self_reflect] %s", lesson[:100])
                    except Exception:
                        pass

                # --- Emotion: log dominant feeling ---
                if emotion_system is not None:
                    try:
                        dom = emotion_system.state.dominant
                        if dom != "neutral":
                            logger.debug("[emotion] feeling: %s", dom)
                    except Exception:
                        pass

                # --- Tier 2: moral evaluation ---
                if moral_connector is not None and value_system is not None and emotion_system is not None:
                    try:
                        drives = homeostatic_drives.drive_levels() if homeostatic_drives else {}
                        moral = moral_connector.evaluate_action(value_system, emotion_system, drives)
                        if moral["evaluation"] not in ("neutral", "maybe good"):
                            logger.info("[moral] %s: %s (conf=%.2f)",
                                       moral["evaluation"], moral["reason"], moral["confidence"])
                    except Exception:
                        pass

                # --- Tier 2: metaphor discovery ---
                if analogizer is not None and concept_graph is not None:
                    try:
                        nodes = list(concept_graph._nodes.values())
                        if len(nodes) > 20:
                            source = nodes[min(len(nodes) - 1, (state.step // 10000) % len(nodes))]
                            if source.name:
                                metaphor = analogizer.generate_metaphor_statement(concept_graph, source.name)
                                if "can't think" not in metaphor:
                                    logger.info("[metaphor] %s", metaphor)
                    except Exception:
                        pass

                # --- Phase 0: number sense training (episode end) ---
                if (number_sense is not None and num_sense_optimizer is not None
                        and len(episode_true_counts) > 0):
                    summary = env.summary()
                    current_ep = summary.get("episodes", 0)
                    if current_ep - num_sense_last_train >= num_sense_train_every:
                        slot_batch = model.encoder(obs_t)
                        true_count = torch.tensor([len(env._objects) if hasattr(env, '_objects') else 3],
                                                  dtype=torch.long, device=device)
                        n_loss = number_sense.loss(slot_batch, true_count)
                        num_sense_optimizer.zero_grad()
                        n_loss.backward()
                        num_sense_optimizer.step()
                        num_sense_last_train = current_ep
                episode_true_counts.clear()

                # --- Phase 0: rule induction (episode end) ---
                if rule_engine is not None and episode_predicates:
                    summary = env.summary()
                    current_ep = summary.get("episodes", 0)
                    if current_ep - rule_induce_last_ep >= rule_induce_every:
                        rule_engine.record_episode(
                            predicates_sequence=episode_predicates,
                            actions=rollout_actions if rollout_actions else [0],
                            outcome=float(ep_ret),
                        )
                        new_rules = rule_engine.induce_rules()
                        if new_rules:
                            logger.info("[rule_induction] %d new rules induced", len(new_rules))
                        rule_induce_last_ep = current_ep
                episode_predicates.clear()

                # --- Phase 0: causal discovery intervention ---
                if (causal_disc is not None and wm is not None
                        and state.step - causal_last_intervene >= causal_intervene_every):
                    if model.use_slots:
                        with torch.no_grad():
                            slot_states = model.encoder(obs_t).squeeze(0)
                            wm_state = wm.initial_state(1, device)
                            effects = causal_disc.intervene(
                                wm, wm_state, int(action.item()),
                                slot_states, state.step,
                            )
                    causal_last_intervene = state.step

            if state.step >= total_steps:
                break

        batch = buffer.as_batch()
        # Estimate last value per-env for GAE (vectorized over N envs)
        with torch.no_grad():
            _, last_value_t = model(_obs_to_tensor(obs, device))
            # P1: value head trained on normalized returns; denormalize
            # back to raw reward scale before GAE (see ReturnNormalizer).
            T = buffer._ptr
            N = buffer.n_envs
            values_raw = reward_ema.denormalize(batch.values).reshape(T, N)
            last_value_raw = reward_ema.denormalize(last_value_t).reshape(N)
        advantages2d, returns2d = compute_gae_vec(
            buffer.rewards[:T].cpu(),
            values_raw.cpu(),
            buffer.dones[:T].cpu(),
            last_value_raw.cpu(),
            gamma,
            gae_lambda,
        )
        advantages = advantages2d.reshape(-1).to(device)
        returns = returns2d.reshape(-1).to(device)
        # P1: update EMA on RAW returns, then produce normalized value target.
        reward_ema.update(returns)
        returns_norm = reward_ema.normalize(returns)
        # P0: standardize advantages with zero-variance guard (see helper).
        adv_norm = _normalize_advantages(advantages)

        # --- DIAG (plateau investigation): log advantage/reward variance ---
        if state.step % 5000 < rollout_capacity:
            with torch.no_grad():
                _rew = buffer.rewards[:T].float()
                _adv_std = float(advantages.std().item())
                _zero_var = bool(
                    advantages.numel() < 2
                    or (not math.isfinite(_adv_std))
                    or _adv_std < 1e-7
                )
                logger.info(
                    "[diag-adv] step=%d raw_adv_std=%.4g raw_adv_mean=%.4g "
                    "rew_std=%.4g rew_mean=%.4g returns_std=%.4g zero_var_guard=%s",
                    state.step, _adv_std, float(advantages.mean().item()),
                    float(_rew.std().item()), float(_rew.mean().item()),
                    float(returns.std().item()), _zero_var,
                )

        # P2: mini-batch PPO — split rollout into shuffled minibatches
        n = batch.obs.shape[0]
        indices = torch.randperm(n, device=device)
        mb_size = max(1, n // ppo_minibatches)
        ppo_losses: dict[str, list[float]] = {"policy": [], "value": [], "entropy": [],
                                               "kl": [], "clipfrac": [], "total": []}
        for _ in range(ppo_epochs):
            for start in range(0, n, mb_size):
                mb_idx = indices[start:start + mb_size]
                logits, values = model(batch.obs[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(batch.actions[mb_idx])
                ratio = (new_logprobs - batch.logprobs[mb_idx]).exp()
                unclipped = ratio * adv_norm[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv_norm[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns_norm[mb_idx])
                entropy = dist.entropy().mean()
                approx_kl = ((batch.logprobs[mb_idx] - new_logprobs).mean()).detach()
                clipfrac = ((ratio - 1.0).abs() > ppo_clip).float().mean().detach()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                if ewc is not None and ewc.has_consolidated():
                    loss = loss + ewc.penalty(model).to(loss.device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                ppo_losses["policy"].append(float(policy_loss.item()))
                ppo_losses["value"].append(float(value_loss.item()))
                ppo_losses["entropy"].append(float(entropy.item()))
                ppo_losses["kl"].append(float(approx_kl.item()))
                ppo_losses["clipfrac"].append(float(clipfrac.item()))
                ppo_losses["total"].append(float(loss.item()))

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
                # TD target with intrinsic-augmented reward + bootstrapped next value.
                # Value head is trained on normalized returns, so offp_values /
                # next_v are in normalized scale. Build TD target in RAW scale
                # (denormalize next_v first), then re-normalize before the loss
                # so both sides of the MSE are in the same (normalized) space.
                with torch.no_grad():
                    _, next_v = model(sample["next_obs"])
                    next_v_raw = reward_ema.denormalize(next_v)
                    td_target_raw = sample["reward"] + gamma * next_v_raw * (1.0 - sample["done"])
                    td_target = reward_ema.normalize(td_target_raw)
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
                # Reward (B, T, 1) for the world-model reward head, if present.
                reward_seq = None
                if "reward" in sample and sample["reward"] is not None:
                    reward_seq = sample["reward"].reshape(bsz, T, 1).float()
                wm_out = wm.compute_loss(obs_flat, actions_onehot, reward_seq=reward_seq)
                wm_optimizer.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)
                wm_optimizer.step()
                wm_last_loss = {
                    "loss": float(wm_out["loss"].item()),
                    "recon": float(wm_out["recon_loss"].item()),
                    "kl": float(wm_out["kl_loss"].item()),
                    "reward": float(
                        wm_out.get("reward_loss", torch.zeros(())).item()
                    ),
                }
            except (ValueError, IndexError, RuntimeError) as exc:
                logger.debug("world model update skipped: %s", exc)

        # --- Phase 1+: Dreamer-style imagination training ---
        if (
            imagination_trainer is not None
            and wm is not None
            and replay is not None
            and len(replay) >= replay_min_size
            and imagination_update_every > 0
            and state.step % imagination_update_every < rollout_capacity
        ):
            try:
                sample, _, _ = replay.sample_prioritized(
                    min(imagination_trainer.config.imagination_batch, len(replay)),
                    alpha=per_alpha,
                )
                imagination_last_loss = imagination_trainer.train_step(
                    actor_critic=model,
                    world_model=wm,
                    replay_sample=sample,
                    num_actions=num_actions,
                )
            except (ValueError, IndexError, RuntimeError) as exc:
                logger.debug("imagination update skipped: %s", exc)

        # --- Phase 1+: knowledge gap update ---
        if (knowledge_gap is not None and wm is not None
                and state.step - knowledge_gap_last_update >= knowledge_gap_update_every):
            try:
                sample_obs = _obs_to_tensor(obs, device).float().reshape(1, -1) / 255.0
                wm_pred_err = wm_last_loss.get("loss", 0.5)
                pred_errors = torch.tensor([wm_pred_err], device=device)
                if model.use_slots:
                    with torch.no_grad():
                        slot_out = model.encoder(_obs_to_tensor(obs, device)).squeeze(0)
                        knowledge_gap.update(slot_out.unsqueeze(0), pred_errors)
                knowledge_gap_last_update = state.step
            except Exception:
                pass

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

        # --- Enhanced memory: consolidation (episodic → semantic) ---
        if memory_manager is not None and memory_manager.should_consolidate(state.step):
            try:
                cons_result = memory_manager.consolidate(state.step)
                if cons_result.get("new_facts", 0) + cons_result.get("updated_facts", 0) > 0:
                    logger.info("[memory] consolidation: %d new, %d updated facts",
                                cons_result["new_facts"], cons_result["updated_facts"])
            except Exception:
                pass

        # --- Phase 0: model growth check ---
        if model_grower_v2 is not None and coverage is not None:
            mean_ret = float(env.summary().get("mean_return", 0.0))
            lp = model_grower_v2.plateau_lp(mean_ret)
            cov = coverage.coverage_ratio()
            if state.step % 50000 == 0:
                logger.info("[growth-debug] step=%d lp=%.4f cov=%.3f layers=%d",
                            state.step, lp, cov, len(model_grower_v2))
            if model_grower_v2.should_grow(state.step, lp, cov):
                try:
                    # Non-disruptive growth: distill the new block on REAL
                    # observations sampled from the replay buffer (so it learns
                    # the identity map on the agent's actual data and the grown
                    # model equals the teacher at growth time). Fall back to
                    # random-noise distillation if the buffer is empty.
                    distill_inputs = None
                    if replay is not None and len(replay) > 0:
                        try:
                            db = min(int(replay_cfg.get("distill_batch", 1024))
                                     if replay_cfg else 1024, len(replay))
                            distill_inputs = replay.sample(db)["obs"].to(device)
                        except Exception as exc:
                            logger.warning("[growth] replay sample failed (%s); "
                                           "falling back to random distill", exc)
                            distill_inputs = None
                    model, optimizer, grow_rec = model_grower_v2.grow(
                        model, optimizer, state.step, n_layers_to_add=1,
                        distill_inputs=distill_inputs,
                    )
                    logger.info("[growth] grown to %d layers (step=%d)%s",
                                grow_rec["new_layers"], state.step,
                                "" if distill_inputs is None else " (non-disruptive, real-data distill)")
                except Exception as exc:
                    logger.error("[growth] failed: %s", exc)

        # --- Phase 0: cross-modal bridge training ---
        if (xmodal_manager is not None and hasattr(env, '_agent')
                and n_envs == 1
                and state.step - xmodal_last_train >= xmodal_train_every):
            try:
                prop = torch.tensor(env._agent.proprio if hasattr(env._agent, 'proprio')
                                    else [0.0]*6, device=device).float()
                lang_emb = xmodal_manager.touch_bridge.touch_to_lang(prop.unsqueeze(0))
                # Simple reconciliation loss: touch→lang→touch should return original
                recon = xmodal_manager.touch_bridge.lang_to_touch(lang_emb)
                t_loss = F.mse_loss(recon.squeeze(0), prop)
                opt_xmodal = torch.optim.Adam(xmodal_manager.parameters(), lr=1e-4)
                opt_xmodal.zero_grad()
                t_loss.backward()
                opt_xmodal.step()
                xmodal_last_train = state.step
            except Exception:
                pass

        # Health sweep after each rollout+update cycle
        health.sweep()

        # --- Concept graph: periodic ingestion ---
        if concept_graph is not None and state.step % 2000 == 0:
            try:
                if memory_manager is not None and len(memory_manager.semantic) > 0:
                    facts = list(memory_manager.semantic._facts.values())[:20]
                    concept_graph.update_from_semantic(facts, state.step)
                if causal_disc is not None:
                    edges = [
                        (e.source, e.target, e.strength)
                        for e in causal_disc._graph.edges.values()
                        if e.strength > 0.3
                    ][:10]
                    if edges:
                        concept_graph.update_from_causal(edges, state.step)
            except Exception:
                pass

        # --- Concept clustering: periodic category discovery ---
        if (concept_clusterer is not None and concept_graph is not None
                and concept_clusterer.should_cluster(state.step)):
            try:
                new_cats = concept_clusterer.cluster(concept_graph, state.step)
                if new_cats:
                    logger.info("[concept] discovered %d new categories", len(new_cats))
            except Exception:
                pass

        # --- Visual Analyzer: classify slots each step (keeps motion
        #     continuous across frames), persist to concept graph every 500 ---
        if (visual_analyzer is not None and concept_graph is not None
                and model.use_slots):
            try:
                obs_t = _obs_to_tensor(obs, device)
                slot_out = model.encoder(obs_t)
                va_out = visual_analyzer(slot_out)  # updates motion vs prev frame
                if state.step % 500 == 0:
                    added = visual_analyzer.feed_to_graph(va_out, slot_out, concept_graph, state.step)
                    if added > 0:
                        logger.debug("[visual] %d objects analyzed", added)
            except Exception:
                logger.debug("[visual] analyze skipped", exc_info=True)

        # --- Neuro-Symbolic Bridge: periodic connection ---
        if state.step > 0 and state.step % 5000 < rollout_capacity:
            if causal2prolog is not None and causal_disc is not None and micro_math is not None:
                try:
                    n = causal2prolog.feed_to_math(micro_math, causal_disc)
                    if n > 0:
                        logger.debug("[bridge] causal2prolog: %d rules", n)
                except Exception:
                    pass
            if number2math is not None and number_sense is not None and micro_math is not None:
                try:
                    obs_t = _obs_to_tensor(obs, device)
                    slot_out = model.encoder(obs_t) if model.use_slots else None
                    if slot_out is not None:
                        number2math.observe(number_sense, slot_out.squeeze(0))
                except Exception:
                    pass
            if schema_detector is not None and rule_engine is not None:
                try:
                    schemas = schema_detector.extract(rule_engine, state.step)
                    if schemas:
                        logger.info("[schema] best: %s", schema_detector.get_best_schema())
                except Exception:
                    pass

            # --- Program Synthesis: active experimentation ---
            if (active_experimenter is not None and active_experimenter.should_test(state.step)
                    and causal_disc is not None and curiosity_director is not None):
                try:
                    exp = active_experimenter.propose_experiment(
                        causal_disc, curiosity_director,
                        rssm_uncertainty=float(np.asarray(int_r).mean()) if curiosity_mode != "none" else 0.0,
                    )
                    if exp:
                        logger.info("[experiment] %s", exp["hypothesis"][:80])
                        active_experimenter.record_result(exp, float(extrinsic_r), state.step)
                except Exception:
                    pass

            # --- Temporal abstraction: record step predicates ---
            if temporal_abstractor is not None and rule_engine is not None:
                try:
                    pred_keys = [k for k, v in episode_predicates[-1].items() if v] if episode_predicates else []
                    temporal_abstractor.record_step(pred_keys, float(extrinsic_r))
                except Exception:
                    pass

        # --- Identity Narrative: periodic self-reflection ---
        if (identity_narrative is not None and memory_manager is not None
                and state.step > 0 and state.step % 50000 < rollout_capacity):
            try:
                events = memory_manager.autobiographical._events
                if len(events) >= 20:
                    identity = identity_narrative(events)
                    logger.info("[identity] %s (events=%d, openness=%.2f)",
                               identity["narrative"][:80], len(events),
                               identity["traits"]["openness"])
            except Exception:
                pass

        # --- J-space monitoring: track reasoning subspace emergence ---
        if state.step > 0 and state.step % 10000 < rollout_capacity:
            try:
                with torch.no_grad():
                    sample_obs = _obs_to_tensor(obs, device)
                    _, _, hidden = model(sample_obs, return_hidden=True)
                    hidden_flat = hidden.reshape(-1)
                    max_act = hidden_flat.abs().max()
                    sparsity = float((hidden_flat.abs() > 0.01 * max_act).float().mean())
                    top_vals, top_dims = hidden_flat.abs().topk(min(16, hidden_flat.shape[0]))
                    normed = hidden_flat.abs() / (hidden_flat.abs().sum() + 1e-8)
                    dim_entropy = float(-(normed * (normed + 1e-8).log()).sum())
                    max_entropy = float(np.log(hidden_flat.shape[0]))
                    dim_concentration = 1.0 - dim_entropy / max_entropy
                    logger.info(
                        "[jspace] sparsity=%.3f concentration=%.3f top_dims=%s",
                        sparsity, dim_concentration,
                        str(top_dims[:8].tolist()),
                    )
            except Exception:
                pass

        # --- Long-range planning: periodic replan ---
        if (long_range_planner is not None and wm is not None
                and state.step % max(1, planner_cfg.get("plan_every_steps", 500)) < rollout_capacity):
            try:
                obs_t = _obs_to_tensor(obs, device)
                wm_obs = obs_t.float().reshape(1, -1) / 255.0
                action_onehot = F.one_hot(torch.tensor([0]), num_actions).float().to(device)
                wm_state = wm.initial_state(1, device)
                wm_state, _ = wm.observe_step(wm_state, action_onehot, wm_obs)

                # Generate base plan
                plan = long_range_planner.plan(wm_state, wm, model, obs_t)

                # Counterfactual validation: evaluate alternatives via the
                # world model's reward head (predicted reward, not policy value).
                if cf_planner is not None:
                    best = cf_planner.select_best(
                        long_range_planner, wm, wm_state, device,
                    )
                    if best is not None:
                        plan = best
                        logger.debug("[cf_plan] validated plan (len=%d)", len(plan))

                if plan:
                    logger.debug("[plan] plan: %s (len=%d)", plan[:5], len(plan))
            except Exception:
                pass

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
                extras.append(f"wm={wm_last_loss['loss']:.3f}(r={wm_last_loss['recon']:.3f},kl={wm_last_loss['kl']:.3f},rew={wm_last_loss['reward']:.4f})")
            if imagination_trainer is not None and imagination_last_loss:
                extras.append(f"img={imagination_last_loss.get('total_loss', 0):.4f}")
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
                "step=%d ep=%d mean_ret=%.3f loss=%.4f(p=%.2f v=%.2f ent=%.3f kl=%.4f cf=%.2f) mem_used=%.2fGB slope=%s %s",
                state.step,
                summary["episodes"],
                summary["mean_return"],
                float(np.mean(ppo_losses["total"])),
                float(np.mean(ppo_losses["policy"])),
                float(np.mean(ppo_losses["value"])),
                float(np.mean(ppo_losses["entropy"])),
                float(np.mean(ppo_losses["kl"])),
                float(np.mean(ppo_losses["clipfrac"])),
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
            if number_sense is not None:
                extra["number_sense_state"] = number_sense.state_dict()
            if rule_engine is not None:
                extra["rule_engine_state"] = rule_engine.state_dict()
            if causal_disc is not None:
                extra["causal_discovery_state"] = causal_disc.state_dict()
            if model_grower_v2 is not None:
                extra["model_grower_v2_state"] = model_grower_v2.state_dict()
            if creativity_orch is not None:
                extra["creativity_state"] = creativity_orch.state_dict()
            if llm_fusion is not None:
                extra["llm_fusion_state"] = {
                    k: v for k, v in llm_fusion.state_dict().items()
                    if not k.startswith("_llm")
                }
            if memory_manager is not None:
                extra["memory_manager_state"] = memory_manager.state_dict()
            if theory_of_mind is not None:
                extra["theory_of_mind_state"] = theory_of_mind.state_dict()
            if homeostatic_drives is not None:
                extra["homeostatic_drives_state"] = homeostatic_drives.state_dict()
            if emotion_system is not None:
                extra["emotion_system_state"] = emotion_system.state_dict()
            if long_range_planner is not None:
                extra["long_range_planner_state"] = long_range_planner.state_dict()
            if concept_graph is not None:
                extra["concept_graph_state"] = concept_graph.state_dict()
            if reflection_validator is not None:
                extra["reflection_validator_state"] = reflection_validator.state_dict()
            if concept_clusterer is not None:
                extra["concept_clusterer_state"] = concept_clusterer.state_dict()
            if cross_domain is not None:
                extra["cross_domain_state"] = {
                    "domains": {k: {"label": v.label} for k, v in cross_domain._domains.items()},
                }
            if cf_regret is not None:
                extra["cf_regret_state"] = {"regrets": cf_regret._regrets[-20:]}
            if value_system is not None:
                extra["value_system_state"] = {"judgments": len(value_system._judgments)}
            if intention_curiosity is not None:
                extra["intention_curiosity_state"] = intention_curiosity.state_dict()
            if knowledge_gap is not None:
                extra["knowledge_gap_state"] = knowledge_gap.state_dict()
            if social_curiosity is not None:
                extra["social_curiosity_state"] = social_curiosity.state_dict()
            if imagination_trainer is not None:
                extra["imagination_trainer_state"] = imagination_trainer.state_dict()
            if audio_encoder is not None:
                extra["audio_encoder_state"] = audio_encoder.state_dict()
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
    if imagination_trainer is not None and imagination_last_loss:
        logger.info("Imagination trainer final: %s", imagination_last_loss)
    if knowledge_gap is not None:
        logger.info("Knowledge gap final: %s", knowledge_gap.summary())

    env.close()

    # Auto-backup checkpoint to prevent data loss
    try:
        import shutil
        ckpt_path = stage_ckpt_path(stage, state.step)
        backup_path = Path(f"/root/backup_stage{stage}_{state.step}.pt")
        shutil.copy2(ckpt_path, backup_path)
        logger.info("Checkpoint backed up: %s", backup_path)
    except Exception as exc:
        logger.warning("Checkpoint backup failed (non-fatal): %s", exc)

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
    8: "stage8_full_cognitive.yaml",
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
