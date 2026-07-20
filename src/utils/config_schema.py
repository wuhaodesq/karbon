"""Config schema validation.

Validates the merged (preset + stage) config dict via dataclasses. Catches
typos, wrong types, and out-of-range values *before* training starts.

配置校验：用 dataclass 严格校验 preset+stage 合并后的 config，
避免拼写错误或非法值静默通过。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, get_type_hints


class ConfigValidationError(ValueError):
    """Raised when a config fails schema validation."""


@dataclass
class ModelSchema:
    hidden_size: int
    num_layers: int = 1
    ttt_backend: str = "pytorch"
    # Stage 2+: Hybrid backbone knobs. Ignored when use_hybrid_backbone=False.
    use_hybrid_backbone: bool = False
    hybrid_n_layers: int = 3
    hybrid_n_heads: int = 4
    hybrid_swa_window: int = 16
    hybrid_ttt_mini_batch: int = 8
    hybrid_ffn_hidden_mult: int = 4
    hybrid_dropout: float = 0.0
    # Stage 3+: Pretrained vision encoder (DINOv2/CLIP). Optional.
    use_vision_encoder: bool = False
    vision_model: str = "dinov2_vits14"
    vision_freeze: bool = True
    vision_target_size: int = 224
    # Stage 3+: Object-centric SlotAttention front-end. Ignored when disabled.
    use_slot_attention: bool = False
    slot_num_slots: int = 7
    slot_dim: int = 128
    slot_num_iterations: int = 3

    def _validate(self) -> None:
        if self.hidden_size <= 0:
            raise ConfigValidationError("model.hidden_size must be positive")
        if self.num_layers <= 0:
            raise ConfigValidationError("model.num_layers must be positive")
        if self.ttt_backend not in ("pytorch", "triton"):
            raise ConfigValidationError(
                f"model.ttt_backend must be 'pytorch' or 'triton', got {self.ttt_backend!r}"
            )
        if self.use_hybrid_backbone:
            if self.hybrid_n_layers <= 0:
                raise ConfigValidationError("model.hybrid_n_layers must be positive")
            if self.hybrid_n_heads <= 0:
                raise ConfigValidationError("model.hybrid_n_heads must be positive")
            if self.hybrid_swa_window <= 0:
                raise ConfigValidationError("model.hybrid_swa_window must be positive")
            if self.hybrid_ttt_mini_batch <= 0:
                raise ConfigValidationError("model.hybrid_ttt_mini_batch must be positive")
            if not (0.0 <= self.hybrid_dropout < 1.0):
                raise ConfigValidationError("model.hybrid_dropout must be in [0, 1)")
        if self.use_vision_encoder:
            if self.vision_target_size < 14:
                raise ConfigValidationError("model.vision_target_size must be >= 14")
        if self.use_slot_attention:
            if self.slot_num_slots <= 0:
                raise ConfigValidationError("model.slot_num_slots must be positive")
            if self.slot_dim <= 0:
                raise ConfigValidationError("model.slot_dim must be positive")
            if self.slot_num_iterations <= 0:
                raise ConfigValidationError("model.slot_num_iterations must be positive")


@dataclass
class MemorySchema:
    gpu_budget_gb: float
    cpu_ram_budget_gb: float
    replay_gpu_capacity: int
    replay_cpu_capacity: int
    skill_gpu_capacity: int
    wm_rollout_max_steps: int

    def _validate(self) -> None:
        if self.gpu_budget_gb < 0:
            raise ConfigValidationError("memory.gpu_budget_gb must be >= 0")
        if self.cpu_ram_budget_gb <= 0:
            raise ConfigValidationError("memory.cpu_ram_budget_gb must be > 0")
        for name in ("replay_gpu_capacity", "replay_cpu_capacity",
                     "skill_gpu_capacity", "wm_rollout_max_steps"):
            v = getattr(self, name)
            if v < 0:
                raise ConfigValidationError(f"memory.{name} must be >= 0, got {v}")


@dataclass
class EnvSchema:
    id: str
    num_envs: int = 1
    max_episode_steps: int | None = None
    # PhysicsSandbox (Stage 3+) knobs. Ignored by envs that don't use them.
    num_objects: int | None = None
    render_size: int | None = None
    gravity: float | None = None
    action_force: float | None = None

    def _validate(self) -> None:
        if not self.id:
            raise ConfigValidationError("env.id is required")
        if self.num_envs <= 0:
            raise ConfigValidationError("env.num_envs must be positive")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ConfigValidationError("env.max_episode_steps must be positive or null")
        if self.num_objects is not None and self.num_objects <= 0:
            raise ConfigValidationError("env.num_objects must be positive")
        if self.render_size is not None and self.render_size <= 0:
            raise ConfigValidationError("env.render_size must be positive")


@dataclass
class TrainSchema:
    batch_size: int
    seq_len: int
    learning_rate: float
    total_steps: int
    log_every_steps: int = 500
    ckpt_every_steps: int = 20_000
    # Stage-specific PPO knobs — optional
    ppo_clip: float | None = None
    ppo_epochs: int | None = None
    entropy_coef: float | None = None
    value_coef: float | None = None
    gamma: float | None = None
    gae_lambda: float | None = None

    def _validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError("train.batch_size must be positive")
        if self.seq_len <= 0:
            raise ConfigValidationError("train.seq_len must be positive")
        if not (0 < self.learning_rate < 1):
            raise ConfigValidationError(
                f"train.learning_rate should be in (0, 1), got {self.learning_rate}"
            )
        if self.total_steps <= 0:
            raise ConfigValidationError("train.total_steps must be positive")
        if self.log_every_steps <= 0:
            raise ConfigValidationError("train.log_every_steps must be positive")
        if self.ckpt_every_steps <= 0:
            raise ConfigValidationError("train.ckpt_every_steps must be positive")
        if self.gamma is not None and not (0 <= self.gamma <= 1):
            raise ConfigValidationError("train.gamma must be in [0, 1]")
        if self.gae_lambda is not None and not (0 <= self.gae_lambda <= 1):
            raise ConfigValidationError("train.gae_lambda must be in [0, 1]")


@dataclass
class MonitorSchema:
    sample_interval_s: float
    slope_alarm_gb_per_hour: float
    empty_cache_every_steps: int

    def _validate(self) -> None:
        if self.sample_interval_s <= 0:
            raise ConfigValidationError("monitor.sample_interval_s must be > 0")
        if self.slope_alarm_gb_per_hour <= 0:
            raise ConfigValidationError("monitor.slope_alarm_gb_per_hour must be > 0")
        if self.empty_cache_every_steps <= 0:
            raise ConfigValidationError("monitor.empty_cache_every_steps must be > 0")


@dataclass
class TopLevelSchema:
    preset: str
    device_preferred: str
    stage: int
    model: ModelSchema
    memory: MemorySchema
    env: EnvSchema
    train: TrainSchema
    monitor: MonitorSchema
    name: str = ""
    # Stage 1+ optional sub-blocks. Validated permissively — deep validation
    # happens inside the trainer once the modules are wired.
    intrinsic: dict | None = None
    replay: dict | None = None
    coverage: dict | None = None
    world_model: dict | None = None
    # Stage 3+: RSSM-uncertainty curiosity, imagination rollouts, model growth.
    curiosity: dict | None = None
    imagination: dict | None = None
    model_growth: dict | None = None
    skills: dict | None = None
    curriculum: dict | None = None
    continual: dict | None = None
    cognitive: dict | None = None
    language: dict | None = None
    advanced: dict | None = None
    # Stage 6+: implemented cognitive modules (activated, safe-skip paths).
    causal_discovery: dict | None = None
    number_sense: dict | None = None
    rule_induction: dict | None = None
    llm_fusion: dict | None = None
    core_knowledge_loss: dict | None = None
    # Count-based exploration bonus (3D deadlock guard). Top-level so
    # enabling it does NOT switch on RND / change curiosity mode.
    exploration_bonus: dict | None = None

    def _validate(self) -> None:
        if not self.preset:
            raise ConfigValidationError("preset is required")
        if self.device_preferred not in ("cpu", "cuda", "xpu", "mps"):   # BOUNDS-OK: whitelist of valid device kinds
            raise ConfigValidationError(
                f"device_preferred must be one of cpu/cuda/xpu/mps, "
                f"got {self.device_preferred!r}"
            )
        if self.stage < 0 or self.stage > 10:
            raise ConfigValidationError(f"stage out of range: {self.stage}")
        # Recurse
        for sub_name in ("model", "memory", "env", "train", "monitor"):
            getattr(self, sub_name)._validate()


# =====================================================================
# Loader from a plain dict (as produced by ``load_config``)
# =====================================================================


_SUB_SCHEMAS = {
    "model": ModelSchema,
    "memory": MemorySchema,
    "env": EnvSchema,
    "train": TrainSchema,
    "monitor": MonitorSchema,
}


def _known_field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _construct(cls: type, data: dict, path: str) -> Any:
    """Construct a dataclass; complain about unknown keys."""
    known = _known_field_names(cls)
    unknown = set(data.keys()) - known
    if unknown:
        raise ConfigValidationError(
            f"unknown keys under {path}: {sorted(unknown)}"
        )
    # Only pass known kwargs
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


def validate_config(cfg: dict) -> TopLevelSchema:
    """Validate a merged config dict; return the typed schema object.

    Raises :class:`ConfigValidationError` on any schema mismatch.
    """
    if not isinstance(cfg, dict):
        raise ConfigValidationError("config must be a dict")

    # Strip loader metadata that isn't part of schema
    cfg = {k: v for k, v in cfg.items() if k != "_meta"}

    # Sub-blocks
    sub_objects = {}
    for name, sub_cls in _SUB_SCHEMAS.items():
        if name not in cfg:
            raise ConfigValidationError(f"missing section: {name}")
        sub = cfg[name]
        if not isinstance(sub, dict):
            raise ConfigValidationError(f"section {name} must be a mapping")
        sub_objects[name] = _construct(sub_cls, sub, path=name)

    # Top-level extras
    top_data = {k: v for k, v in cfg.items() if k not in _SUB_SCHEMAS}
    top_data.update(sub_objects)
    top = _construct(TopLevelSchema, top_data, path="<top-level>")
    top._validate()
    return top


def validate_and_dump(cfg: dict) -> dict:
    """Validate ``cfg`` then round-trip back into a plain dict.

    Useful when callers want the *canonicalized* config (schema-normalized).
    """
    schema = validate_config(cfg)
    return asdict(schema)
