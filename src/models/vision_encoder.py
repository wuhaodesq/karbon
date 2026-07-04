"""Pretrained vision encoder for semantic object recognition.

Wraps DINOv2 / CLIP / timm models as a drop-in replacement for the
from-scratch CNN encoder in ``HybridActorCritic``. The pretrained backbone
is **frozen** — only the projection layer trains.

Effect: instead of learning "pixel pattern → action" from scratch, the agent
gets "semantic features (cat/dog/table/key/door) → action". Combined with
TTT-Linear's in-context adaptation, this enables recognizing **new objects
at test time** without retraining.

Usage in config:
    model:
      use_hybrid_backbone: true
      use_vision_encoder: true      # ← enables pretrained encoder
      vision_model: dinov2_vits14   # ← which pretrained model
      vision_freeze: true            # ← freeze backbone, only train proj

Bounded: backbone is frozen → VRAM is constant. Projection layer is tiny
(d_model × embed_dim ≈ 150k params). Axiom 1 satisfied.

预训练视觉编码器：用 DINOv2/CLIP 替代从零学的 CNN。
骨干网络冻结，只训练投影层。让智能体从"看像素"升级到"理解语义"。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Known pretrained models and their output dims
_VISION_MODELS: dict[str, dict[str, Any]] = {
    "dinov2_vits14": {"embed_dim": 384, "min_input": 14, "source": "torch.hub"},
    "dinov2_vitb14": {"embed_dim": 768, "min_input": 14, "source": "torch.hub"},
    "dinov2_vitl14": {"embed_dim": 1024, "min_input": 14, "source": "torch.hub"},
    # CLIP visual encoders (via timm)
    "clip_vit_base_patch16_224": {"embed_dim": 512, "min_input": 224, "source": "timm"},
    "clip_vit_small_patch16_224": {"embed_dim": 384, "min_input": 224, "source": "timm"},
    # siglip
    "siglip_base_patch16_224": {"embed_dim": 768, "min_input": 224, "source": "timm"},
}


def list_available_vision_models() -> list[str]:
    """Return the list of known vision model names."""
    return list(_VISION_MODELS.keys())


class VisionEncoder(nn.Module):
    """Pretrained vision encoder with a trainable projection layer.

    Architecture:
        obs (B, H, W, C) uint8
            → resize to ≥ min_input × min_input
            → frozen pretrained backbone → (B, embed_dim)
            → trainable projection → (B, d_model)

    The pretrained backbone is **frozen** (requires_grad=False). Only the
    projection layer trains. This keeps VRAM bounded (Axiom 1) and training
    fast (backward only through the projection, not the 21M-param backbone).

    For MiniGrid's tiny 7×7 observations, images are upscaled to 224×224 via
    bilinear interpolation. This is suboptimal (DINOv2 wasn't trained on
    upscaled pixel art) but functional. For Crafter's 64×64 observations,
    the upscaling is gentler and features are more meaningful.

    Bounded: backbone frozen → constant VRAM. Projection is (embed_dim × d_model)
    ≈ 150k params. No unbounded state.
    """

    def __init__(
        self,
        d_model: int = 384,
        model_name: str = "dinov2_vits14",
        freeze: bool = True,
        target_size: int = 224,
    ) -> None:
        super().__init__()
        if model_name not in _VISION_MODELS:
            raise ValueError(
                f"Unknown vision model: {model_name!r}. "
                f"Available: {list_available_vision_models()}"
            )

        spec = _VISION_MODELS[model_name]
        self._model_name = model_name
        self._embed_dim = int(spec["embed_dim"])
        self._min_input = int(spec["min_input"])
        self._target_size = int(target_size)
        self._source = spec["source"]

        # Load pretrained backbone
        self._backbone = self._load_backbone(model_name, self._source)

        if freeze:
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            self._backbone.eval()
            logger.info(
                "VisionEncoder: backbone=%s frozen (embed_dim=%d, params=%d)",
                model_name, self._embed_dim,
                sum(p.numel() for p in self._backbone.parameters()),
            )
        else:
            logger.info(
                "VisionEncoder: backbone=%s trainable (embed_dim=%d, params=%d)",
                model_name, self._embed_dim,
                sum(p.numel() for p in self._backbone.parameters()),
            )

        # Trainable projection: embed_dim → d_model
        self.proj = nn.Linear(self._embed_dim, d_model)
        self._d_model = d_model

    def _load_backbone(self, model_name: str, source: str) -> nn.Module:
        """Load the pretrained backbone. Lazy import + download."""
        if source == "torch.hub":
            try:
                backbone = torch.hub.load(
                    "facebookresearch/dinov2", model_name,
                    pretrained=True, trust_repo=True,
                )
                return backbone
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {model_name} via torch.hub. "
                    f"Possible causes: no internet, GitHub blocked, or model name wrong. "
                    f"Error: {exc}"
                ) from exc
        elif source == "timm":
            try:
                import timm
                backbone = timm.create_model(model_name, pretrained=True)
                return backbone
            except ImportError:
                raise RuntimeError(
                    f"timm not installed. Install with: pip install timm"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {model_name} via timm. Error: {exc}"
                ) from exc
        else:
            raise ValueError(f"Unknown source: {source}")

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """Encode observations to semantic features.

        Args:
            obs_u8: (B, H, W, C) uint8 image observations.

        Returns:
            (B, d_model) semantic feature vectors.
        """
        # (B, H, W, C) uint8 → (B, C, H, W) float
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0

        # Resize to target_size × target_size if too small
        h, w = x.shape[-2], x.shape[-1]
        if h < self._target_size or w < self._target_size:
            x = F.interpolate(
                x, size=(self._target_size, self._target_size),
                mode="bilinear", align_corners=False,
            )

        # Forward through frozen backbone
        if self._backbone.training:
            self._backbone.eval()
        with torch.no_grad():
            feats = self._backbone(x)  # (B, embed_dim)

        # Trainable projection
        return self.proj(feats)  # (B, d_model)

    def extra_repr(self) -> str:
        return (
            f"model={self._model_name}, embed_dim={self._embed_dim}, "
            f"d_model={self._d_model}, target_size={self._target_size}"
        )


class CNNEncoder(nn.Module):
    """Default from-scratch CNN encoder (current Stage 0-3 default).

    Kept as a fallback when no pretrained model is available or for
    environments where pretrained features are not useful (e.g., MiniGrid's
    7×7 pixel art).
    """

    def __init__(self, obs_shape: tuple[int, ...], d_model: int = 384) -> None:
        super().__init__()
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
        self._d_model = d_model

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        x = obs_u8.permute(0, 3, 1, 2).float() / 255.0
        return self.encoder(x)  # (B, d_model)


def build_encoder(
    config: dict,
    obs_shape: tuple[int, ...],
    device: torch.device,
) -> nn.Module:
    """Factory: build the right encoder based on config.

    If ``config.model.use_vision_encoder`` is True, load a pretrained
    VisionEncoder. Otherwise fall back to CNNEncoder.

    Returns the encoder module on ``device``.
    """
    model_cfg = config.get("model", {})
    use_vision = bool(model_cfg.get("use_vision_encoder", False))

    if use_vision:
        model_name = model_cfg.get("vision_model", "dinov2_vits14")
        freeze = bool(model_cfg.get("vision_freeze", True))
        target_size = int(model_cfg.get("vision_target_size", 224))
        d_model = int(model_cfg.get("hidden_size", 384))

        try:
            encoder = VisionEncoder(
                d_model=d_model,
                model_name=model_name,
                freeze=freeze,
                target_size=target_size,
            ).to(device)
            logger.info("Using pretrained VisionEncoder: %s", model_name)
            return encoder
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed to load VisionEncoder (%s). Falling back to CNN.", exc
            )

    # Fallback: from-scratch CNN
    d_model = int(model_cfg.get("hidden_size", 384))
    encoder = CNNEncoder(obs_shape, d_model=d_model).to(device)
    logger.info("Using CNNEncoder (from scratch, d_model=%d)", d_model)
    return encoder
