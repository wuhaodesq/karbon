"""Language encoder + multimodal fusion for instruction-conditioned RL.

Enables the agent to understand natural-language instructions like:
  - "go to the goal"
  - "pick up the key"
  - "open the door"

Three components:

1. :class:`LanguageEncoder` — wraps CLIP's text branch (frozen) to encode
   instructions into embedding vectors. Pre-trained on 400M image-text pairs.

2. :class:`MultimodalFusion` — fuses vision features + language features
   into a joint representation. Uses FiLM (Feature-wise Linear Modulation):
   the language embedding generates scale/shift parameters that modulate
   the vision features. This lets the instruction *steer* what the agent
   attends to.

3. :class:`InstructionConditionedActorCritic` — a full policy network that
   takes (obs, instruction) → action. Wraps the Hybrid backbone with
   vision + language encoders + FiLM fusion.

All pretrained backbones are frozen → bounded VRAM (Axiom 1).

Bounded: CLIP text encoder is frozen (63M params, ~250MB VRAM constant).
FiLM layers are tiny (2 × d_model² ≈ 300k params). Axiom 1 satisfied.

语言编码 + 多模态融合：让智能体听懂"去拿钥匙"之类的指令。
CLIP 文本编码器冻结，FiLM 调制层让指令引导视觉注意力。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import CNNEncoder, VisionEncoder

logger = logging.getLogger(__name__)


# =====================================================================
# Language encoder (CLIP text branch, frozen)
# =====================================================================


class LanguageEncoder(nn.Module):
    """Frozen CLIP text encoder + trainable projection.

    Encodes a natural-language instruction string into a dense embedding
    that lives in the same space as CLIP's visual features. This enables
    zero-shot alignment: the word "key" and the image of a key will have
    similar embeddings.

    The CLIP backbone is **frozen** (requires_grad=False). Only the
    projection layer trains.

    Usage:
        enc = LanguageEncoder(d_model=384)
        text = enc.encode_text("pick up the key")  # (1, d_model)
        # or batch:
        texts = enc.encode_text(["go to goal", "pick up key"])  # (2, d_model)

    Bounded: CLIP text encoder is frozen → constant VRAM (~250 MB).
    """

    def __init__(
        self,
        d_model: int = 384,
        model_name: str = "clip_text_vit_base_patch16",
        freeze: bool = True,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._max_length = max_length
        self._backbone = self._load_clip_text(model_name)

        if freeze:
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            self._backbone.eval()
            logger.info(
                "LanguageEncoder: %s frozen (embed_dim=%d)",
                model_name, self._embed_dim,
            )

        self.proj = nn.Linear(self._embed_dim, d_model)

    def _load_clip_text(self, model_name: str) -> nn.Module:
        """Load CLIP text encoder via timm or torch.hub."""
        # Try timm first (more reliable on AutoDL)
        try:
            import timm
            model = timm.create_model(model_name, pretrained=True)
            # timm CLIP models expose .text_cfg and .visual / .text branches
            if hasattr(model, "text"):
                backbone = model.text
                self._embed_dim = int(model.text_embed_dim)
            elif hasattr(model, "encode_text"):
                backbone = model
                self._embed_dim = int(getattr(model, "text_embed_dim", 512))
            else:
                raise RuntimeError("model has no text branch")
            self._full_model = model  # keep ref for tokenizer
            return backbone
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("timm load failed (%s), trying torch.hub", exc)

        # Fallback: torch.hub CLIP
        try:
            import clip as clip_mod
            model, _ = clip_mod.load("ViT-B/32", device="cpu")
            self._full_model = model
            self._embed_dim = 512
            return model
        except ImportError:
            raise RuntimeError(
                "No CLIP backend available. Install with: pip install timm "
                "OR pip install git+https://github.com/openai/CLIP.git"
            )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Encode instruction text → (B, d_model) embedding.

        Uses the CLIP tokenizer internally. The text is tokenized, passed
        through the frozen text encoder, then projected to d_model.
        """
        import clip as clip_mod
        if isinstance(text, str):
            text = [text]
        tokens = clip_mod.tokenize(text, context_length=self._max_length)
        with torch.no_grad():
            text_feats = self._full_model.encode_text(tokens)  # (B, embed_dim)
        return self.proj(text_feats)  # (B, d_model)

    def forward(self, text: str | list[str]) -> torch.Tensor:
        return self.encode_text(text)


# =====================================================================
# FiLM: Feature-wise Linear Modulation
# =====================================================================


class FiLMLayer(nn.Module):
    """FiLM: language-conditioned scale/shift of vision features.

    Given a language embedding ``l`` (B, d_lang) and vision features ``v``
    (B, d_vis), produces:

        v' = γ(l) * v + β(l)

    where γ and β are learned linear projections from d_lang → d_vis.

    This lets the instruction modulate WHAT the agent attends to in the
    visual scene. E.g., instruction "pick up key" → γ emphasizes key-like
    features and suppresses wall/door features.

    Bounded: just two Linear layers (d_lang × d_vis × 2 ≈ 300k params).
    """

    def __init__(self, d_vis: int, d_lang: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(d_lang, d_vis)
        self.beta = nn.Linear(d_lang, d_vis)
        # Initialize to identity (γ=1, β=0) so it starts as a no-op.
        # γ(l) = W_γ·l + b_γ = 1  →  W_γ=0, b_γ=1
        # β(l) = W_β·l + b_β = 0  →  W_β=0, b_β=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, v: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """v: (B, d_vis), l: (B, d_lang) → (B, d_vis)"""
        gamma = self.gamma(l)  # (B, d_vis)
        beta = self.beta(l)    # (B, d_vis)
        return gamma * v + beta


# =====================================================================
# Multimodal fusion: vision + language → joint features
# =====================================================================


class MultimodalFusion(nn.Module):
    """Fuses vision features and language features via FiLM + cross-attention.

    Pipeline:
        1. Vision encoder → v (B, d_model)
        2. Language encoder → l (B, d_model)
        3. FiLM: v' = γ(l) * v + β(l)
        4. Cross-attention: v'' = Attention(v', l)  (optional)
        5. Output: v'' (B, d_model)

    The cross-attention step lets the agent dynamically attend to different
    parts of the instruction for different visual inputs. Disabled by default
    (FiLM alone is usually sufficient).

    Bounded: no unbounded state. FiLM + optional cross-attn are fixed-size.
    """

    def __init__(
        self,
        d_model: int = 384,
        use_cross_attention: bool = False,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.film = FiLMLayer(d_vis=d_model, d_lang=d_model)
        self.use_cross_attn = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, batch_first=True,
            )
            self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        vision_feats: torch.Tensor,   # (B, d_model)
        lang_feats: torch.Tensor,      # (B, d_model)
    ) -> torch.Tensor:
        # FiLM modulation
        fused = self.film(vision_feats, lang_feats)  # (B, d_model)

        if self.use_cross_attn:
            # Cross-attention: vision queries attend to language keys
            v = fused.unsqueeze(1)  # (B, 1, d)
            l = lang_feats.unsqueeze(1)  # (B, 1, d)
            attn_out, _ = self.cross_attn(v, l, l)
            fused = self.norm(fused + attn_out.squeeze(1))

        return fused  # (B, d_model)


# =====================================================================
# Instruction-conditioned actor-critic
# =====================================================================


class InstructionConditionedActorCritic(nn.Module):
    """Full (obs, instruction) → action model.

    Architecture:
        obs (B, H, W, C) uint8
            → VisionEncoder → v (B, d_model)
        instruction text (str / list[str])
            → LanguageEncoder → l (B, d_model)
        (v, l) → MultimodalFusion → fused (B, d_model)
            → HybridBackbone → (B, 1, d_model)
            → policy_head + value_head

    When language encoder is unavailable (no internet to download CLIP),
    falls back to vision-only mode (equivalent to HybridActorCritic).

    Bounded: all components have fixed state. CLIP is frozen.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        d_model: int = 384,
        n_layers: int = 3,
        n_heads: int = 4,
        swa_window: int = 16,
        ttt_mini_batch: int = 8,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        use_vision_encoder: bool = False,
        vision_model_name: str = "dinov2_vits14",
        use_language_encoder: bool = False,
        language_model_name: str = "clip_text_vit_base_patch16",
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        from .hybrid_backbone import HybridBackbone

        # Snap d_model
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        if d_model % 2 != 0:
            d_model += 1
        self.d_model = d_model

        # Vision encoder
        self.use_vision = use_vision_encoder
        if use_vision_encoder:
            try:
                self.vision_encoder = VisionEncoder(
                    d_model=d_model, model_name=vision_model_name,
                )
            except (RuntimeError, ValueError):
                self.vision_encoder = CNNEncoder(obs_shape, d_model=d_model)
                self.use_vision = False
        else:
            self.vision_encoder = CNNEncoder(obs_shape, d_model=d_model)

        # Language encoder (optional)
        self.use_language = use_language_encoder
        if use_language_encoder:
            try:
                self.language_encoder = LanguageEncoder(
                    d_model=d_model, model_name=language_model_name,
                )
                self.fusion = MultimodalFusion(
                    d_model=d_model,
                    use_cross_attention=use_cross_attention,
                    n_heads=n_heads,
                )
            except (RuntimeError, ImportError) as exc:
                logger.warning("LanguageEncoder load failed (%s), using vision-only", exc)
                self.use_language = False

        # Hybrid backbone
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

    def forward(
        self,
        obs_u8: torch.Tensor,
        instruction: str | list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs_u8: (B, H, W, C) uint8 observations.
            instruction: natural-language instruction string (or batch of).
                If None and language encoder is enabled, uses a default
                empty instruction.

        Returns:
            logits: (B, num_actions)
            value: (B,)
        """
        # Vision features
        v = self.vision_encoder(obs_u8)  # (B, d_model)

        # Language fusion (if enabled and instruction provided)
        if self.use_language:
            if instruction is None:
                instruction = ""
            l = self.language_encoder.encode_text(instruction)  # (B, d_model)
            # If batch sizes differ, broadcast language to match vision batch
            if l.shape[0] == 1 and v.shape[0] > 1:
                l = l.expand(v.shape[0], -1)
            v = self.fusion(v, l)  # FiLM-modulated features

        # Hybrid backbone (seq_len=1 per obs)
        seq = v.unsqueeze(1)       # (B, 1, d_model)
        seq_out = self.backbone(seq)  # (B, 1, d_model)
        z = seq_out.squeeze(1)     # (B, d_model)

        return self.policy_head(z), self.value_head(z).squeeze(-1)
