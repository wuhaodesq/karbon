"""TTT backend protocol.

Abstract layer that decouples the TTT-Linear layer's public API from its
implementation. Two implementations are planned:

- :mod:`src.models.ttt_linear` (this module's pure-PyTorch teaching backend)
- :mod:`src.models.ttt_linear_triton` (Stage 2b, Linux+CUDA only)

The Hybrid backbone must import via this module, never the concrete backends
directly. Stage 2's hot-swap between backends is controlled by
``config.model.ttt_backend``.

TTT 后端抽象。所有 TTT-Linear 使用方通过此模块获取实现，
让 PyTorch 教学版和 Triton 版可无痛热切换。
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)

BackendName = Literal["pytorch", "triton"]


@runtime_checkable
class TTTLinearBackend(Protocol):
    """Signature required by any TTT-Linear implementation.

    ``ttt_linear_forward(x, theta_K, theta_V, theta_Q, eta, mini_batch,
    detach_every_n_segments)`` returns:

    - ``y``: (B, T, d_out) — the layer's output for each token.
    - ``final_W``: (B, d_out, d_in) — the final inner-model weight after the
      sequence. Kept for optional consolidation.

    输入形状：
        x         : (B, T, d_in)   float
        theta_K/V/Q: (d_in, d_key/val)  float, shared across the batch
        eta       : () or (B,) float — inner-model learning rate
        mini_batch: int — mini-batch TTT segment size
        detach_every_n_segments: Optional[int] — BPTT chunk boundary

    Outputs a *bounded* rollout: no per-step W snapshots are stored beyond the
    last one (Axiom 1).
    """

    def __call__(
        self,
        x: torch.Tensor,
        theta_K: torch.Tensor,
        theta_V: torch.Tensor,
        theta_Q: torch.Tensor,
        eta: torch.Tensor,
        mini_batch: int,
        detach_every_n_segments: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# ---------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------


def get_backend(name: BackendName | None = None) -> TTTLinearBackend:
    """Return a TTT-Linear implementation.

    Selection order:
      1. explicit ``name`` argument
      2. ``DEVAGI_TTT_BACKEND`` environment variable
      3. auto: prefer Triton on CUDA-Linux, else PyTorch

    根据显式参数 / 环境变量 / 自动探测 三级顺序选择后端。
    """
    forced = name or os.environ.get("DEVAGI_TTT_BACKEND")
    if forced:
        forced = forced.strip().lower()
        if forced == "pytorch":
            return _get_pytorch_backend()
        if forced == "triton":
            return _get_triton_backend_or_fallback()
        raise ValueError(f"Unknown TTT backend: {forced!r}")

    # Auto
    if _triton_available():
        return _get_triton_backend_or_fallback()
    return _get_pytorch_backend()


def _get_pytorch_backend() -> TTTLinearBackend:
    from .ttt_linear import ttt_linear_forward_pytorch

    return ttt_linear_forward_pytorch  # type: ignore[return-value]


def _get_triton_backend_or_fallback() -> TTTLinearBackend:
    try:
        from .ttt_linear_triton import ttt_linear_forward_triton  # type: ignore

        return ttt_linear_forward_triton  # type: ignore[return-value]
    except ImportError:
        logger.warning("Triton backend unavailable — falling back to PyTorch.")
        return _get_pytorch_backend()


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False
