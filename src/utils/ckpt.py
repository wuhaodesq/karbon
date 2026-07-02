"""Checkpoint save/load helpers (upgrade-oriented; see Axiom 6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


CKPT_FORMAT_VERSION = 1


def save_ckpt(
    path: Path,
    *,
    stage: int,
    step: int,
    model_state: dict,
    optim_state: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint with a schema-versioned envelope.

    带 schema 版本号的 checkpoint 存档，用于跨阶段升级。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_format_version": CKPT_FORMAT_VERSION,
        "stage": stage,
        "step": step,
        "model_state": model_state,
        "optim_state": optim_state,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s", path)
    return path


def load_ckpt(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload. Validates the format version.

    加载 checkpoint 并校验 schema 版本。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    v = payload.get("_format_version")
    if v != CKPT_FORMAT_VERSION:
        raise RuntimeError(
            f"Incompatible checkpoint format: got v{v}, expected v{CKPT_FORMAT_VERSION}"
        )
    return payload
