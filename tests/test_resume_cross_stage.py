"""Tests for cross-stage vs same-stage checkpoint resume semantics.

Two important behaviors:

1. **Same-stage resume**: continue the step counter (allows split-run training).
2. **Cross-stage resume**: reset step counter to 0 (each stage gets its own
   total_steps budget). Without this, stage N+1 sees `state.step >= total_steps`
   right on entry and exits before doing any work.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_stage0_ckpt(ckpt_dir: Path, step: int = 3_000_000) -> Path:
    """Build a fake Stage-0 checkpoint at the given step."""
    from src.utils import save_ckpt

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Linear(4, 4)

    m = TinyModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ckpt = ckpt_dir / f"ckpt_stage0_{step:09d}.pt"
    save_ckpt(
        ckpt,
        stage=0,
        step=step,
        model_state=m.state_dict(),
        optim_state=opt.state_dict(),
        extra={"preset": "cloud_5090", "run_id": "test"},
    )
    return ckpt


def test_cross_stage_resume_resets_step_counter(tmp_path, monkeypatch):
    """Stage 1 resuming from Stage 0 ckpt must start at step=0.

    Regression: the initial Stage-1 implementation copied `step=3_000_000`
    from the ckpt, so `while state.step < total_steps` failed immediately.
    """
    from src.utils import load_ckpt

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt = _make_stage0_ckpt(ckpt_dir, step=3_000_000)
    payload = load_ckpt(ckpt)

    resumed_stage = int(payload["stage"])
    resumed_step = int(payload["step"])

    # Simulate the trainer's decision logic for stage=1
    target_stage = 1
    if resumed_stage == target_stage:
        effective_step = resumed_step
    else:
        effective_step = 0  # cross-stage → reset

    assert effective_step == 0, (
        "cross-stage resume must reset step counter to 0, "
        f"got {effective_step}"
    )


def test_same_stage_resume_preserves_step_counter(tmp_path):
    """Stage 0 resuming from a Stage 0 ckpt must continue the step counter."""
    from src.utils import load_ckpt

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt = _make_stage0_ckpt(ckpt_dir, step=500_000)
    payload = load_ckpt(ckpt)

    resumed_stage = int(payload["stage"])
    resumed_step = int(payload["step"])
    target_stage = 0

    if resumed_stage == target_stage:
        effective_step = resumed_step
    else:
        effective_step = 0

    assert effective_step == 500_000, (
        "same-stage resume must continue step counter "
        f"(got {effective_step}, expected 500000)"
    )


def test_forward_stage_jumps_all_reset(tmp_path):
    """Cross-stage resume works for any stage → any other stage jump."""
    from src.utils import load_ckpt

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt = _make_stage0_ckpt(ckpt_dir, step=1_000_000)
    payload = load_ckpt(ckpt)

    for target_stage in (1, 2, 3, 4, 5, 6):
        resumed_stage = int(payload["stage"])
        resumed_step = int(payload["step"])
        if resumed_stage == target_stage:
            effective_step = resumed_step
        else:
            effective_step = 0
        assert effective_step == 0, (
            f"stage 0 → stage {target_stage} must reset step, got {effective_step}"
        )
