"""Generate a demo HybridBackbone checkpoint and export it to HF format.

Produces two directories under ``exports/``:
    demo-hybrid-fp32/   — float32 export (largest, safest for verification)
    demo-hybrid-fp16/   — float16 export (half size, deployment-friendly)

These are ready-to-upload artefacts for TOS / HuggingFace Hub / ARK custom
model import.

Usage:
    python scripts/build_demo_export.py                    # both dtypes
    python scripts/build_demo_export.py --dtype float16    # single dtype
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from scripts.export_hf import export_checkpoint
from src.models import HybridBackbone
from src.utils import save_ckpt

logger = logging.getLogger(__name__)


def build_demo_ckpt(ckpt_path: Path, seed: int = 0) -> tuple[Path, int]:
    """Build a small deterministic HybridBackbone and save its ckpt."""
    torch.manual_seed(seed)
    model = HybridBackbone(
        d_model=64,
        n_layers=2,
        vocab_size=128,
        n_heads=4,
        swa_window_size=16,
        ttt_mini_batch=8,
        max_seq_len=128,
    )
    n_params = model.num_parameters()

    save_ckpt(
        ckpt_path,
        stage=2,
        step=1_000,
        model_state=model.state_dict(),
        optim_state=None,
        extra={
            "preset": "cloud_24g",
            "run_id": "demo",
            "d_model": 64,
            "n_layers": 2,
            "vocab_size": 128,
            "n_heads": 4,
            "swa_window_size": 16,
            "ttt_mini_batch": 8,
            "max_seq_len": 128,
        },
    )
    return ckpt_path, n_params


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", type=str, default="both",
                    choices=("both", "float32", "float16", "bfloat16"))
    ap.add_argument("--out-root", type=Path, default=_ROOT / "exports")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_root / "_demo_ckpt.pt"

    logger.info("Building demo HybridBackbone checkpoint...")
    _, n_params = build_demo_ckpt(ckpt_path)
    logger.info("HybridBackbone has %d parameters", n_params)

    dtypes = ["float32", "float16"] if args.dtype == "both" else [args.dtype]

    for dt in dtypes:
        suffix = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dt]
        out_dir = args.out_root / f"demo-hybrid-{suffix}"
        logger.info("Exporting to %s (dtype=%s)", out_dir, dt)
        export_checkpoint(
            ckpt_path=ckpt_path,
            output_dir=out_dir,
            model_name=f"devagi-hybrid-demo-{suffix}",
            arch="hybrid_backbone",
            dtype=dt,
        )
        files = sorted(p.name for p in out_dir.iterdir())
        logger.info("  files: %s", files)

    logger.info("Done. Ready for TOS upload from: %s", args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
