"""Export a devagi checkpoint to Hugging Face-compatible layout.

Produces a directory that can be uploaded as-is to TOS / HuggingFace Hub / ARK
custom-model import:

.. code-block:: text

    <output_dir>/
      config.json              # HF-style config with model_type + custom fields
      model.safetensors        # weights (safetensors, sharded if > 5 GB)
      generation_config.json   # inference-time defaults (optional)
      tokenizer.json           # only when a tokenizer is present (skipped for our RL agents)
      README.md                # model card (bilingual)

Usage:
    python -m scripts.export_hf \\
        --ckpt checkpoints/ckpt_stage2_000500000.pt \\
        --output-dir exports/devagi-hybrid-stage2 \\
        --model-name "devagi-hybrid" \\
        --arch hybrid_backbone

Currently supported ``--arch`` values (extend as new stages ship):
    - ``hybrid_backbone``  (Stage 2, TTT-Linear + SWA + FFN stack)
    - ``rssm``             (Stage 3, world model)
    - ``rnd``              (Stage 1, curiosity module)
    - ``ttt_linear``       (Stage 2, standalone TTT-Linear layer)

The exporter is intentionally lightweight — it does NOT depend on
``transformers``. HF Hub / TOS will happily accept the ``model_type: custom``
layout for research artefacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `src` importable when invoked as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from safetensors.torch import save_file

from src.utils import load_ckpt

logger = logging.getLogger(__name__)


SUPPORTED_ARCHS = ("hybrid_backbone", "rssm", "rnd", "ttt_linear")

# Safetensors shard size cap (5 GB). Anything larger gets split.
_SHARD_LIMIT_BYTES = 5 * 1024 * 1024 * 1024


# =====================================================================
# HF-style config
# =====================================================================


@dataclass
class HFExportConfig:
    """The ``config.json`` payload we ship to HF/TOS."""

    model_name: str
    architectures: list[str]
    model_type: str            # "devagi_hybrid", "devagi_rssm", ...
    torch_dtype: str           # "float32" | "bfloat16" | ...
    devagi_stage: int
    devagi_arch: str
    devagi_ckpt_format_version: int
    devagi_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Sharded safetensors writer
# =====================================================================


def _flatten_state_dict(sd: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten a nested state dict of tensors."""
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_state_dict(v, prefix=key))
        elif isinstance(v, torch.Tensor):
            out[key] = v.detach().cpu().contiguous()
        # non-tensor entries silently dropped — they go into the config JSON
    return out


def _save_safetensors_sharded(
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
    max_shard_bytes: int = _SHARD_LIMIT_BYTES,
) -> dict[str, str]:
    """Save tensors as one or more safetensors shards.

    Returns a mapping ``{tensor_name: shard_filename}`` for the index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_sizes: list[int] = [0]
    shard_buckets: list[dict[str, torch.Tensor]] = [{}]

    # Simple greedy packing by declaration order
    for name, t in tensors.items():
        sz = t.element_size() * t.numel()
        if shard_sizes[-1] + sz > max_shard_bytes and shard_buckets[-1]:
            shard_sizes.append(0)
            shard_buckets.append({})
        shard_buckets[-1][name] = t
        shard_sizes[-1] += sz

    total_shards = len(shard_buckets)
    index_map: dict[str, str] = {}

    if total_shards == 1:
        # Single file
        fname = "model.safetensors"
        save_file(shard_buckets[0], str(out_dir / fname))
        for k in shard_buckets[0]:
            index_map[k] = fname
        return index_map

    for i, bucket in enumerate(shard_buckets, start=1):
        fname = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        save_file(bucket, str(out_dir / fname))
        for k in bucket:
            index_map[k] = fname

    # HF-style index file
    total_size = sum(t.element_size() * t.numel() for t in tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": index_map,
    }
    with (out_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index_map


# =====================================================================
# Model-card writer
# =====================================================================


_README_TEMPLATE = """---
library_name: pytorch
tags:
- devagi
- {arch}
- research
model_name: {model_name}
model_type: {model_type}
---

# {model_name}

**Architecture**: `{arch}`
**devagi stage**: {stage}
**ckpt format version**: {ckpt_fmt}
**Exported at**: {ts}

## What is this? / 这是什么？

Research artefact from the **devagi** project — a bounded-memory, perpetually
runnable developmental AI research platform. See the project repo for context:
<https://github.com/wuhaodesq/karbon>.

This checkpoint is **not** a general-purpose language / vision model. It is a
component of a larger developmental agent (TTT-Hybrid backbone, world model,
curiosity, curriculum, EWC, sleep consolidation).

发育式 AI 研究项目 `devagi` 的模型工件。**不是**通用对话/视觉大模型，
而是一个"边跑边学"的智能体子模块。

## Files

- `config.json` — architecture + devagi metadata
- `model.safetensors` (or sharded) — weights in safetensors format
- `README.md` — this card

## Loading

The weights use plain PyTorch tensor names. To reload:

```python
from safetensors.torch import load_file
tensors = load_file("model.safetensors")
# ... instantiate the matching devagi module and load with strict=False
```

Refer to `src/models/{arch_source}.py` in the source repo for the module
definition.

## Bounded Design Axioms

This model was trained under the six bounded design axioms — see
`DESIGN_PRINCIPLES.md` in the source repo. Every component declares a fixed
capacity; no unbounded growth over training time.

## License

Refer to the source repository.
"""


def _write_readme(cfg: HFExportConfig, out_dir: Path) -> None:
    arch_source = {
        "hybrid_backbone": "hybrid_backbone",
        "rssm": "world_model",
        "rnd": "../intrinsic/rnd",
        "ttt_linear": "ttt_linear",
    }.get(cfg.devagi_arch, cfg.devagi_arch)

    md = _README_TEMPLATE.format(
        model_name=cfg.model_name,
        model_type=cfg.model_type,
        arch=cfg.devagi_arch,
        stage=cfg.devagi_stage,
        ckpt_fmt=cfg.devagi_ckpt_format_version,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        arch_source=arch_source,
    )
    (out_dir / "README.md").write_text(md, encoding="utf-8")


# =====================================================================
# Main export flow
# =====================================================================


def export_checkpoint(
    ckpt_path: Path,
    output_dir: Path,
    model_name: str,
    arch: str,
    dtype: str = "float32",
) -> HFExportConfig:
    """Export ``ckpt_path`` to ``output_dir`` in HF-compatible layout.

    Returns the config that was written.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"arch {arch!r} not supported. Choose from: {SUPPORTED_ARCHS}"
        )

    logger.info("Loading checkpoint: %s", ckpt_path)
    payload = load_ckpt(ckpt_path)

    # Extract model state — checkpoints from `src.utils.ckpt.save_ckpt` use
    # `model_state`. Fall back to top-level dict if user hand-crafted it.
    model_state = payload.get("model_state")
    if model_state is None:
        if all(isinstance(k, str) and isinstance(v, torch.Tensor)
               for k, v in payload.items()):
            model_state = payload  # already a flat state dict
        else:
            raise ValueError(
                "checkpoint has no 'model_state' key and isn't a flat tensor dict"
            )

    # Optional dtype cast (safetensors preserves whatever dtype the tensor has)
    tensors = _flatten_state_dict(model_state)
    if dtype != "float32":
        target = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype)
        if target is None:
            raise ValueError(f"unknown dtype {dtype!r}")
        tensors = {k: v.to(target) if v.is_floating_point() else v
                   for k, v in tensors.items()}

    # Build config
    stage = int(payload.get("stage", 0))
    fmt_version = int(payload.get("_format_version", 1))
    cfg = HFExportConfig(
        model_name=model_name,
        architectures=[f"Devagi{arch.replace('_', ' ').title().replace(' ', '')}"],
        model_type=f"devagi_{arch}",
        torch_dtype=dtype,
        devagi_stage=stage,
        devagi_arch=arch,
        devagi_ckpt_format_version=fmt_version,
        devagi_meta={
            "step": int(payload.get("step", 0)),
            "extra": payload.get("extra", {}),
            "num_parameters": sum(t.numel() for t in tensors.values()),
            "num_tensors": len(tensors),
        },
    )

    # Write everything
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing config.json to %s", output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Writing safetensors weights (%d tensors)", len(tensors))
    _save_safetensors_sharded(tensors, output_dir)

    logger.info("Writing README.md")
    _write_readme(cfg, output_dir)

    total_bytes = sum(t.element_size() * t.numel() for t in tensors.values())
    logger.info(
        "Export complete. %s (%d tensors, %.2f MiB) → %s",
        model_name,
        len(tensors),
        total_bytes / 1024**2,
        output_dir,
    )
    return cfg


# =====================================================================
# CLI
# =====================================================================


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export a devagi checkpoint to Hugging Face format."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to the .pt checkpoint (from src.utils.save_ckpt).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the HF-formatted directory.")
    ap.add_argument("--model-name", type=str, required=True,
                    help="Human-readable model name for config.json / README.")
    ap.add_argument("--arch", type=str, required=True, choices=SUPPORTED_ARCHS,
                    help="Which devagi architecture this checkpoint is for.")
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=("float32", "float16", "bfloat16"),
                    help="Cast weights to this dtype before saving. Default: float32.")
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    export_checkpoint(
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        model_name=args.model_name,
        arch=args.arch,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
