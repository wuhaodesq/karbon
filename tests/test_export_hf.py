"""Tests for :mod:`scripts.export_hf`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Make scripts/ importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_hf import (  # noqa: E402
    SUPPORTED_ARCHS,
    _flatten_state_dict,
    _save_safetensors_sharded,
    export_checkpoint,
)
from src.utils import save_ckpt  # noqa: E402


def _make_dummy_ckpt(path: Path, stage: int = 2, step: int = 12345) -> Path:
    """Build a tiny checkpoint that mimics a HybridBackbone-like state dict."""
    state = {
        "embedding.weight": torch.randn(16, 8),
        "blocks.0.ttt.theta_K": torch.randn(8, 8),
        "blocks.0.ttt.theta_V": torch.randn(8, 8),
        "blocks.0.ttt.theta_Q": torch.randn(8, 8),
        "blocks.0.ttt.log_eta": torch.tensor(-4.0),
        "blocks.0.attn.qkv_proj.weight": torch.randn(24, 8),
        "blocks.0.attn.qkv_proj.bias": torch.randn(24),
        "blocks.0.ffn.fc1.weight": torch.randn(32, 8),
        "blocks.0.ffn.fc1.bias": torch.randn(32),
        "norm_out.weight": torch.ones(8),
        "norm_out.bias": torch.zeros(8),
    }
    save_ckpt(
        path,
        stage=stage,
        step=step,
        model_state=state,
        optim_state=None,
        extra={"preset": "cloud_24g", "run_id": "test"},
    )
    return path


# =====================================================================
# Flat state dict
# =====================================================================


def test_flatten_state_dict_preserves_leaf_tensors():
    nested = {
        "a": torch.zeros(3),
        "sub": {
            "b": torch.ones(2),
            "deep": {"c": torch.tensor([1.0, 2.0])},
        },
        "non_tensor": "ignored",   # non-tensor entries dropped
    }
    flat = _flatten_state_dict(nested)
    assert set(flat.keys()) == {"a", "sub.b", "sub.deep.c"}
    assert flat["a"].shape == (3,)
    assert flat["sub.b"].shape == (2,)
    assert flat["sub.deep.c"].shape == (2,)


# =====================================================================
# Safetensors writing
# =====================================================================


def test_safetensors_single_shard(tmp_path):
    tensors = {"w": torch.randn(4, 4), "b": torch.zeros(4)}
    idx = _save_safetensors_sharded(tensors, tmp_path)
    # Both tensors in the single shard
    assert set(idx.values()) == {"model.safetensors"}
    assert (tmp_path / "model.safetensors").exists()
    # No index file for single-shard case
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_safetensors_multi_shard(tmp_path):
    """Force sharding by setting a very small size limit."""
    tensors = {
        "a": torch.randn(64, 64),   # 16 KiB
        "b": torch.randn(64, 64),
        "c": torch.randn(64, 64),
    }
    idx = _save_safetensors_sharded(tensors, tmp_path, max_shard_bytes=20_000)
    shard_files = {v for v in idx.values()}
    assert len(shard_files) >= 2
    for f in shard_files:
        assert (tmp_path / f).exists()
    # Index json present
    index_json = tmp_path / "model.safetensors.index.json"
    assert index_json.exists()
    with index_json.open() as f:
        idx_data = json.load(f)
    assert "weight_map" in idx_data
    assert "metadata" in idx_data
    assert idx_data["metadata"]["total_size"] > 0


# =====================================================================
# End-to-end export
# =====================================================================


def test_end_to_end_export_hybrid(tmp_path):
    ckpt_path = _make_dummy_ckpt(tmp_path / "ckpt.pt", stage=2, step=500_000)
    out_dir = tmp_path / "hf_export"

    cfg = export_checkpoint(
        ckpt_path=ckpt_path,
        output_dir=out_dir,
        model_name="devagi-hybrid-test",
        arch="hybrid_backbone",
    )

    # Files present
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "README.md").exists()

    # Config content
    with (out_dir / "config.json").open() as f:
        loaded = json.load(f)
    assert loaded["model_type"] == "devagi_hybrid_backbone"
    assert loaded["devagi_stage"] == 2
    assert loaded["devagi_arch"] == "hybrid_backbone"
    assert loaded["devagi_meta"]["step"] == 500_000
    assert loaded["devagi_meta"]["num_parameters"] > 0

    # Config dataclass
    assert cfg.model_type == "devagi_hybrid_backbone"

    # README mentions the arch and stage
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "hybrid_backbone" in readme
    assert "Stage" in readme or "stage" in readme


def test_end_to_end_export_reload_matches(tmp_path):
    """Weights loaded from the exported safetensors must equal the source."""
    from safetensors.torch import load_file

    ckpt_path = _make_dummy_ckpt(tmp_path / "ckpt.pt")
    out_dir = tmp_path / "hf_export"

    payload_before = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state = payload_before["model_state"]

    export_checkpoint(
        ckpt_path=ckpt_path,
        output_dir=out_dir,
        model_name="devagi-hybrid-test",
        arch="hybrid_backbone",
    )

    reloaded = load_file(str(out_dir / "model.safetensors"))
    assert set(reloaded.keys()) == set(source_state.keys())
    for k in source_state:
        torch.testing.assert_close(reloaded[k], source_state[k])


def test_dtype_cast(tmp_path):
    ckpt_path = _make_dummy_ckpt(tmp_path / "ckpt.pt")
    out_dir = tmp_path / "hf_export"

    export_checkpoint(
        ckpt_path=ckpt_path,
        output_dir=out_dir,
        model_name="devagi-fp16",
        arch="hybrid_backbone",
        dtype="float16",
    )
    from safetensors.torch import load_file
    reloaded = load_file(str(out_dir / "model.safetensors"))
    # Float tensors should now be fp16; the scalar log_eta is fp32→fp16 too
    for k, v in reloaded.items():
        if v.is_floating_point():
            assert v.dtype == torch.float16, f"{k} not fp16: {v.dtype}"


def test_rejects_unknown_arch(tmp_path):
    ckpt_path = _make_dummy_ckpt(tmp_path / "ckpt.pt")
    out_dir = tmp_path / "hf_export"
    with pytest.raises(ValueError, match="not supported"):
        export_checkpoint(
            ckpt_path=ckpt_path,
            output_dir=out_dir,
            model_name="x",
            arch="mystery_arch",
        )


def test_supported_archs_list_stable():
    assert set(SUPPORTED_ARCHS) == {"hybrid_backbone", "rssm", "rnd", "ttt_linear"}
