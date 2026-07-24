#!/usr/bin/env python
"""On-demand independent evaluator (3-axis: curiosity/drive/task)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.independent_evaluator import IndependentEvaluator
from src.platform import get_device
from src.utils.ckpt import load_ckpt
from src.train import HybridActorCritic, _ckpt_layer_count


def build_model(obs_shape, num_actions, model_cfg, device, n_layers):
    return HybridActorCritic(
        obs_shape=obs_shape, num_actions=num_actions,
        d_model=int(model_cfg.get("hidden_size", 128)),
        n_layers=n_layers, n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage6_consolidation.yaml")
    ap.add_argument("--preset", type=str, default="home_64g")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = get_device(None)
    print(f"[ie] device={device}")

    from src.utils import load_config
    cfg = load_config(args.config, args.preset)
    model_cfg = cfg["model"]

    n_layers = _ckpt_layer_count(args.ckpt)
    if n_layers <= 0:
        n_layers = int(model_cfg.get("hybrid_n_layers", 7))

    model = build_model((64, 64, 3), 8, model_cfg, device, n_layers)
    payload = load_ckpt(args.ckpt)
    sd = payload["model_state"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    step = payload.get("step", 0)

    ie = IndependentEvaluator(cfg, device)
    report = ie.evaluate(model, drives_module=None, step=step)
    print(f"\n[ie] step={step} | curiosity={report.curiosity:.3f} "
          f"drive={report.drive:.3f} task={report.task:.3f} "
          f"(vs_random={report.task_vs_random:.2f}) | total={report.total:.3f} "
          f"(w={ie.weights}) advisory='{report.advisory}'")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "step": step, "curiosity": report.curiosity, "drive": report.drive,
            "task": report.task, "total": report.total,
            "task_vs_random": report.task_vs_random, "advisory": report.advisory,
        }, indent=2) + "\n")
        print(f"[ie] report -> {args.out}")


if __name__ == "__main__":
    main()
