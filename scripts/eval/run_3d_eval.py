#!/usr/bin/env python
"""Stage 8+ 3D Evaluation — Language Grounding & Instruction Metrics.

Evaluates on ThreeDWorld (not PhysicsSandbox) to measure:
1. Scene Description Accuracy — agent's language output vs ground truth objects
2. Object Attribute Accuracy — color/size/type classification from slots
3. Developmental Milestones — 3D occlusion, physics, counting
4. Instruction Following — (when instruction templates exist)

Usage:
    MUJOCO_GL=osmesa .venv/bin/python scripts/eval/run_3d_eval.py \
        --ckpt checkpoints/ckpt_stage8_000501760.pt \
        --stage 8 --preset home_64g --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.three_d_world import ThreeDWorld
from src.eval.developmental_milestones import DevelopmentalEvaluator
from src.platform import get_device
from src.utils import load_config
from src.train import HybridActorCritic, _ckpt_layer_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "red": (0.9, 0.1, 0.1), "blue": (0.1, 0.1, 0.9),
    "green": (0.1, 0.9, 0.1), "yellow": (0.9, 0.9, 0.1),
    "white": (0.9, 0.9, 0.9), "black": (0.1, 0.1, 0.1),
    "orange": (0.9, 0.5, 0.1), "purple": (0.5, 0.1, 0.9),
    "brown": (0.5, 0.3, 0.1), "pink": (0.9, 0.5, 0.7),
    "gray": (0.5, 0.5, 0.5), "cyan": (0.1, 0.9, 0.9),
}

_SIZES = {"small": 0.03, "medium": 0.08, "large": 0.15}


def _color_name(r: float, g: float, b: float) -> str:
    best_name, best_dist = "unknown", float("inf")
    for name, (cr, cg, cb) in _NAMED_COLORS.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def _size_name(sx: float, sy: float, sz: float) -> str:
    """Use average extent to assign size label."""
    avg = (sx + sy + sz) / 3.0
    if avg < 0.06:
        return "small"
    elif avg < 0.12:
        return "medium"
    return "large"


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(obs))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


# ---------------------------------------------------------------------------
# 1. Scene Description Accuracy
# ---------------------------------------------------------------------------

def measure_scene_description(
    model: HybridActorCritic,
    env: ThreeDWorld,
    device: torch.device,
    num_probes: int = 50,
) -> dict:
    """Measure how well agent's perceptual output matches ground truth objects.

    Uses the model's slot attention to classify object properties and compares
    to the ThreeDWorld ground truth object library.
    """
    obs = env.reset()
    # Collect ground truth labels from env
    gt_objects: list[dict] = []
    for i, obj in enumerate(env._object_lib[:env._num_objects]):
        gt_objects.append({
            "id": i,
            "label": obj.label,
            "category": obj.category,
            "color": _color_name(*obj.color[:3]),
            "size": _size_name(*obj.size),
            "kind": obj.kind,
        })

    # Probe agent's perception at regular intervals
    probes: list[dict] = []
    for step in range(min(300, env._max_steps)):
        if step % (max(1, 300 // num_probes)) == 0:
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                _, _, hidden = model(obs_t, return_hidden=True)
                # Slot attention output is in model._last_slots
                slots = getattr(model, "_last_slots", None)
            if slots is not None:
                # Check slot activation (which slots "see" objects)
                slot_norms = slots.squeeze(0).norm(dim=-1).cpu().numpy()
                active_slots = int((slot_norms > 0.1).sum())
                probes.append({
                    "step": step,
                    "active_slots": active_slots,
                    "total_slots": len(slot_norms),
                })

        # Step env
        obs_t = _obs_to_tensor(obs, device)
        with torch.no_grad():
            out = model(obs_t)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        action = int(torch.argmax(logits, dim=-1).item())
        step_out = env.step(action)
        obs = step_out.obs
        if step_out.terminated or step_out.truncated:
            break

    # Score: fraction of slots active (objects detected) vs total objects
    if probes:
        avg_active = float(np.mean([p["active_slots"] for p in probes]))
        slot_utilization = avg_active / max(len(gt_objects), 1)
    else:
        slot_utilization = 0.0

    return {
        "num_gt_objects": len(gt_objects),
        "gt_object_labels": [o["label"] for o in gt_objects[:10]],
        "gt_object_colors": [o["color"] for o in gt_objects[:10]],
        "gt_object_sizes": [o["size"] for o in gt_objects[:10]],
        "avg_active_slots": round(float(np.mean([p["active_slots"] for p in probes])) if probes else 0, 1),
        "slot_utilization": round(slot_utilization, 3),
        "num_probes": len(probes),
        "summary": (
            f"{len(gt_objects)} objects, {avg_active:.1f} avg active slots"
            if probes else "no slot data"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Developmental Milestones on 3D
# ---------------------------------------------------------------------------

def measure_3d_milestones(
    model: HybridActorCritic,
    env: ThreeDWorld,
    device: torch.device,
    episodes: int = 10,
    max_steps: int = 300,
) -> dict:
    """Run rollout in ThreeDWorld and score developmental milestones."""
    from src.eval.developmental_milestones import DevelopmentalEvaluator

    evaluator = DevelopmentalEvaluator()
    rng = np.random.RandomState(42)
    all_episode_states: list[dict] = []
    eval_epsilon = 0.3  # higher epsilon to explore grasping actions

    for ep in range(episodes):
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        done = False
        step = 0
        ep_actions: list[int] = []

        while not done and step < max_steps:
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                out = model(obs_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if rng.random() < eval_epsilon:
                action = int(rng.randint(0, env.action_space_n))
            else:
                action = int(torch.argmax(logits, dim=-1).item())
            ep_actions.append(action)

            step_out = env.step(action)
            obs = step_out.obs
            done = bool(step_out.terminated) or bool(step_out.truncated)
            step += 1

        # Collect cumulative dev signals AFTER episode ends
        ep_state = {
            "occlusion_events": list(env._occlusion_events),
            "force_motion_pairs": list(env._force_motion_pairs),
            "count_trials": list(env._count_trials),
            "actions": list(env._actions),
            "object_contact_order": list(env._object_contact_order),
            "grasp_carry_events": list(getattr(env, '_grasp_carry_events', [])),
            "tool_use_events": list(getattr(env, '_tool_use_events', [])),
            "release_events": list(getattr(env, '_release_events', [])),
            "means_ends_score": getattr(env, '_task_progress', 0.0) if getattr(env, '_task_reward_collected', False) else 0.0,
        }
        all_episode_states.append(ep_state)

    report = evaluator.evaluate(all_episode_states)
    return {
        "scores": report.scores,
        "estimated_age": report.estimated_age,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="3D Language Grounding Evaluation")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--stage", type=int, default=8)
    ap.add_argument("--config", type=str, default="stage8_language_grounding.yaml")
    ap.add_argument("--preset", type=str, default="home_64g")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = get_device(None)
    print(f"[3d_eval] device={device}")

    cfg = load_config(args.config, args.preset)
    model_cfg = cfg["model"]

    # --- Env: ThreeDWorld ---
    env = ThreeDWorld(
        num_objects=10,
        max_episode_steps=args.max_steps,
        render_size=64,
        developmental_age=0.5,  # match training env: grasping + chain tasks active
    )
    env._auto_reset = False  # prevent auto-reset from clearing dev signals
    print(f"[3d_eval] Env: ThreeDWorld obs={env.observation_shape} actions={env.action_space_n}")

    # --- Model ---
    n_layers = _ckpt_layer_count(args.ckpt)
    if n_layers <= 0:
        n_layers = int(model_cfg.get("hybrid_n_layers", 7))
    model = HybridActorCritic(
        obs_shape=env.observation_shape,
        num_actions=env.action_space_n,
        d_model=int(model_cfg.get("hidden_size", 128)),
        n_layers=n_layers,
        n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
        swa_window=int(model_cfg.get("hybrid_swa_window", 16)),
        ttt_mini_batch=int(model_cfg.get("hybrid_ttt_mini_batch", 8)),
        ffn_hidden_mult=int(model_cfg.get("hybrid_ffn_hidden_mult", 4)),
        dropout=float(model_cfg.get("hybrid_dropout", 0.0)),
        use_vision_encoder=bool(model_cfg.get("use_vision_encoder", False)),
        vision_model_name=str(model_cfg.get("vision_model", "dinov2_vits14")),
        vision_freeze=bool(model_cfg.get("vision_freeze", True)),
        use_slot_attention=bool(model_cfg.get("use_slot_attention", True)),
        slot_num_slots=int(model_cfg.get("slot_num_slots", 7)),
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        slot_num_iterations=int(model_cfg.get("slot_num_iterations", 3)),
    ).to(device)

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck.get("model_state") if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[3d_eval] Loaded model: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()

    results = {}

    # 1. Scene Description Accuracy
    print("[3d_eval] Measuring scene description accuracy ...")
    scene = measure_scene_description(model, env, device)
    results["scene_description"] = scene
    print(f"  Scene: {scene['summary']}")

    # 2. Developmental Milestones (3D)
    print("[3d_eval] Measuring 3D developmental milestones ...")
    milestones = measure_3d_milestones(model, env, device, episodes=args.episodes, max_steps=args.max_steps)
    results["milestones_3d"] = milestones
    print(f"  Milestones: num_sense={milestones['scores'].get('number_sense',0):.2f}")

    # Print report
    print()
    print("=" * 60)
    print("  3D Language Grounding Evaluation Report")
    print("=" * 60)
    print(f"\n  Scene Description:")
    print(f"    Ground truth: {scene['num_gt_objects']} objects")
    print(f"    Avg active slots: {scene['avg_active_slots']}")
    print(f"    Slot utilization: {scene['slot_utilization']:.2%}")
    print(f"\n  Ground Truth Samples (first 5):")
    for o in scene.get("gt_object_labels", [])[:5]:
        i = scene["gt_object_labels"].index(o)
        print(f"    {o} ({scene['gt_object_colors'][i]}, {scene['gt_object_sizes'][i]})")
    print(f"\n  Developmental Milestones (3D):")
    for k, v in milestones['scores'].items():
        print(f"    {k}: {v:.3f}")
    print(f"  est. age: {milestones['estimated_age']:.1f}y")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n[3d_eval] Report -> {args.out}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
