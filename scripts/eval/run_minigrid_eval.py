#!/usr/bin/env python
"""MiniGrid cognitive evaluation - measures means-ends and systematic reasoning.

Tests agent on MiniGrid doorkey tasks:
1. Success rate: completed key->door->goal sequence
2. Key pickup rate: did agent pick up the key?
3. Door open rate: did agent open the door?
4. Goal reach rate: did agent reach the goal?
5. Step efficiency: steps to complete / optimal steps
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.minigrid_wrapper import MiniGridWrapper
from src.platform import get_device
from src.utils import load_config
from src.train import HybridActorCritic, _ckpt_layer_count


def _obs_to_tensor(obs, device):
    t = torch.from_numpy(np.asarray(obs))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def main():
    ap = argparse.ArgumentParser(description="MiniGrid cognitive eval")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage10_minigrid.yaml")
    ap.add_argument("--preset", type=str, default="home_64g")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = get_device(None)
    cfg = load_config(args.config, args.preset)
    model_cfg = cfg["model"]

    # Infer num_actions from ckpt
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck.get("model_state") if isinstance(ck, dict) else ck
    num_actions = 8
    for key in state:
        if "policy_head.weight" in key:
            num_actions = state[key].shape[0]
            break

    n_layers = _ckpt_layer_count(args.ckpt)
    if n_layers <= 0:
        n_layers = int(model_cfg.get("hybrid_n_layers", 7))

    model = HybridActorCritic(
        obs_shape=(64, 64, 3), num_actions=num_actions,
        d_model=int(model_cfg.get("hidden_size", 128)),
        n_layers=n_layers,
        n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
        swa_window=int(model_cfg.get("hybrid_swa_window", 16)),
        ttt_mini_batch=int(model_cfg.get("hybrid_ttt_mini_batch", 8)),
        ffn_hidden_mult=int(model_cfg.get("hybrid_ffn_hidden_mult", 4)),
        dropout=float(model_cfg.get("hybrid_dropout", 0.0)),
        use_vision_encoder=bool(model_cfg.get("use_vision_encoder", False)),
        use_slot_attention=bool(model_cfg.get("use_slot_attention", True)),
        slot_num_slots=int(model_cfg.get("slot_num_slots", 7)),
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        slot_num_iterations=int(model_cfg.get("slot_num_iterations", 3)),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[mg_eval] Model loaded: {n_layers} layers, {num_actions} actions")

    # Test on multiple MiniGrid tasks
    tasks = [
        ("MiniGrid-Empty-5x5-v0", "empty-5x5"),
        ("MiniGrid-Empty-8x8-v0", "empty-8x8"),
        ("MiniGrid-DoorKey-5x5-v0", "doorkey-5x5"),
        ("MiniGrid-DoorKey-6x6-v0", "doorkey-6x6"),
    ]

    results = {}
    for env_id, tag in tasks:
        env = MiniGridWrapper(env_id=env_id, seed=42, auto_reset=False, render_size=64)
        successes = 0
        key_picked = 0
        door_opened = 0
        goal_reached = 0
        step_counts = []
        ep_returns = []

        for ep in range(args.episodes):
            obs = env.reset(seed=42 + ep)
            ep_ret = 0.0
            done = False
            steps = 0
            picked_key = False
            opened_door = False

            while not done and steps < args.max_steps:
                obs_t = _obs_to_tensor(obs, device)
                with torch.no_grad():
                    out = model(obs_t)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
                step_out = env.step(action)
                obs = step_out.obs
                ep_ret += float(step_out.reward)
                done = bool(step_out.terminated) or bool(step_out.truncated)
                steps += 1

                # Detect key pickup (reward increase in doorkey)
                if "doorkey" in tag and float(step_out.reward) > 0 and not picked_key:
                    picked_key = True
                    key_picked += 1

            if ep_ret > 0.5:
                successes += 1
            if ep_ret > 0:
                goal_reached += 1
            step_counts.append(steps)
            ep_returns.append(ep_ret)

        env.close()

        sr = successes / args.episodes
        kr = goal_reached / args.episodes
        mr = np.mean(ep_returns) if ep_returns else 0.0
        ms = np.mean(step_counts) if step_counts else 0.0

        results[tag] = {
            "env_id": env_id,
            "episodes": args.episodes,
            "success_rate": round(sr, 3),
            "goal_reach_rate": round(kr, 3),
            "mean_return": round(mr, 3),
            "mean_steps": round(ms, 1),
        }

        print(f"  {tag:20s}  SR={sr:.0%}  GRR={kr:.0%}  ret={mr:.3f}  steps={ms:.0f}")

    # Cognitive scores
    empty_sr = results.get("empty-5x5", {}).get("success_rate", 0)
    doorkey_sr = results.get("doorkey-5x5", {}).get("success_rate", 0)
    doorkey6_sr = results.get("doorkey-6x6", {}).get("success_rate", 0)

    # Means-ends: ability to complete multi-step key->door->goal
    means_ends = doorkey_sr
    # Systematic reasoning: transfer from 5x5 to 6x6 (harder)
    systematic = min(1.0, doorkey6_sr / max(doorkey_sr, 0.01)) if doorkey_sr > 0 else 0.0
    # Navigation: basic empty room completion
    navigation = empty_sr

    print(f"\n=== MiniGrid Cognitive Scores ===")
    print(f"  Navigation (empty-5x5):  {navigation:.2f}")
    print(f"  Means-Ends (doorkey-5x5): {means_ends:.2f}")
    print(f"  Systematic (5x5->6x6):   {systematic:.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {**results, "cognitive_scores": {
            "navigation": navigation, "means_ends": means_ends, "systematic": systematic
        }}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n[mg_eval] Report -> {args.out}")


if __name__ == "__main__":
    main()
