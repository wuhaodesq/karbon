"""Evaluate a trained devagi agent by running it in MiniGrid.

Loads a checkpoint, builds the model, runs N episodes in eval mode
(no learning, no RND, no replay — just the policy network choosing actions),
and prints a per-step trajectory.

Usage:
    python -m scripts.eval_agent \
        --ckpt /root/autodl-tmp/karbon/ckpts/ckpt_stage3_001440256.pt \
        --stage 3 \
        --preset cloud_5090 \
        --episodes 5 \
        --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.envs import MiniGridWrapper
from src.platform import get_device, get_device_info
from src.train import ActorCritic, HybridActorCritic
from src.utils import load_ckpt, load_config


# MiniGrid action names (standard 7 actions)
ACTION_NAMES = [
    "left",       # 0: turn left
    "right",      # 1: turn right
    "forward",    # 2: move forward
    "pickup",     # 3: pick up object
    "drop",       # 4: drop object
    "toggle",     # 5: toggle/activate
    "done",       # 6: done task
]


def _build_model(config: dict, obs_shape: tuple, num_actions: int, device: torch.device):
    """Build the right model based on config."""
    model_cfg = config["model"]
    if bool(model_cfg.get("use_hybrid_backbone", False)):
        model = HybridActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            d_model=int(model_cfg.get("hidden_size", 128)),
            n_layers=int(model_cfg.get("hybrid_n_layers", 3)),
            n_heads=int(model_cfg.get("hybrid_n_heads", 4)),
            swa_window=int(model_cfg.get("hybrid_swa_window", 16)),
            ttt_mini_batch=int(model_cfg.get("hybrid_ttt_mini_batch", 8)),
            ffn_hidden_mult=int(model_cfg.get("hybrid_ffn_hidden_mult", 4)),
            dropout=0.0,  # eval mode, no dropout
        ).to(device)
        model_type = "HybridActorCritic"
    else:
        model = ActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            hidden=int(model_cfg.get("hidden_size", 64)),
        ).to(device)
        model_type = "ActorCritic"
    return model, model_type


def _render_grid(obs: np.ndarray) -> str:
    """Render a simple ASCII representation of the MiniGrid observation.

    Very crude — just shows the 7×7 grid with agent position.
    A proper render would use minigrid's env.render() but that needs a display.
    """
    # obs is (H, W, C) uint8. MiniGrid encodes:
    # Channel 0: object type (0=unseen, 1=empty, 2=wall, 5=agent, 8=goal, ...)
    # Channel 1: color
    # Channel 2: state
    h, w, c = obs.shape
    chars = []
    type_map = {0: '?', 1: '.', 2: '#', 4: 'D', 5: 'A', 6: 'K', 8: 'G', 10: 'B'}
    for row in range(h):
        line = ""
        for col in range(w):
            obj_type = obs[row, col, 0]
            line += type_map.get(int(obj_type), '.')
        chars.append(line)
    return "\n".join(chars)


def evaluate(
    ckpt_path: Path,
    stage: int,
    preset: str,
    num_episodes: int,
    render: bool,
    seed: int,
) -> int:
    device = get_device_info().device if get_device_info().kind == "cuda" else torch.device("cpu")
    if get_device_info().kind != "cuda":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device: {device}")

    # Load config
    config = load_config(f"stage{stage}_baseline.yaml" if stage == 0 else
                         f"stage{stage}_{'baseline' if stage == 0 else 'curiosity' if stage == 1 else 'hybrid' if stage == 2 else 'world_model' if stage == 3 else 'skills' if stage == 4 else 'curriculum' if stage == 5 else 'consolidation'}.yaml",
                         preset)
    config.setdefault("stage", stage)

    # Build env
    env_cfg = config["env"]
    env = MiniGridWrapper(
        env_id=env_cfg["id"],
        seed=seed,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        auto_reset=False,
    )

    # Build model
    obs = env.reset()
    obs_shape = env.observation_shape
    num_actions = env.action_space_n
    model, model_type = _build_model(config, obs_shape, num_actions, device)

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    payload = load_ckpt(ckpt_path)
    try:
        model.load_state_dict(payload["model_state"])
        print(f"  weights loaded successfully")
    except RuntimeError as exc:
        print(f"  WARNING: state mismatch ({exc})")
        return 1
    model.eval()

    ckpt_stage = int(payload.get("stage", stage))
    ckpt_step = int(payload.get("step", 0))
    print(f"  checkpoint: stage={ckpt_stage}, step={ckpt_step}")
    print(f"  model type: {model_type}")
    print(f"  model params: {sum(p.numel() for p in model.parameters())}")

    # Run episodes
    print(f"\n{'='*60}")
    print(f"Running {num_episodes} eval episodes on {env_cfg['id']}")
    print(f"{'='*60}\n")

    returns = []
    lengths = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        ep_steps = 0
        done = False

        if render:
            print(f"--- Episode {ep+1} ---")
            print(_render_grid(obs))
            print()

        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = model(obs_t)
                # Use argmax (greedy) for eval, not sampling
                action = int(logits.argmax(dim=-1).item())
                # Also show action probabilities
                probs = torch.softmax(logits, dim=-1).squeeze(0)

            step_out = env.step(action)
            obs = step_out.obs
            ep_return += step_out.reward
            ep_steps += 1
            done = step_out.terminated or step_out.truncated

            if render and ep_steps <= 30:
                action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
                top_prob = float(probs[action].item())
                print(f"  step {ep_steps:3d}: action={action_name:8s} (p={top_prob:.2f}) "
                      f"reward={step_out.reward:+.3f} done={done}")
                if ep_steps <= 10 or done:
                    print(_render_grid(obs))
                    print()

        returns.append(ep_return)
        lengths.append(ep_steps)
        print(f"  Episode {ep+1}: return={ep_return:.3f}  steps={ep_steps}")
        print()

    # Summary
    print(f"{'='*60}")
    print(f"EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Mean return:   {np.mean(returns):.3f}")
    print(f"  Std return:    {np.std(returns):.3f}")
    print(f"  Mean length:   {np.mean(lengths):.1f} steps")
    print(f"  Returns:       {[round(r, 3) for r in returns]}")
    print(f"  (MiniGrid-5x5 optimal return ≈ 0.94, optimal length ≈ 5-10 steps)")

    env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained devagi agent")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--stage", type=int, default=3, help="Stage number (for config)")
    ap.add_argument("--preset", type=str, default="cloud_5090")
    ap.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    ap.add_argument("--render", action="store_true", help="Print grid + actions")
    ap.add_argument("--seed", type=int, default=42)
    return evaluate(
        ckpt_path=ap.parse_args().ckpt,
        stage=ap.parse_args().stage,
        preset=ap.parse_args().preset,
        num_episodes=ap.parse_args().episodes,
        render=ap.parse_args().render,
        seed=ap.parse_args().seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
