"""Copy-baseline experiment: measure how much obs changes between consecutive
steps in Stage-14 training data. If MSE(obs_t, obs_{t+1}) is large while the
WM's next-loss is ~3.5e-05, the WM has genuinely learned to predict frames.
If the copy baseline is also ~0, transitions are dominated by no-ops and the
next-frame signal is intrinsically weak.

Also reports action statistics (how often the agent actually moves) and
reward frequency, which tells us how much "real dynamics" the sequences contain.

Usage: .venv/bin/python scripts/eval/copy_baseline.py --config configs/stage14_causal_reasoning.yaml --preset cloud_24g --steps 4096
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from src.envs.minigrid_wrapper import MiniGridWrapper
from src.platform import get_device


def _load_cfg(config_path: str, preset: str) -> dict:
    from src.utils.logging import load_config
    try:
        return load_config(Path(config_path).name, preset=preset)
    except Exception:
        pass
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage14_causal_reasoning.yaml")
    ap.add_argument("--preset", default="cloud_24g")
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--env-id", default=None, help="MiniGrid env id (default: from config env.id)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config, args.preset)
    rng = random.Random(args.seed)
    device = get_device()

    env_id = args.env_id or str(cfg["env"].get("id", "MiniGrid-Empty-5x5-v0"))
    max_episode_steps = cfg["env"].get("max_episode_steps")
    render_size = int(cfg["env"].get("render_size", 64))

    env = MiniGridWrapper(
        env_id=env_id,
        seed=rng.randrange(2**31),
        max_episode_steps=max_episode_steps,
        auto_reset=True,
        render_size=render_size,
    )
    print(f"env_id: {env_id}  steps={args.steps}  device={device}")

    obs = []
    nxt = []
    acts = []
    rews = []
    same_frames = 0
    reset_count = 0

    o = env.reset()
    for i in range(args.steps):
        a = rng.randrange(env.action_space_n)
        prev = np.asarray(o, dtype=np.float32)
        step = env.step(a)
        o = step.obs
        obs.append(prev)
        nxt.append(np.asarray(o, dtype=np.float32))
        acts.append(a)
        rews.append(float(step.reward))
        if np.array_equal(prev, np.asarray(o, dtype=np.float32)):
            same_frames += 1
        if step.terminated or step.truncated:
            reset_count += 1
        if i % 1024 == 0:
            print(f"  collected {i}/{args.steps}")

    obs_t = torch.tensor(np.stack(obs), dtype=torch.float32, device=device)
    nxt_t = torch.tensor(np.stack(nxt), dtype=torch.float32, device=device)
    acts = torch.tensor(acts, dtype=torch.long)
    rews = torch.tensor(rews, dtype=torch.float32)

    copy_mse = ((obs_t - nxt_t) ** 2).mean().item()
    copy_mse_norm = copy_mse / 65025.0  # training divides obs by 255; /255^2
    obs_std = obs_t.std().item()
    noop_frac = same_frames / args.steps
    reward_frac = (rews > 0).float().mean().item()
    act_counts = torch.bincount(acts).tolist()

    print("=" * 60)
    print(json.dumps({
        "copy_mse_uint8": copy_mse,
        "copy_mse_norm255": copy_mse_norm,
        "obs_std_uint8": obs_std,
        "same_frame_frac": noop_frac,
        "reward_frac": reward_frac,
        "reset_count": reset_count,
        "action_counts": act_counts,
    }, indent=2))
    print("=" * 60)
    print(f"NOTE: training obs are uint8/255, so compare wm next-loss against copy_mse_norm255={copy_mse_norm:.2e}")
    if copy_mse_norm < 3e-5:
        print(f"VERDICT: copy baseline {copy_mse_norm:.2e} <= wm next-loss (~3.5e-05) -> WM is NOT beating frame-copy; no real prediction yet")
    else:
        print(f"VERDICT: copy baseline {copy_mse_norm:.2e} >> wm next-loss (~3.5e-05) -> WM genuinely predicts frames")


if __name__ == "__main__":
    main()
