"""Diagnose WM training dynamics on the live checkpoint.

Answers:
1. Does a backward through compute_loss actually produce gradients in the
   posterior/encoder/decoder (gradient path intact)?
2. Is recon loss already near the "mean-frame floor" (~3.5e-05)?
3. What is the real KL before the free-nats clamp (is posterior really
   collapsing, or is it the clamp hiding a healthy KL)?

Usage (on the training box, while training runs):
  PYTHONPATH=/root/karbon .venv/bin/python scripts/eval/wm_diag.py \
      --checkpoint checkpoints/ckpt_stage14_000251904.pt \
      --config configs/stage14_causal_reasoning.yaml --preset cloud_24g
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.minigrid_wrapper import MiniGridWrapper
from src.models.world_model import RSSM, RSSMConfig, RSSMState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--inspect", action="store_true", help="only print ckpt key structure")
    ap.add_argument("--config", default="configs/stage14_causal_reasoning.yaml")
    ap.add_argument("--preset", default="cloud_24g")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--env-id", default="MiniGrid-RedBlueDoors-6x6-v0")
    ap.add_argument("--n-batches", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.inspect:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        print(type(ckpt))
        if isinstance(ckpt, dict):
            ms = ckpt.get("model_state")
            extra = ckpt.get("extra")
            if isinstance(ms, dict):
                print(f"model_state: {len(ms)} keys")
            if isinstance(extra, dict):
                print(f"extra keys: {list(extra.keys())[:15]}")
                wm = extra.get("wm_state")
                if isinstance(wm, dict):
                    print(f"wm_state: {len(wm)} keys, sample {list(wm.keys())[:6]}")
        return

    wm_cfg = cfg["world_model"]
    env_cfg = cfg["env"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    env = MiniGridWrapper(
        env_id=args.env_id,
        seed=42,
        max_episode_steps=env_cfg.get("max_episode_steps"),
        auto_reset=True,
        render_size=int(env_cfg.get("render_size", 64)),
    )
    obs_shape = env.observation_shape  # (H, W, C)
    num_actions = env.action_space_n

    rssm_cfg = RSSMConfig(
        obs_dim=int(obs_shape[0] * obs_shape[1] * obs_shape[2]),
        action_dim=num_actions,
        z_dim=int(wm_cfg["z_dim"]),
        h_dim=int(wm_cfg["h_dim"]),
        embed_dim=int(wm_cfg["embed_dim"]),
        hidden=int(wm_cfg["hidden"]),
        max_rollout_steps=int(wm_cfg["max_rollout_steps"]),
        kl_free_nats=float(wm_cfg["kl_free_nats"]),
        recon_loss_weight=float(wm_cfg.get("recon_loss_weight", 1.0)),
        reward_loss_weight=float(wm_cfg.get("reward_loss_weight", 1.0)),
        next_loss_coef=float(wm_cfg.get("next_loss_coef", 1.0)),
    )
    wm = RSSM(rssm_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    sd = extra.get("wm_state") or extra.get("world_model_state")
    if not isinstance(sd, dict):
        sd = {}
    if not sd:
        print("FATAL: no wm_state in checkpoint extra")
        return
    missing, unexpected = wm.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARNING missing keys: {missing[:5]}")

    # --- collect a real trajectory ---
    print("collecting trajectory...")
    obs_list, act_list, rew_list = [], [], []
    o = env.reset()
    for _ in range(args.seq_len * args.n_batches * 20):
        a = int(torch.randint(0, num_actions, (1,)).item())
        st = env.step(a)
        obs_list.append(o.copy())
        act_list.append(a)
        rew_list.append(float(st.reward))
        o = st.obs
        if st.terminated or st.truncated:
            o = env.reset()

    obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device) / 255.0
    obs_t = obs_t.reshape(len(obs_list), -1)
    acts_t = torch.tensor(act_list, dtype=torch.long, device=device)
    rews_t = torch.tensor(rew_list, dtype=torch.float32, device=device)

    # --- run compute_loss + backward, check grads ---
    all_grad_norms: dict[str, float] = {}
    kl_before: float = float("nan")
    recon_vals, next_vals, copy_vals = [], [], []
    for bi in range(args.n_batches):
        s = bi * args.seq_len
        obs_seq = obs_t[s : s + args.seq_len].unsqueeze(0)
        act_seq = F.one_hot(acts_t[s : s + args.seq_len], num_classes=num_actions).float().unsqueeze(0)
        next_seq = obs_t[s + 1 : s + 1 + args.seq_len].unsqueeze(0)
        rew_seq = rews_t[s : s + args.seq_len].unsqueeze(0).unsqueeze(-1)

        wm.zero_grad()
        out = wm.compute_loss(obs_seq, act_seq, reward_seq=rew_seq, next_obs_seq=next_seq)
        out["loss"].backward()
        for name, p in wm.named_parameters():
            if p.grad is not None:
                gn = p.grad.norm().item()
                all_grad_norms[name] = max(all_grad_norms.get(name, 0.0), gn)
        kl_before = float(out["kl_loss"].item())
        recon_vals.append(float(out["recon_loss"].item()))
        next_vals.append(float(out["next_loss"].item()))
        copy_mse = float((obs_seq[:, :-1, :] - next_seq[:, :-1, :]).pow(2).mean().item())
        copy_vals.append(copy_mse)
        print(f"batch {bi}: loss={out['loss'].item():.6f} recon={out['recon_loss'].item():.3e} "
              f"kl(clamped)={out['kl_loss'].item():.6f} kl_raw={out['kl_raw'].item():.6f} "
              f"next={out.get('next_loss', torch.tensor(0.0)).item():.3e} copy={copy_mse:.3e}")

    print(f"\nAGGREGATE: mean recon={sum(recon_vals)/len(recon_vals):.3e} "
          f"mean next={sum(next_vals)/len(next_vals):.3e} mean copy={sum(copy_vals)/len(copy_vals):.3e}")
    print(f"VERDICT: wm beats copy if next < copy ({sum(next_vals)/len(next_vals):.3e} vs {sum(copy_vals)/len(copy_vals):.3e})")

    # report the biggest gradient norms by module group
    groups = {"encoder": 0.0, "posterior": 0.0, "prior": 0.0, "decoder": 0.0, "recurrent": 0.0, "reward_head": 0.0}
    for name, gn in all_grad_norms.items():
        for g in groups:
            if g in name:
                groups[g] = max(groups[g], gn)
    print(json.dumps({"kl_clamped": kl_before, "max_grad_norms": groups}, indent=2))
    print("HINT: if encoder/posterior grads are tiny (<1e-4) the recon gradient never reaches z;")
    print("      if they are large (>1e-2) the path is alive and it is a scale/learning-rate issue.")


if __name__ == "__main__":
    main()
