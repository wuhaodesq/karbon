#!/usr/bin/env python
"""C#8 发育里程碑评测脚本 (open-gap C#8 evaluation harness).

加载一个训练好的 checkpoint,在 PhysicsSandbox 上跑若干 episode 的 rollout,
收集 env 暴露的发育信号 (occlusion_events / force_motion_pairs / count_trials),
然后交给 ``src.eval.developmental_milestones.DevelopmentalEvaluator`` 打分量表,
输出 *估计认知年龄* (estimated_age)。

用途:
    - Stage 5 / Stage 6 退出复验:量化 "认知到达几岁",给训练路线闭环反馈。
    - 对比不同 ckpt 的发育年龄,作为 exit criterion 的 *认知能力* 维度补充
      (现有 exit 标准只衡量系统韧性: 30 天 / 10 任务 / 显存趋平)。

数感 (number sense) 默认接 **真实 NumberSense 头** (从 ckpt 的
``extra.number_sense_state`` 加载权重), 比 env 的 "接触不同物体数" 行为代理更准;
若 ckpt 无该权重则回退到行为代理, 并在诊断中标注来源。

用法:
    python scripts/eval/run_developmental_eval.py \\
        --ckpt /root/autodl-tmp/karbon_ckpts/checkpoints/ckpt_stage5_002000000.pt \\
        --stage 6 --preset home_64g --episodes 20 --max-steps 1000

注意:
    - 只评测 PhysicsSandbox 可控域内能测的 3 个里程碑 (客体永存 / 直觉物理 /
      数感); 心智理论 / 手段-目的 / 系统推理 在 2D 沙盒无信号, 保持 0
      (占位接口, 待 3D / Y1 环境接入)。
    - 有界: rollout 步数 / episode 数固定, 不创建无界缓冲。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 让 `src` 可导入 (脚本位于 <root>/scripts/eval/)
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.developmental_milestones import DevelopmentalEvaluator  # noqa: E402
from src.envs.physics_sandbox import PhysicsSandbox  # noqa: E402
from src.models.number_sense import NumberSense  # noqa: E402
from src.platform import get_device  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.train import (  # noqa: E402
    HybridActorCritic,
    _ckpt_layer_count,
)


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(obs))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def build_model(
    obs_shape: tuple[int, ...],
    num_actions: int,
    model_cfg: dict,
    device: torch.device,
    n_layers: int,
) -> HybridActorCritic:
    model = HybridActorCritic(
        obs_shape=obs_shape,
        num_actions=num_actions,
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
        use_slot_attention=bool(model_cfg.get("use_slot_attention", False)),
        slot_num_slots=int(model_cfg.get("slot_num_slots", 7)),
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        slot_num_iterations=int(model_cfg.get("slot_num_iterations", 3)),
    ).to(device)
    return model


def build_number_sense(model_cfg: dict, num_sense_cfg: dict, device: torch.device,
                       state: dict | None):
    """Build NumberSense head; load weights if provided. Returns (module|None).

    Infers max_count from the checkpoint weights when available (handles
    the config change from max_count=10→20 across ckpt versions).
    """
    if num_sense_cfg is None:
        return None
    max_count = int(num_sense_cfg.get("max_count", 10))
    if state is not None:
        # Infer actual max_count from saved weight shape
        for key, tensor in state.items():
            if "net.2" in key and tensor.dim() == 2:
                max_count = tensor.shape[0] - 1  # output classes - 1
                break
    ns = NumberSense(
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        max_count=max_count,
        hidden=int(num_sense_cfg.get("hidden", 32)),
    ).to(device)
    if state:
        ns.load_state_dict(state, strict=False)
        return ns
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="C#8 developmental milestone eval")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--stage", type=int, default=6)
    ap.add_argument("--preset", type=str, default="home_64g")
    ap.add_argument("--config", type=str, default=None,
                    help="Stage config filename (default: pick by stage)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="Max env steps per episode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-number-head", action="store_true",
                    help="Disable NumberSense head; use env behavioral proxy")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write JSON report")
    args = ap.parse_args()

    device = get_device(None)
    print(f"[eval] device={device}")

    # --- Config + model build ---
    if args.config:
        cfg_name = args.config
    elif args.stage == 6:
        cfg_name = "stage6_consolidation.yaml"
    else:
        cfg_name = f"stage{args.stage}_curriculum.yaml"
    cfg = load_config(cfg_name, args.preset)
    model_cfg = cfg["model"]
    num_sense_cfg = cfg.get("number_sense")

    # --- Env (PhysicsSandbox, matching resume ckpt obs shape) ---
    env_cfg = cfg.get("env", {})
    env = PhysicsSandbox(
        num_objects=int(env_cfg.get("num_objects", 10)),
        seed=args.seed,
        max_episode_steps=int(env_cfg.get("max_episode_steps", 300)),
        render_size=int(env_cfg.get("render_size", 64)),
        gravity=float(env_cfg.get("gravity", -9.8)),
        action_force=float(env_cfg.get("action_force", 50.0)),
    )
    obs_shape = env.observation_shape
    num_actions = env.action_space_n
    print(f"[eval] Env: PhysicsSandbox obs_shape={obs_shape} actions={num_actions}")

    # --- Build model with SAME layer count as ckpt, load weights ---
    n_layers = _ckpt_layer_count(args.ckpt)
    if n_layers <= 0:
        n_layers = int(model_cfg.get("hybrid_n_layers", 7))
    model = build_model(obs_shape, num_actions, model_cfg, device, n_layers)
    print(f"[eval] Model: HybridActorCubit layers={n_layers}")

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck.get("model_state") if isinstance(ck, dict) else None
    if state is None:
        state = ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] loaded model_state; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # --- NumberSense head (real counting) ---
    number_sense = None
    if not args.no_number_head:
        ns_state = (ck.get("extra") or {}).get("number_sense_state")
        number_sense = build_number_sense(model_cfg, num_sense_cfg, device, ns_state)
    use_head = number_sense is not None
    print(f"[eval] number_sense source: {'NumberSense head' if use_head else 'env behavioral proxy'}")

    # --- Rollout, collect developmental signals ---
    rng = np.random.RandomState(args.seed)
    env_states: list[dict] = []
    head_count_trials: list[dict] = []  # filled only when use_head
    for ep in range(args.episodes):
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        done = False
        steps = 0
        ep_est: list[int] = []
        while not done and steps < args.max_steps:
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                out = model(obs_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            action = int(torch.argmax(logits, dim=-1).item())
            if use_head:
                slots = model._last_slots
                if slots is not None:
                    ep_est.append(int(number_sense.predict_count(slots).item()))
            step_out = env.step(action)
            info = dict(step_out.info) if step_out.info else {}
            if use_head:
                # replace env behavioral proxy with head-based (fed at ep end)
                info["count_trials"] = []
            env_states.append(info)
            obs = step_out.obs
            done = bool(step_out.terminated) or bool(step_out.truncated)
            steps += 1
        # episode-end true count from env public snapshot
        true_count = int(env.read_states()["num_objects"])
        if use_head:
            est = int(round(float(np.mean(ep_est)))) if ep_est else 0
            head_count_trials.append(
                {"true_count": true_count, "estimated_count": est})
        print(f"[eval] episode {ep + 1}/{args.episodes} steps={steps} done={done}")
    if use_head:
        env_states.append({
            "occlusion_events": [],
            "force_motion_pairs": [],
            "count_trials": head_count_trials,
        })

    # --- Diagnostics: how many signal samples did we actually collect? ---
    n_occ = sum(len(s.get("occlusion_events", [])) for s in env_states)
    n_fm = sum(len(s.get("force_motion_pairs", [])) for s in env_states)
    n_ct = sum(len(s.get("count_trials", [])) for s in env_states)
    print(f"[diag] signal samples: occlusion={n_occ} force_motion={n_fm} "
          f"count_trials={n_ct} (n_states={len(env_states)})")

    # --- Evaluate milestones ---
    report = DevelopmentalEvaluator().evaluate(env_states)
    print()
    print(report.summary())

    if args.out:
        payload = {
            "ckpt": args.ckpt,
            "stage": args.stage,
            "preset": args.preset,
            "episodes": args.episodes,
            "n_states": len(env_states),
            "signal_samples": {"occlusion": n_occ, "force_motion": n_fm,
                               "count_trials": n_ct},
            "number_sense_source": "head" if use_head else "env_proxy",
            "scores": report.scores,
            "passed": report.passed,
            "estimated_age": report.estimated_age,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n[eval] JSON report -> {args.out}")


if __name__ == "__main__":
    main()
