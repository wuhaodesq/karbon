#!/usr/bin/env python
"""Stage 18 Full Evaluation at 500K — 全量评测脚本.

与 Stage 18 训练配置精确匹配:
    - Env:  ThreeDWorld(num_objects=8, max_episode_steps=300,
            render_size=128, action_force=50.0, developmental_age=0.5)
    - Model: HierarchicalActorCritic (12 actions, 7 slots, d_model=128,
            slot_dim=128, sub_goal_every=10, layers from ckpt)

评测内容:
    1. 3D 发育里程碑 (est. age / object_permanence / means_ends /
       intuitive_physics / number_sense / theory_of_mind / systematic_reasoning)
    2. ToM 模块专项 (加载 theory_of_mind_state, false-belief 测试)
    3. NumberSense 头 (加载 number_sense_state, 数量预测准确率)
    4. Scene description (slot 利用率)
    5. 训练状态摘要 (因果边/规则数/技能/想象更新/eval_trajectory)

用法:
    python scripts/eval/run_stage18_full_eval.py \
        --ckpt /root/autodl-tmp/karbon/ckpts/ckpt_stage18_000501760.pt \
        --out /root/stage18_500k_full_eval.json

有界: rollout 步数/episode 数固定, 不创建无界缓冲。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.three_d_world import ThreeDWorld
from src.eval.developmental_milestones import DevelopmentalEvaluator
from src.models.hierarchical_policy import HierarchicalActorCritic
from src.models.number_sense import NumberSense
from src.models.theory_of_mind import TheoryOfMind
from src.platform import get_device
from src.utils import load_config
from src.train import _ckpt_layer_count


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(obs)).float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def build_model(obs_shape, num_actions, model_cfg, device, n_layers):
    return HierarchicalActorCritic(
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
        use_slot_attention=bool(model_cfg.get("use_slot_attention", False)),
        slot_num_slots=int(model_cfg.get("slot_num_slots", 7)),
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        slot_num_iterations=int(model_cfg.get("slot_num_iterations", 3)),
        sub_goal_every=int(model_cfg.get("sub_goal_every", 10)),
    ).to(device)


def measure_milestones(model, env, device, episodes=20, max_steps=300, epsilon=0.1,
                       rule_count=0, seed=42):
    """Rollout in 3D env, collect dev signals, score milestones.

    ``rule_count`` (from ckpt symbolic state) is passed through to the
    systematic_reasoning milestone; without it that term is always 0 and the
    milestone can never reach its 0.6 threshold (Stage 18 eval fix).
    ``seed`` controls the rollout RNG (multi-seed eval for stability checks).
    """
    evaluator = DevelopmentalEvaluator()
    rng = np.random.RandomState(seed)
    all_states = []
    env._auto_reset = False  # prevent auto-reset from clearing dev signals
    for ep in range(episodes):
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        done = False
        step = 0
        while not done and step < max_steps:
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                out = model(obs_t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if rng.random() < epsilon:
                action = int(rng.randint(0, env.action_space_n))
            else:
                action = int(torch.argmax(logits, dim=-1).item())
            step_out = env.step(action)
            obs = step_out.obs
            done = bool(step_out.terminated) or bool(step_out.truncated)
            step += 1
        all_states.append({
            "occlusion_events": list(env._occlusion_events),
            "force_motion_pairs": list(env._force_motion_pairs),
            "count_trials": list(env._count_trials),
            "actions": list(env._actions),
            "object_contact_order": list(env._object_contact_order),
            "grasp_carry_events": list(getattr(env, "_grasp_carry_events", [])),
            "tool_use_events": list(getattr(env, "_tool_use_events", [])),
            "release_events": list(getattr(env, "_release_events", [])),
            "means_ends_score": (
                getattr(env, "_task_progress", 0.0)
                if getattr(env, "_task_reward_collected", False) else 0.0),
            "rule_count": int(rule_count),
            "num_actions": int(env.action_space_n),
        })
    report = evaluator.evaluate(all_states)
    return report


def measure_tom(tom_state, device, d_model=128, num_actions=12, num_slots=7):
    """Load ToM weights from ckpt and run targeted tests."""
    tom = TheoryOfMind(d_model=d_model, num_actions=num_actions,
                       num_slots=num_slots).to(device)
    if tom_state:
        tom.load_state_dict(tom_state)
    tom.eval()

    rng = np.random.RandomState(7)
    results = {}

    # Test 1: Perspective taking — caregiver far away sees fewer objects
    with torch.no_grad():
        self_slots = torch.randn(1, num_slots, d_model, device=device)
        near = torch.tensor([[0.0, 0.2, 0.0]], device=device)
        far = torch.tensor([[5.0, 5.0, 0.0]], device=device)
        obj_pos = torch.randn(8, 3, device=device) * 0.5
        near_vis = tom.predict_perspective(self_slots, near, obj_pos)
        far_vis = tom.predict_perspective(self_slots, far, obj_pos)
        # Visibility scaling: far should attenuate more (lower norm scale)
        n_near = near_vis.norm(dim=(-2, -1)).mean().item()
        n_far = far_vis.norm(dim=(-2, -1)).mean().item()
        results["perspective_distance_attenuation"] = round(float(n_far / max(n_near, 1e-6)), 3)
        results["perspective_ok"] = bool(n_far < n_near)

    # Test 2: Belief update + action prediction
    with torch.no_grad():
        tom.reset_beliefs()
        agg = torch.randn(1, d_model, device=device)
        tom.update_belief("caregiver", agg)
        act = tom.predict_other_action("caregiver")
        results["action_prediction_shape_ok"] = bool(act.shape == (1, num_actions))

    # Test 3: False-belief — belief should NOT encode hidden object
    with torch.no_grad():
        tom.reset_beliefs()
        visible = torch.randn(1, d_model, device=device) * 0.3
        tom.update_belief("caregiver", visible)  # only saw visible objects
        hidden_slot = torch.randn(d_model, device=device) * 2.0  # never seen
        ok = tom.false_belief_test("caregiver", hidden_slot)
        results["false_belief_ok"] = bool(ok)

    # Test 4: Surprise — low sim (surprising obs) → higher surprise
    with torch.no_grad():
        surprise_low = tom.predict_other_surprise("caregiver", visible.squeeze(0))
        surprise_high = tom.predict_other_surprise("caregiver", hidden_slot)
        results["surprise_discrimination"] = round(float(surprise_high - surprise_low), 3)
        results["surprise_ok"] = bool(surprise_high > surprise_low)

    return results


def measure_number_sense(ns_state, model, env, device, model_cfg, num_sense_cfg, trials=30):
    """NumberSense head accuracy: predict count from slots vs env truth."""
    max_count = int(num_sense_cfg.get("max_count", 10)) if num_sense_cfg else 10
    if ns_state:
        for key, tensor in ns_state.items():
            if "net.2" in key and tensor.dim() == 2:
                max_count = tensor.shape[0] - 1
                break
    ns = NumberSense(slot_dim=int(model_cfg.get("slot_dim", 128)),
                     max_count=max_count,
                     hidden=int(num_sense_cfg.get("hidden", 32)) if num_sense_cfg else 32).to(device)
    ns.load_state_dict(ns_state)
    ns.eval()

    rng = np.random.RandomState(3)
    correct = 0
    errs = []
    for _ in range(trials):
        env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        obs = env._render()
        obs_t = _obs_to_tensor(obs, device)
        with torch.no_grad():
            model(obs_t)
            slots = model._last_slots
            if slots is None:
                continue
            est = int(ns.predict_count(slots).item())
        true_count = len(env.objects)
        if est == true_count:
            correct += 1
        errs.append(abs(est - true_count))
    n = max(len(errs), 1)
    return {"trials": len(errs),
            "accuracy": round(correct / n, 3),
            "mean_abs_err": round(float(np.mean(errs)), 3),
            "true_count": len(env.objects)}


def measure_scene(model, env, device, num_probes=30):
    """Slot utilization on the current scene."""
    obs = env._render()
    probes = []
    for _ in range(num_probes):
        obs_t = _obs_to_tensor(obs, device)
        with torch.no_grad():
            model(obs_t)
            slots = getattr(model, "_last_slots", None)
        if slots is not None:
            norms = slots.squeeze(0).norm(dim=-1).cpu().numpy()
            probes.append(int((norms > 0.1).sum()))
        # step with random action to vary view
        step_out = env.step(int(rng.randint(0, env.action_space_n)))
        obs = step_out.obs
    return {
        "num_objects": len(env.objects),
        "num_slots": 7,
        "avg_active_slots": round(float(np.mean(probes)), 2) if probes else 0,
        "slot_utilization": round(float(np.mean(probes)) / 7.0, 3) if probes else 0.0,
    }


rng = np.random.RandomState(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="stage16_neuro_symbolic.yaml")
    ap.add_argument("--preset", type=str, default="cloud_24g")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42,
                    help="rollout RNG seed (multi-seed stability eval)")
    ap.add_argument("--out", type=str, default="/root/stage18_500k_full_eval.json")
    ap.add_argument("--tasks", type=str, default="all",
                    help="comma list of curriculum task ids to evaluate, or 'all'")
    args = ap.parse_args()

    device = get_device(None)
    print(f"[s18e] device={device}")

    cfg = load_config(args.config, args.preset)
    model_cfg = cfg["model"]
    num_sense_cfg = cfg.get("number_sense")

    def make_env(num_objects, action_force):
        return ThreeDWorld(
            num_objects=int(num_objects),
            max_episode_steps=args.max_steps,
            render_size=int(cfg["env"].get("render_size", 128)),
            action_force=float(action_force),
            developmental_age=float(cfg["env"].get("developmental_age", 0.5)),
            num_occluders=int(cfg["env"].get("num_occluders", 0)),
            # occluder_trace intentionally NOT forwarded: eval must measure
            # true memory-based tracking without trace feedback (train/eval
            # env parity fix, 2026-08-13).
        )

    env = make_env(cfg["env"].get("num_objects", 8),
                   cfg["env"].get("action_force", 50.0))
    env._auto_reset = False
    print(f"[s18e] Env: ThreeDWorld obs={env.observation_shape} "
          f"actions={env.action_space_n} objects={env._num_objects}")
    n_layers = _ckpt_layer_count(args.ckpt)
    if n_layers <= 0:
        n_layers = int(model_cfg.get("hybrid_n_layers", 7))
    model = build_model(env.observation_shape, env.action_space_n, model_cfg, device, n_layers)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    step = ck.get("step", 0) if isinstance(ck, dict) else 0
    print(f"[s18e] Loaded step={step} | {len(missing)} missing, {len(unexpected)} unexpected")

    extra = ck.get("extra", {}) if isinstance(ck, dict) else {}
    rule_count = 0
    sym = extra.get("symbolic_state")
    if isinstance(sym, dict):
        rule_count = int(sym.get("next_id", 0))
    print(f"[s18e] rule_count={rule_count}")
    results = {"step": step, "ckpt": args.ckpt}

    # 1. Developmental milestones (3D), per curriculum task
    tasks_cfg = (cfg.get("curriculum") or {}).get("tasks", [])
    task_ids = [t["id"] for t in tasks_cfg] if args.tasks == "all" else \
        [int(x) for x in args.tasks.split(",")]
    per_task: dict[str, dict] = {}
    env0 = env
    for tid in task_ids:
        spec = next((t for t in tasks_cfg if int(t["id"]) == tid), None)
        if spec is None:
            print(f"[s18e] !! unknown task id {tid}, skipped")
            continue
        env = make_env(spec.get("num_objects", 8), spec.get("action_force", 50.0))
        print(f"[s18e] Measuring task {tid} ({spec.get('tag')}): "
              f"{spec.get('num_objects')} objects, force={spec.get('action_force')} ...")
        rep = measure_milestones(model, env, device, episodes=args.episodes,
                                 max_steps=args.max_steps, epsilon=args.epsilon,
                                 rule_count=rule_count, seed=args.seed)
        per_task[str(tid)] = {
            "tag": spec.get("tag"),
            "num_objects": spec.get("num_objects"),
            "action_force": spec.get("action_force"),
            "scores": {k: round(float(v), 4) for k, v in rep.scores.items()},
            "passed": {k: bool(v) for k, v in rep.passed.items()},
            "estimated_age": rep.estimated_age,
        }
        print(f"  task {tid}: est. age={rep.estimated_age:.1f}y | "
              + " ".join(f"{k}={v:.2f}" for k, v in rep.scores.items()))
        env.close()
    env = env0
    # Aggregate report over ALL tasks (merged states)
    # Re-roll once more on the base env to keep merged_states deterministic
    rep_agg = None
    if per_task:
        rep_agg = None
        # simplest deterministic aggregate: re-evaluate on base env only
        rep_base = measure_milestones(model, env, device, episodes=args.episodes,
                                      max_steps=args.max_steps,
                                      epsilon=args.epsilon, rule_count=rule_count)
        results["milestones_3d"] = {
            "scores": {k: round(float(v), 4) for k, v in rep_base.scores.items()},
            "passed": {k: bool(v) for k, v in rep_base.passed.items()},
            "estimated_age": rep_base.estimated_age,
            "per_task": per_task,
        }
        rep_agg = rep_base
    else:
        rep = measure_milestones(model, env, device, episodes=args.episodes,
                                 max_steps=args.max_steps, epsilon=args.epsilon,
                                 rule_count=rule_count)
        results["milestones_3d"] = {
            "scores": {k: round(float(v), 4) for k, v in rep.scores.items()},
            "passed": {k: bool(v) for k, v in rep.passed.items()},
            "estimated_age": rep.estimated_age,
        }
        rep_agg = rep
    print(f"  BASE est. age={rep_agg.estimated_age:.1f}y")

    # 2. ToM module targeted tests
    print("[s18e] Measuring ToM module ...")
    tom_state = extra.get("theory_of_mind_state")
    results["tom_module"] = measure_tom(tom_state, device,
                                        d_model=int(model_cfg.get("hidden_size", 128)),
                                        num_actions=env.action_space_n,
                                        num_slots=int(model_cfg.get("slot_num_slots", 7)))

    # 3. NumberSense head
    print("[s18e] Measuring NumberSense head ...")
    ns_state = extra.get("number_sense_state")
    results["number_sense"] = measure_number_sense(ns_state, model, env, device,
                                                   model_cfg, num_sense_cfg)

    # 4. Scene description
    print("[s18e] Measuring scene description ...")
    results["scene_description"] = measure_scene(model, env, device)

    # 5. Training state summary
    print("[s18e] Summarizing training state ...")
    summary = {"preset": extra.get("preset"), "run_id": extra.get("run_id")}
    for key in ["symbolic_state", "symbol_backend_state", "logic_engine_state",
                "self_model_state", "reflection_state", "creativity_state",
                "imagination_trainer_state", "curriculum_active_task_id",
                "causal_discovery_state", "sleep_loop_state",
                "narrative_loop_state"]:
        v = extra.get(key)
        if isinstance(v, dict):
            safe = {}
            for kk, vv in v.items():
                if isinstance(vv, (int, float, str, bool)):
                    safe[kk] = vv
                elif isinstance(vv, list) and len(vv) <= 64:
                    safe[kk] = vv
                else:
                    safe[kk] = f"<{type(vv).__name__}>"
            summary[key] = safe
        elif v is not None:
            summary[key] = str(v)[:200]
    summary["eval_trajectory"] = extra.get("eval_trajectory")
    results["training_state"] = summary

    print()
    print("=" * 70)
    print("  Stage 18 FULL EVALUATION @ step", step)
    print("=" * 70)
    print(f"  est. age            : {results['milestones_3d']['estimated_age']:.1f}y")
    for k, v in results["milestones_3d"]["scores"].items():
        mark = "✅" if results["milestones_3d"]["passed"].get(k) else "⬜"
        print(f"    {mark} {k}: {v:.3f}")
    pt = results["milestones_3d"].get("per_task")
    if pt:
        print("  per task:")
        for tid, info in pt.items():
            print(f"    t{tid} ({info.get('tag')}, {info.get('num_objects')}obj): "
                  f"age={info['estimated_age']:.1f}y "
                  + " ".join(f"{k}={v:.2f}" for k, v in info["scores"].items()))
    print(f"  ToM module          : {results['tom_module']}")
    print(f"  NumberSense head    : {results['number_sense']}")
    print(f"  Scene description   : {results['scene_description']}")
    print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=lambda o: f"<{type(o).__name__}>"))
    print(f"[s18e] report -> {out_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())