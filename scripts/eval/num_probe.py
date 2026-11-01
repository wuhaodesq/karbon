"""num_probe.py — behavioural confirmation of the number-sense trade-off.

Numbers-sense milestone = "at episode end, nearly all objects are within
3 m" (the env's awareness proxy). The 46.15M eval dropped it in sparse
scenes while crowded scenes improved. This probe measures the actual
end-of-episode behaviour for two checkpoints:

  nearby3      : objects within 3 m (the milestone quantity)
  min_dist     : distance to the nearest object
  centroid_dist: distance to the object centroid
  n_contacted  : objects contacted during the episode

Usage:
    python scripts/eval/num_probe.py \
        --ckpt /root/autodl-tmp/karbon_ckpts/backup_stage20_39350000.pt \
        --ckpt2 /root/karbon/checkpoints/ckpt_stage20_046151200.pt \
        --out /root/num_probe.json
"""
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
from src.envs.extended_3d_world import ExtendedThreeDWorld
from src.models.hierarchical_policy import HierarchicalActorCritic
from src.platform import get_device
from src.train import _ckpt_layer_count
from src.utils import load_config


def _obs_t(obs, device):
    t = torch.from_numpy(np.asarray(obs)).float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def _prop_t(env, device):
    dim = int(getattr(env, "proprio_dim", 0))
    if dim <= 0:
        return None
    raw = env._proprio() if hasattr(env, "_proprio") else None
    if raw is None:
        return None
    t = torch.from_numpy(np.asarray(raw, dtype=np.float32)).reshape(1, -1)
    if t.shape[-1] < dim:
        t = torch.cat([t, torch.zeros(1, dim - t.shape[-1])], dim=-1)
    return t.to(device)


def _build_model(obs_shape, num_actions, mcfg, device, n_layers, proprio_dim):
    return HierarchicalActorCritic(
        obs_shape=obs_shape, num_actions=num_actions,
        d_model=int(mcfg.get("hidden_size", 128)), n_layers=n_layers,
        n_heads=int(mcfg.get("hybrid_n_heads", 4)),
        swa_window=int(mcfg.get("hybrid_swa_window", 16)),
        ttt_mini_batch=int(mcfg.get("hybrid_ttt_mini_batch", 8)),
        ffn_hidden_mult=int(mcfg.get("hybrid_ffn_hidden_mult", 4)),
        dropout=float(mcfg.get("hybrid_dropout", 0.0)),
        use_vision_encoder=bool(mcfg.get("use_vision_encoder", False)),
        vision_model_name=str(mcfg.get("vision_model", "dinov2_vits14")),
        use_slot_attention=bool(mcfg.get("use_slot_attention", False)),
        slot_num_slots=int(mcfg.get("slot_num_slots", 7)),
        slot_dim=int(mcfg.get("slot_dim", 128)),
        slot_num_iterations=int(mcfg.get("slot_num_iterations", 3)),
        sub_goal_every=int(mcfg.get("sub_goal_every", 10)),
        proprio_dim=int(proprio_dim),
    ).to(device)


def run_ckpt(ckpt: str, cfg: dict, episodes: int, seed: int,
             device, env_type: str = "extended") -> dict:
    ec = cfg["env"]
    mcfg = cfg["model"]
    tasks = cfg["curriculum"]["tasks"]
    env0 = None
    out = {"ckpt": ckpt, "env_type": env_type, "per_task": {}}

    def _make(spec):
        kw = dict(
            num_objects=int(spec["num_objects"]),
            max_episode_steps=300,
            render_size=int(ec.get("render_size", 128)),
            action_force=float(spec["action_force"]),
            developmental_age=float(ec.get("developmental_age", 0.5)),
            num_occluders=int(ec.get("num_occluders", 0)),
            occluder_obs_slots=int(ec.get("occluder_obs_slots", 0)),
            object_crossing_every=int(ec.get("object_crossing_every", 0)),
            object_crossing_hold_steps=int(ec.get("object_crossing_hold_steps", 0)),
        )
        if env_type == "extended":
            kw.update(
                num_siblings=int(ec.get("num_siblings", 2)),
                season_cycle_steps=int(ec.get("season_cycle_steps", 500000)),
            )
            return ExtendedThreeDWorld(**kw)
        return ThreeDWorld(**kw)

    for spec in tasks:
        env = _make(spec)
        n_layers = _ckpt_layer_count(ckpt)
        if n_layers <= 0:
            n_layers = int(mcfg.get("hybrid_n_layers", 7))
        if env0 is None:
            env0 = env
            model = _build_model(env.observation_shape, env.action_space_n,
                                 mcfg, device, n_layers,
                                 int(getattr(env, "proprio_dim", 0)))
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ck.get("model_state") if isinstance(ck, dict) else ck
            model.load_state_dict(sd, strict=False)
            model.eval()
        rng = np.random.RandomState(seed + int(spec["id"]))
        rows = []
        for _ in range(episodes):
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            steps = 0
            done = False
            while not done and steps < 300:
                with torch.no_grad():
                    logits, _ = model(_obs_t(obs, device),
                                      proprio=_prop_t(env, device))
                if rng.rand() < 0.1:
                    a = int(rng.randint(0, env.action_space_n))
                else:
                    a = int(torch.argmax(logits, dim=-1).item())
                st = env.step(a)
                obs = st.obs
                done = bool(st.terminated or st.truncated)
                steps += 1
            # metrics from env internals
            try:
                b = env._model.body("learner")
                ax, ay = float(env._data.xpos[b.id, 0]), float(env._data.xpos[b.id, 1])
                pos = []
                nearby = 0
                for i in range(env._num_objects):
                    ob = env._model.body(f"obj_{i}")
                    ox, oy = float(env._data.xpos[ob.id, 0]), float(env._data.xpos[ob.id, 1])
                    pos.append((ox, oy))
                    if ((ax - ox) ** 2 + (ay - oy) ** 2) ** 0.5 < 3.0:
                        nearby += 1
                pos = np.asarray(pos)
                centroid = pos.mean(axis=0)
                rows.append({
                    "nearby3": nearby,
                    "min_dist": float(np.min(np.hypot(pos[:, 0] - ax, pos[:, 1] - ay))),
                    "centroid_dist": float(np.hypot(centroid[0] - ax, centroid[1] - ay)),
                    "n_contacted": int(len(getattr(env, "_contacted", ()))),
                    "n_obj": int(env._num_objects),
                })
            except Exception as ex:
                rows.append({"error": str(ex)})
        ok = [r for r in rows if "nearby3" in r]
        agg = {}
        if ok:
            for k in ("nearby3", "min_dist", "centroid_dist", "n_contacted"):
                agg[k] = round(float(np.mean([r[k] for r in ok])), 3)
            agg["frac_near_all"] = round(float(np.mean(
                [r["nearby3"] >= r["n_obj"] - 1 for r in ok])), 3)
            agg["n_obj"] = int(ok[0]["n_obj"])
        out["per_task"][str(spec["id"])] = {"tag": spec.get("tag"),
                                            "agg": agg, "n": len(ok)}
        env.close()
        print(f"[probe] task {spec['id']} ({spec.get('tag')}): {agg}", flush=True)
    if env0 is not None and env0 is not env:
        env0.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-type", default="extended",
                    choices=["threed", "extended"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/root/num_probe.json")
    args = ap.parse_args()
    cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
    device = get_device()
    res = {"env": cfg["env"]["id"], "seed": args.seed,
           "env_type": args.env_type,
           "run": run_ckpt(args.ckpt, cfg, args.episodes, args.seed, device,
                            env_type=args.env_type)}
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[probe] saved", args.out)


if __name__ == "__main__":
    main()
