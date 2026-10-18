"""analysis_b2.py — B2 exp1+exp2: input attribution + representation stats.

Locked ckpt (22.85M) vs burst-era ckpt (20.35M), on REAL game observations:
  Exp1a: fix image, vary lk-slot dims (18..26) -> argmax sensitivity.
  Exp1b: fix proprio, vary image (10 real frames) -> argmax sensitivity.
  Exp2:  slot-vector stats (norm spread, inter-slot cosine) per ckpt.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.platform import get_device
from src.utils import load_config

_ev = Path(__file__).resolve().parent / "run_stage18_full_eval.py"
_spec = importlib.util.spec_from_file_location("s18e", str(_ev))
_s18e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s18e)
_dp = Path(__file__).resolve().parent / "op_failure_diag.py"
_dspec = importlib.util.spec_from_file_location("opdiag", str(_dp))
_opdiag = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_opdiag)

CK_LOCKED = "/root/autodl-tmp/karbon_ckpts/backup_stage20_22850000.pt"
CK_BURST = "/root/autodl-tmp/karbon_ckpts/backup_stage20_20350000.pt"

cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
device = get_device()
spec = (cfg.get("curriculum") or {}).get("tasks", [])[2]  # 3d-base
env = _opdiag._make_env(cfg, int(spec["num_objects"]), 50.0, 300)


def build(ckpt):
    n_layers = int(_s18e._ckpt_layer_count(ckpt) or 7)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model_state") if isinstance(ck, dict) else ck
    step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0
    m = _s18e.build_model(env.observation_shape, env.action_space_n, cfg["model"],
                          device, n_layers, proprio_dim=int(getattr(env, "proprio_dim", 0)))
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, step


def argmax_of(m, img, prop):
    obs_t = _s18e._obs_to_tensor(img, device)
    prop_t = torch.as_tensor(np.asarray(prop, dtype=np.float32),
                             dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        out = m(obs_t, proprio=prop_t)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return int(torch.argmax(logits.reshape(-1)).item())


imgs, props = [], []
for i in range(10):
    obs = env.reset(seed=5000 + i)
    for _ in range(5):
        so = env.step(3)
        obs = so.obs
    imgs.append(obs)
    props.append(np.asarray(env._proprio(), dtype=np.float32))

res = {}
rng = np.random.RandomState(0)
for name, ck in (("locked_22.85M", CK_LOCKED), ("burst_20.35M", CK_BURST)):
    m, step = build(ck)
    img0, p0 = imgs[0], props[0].copy()
    acts_slot = []
    for k in range(10):
        p = p0.copy()
        for s in range(3):
            ang = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(0.05, 0.9)
            p[18 + 3 * s] = np.cos(ang) * d / 4.0
            p[19 + 3 * s] = np.sin(ang) * d / 4.0
            p[20 + 3 * s] = d / 4.0
        acts_slot.append(argmax_of(m, img0, p))
    acts_img = [argmax_of(m, im, props[0]) for im in imgs]
    slot_stats = []
    for im in imgs:
        obs_t = _s18e._obs_to_tensor(im, device)
        prop_t = torch.as_tensor(props[0], dtype=torch.float32,
                                 device=device).reshape(1, -1)
        with torch.no_grad():
            m(obs_t, proprio=prop_t)
        sl = getattr(m, "_last_slots", None)
        if sl is not None:
            v = sl.squeeze(0).float().cpu().numpy()
            norms = np.linalg.norm(v, axis=1)
            pair = [float(np.dot(v[a], v[b]) / (norms[a] * norms[b] + 1e-9))
                    for a in range(len(v)) for b in range(a + 1, len(v))]
            slot_stats.append({"norm_mean": float(norms.mean()),
                               "norm_std": float(norms.std()),
                               "pair_cos_mean": float(np.mean(pair))})
    res[name] = {"step": step, "acts_slot": acts_slot, "acts_img": acts_img,
                 "slot_stats": slot_stats}
    print(name, "slot-var uniq:", sorted(set(acts_slot)),
          "| img-var uniq:", sorted(set(acts_img)), flush=True)

summary = {k: {"acts_slot_uniq": sorted(set(v["acts_slot"])),
               "acts_img_uniq": sorted(set(v["acts_img"])),
               "slot_norm_mean": round(float(np.mean([s["norm_mean"] for s in v["slot_stats"]])), 3),
               "slot_norm_std": round(float(np.mean([s["norm_std"] for s in v["slot_stats"]])), 3),
               "pair_cos_mean": round(float(np.mean([s["pair_cos_mean"] for s in v["slot_stats"]])), 3)}
           for k, v in res.items()}
print("SUMMARY:", json.dumps(summary))
with open("/root/b2_analysis.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print("saved /root/b2_analysis.json")
