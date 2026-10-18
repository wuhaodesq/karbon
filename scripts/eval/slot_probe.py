"""slot_probe.py — slot-stat trajectory across era checkpoints.

For each ckpt: real game obs -> slot norms / inter-slot cosine / argmax
sensitivity (varied lk-slots; varied images). Detects when the slot path
blew up (if the golden-era ckpt is normal, the blow-up is a tracked event).
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

CKPTS = [
    ("golden_12.05M", "/root/autodl-tmp/karbon_ckpts/backup_stage20_12051552.pt"),
    ("era_15.35M", "/root/autodl-tmp/karbon_ckpts/backup_stage20_15350000.pt"),
    ("camp_18.10M", "/root/autodl-tmp/karbon_ckpts/backup_stage20_18100000.pt"),
    ("burst_20.35M", "/root/autodl-tmp/karbon_ckpts/backup_stage20_20350000.pt"),
    ("locked_22.85M", "/root/autodl-tmp/karbon_ckpts/backup_stage20_22850000.pt"),
]

cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
device = get_device()
spec = (cfg.get("curriculum") or {}).get("tasks", [])[2]
env = _opdiag._make_env(cfg, int(spec["num_objects"]), 50.0, 300)

imgs, props = [], []
for i in range(8):
    obs = env.reset(seed=5000 + i)
    for _ in range(5):
        so = env.step(3)
        obs = so.obs
    imgs.append(obs)
    props.append(np.asarray(env._proprio(), dtype=np.float32))


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


rng = np.random.RandomState(0)
for name, ck in CKPTS:
    if not Path(ck).exists():
        print(name, "MISSING", ck, flush=True)
        continue
    m, step = build(ck)
    norms, pairc = [], []
    for im in imgs:
        obs_t = _s18e._obs_to_tensor(im, device)
        prop_t = torch.as_tensor(props[0], dtype=torch.float32,
                                 device=device).reshape(1, -1)
        with torch.no_grad():
            m(obs_t, proprio=prop_t)
        sl = getattr(m, "_last_slots", None)
        if sl is not None:
            v = sl.squeeze(0).float().cpu().numpy()
            nm = np.linalg.norm(v, axis=1)
            norms.append(float(nm.mean()))
            pairc.append(float(np.mean([np.dot(v[a], v[b]) /
                                        (nm[a] * nm[b] + 1e-9)
                                        for a in range(len(v))
                                        for b in range(a + 1, len(v))])))
    acts_slot = []
    img0, p0 = imgs[0], props[0].copy()
    for k in range(6):
        p = p0.copy()
        for s in range(3):
            ang = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(0.05, 0.9)
            p[18 + 3 * s] = np.cos(ang) * d / 4.0
            p[19 + 3 * s] = np.sin(ang) * d / 4.0
            p[20 + 3 * s] = d / 4.0
        acts_slot.append(argmax_of(m, img0, p))
    acts_img = [argmax_of(m, im, props[0]) for im in imgs]
    print(f"{name} step={step} | slot_norm={np.mean(norms):.1f}±{np.std(norms):.1f} "
          f"| pair_cos={np.mean(pairc):.3f} | acts_slot={sorted(set(acts_slot))} "
          f"| acts_img={sorted(set(acts_img))}", flush=True)
print("done")
