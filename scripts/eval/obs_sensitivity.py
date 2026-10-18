"""obs_sensitivity2.py — does argmax depend on the observation? (no MuJoCo)

Feeds 30 RANDOM rendered-image-shaped inputs (uint8 HWC noise) + random
proprio to the policy and counts distinct argmax actions. A constant policy
(1 unique action) = network-level input-insensitivity, independent of any
environment/rendering concerns.
"""
from __future__ import annotations

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

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/karbon/checkpoints/ckpt_stage20_022251552.pt"
cfg = load_config("stage20_hypothesis_v2.yaml", "cloud_24g")
device = get_device()
rs = int(cfg["env"].get("render_size", 128))
obs_shape = (rs, rs, 3)
n_actions = 12
proprio_dim = 27

n_layers = int(_s18e._ckpt_layer_count(CKPT) or int(cfg["model"].get("hybrid_n_layers", 7)))
ck = torch.load(CKPT, map_location="cpu")
sd = ck.get("model_state") if isinstance(ck, dict) else ck
step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0
model = _s18e.build_model(obs_shape, n_actions, cfg["model"], device, n_layers,
                          proprio_dim=proprio_dim)
model.load_state_dict(sd, strict=False)
model.eval()

rng = np.random.RandomState(0)
acts, spreads = [], []
for i in range(30):
    img = rng.randint(0, 255, obs_shape).astype(np.uint8)
    obs_t = _s18e._obs_to_tensor(img, device)
    prop = torch.randn(1, proprio_dim, device=device)
    with torch.no_grad():
        out = model(obs_t, proprio=prop)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = logits.reshape(-1, n_actions)[0]
    acts.append(int(torch.argmax(logits).item()))
    spreads.append(float(logits.max() - logits.min()))

uniq = sorted(set(acts))
print(json.dumps({"ckpt_step": step, "n_inputs": 30, "unique_actions": uniq,
                  "n_unique": len(uniq),
                  "action_counts": {str(a): acts.count(a) for a in uniq},
                  "logit_spread_mean": round(float(np.mean(spreads)), 3)}))
