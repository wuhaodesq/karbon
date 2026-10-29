"""m2p2_smoke.py — Stage 23 M2-p2 pre-flight for ExtendedThreeDWorld.

Builds the extended world with the FULL M2-p2 kwarg set (mirroring the train.py
branch), verifies the model contract (obs shape / proprio 27 / 12 actions) and
runs a short random rollout (no crash, rewards + occlusion events observable).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.extended_3d_world import ExtendedThreeDWorld

env = ExtendedThreeDWorld(
    num_objects=20,
    num_siblings=2,
    season_cycle_steps=500_000,
    seed=42,
    max_episode_steps=300,
    render_size=128,
    developmental_age=0.5,
    num_occluders=4,
    occluder_trace=False,
    occluder_target_reward=0.3,
    object_crossing_every=100,
    object_crossing_hold_steps=10,
    object_crossing_fixed_object=-1,
    object_crossing_fixed_wall=-1,
    occluder_obs_slots=3,
    focus_op_only=False,
    occluder_shaping_weight=0.3,
    occluder_reveal_bonus=2.0,
    occluder_reveal_ratio=0.85,
    occluder_reach_reward=100.0,
    occluder_reach_radius=0.8,
    occluder_reward_radius=0.8,
    occluder_reach_hold=5.0,
    occluder_reach_hold_max_steps=15,
    occluder_orient_weight=0.1,
    occluder_orient_bonus=2.0,
    occluder_teacher_force=0.0,
    occluder_search_min_start=1.2,
    op_reward_clamp=100.0,
)

obs = env.reset()
print("obs shape:", obs.shape, "| observation_shape:", env.observation_shape)
print("proprio_dim:", env.proprio_dim, "| action_n:", env.action_space_n)
assert tuple(env.observation_shape) == (128, 128, 3), env.observation_shape
assert int(env.proprio_dim) == 27, env.proprio_dim
assert int(env.action_space_n) == 12, env.action_space_n

rng = np.random.RandomState(0)
total_r = 0.0
max_r = 0.0
n_events = 0
for i in range(200):
    a = int(rng.randint(0, env.action_space_n))
    so = env.step(a)
    obs = so.obs
    total_r += float(so.reward)
    max_r = max(max_r, float(so.reward))
    if so.terminated or so.truncated:
        try:
            n_events += len(env._occlusion_events)
        except Exception:
            pass  # legit: event list optional
        obs = env.reset()

try:
    n_events += len(env._occlusion_events)
except Exception:
    pass  # legit: event list optional

print("200 steps ok | total_r=", round(total_r, 2), "| max_r=", round(max_r, 2),
      "| occlusion events:", n_events)
assert np.isfinite(total_r), total_r
assert total_r > 0.0, "no reward at all in 200 steps"
print("SMOKE OK")
env.close()
