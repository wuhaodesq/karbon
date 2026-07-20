"""Tests that PhysicsSandbox emits C#8 developmental signals in info."""

from __future__ import annotations

import numpy as np

from src.envs.physics_sandbox import PhysicsSandbox


def _run_episodes(env: PhysicsSandbox, n: int = 3, seed: int = 0) -> list[dict]:
    infos = []
    obs = env.reset(seed=seed)
    for _ in range(n):
        done = False
        while not done:
            act = int(np.random.RandomState(seed).randint(0, 8))
            step_out = env.step(act)
            infos.append(step_out.info)
            done = step_out.terminated or step_out.truncated
        obs = env.reset(seed=seed + 1)
    return infos


def test_info_has_signal_fields():
    env = PhysicsSandbox(num_objects=4, seed=1, max_episode_steps=50)
    env.reset(seed=1)
    out = env.step(0)
    for key in ("occlusion_events", "force_motion_pairs", "count_trials"):
        assert key in out.info, f"missing {key} in info"
        assert isinstance(out.info[key], list)


def test_force_motion_pairs_populate():
    env = PhysicsSandbox(num_objects=3, seed=2, max_episode_steps=80)
    infos = _run_episodes(env, n=2, seed=2)
    pairs = [p for info in infos for p in info["force_motion_pairs"]]
    # Agent applies forces every step (action 0-7), so pairs must accumulate.
    assert len(pairs) > 0
    sample = pairs[0]
    assert "force" in sample and "velocity_after" in sample
    f = sample["force"]
    v = sample["velocity_after"]
    assert len(f) == 2 and len(v) == 2


def test_count_trials_finalized_per_episode():
    env = PhysicsSandbox(num_objects=5, seed=3, max_episode_steps=40)
    infos = _run_episodes(env, n=2, seed=3)
    trials = [t for info in infos for t in info["count_trials"]]
    # One count trial finalized per completed episode.
    assert len(trials) >= 1
    t = trials[0]
    assert t["true_count"] == 5
    assert 0.0 <= t["estimated_count"] <= 5.0


def test_occlusion_events_optional():
    # Occlusion requires objects behind the central occluder; with small worlds
    # it may or may not trigger, but the field must always be a list.
    env = PhysicsSandbox(num_objects=4, seed=4, max_episode_steps=30)
    infos = _run_episodes(env, n=1, seed=4)
    occ = [e for info in infos for e in info["occlusion_events"]]
    assert isinstance(occ, list)
    if occ:
        ev = occ[0]
        assert "last_known" in ev and "agent_traj_during_occ" in ev
