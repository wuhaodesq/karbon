# -*- coding: utf-8 -*-
"""Unit tests for the bounded occluder hold-credit window (bug-ledger B10).

2026-10-15: unbounded in-circle per-step credit (occluder_reach_hold) made
"camp at last_known forever" dominant; combined with the grasp stand-still
primitive the policy froze (grasp action 10.8%->100% between 16.85M-17.35M).
The cap bounds the per-event payout window; arrival/reveal keep paying.
"""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

from src.envs.three_d_world import _mj_available

if not _mj_available:
    raise unittest.SkipTest("mujoco not available locally")


class TestHoldCap(unittest.TestCase):
    def _make(self, hold=5.0, cap=3):
        from src.envs.three_d_world import ThreeDWorld
        return ThreeDWorld(
            num_objects=8, seed=42, max_episode_steps=60, render_size=64,
            num_occluders=4,
            occluder_target_reward=0.0, occluder_shaping_weight=0.0,
            occluder_reveal_bonus=0.0, occluder_orient_weight=0.0,
            occluder_orient_bonus=0.0, occluder_reach_reward=0.0,
            occluder_reach_hold=hold, occluder_reach_hold_max_steps=cap,
            focus_op_only=True, action_force=50.0, developmental_age=0.3,
        )

    def _inject(self, env):
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()
        ax = float(env._data.body("learner").xpos[0])
        ay = float(env._data.body("learner").xpos[1])
        env._active_occlusions_3d["occ_0"] = {
            "last_known": (ax + 0.2, ay),  # dist 0.2m < reward radius 0.8
            "agent_traj_during_occ": [(ax, ay), (ax, ay), (ax, ay)],
            "truly_occluded": True,
            "far_start": True,
        }

    def test_cap_limits_payout_steps(self):
        env = self._make(hold=5.0, cap=3)
        self._inject(env)
        rewards = [env._compute_reward(action=0) for _ in range(6)]
        paid = sum(1 for r in rewards if r > 1e-9)
        self.assertEqual(paid, 3, rewards)
        expected = (0.8 - 0.2) * 5.0  # 3.0 per paid step
        self.assertAlmostEqual(rewards[0], expected, places=6)
        # after the window the camp pays nothing
        self.assertAlmostEqual(rewards[-1], 0.0, places=9)

    def test_unbounded_when_cap_zero(self):
        env = self._make(hold=5.0, cap=0)
        self._inject(env)
        rewards = [env._compute_reward(action=0) for _ in range(6)]
        self.assertTrue(all(r > 1e-9 for r in rewards), rewards)

    def test_oob_does_not_pay(self):
        env = self._make(hold=5.0, cap=3)
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()
        ax = float(env._data.body("learner").xpos[0])
        ay = float(env._data.body("learner").xpos[1])
        env._active_occlusions_3d["occ_0"] = {
            "last_known": (ax + 1.5, ay),  # 1.5m > radius 0.8: outside circle
            "agent_traj_during_occ": [(ax, ay), (ax, ay), (ax, ay)],
            "truly_occluded": True,
            "far_start": True,
        }
        r = env._compute_reward(action=0)
        self.assertAlmostEqual(r, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
