# -*- coding: utf-8 -*-
"""Unit tests for Stage 20d reveal attribution bonus."""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

from src.envs.three_d_world import _mj_available

if not _mj_available:
    raise unittest.SkipTest("mujoco not available locally")


class TestRevealBonus(unittest.TestCase):
    def _make(self, reveal_bonus=1.0, ratio=0.7, target=1.5, shaping=0.0):
        from src.envs.three_d_world import ThreeDWorld
        return ThreeDWorld(
            num_objects=8, seed=42, max_episode_steps=60, render_size=64,
            num_occluders=4, occluder_target_reward=target,
            occluder_shaping_weight=shaping,
            occluder_reveal_bonus=reveal_bonus, occluder_reveal_ratio=ratio,
            object_crossing_every=10, object_crossing_hold_steps=8,
            focus_op_only=True, action_force=50.0,
        )

    def test_disabled_when_bonus_zero(self):
        env = self._make(reveal_bonus=0.0)
        env._track_3d_developmental_signals(0.0, 0.0)
        self.assertEqual(env._reveal_bonus_pending, 0.0)

    def test_pending_attribution_and_consume(self):
        env = self._make(reveal_bonus=1.0)
        # Fake an occlusion record with a trajectory that clearly approached
        # last_known (agent started far, ends close -> end << 0.7*start).
        env._active_occlusions_3d["occ_0"] = {
            "last_known": (2.0, 2.0),
            "agent_traj_during_occ": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
            "truly_occluded": True,
        }
        # Place learner near last_known (dist ~0.28 < 0.7*2.83=1.98)
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()
        env._data.body("learner").xpos[0] = 1.9
        env._data.body("learner").xpos[1] = 1.9
        env._maybe_reveal_bonus("occ_0")
        self.assertEqual(env._reveal_bonus_pending, 1.0)
        # Reward consumes it exactly once
        r1 = env._occluder_only_reward()
        self.assertEqual(r1, 1.0)
        self.assertEqual(env._reveal_bonus_pending, 0.0)
        r2 = env._occluder_only_reward()
        self.assertEqual(r2, 0.0)

    def test_no_bonus_when_not_approached(self):
        env = self._make(reveal_bonus=1.0)
        env._active_occlusions_3d["occ_0"] = {
            "last_known": (2.0, 2.0),
            "agent_traj_during_occ": [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)],
            "truly_occluded": True,
        }
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()
        env._data.body("learner").xpos[0] = 0.3
        env._data.body("learner").xpos[1] = 0.3
        env._maybe_reveal_bonus("occ_0")
        self.assertEqual(env._reveal_bonus_pending, 0.0)

    def test_short_traj_no_attribute(self):
        env = self._make(reveal_bonus=1.0)
        env._active_occlusions_3d["occ_0"] = {
            "last_known": (2.0, 2.0),
            "agent_traj_during_occ": [(0.0, 0.0), (1.9, 1.9)],
            "truly_occluded": True,
        }
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()
        env._data.body("learner").xpos[0] = 1.9
        env._data.body("learner").xpos[1] = 1.9
        env._maybe_reveal_bonus("occ_0")
        self.assertEqual(env._reveal_bonus_pending, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)