# -*- coding: utf-8 -*-
"""Contract test: manipulation actions must NOT freeze the body (M3.5).

Stage 23 M3.5 momentum semantics: actions 8-11 preserve the last locomotion
command. The old zero-force semantics ("manipulation = stand still") was the
mechanical base of the constant-action freeze family.
"""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

from src.envs.three_d_world import _mj_available

if not _mj_available:
    raise unittest.SkipTest("mujoco not available locally")


class TestMomentumSemantics(unittest.TestCase):
    def _make(self):
        from src.envs.three_d_world import ThreeDWorld
        return ThreeDWorld(
            num_objects=4, seed=1, max_episode_steps=80, render_size=64,
            developmental_age=0.5, action_force=50.0,
        )

    def test_release_keeps_moving(self):
        env = self._make()
        try:
            env.reset(seed=1)
            for _ in range(4):          # build locomotion (action 3 = East)
                env.step(3)
            x0 = float(env._data.body("learner").xpos[0])
            for _ in range(4):          # release must PRESERVE motion
                env.step(9)
            x1 = float(env._data.body("learner").xpos[0])
            self.assertGreater(x1, x0 + 1e-4, (x0, x1))
        finally:
            env.close()

    def test_rotate_keeps_moving(self):
        env = self._make()
        try:
            env.reset(seed=1)
            for _ in range(4):
                env.step(3)
            x0 = float(env._data.body("learner").xpos[0])
            for _ in range(4):
                env.step(11)        # rotate must PRESERVE motion
            x1 = float(env._data.body("learner").xpos[0])
            self.assertGreater(x1, x0 + 1e-4, (x0, x1))
        finally:
            env.close()

    def test_last_command_tracking(self):
        env = self._make()
        try:
            env.reset(seed=1)
            env.step(3)             # East
            self.assertGreater(env._last_dx, 0.0)
            env.step(1)             # South
            self.assertAlmostEqual(env._last_dy, -float(env._action_force), places=6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
