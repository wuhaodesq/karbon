# -*- coding: utf-8 -*-
"""Contract test: ExtendedThreeDWorld must forward ThreeDWorld kwargs.

Stage 23 M2-p2: the train.py branch used to pass only 5 kwargs, leaving
occluder_obs_slots=0 -> proprio_dim=18 while the checkpoint's model expects
27 (strict=False would silently randomize the proprio pathway).
"""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

from src.envs.three_d_world import _mj_available

if not _mj_available:
    raise unittest.SkipTest("mujoco not available locally")


class TestExtendedEnvKwargs(unittest.TestCase):
    def test_contract_matches_training_model(self):
        from src.envs.extended_3d_world import ExtendedThreeDWorld
        env = ExtendedThreeDWorld(
            num_objects=8, num_siblings=1, seed=1,
            max_episode_steps=60, render_size=64, developmental_age=0.5,
            num_occluders=4, occluder_obs_slots=3, focus_op_only=False,
            occluder_target_reward=0.3, occluder_shaping_weight=0.3,
            occluder_reach_reward=100.0, occluder_reach_hold=5.0,
            occluder_reach_hold_max_steps=15,
            op_reward_clamp=100.0,
        )
        try:
            self.assertEqual(int(env.proprio_dim), 27)  # 18 + 3*3 slots
            self.assertEqual(int(env.action_space_n), 12)
            self.assertEqual(tuple(env.observation_shape), (64, 64, 3))
            obs = env.reset()
            self.assertEqual(tuple(obs.shape), (64, 64, 3))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
