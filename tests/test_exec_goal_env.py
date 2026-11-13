# -*- coding: utf-8 -*-
"""Unit tests for the Stage 27 M1 executive goal env-side machinery:

- set_exec_goal maps the observation slot index to the event key with the
  SAME distance sort as the obs slots (nearest first);
- commitment shaping pays only toward the CHOSEN event's last_known;
- the settlement bonus is one-shot, success-gated, and releases the goal.
"""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

from src.envs.three_d_world import _mj_available

if not _mj_available:
    raise unittest.SkipTest("mujoco not available locally")


class TestExecGoalEnv(unittest.TestCase):
    def _make(self, enabled=True, commit=0.1, settle=10.0, reveal=2.0):
        from src.envs.three_d_world import ThreeDWorld
        return ThreeDWorld(
            num_objects=8, seed=42, max_episode_steps=60, render_size=64,
            num_occluders=4,
            occluder_target_reward=0.0, occluder_shaping_weight=0.0,
            occluder_reveal_bonus=reveal, occluder_orient_weight=0.0,
            occluder_orient_bonus=0.0, occluder_reach_reward=0.0,
            occluder_reach_hold=0.0, occluder_reach_hold_max_steps=0,
            focus_op_only=True, action_force=50.0, developmental_age=0.3,
            occluder_obs_slots=3,
            exec_goal_enabled=enabled,
            exec_goal_commit_weight=commit,
            exec_goal_settle_bonus=settle,
        )

    def _ensure(self, env):
        if not hasattr(env, "_data") or env._data is None:
            env._rebuild_scene()

    def _inject(self, env, key, dx, dy, traj_start=None):
        self._ensure(env)
        ax = float(env._data.body("learner").xpos[0])
        ay = float(env._data.body("learner").xpos[1])
        start = traj_start or (ax, ay)
        env._active_occlusions_3d[key] = {
            "last_known": (ax + dx, ay + dy),
            "agent_traj_during_occ": [start, start, start],
            "truly_occluded": True,
            "far_start": True,
        }

    def _set_velocity(self, env, vx, vy):
        self._ensure(env)
        bid = env._model.body("learner").id
        dof = env._model.body_dofadr[bid]
        env._data.qvel[dof] = vx
        env._data.qvel[dof + 1] = vy

    def test_slot_mapping_nearest_first(self):
        env = self._make()
        self._inject(env, "occ_0", 2.0, 0.0)   # farther
        self._inject(env, "occ_1", 1.0, 0.0)   # nearer
        env.set_exec_goal(0)
        self.assertEqual(env._exec_goal_key, "occ_1")  # nearest first
        env.set_exec_goal(1)
        self.assertEqual(env._exec_goal_key, "occ_0")
        env.set_exec_goal(3)  # none choice (K=3)
        self.assertIsNone(env._exec_goal_key)
        env.set_exec_goal(1)
        env.set_exec_goal(-1)
        self.assertIsNone(env._exec_goal_key)

    def test_commit_shaping_targets_chosen_event(self):
        env = self._make(commit=0.1, settle=0.0)
        self._inject(env, "occ_0", -1.0, 0.0)
        self._inject(env, "occ_1", 2.0, 0.0)
        self._set_velocity(env, -0.05, 0.0)  # moving toward -x
        env.set_exec_goal(0)  # chosen = -1.0 (nearest)
        r_toward = env._compute_reward(action=0)
        self.assertAlmostEqual(r_toward, 0.1, places=6)
        env.set_exec_goal(1)  # chosen = +2.0, now moving AWAY from it
        r_away = env._compute_reward(action=0)
        self.assertAlmostEqual(r_away, 0.0, places=6)

    def test_commit_off_when_disabled(self):
        env = self._make(enabled=False, commit=0.1)
        self._inject(env, "occ_0", -1.0, 0.0)
        env.set_exec_goal(0)
        self.assertIsNone(env._exec_goal_key)
        self._set_velocity(env, -0.05, 0.0)
        self.assertAlmostEqual(env._compute_reward(action=0), 0.0, places=6)

    def test_settle_bonus_one_shot(self):
        env = self._make(commit=0.0, settle=10.0, reveal=2.0)
        self._ensure(env)
        ax = float(env._data.body("learner").xpos[0])
        ay = float(env._data.body("learner").xpos[1])
        # started 1.5m away, now standing on last_known -> approach success
        self._inject(env, "occ_0", 0.0, 0.0, traj_start=(ax + 1.5, ay))
        env.set_exec_goal(0)
        self.assertEqual(env._exec_goal_key, "occ_0")
        env._maybe_reveal_bonus("occ_0")
        self.assertIsNone(env._exec_goal_key)  # released on settlement
        # pending channel is a single MAX slot (existing reveal semantics):
        # the larger of reveal(2.0) / settle(10.0) is delivered once.
        self.assertAlmostEqual(env._reveal_bonus_pending, 10.0, places=6)
        r1 = env._compute_reward(action=0)
        self.assertAlmostEqual(r1, 10.0, places=6)
        # no double payout on a second settlement call (production pops the
        # event right after settlement; the exec bonus must not re-fire)
        env._active_occlusions_3d.pop("occ_0", None)
        env._maybe_reveal_bonus("occ_0")
        self.assertAlmostEqual(env._reveal_bonus_pending, 0.0, places=6)
        self.assertAlmostEqual(env._compute_reward(action=0), 0.0, places=6)

    def test_settle_requires_success(self):
        env = self._make(commit=0.0, settle=10.0, reveal=2.0)
        self._ensure(env)
        ax = float(env._data.body("learner").xpos[0])
        ay = float(env._data.body("learner").xpos[1])
        # never moved: d_now == d0 -> not a success -> no payout
        self._inject(env, "occ_0", 3.0, 0.0)
        env.set_exec_goal(0)
        env._maybe_reveal_bonus("occ_0")
        self.assertIsNone(env._exec_goal_key)
        self.assertAlmostEqual(env._reveal_bonus_pending, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
