# -*- coding: utf-8 -*-
"""Unit tests for the Stage 27 M1 executive goal head (GoalExecutive)."""
import sys
import unittest

sys.path.insert(0, r"D:\karbon")

import torch

from src.models.goal_executive import (
    SLOT_FEATURES,
    GoalExecutive,
    slot_features_from_proprio,
)


class TestGoalExecutive(unittest.TestCase):
    def test_bounds_and_capacity(self):
        head = GoalExecutive(hidden_size=32, num_slots=9)
        self.assertEqual(head.num_slots, 3)  # clamped to MAX_SLOTS
        self.assertEqual(head.capacity, 3)
        self.assertEqual(len(head), 3)
        self.assertEqual(head.num_choices, 4)
        self.assertEqual(head.none_index, 3)

    def test_forward_shape_and_zero_init_uniform(self):
        torch.manual_seed(0)
        head = GoalExecutive(hidden_size=16, num_slots=3)
        b = 5
        h = torch.randn(b, 16)
        feats = torch.randn(b, 3, SLOT_FEATURES)
        valid = torch.ones(b, 3)
        logits = head(h, feats, valid)
        self.assertEqual(tuple(logits.shape), (b, 4))
        # zero-init output layer -> exactly uniform first-step choice
        self.assertTrue(torch.allclose(logits, torch.zeros_like(logits)))

    def test_invalid_slots_masked(self):
        head = GoalExecutive(hidden_size=8, num_slots=3)
        b = 2
        h = torch.randn(b, 8)
        feats = torch.zeros(b, 3, SLOT_FEATURES)
        valid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        logits = head(h, feats, valid)
        self.assertTrue(torch.isinf(logits[0, 1]) and logits[0, 1] < 0)
        self.assertTrue(torch.isinf(logits[1, 0]) and logits[1, 0] < 0)
        dist = torch.distributions.Categorical(logits=logits[0:1])
        for _ in range(20):
            a = dist.sample()
            self.assertIn(int(a.item()), (0, 3))  # row0: slot0 or none
        dist1 = torch.distributions.Categorical(logits=logits[1:2])
        self.assertEqual(int(dist1.sample()), 3)  # row1: only none is legal

    def test_slot_extraction_from_proprio(self):
        # proprio = body block (7) + slots (K*3); slots are the LAST dims.
        k = 3
        prop = torch.zeros(2, 7 + k * SLOT_FEATURES)
        # slot 0 valid at dist 2/4; slot 1 empty; slot 2 valid at 1/4
        prop[0, 7 + 0:7 + 3] = torch.tensor([0.5, 0.0, 0.5])
        prop[0, 7 + 6:7 + 9] = torch.tensor([-0.25, 0.1, 0.25])
        prop[1, 7:] = 0.0  # all empty
        feats, valid = slot_features_from_proprio(prop, k)
        self.assertEqual(tuple(feats.shape), (2, 3, 3))
        self.assertEqual(valid[0].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(valid[1].tolist(), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(feats[0, 0, 2]), 0.5)
        with self.assertRaises(ValueError):
            slot_features_from_proprio(torch.zeros(2, 5), 3)

    def test_state_dict_roundtrip(self):
        head = GoalExecutive(hidden_size=16, num_slots=3)
        # perturb output layer so the roundtrip is non-trivial
        with torch.no_grad():
            head.mlp[-1].weight.normal_(0.0, 0.1)
            head.mlp[-1].bias.normal_(0.0, 0.1)
        sd = head.state_dict()
        clone = GoalExecutive(hidden_size=16, num_slots=3)
        missing, unexpected = clone.load_state_dict(sd, strict=False)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])
        h = torch.randn(3, 16)
        f = torch.randn(3, 3, SLOT_FEATURES)
        v = torch.ones(3, 3)
        self.assertTrue(torch.allclose(head(h, f, v), clone(h, f, v)))


if __name__ == "__main__":
    unittest.main()
