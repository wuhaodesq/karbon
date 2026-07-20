"""Tests for core-knowledge auxiliary losses (open-gap A#4 / P2)."""

from __future__ import annotations

import torch

from src.intrinsic.core_knowledge_loss import (
    CoreKnowledgeAuxLoss,
    CoreKnowledgeLossConfig,
)


def _rec(object_permanence_bad: bool = False, physics_misaligned: bool = False,
         count_err: float = 0.0, device="cpu") -> dict:
    B, K = 2, 3
    belief = torch.ones(B, K)
    removed = torch.zeros(B, K)
    if object_permanence_bad:
        belief[0, 0] = 0.0  # dropped belief on a non-removed object
    force = torch.zeros(B, 2)
    ovel = torch.zeros(B, K, 2)
    if physics_misaligned:
        force[0] = torch.tensor([1.0, 0.0])
        ovel[0, 0] = torch.tensor([-1.0, 0.0])  # moves opposite to force
    else:
        force[0] = torch.tensor([1.0, 0.0])
        ovel[0, 0] = torch.tensor([0.9, 0.05])  # aligned
    count_est = torch.tensor([3.0, 2.0]) + count_err
    true_count = torch.tensor([3.0, 2.0])
    return {
        "existence_belief": belief,
        "removed_mask": removed,
        "force": force,
        "object_vel": ovel,
        "count_est": count_est,
        "true_count": true_count,
    }


def test_physics_aligned_low_loss():
    out = CoreKnowledgeAuxLoss()( _rec(physics_misaligned=False))
    assert out["intuitive_physics"].item() == 0.0


def test_physics_misaligned_nonzero_loss():
    out = CoreKnowledgeAuxLoss()(_rec(physics_misaligned=True))
    assert out["intuitive_physics"].item() > 0.0


def test_object_permanence_penalizes_dropped_belief():
    out = CoreKnowledgeAuxLoss()(_rec(object_permanence_bad=True))
    assert out["object_permanence"].item() > 0.0
    # without the bad case, loss is ~0
    out_ok = CoreKnowledgeAuxLoss()(_rec(object_permanence_bad=False))
    assert out_ok["object_permanence"].item() == 0.0


def test_number_sense_error_proportional():
    out = CoreKnowledgeAuxLoss()(_rec(count_err=1.0))
    # env0: est4/true3 -> 1/3 ; env1: est3/true2 -> 1/2 ; mean = 5/12
    assert abs(out["number_sense"].item() - (5.0 / 12.0)) < 1e-5


def test_total_is_weighted_sum():
    cfg = CoreKnowledgeLossConfig(coef_object_permanence=0.1, coef_intuitive_physics=0.2,
                                  coef_number_sense=0.3)
    out = CoreKnowledgeAuxLoss(cfg)(_rec(object_permanence_bad=True,
                                        physics_misaligned=True, count_err=1.0))
    expected = 0.1 * out["object_permanence"] + 0.2 * out["intuitive_physics"] + 0.3 * out["number_sense"]
    assert torch.allclose(out["total"], expected, atol=1e-6)
