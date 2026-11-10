"""Tests for Stage 24 M2 goal conditioning: zero-init model path + buffer."""
import torch

from src.models.hierarchical_policy import HierarchicalActorCritic
from src.train import RolloutBuffer


def _model(goal_dim=21):
    torch.manual_seed(0)
    return HierarchicalActorCritic(
        obs_shape=(8, 8, 3), num_actions=4, d_model=32, n_layers=1,
        n_heads=4, sub_goal_every=4, proprio_dim=6, goal_dim=goal_dim,
        obj_slot_dim=56,
    )


def test_zero_init_goal_is_identity():
    m = _model().eval()
    obs = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    prop = torch.randn(2, 6)
    with torch.no_grad():
        l1, v1 = m(obs, proprio=prop)
        l2, v2 = m(obs, proprio=prop, goal=torch.zeros(2, 21))
    assert torch.allclose(l1, l2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_nonzero_goal_changes_logits():
    m = _model()
    with torch.no_grad():
        m.goal_proj.weight.normal_()
    m.eval()
    obs = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    prop = torch.randn(2, 6)
    goal = torch.zeros(2, 21)
    goal[:, 1] = 1.0  # practice
    with torch.no_grad():
        l1, _ = m(obs, proprio=prop)
        l2, _ = m(obs, proprio=prop, goal=goal)
    assert not torch.allclose(l1, l2, atol=1e-6)


def test_buffer_goal_roundtrip():
    device = torch.device("cpu")
    buf = RolloutBuffer(4, (8, 8, 3), device=device, n_envs=1,
                        proprio_dim=6, goal_dim=21)
    for i in range(3):
        g = torch.zeros(1, 21)
        g[0, i] = 1.0
        buf.add(
            obs=torch.zeros(1, 8, 8, 3, dtype=torch.uint8).numpy(),
            action=[0], logprob=[0.0], value=[0.0], reward=[0.0],
            done=[False], goal=g.numpy(),
        )
    batch = buf.as_batch()
    assert batch.goal.shape == (3, 21)
    assert batch.goal[1, 1].item() == 1.0
    assert batch.goal[2, 2].item() == 1.0


def test_obj_slots_zero_init_identity():
    m = _model().eval()
    obs = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    prop = torch.randn(2, 6)
    slots = torch.randn(2, 56)
    with torch.no_grad():
        l1, v1 = m(obs, proprio=prop)
        l2, v2 = m(obs, proprio=prop, obj_slots=torch.zeros(2, 56))
    assert torch.allclose(l1, l2, atol=1e-6)
    with torch.no_grad():
        m.obj_proj.weight.normal_()
    m.eval()
    with torch.no_grad():
        l3, _ = m(obs, proprio=prop, obj_slots=slots)
    assert not torch.allclose(l1, l3, atol=1e-6)


def test_buffer_objslots_roundtrip():
    device = torch.device("cpu")
    buf = RolloutBuffer(4, (8, 8, 3), device=device, n_envs=1,
                        proprio_dim=6, goal_dim=21, obj_slot_dim=56)
    g = torch.zeros(1, 21)
    s = torch.zeros(1, 56)
    s[0, 2] = 0.5
    buf.add(
        obs=torch.zeros(1, 8, 8, 3, dtype=torch.uint8).numpy(),
        action=[0], logprob=[0.0], value=[0.0], reward=[0.0],
        done=[False], goal=g.numpy(), obj_slots=s.numpy(),
    )
    batch = buf.as_batch()
    assert batch.objslots.shape == (1, 56)
    assert batch.objslots[0, 2].item() == 0.5
