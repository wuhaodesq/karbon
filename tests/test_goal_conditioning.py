"""Tests for Stage 24 M2 goal conditioning: zero-init model path + buffer."""
import torch

from src.models.hierarchical_policy import HierarchicalActorCritic
from src.train import RolloutBuffer


def _model(goal_dim=14):
    torch.manual_seed(0)
    return HierarchicalActorCritic(
        obs_shape=(8, 8, 3), num_actions=4, d_model=32, n_layers=1,
        n_heads=4, sub_goal_every=4, proprio_dim=6, goal_dim=goal_dim,
    )


def test_zero_init_goal_is_identity():
    m = _model().eval()
    obs = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    prop = torch.randn(2, 6)
    with torch.no_grad():
        l1, v1 = m(obs, proprio=prop)
        l2, v2 = m(obs, proprio=prop, goal=torch.zeros(2, 14))
    assert torch.allclose(l1, l2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_nonzero_goal_changes_logits():
    m = _model()
    with torch.no_grad():
        m.goal_proj.weight.normal_()
    m.eval()
    obs = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    prop = torch.randn(2, 6)
    goal = torch.zeros(2, 14)
    goal[:, 1] = 1.0  # practice
    with torch.no_grad():
        l1, _ = m(obs, proprio=prop)
        l2, _ = m(obs, proprio=prop, goal=goal)
    assert not torch.allclose(l1, l2, atol=1e-6)


def test_buffer_goal_roundtrip():
    device = torch.device("cpu")
    buf = RolloutBuffer(4, (8, 8, 3), device=device, n_envs=1,
                        proprio_dim=6, goal_dim=14)
    for i in range(3):
        g = torch.zeros(1, 14)
        g[0, i] = 1.0
        buf.add(
            obs=torch.zeros(1, 8, 8, 3, dtype=torch.uint8).numpy(),
            action=[0], logprob=[0.0], value=[0.0], reward=[0.0],
            done=[False], goal=g.numpy(),
        )
    batch = buf.as_batch()
    assert batch.goal.shape == (3, 14)
    assert batch.goal[1, 1].item() == 1.0
    assert batch.goal[2, 2].item() == 1.0
