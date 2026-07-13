"""Tests that the actor-critic vision encoder no longer builds a GB-scale
Linear (the source of the 3D per-step memory swing).

Before the fix the encoder flattened the *full* 256x256 feature map and
fed it to ``Linear(32*h*w, …)`` = ``Linear(2_097_152, …)`` (~0.5-1 GB
of weights + an equally large gradient, allocated every PPO update). After
the fix an ``AdaptiveAvgPool2d((8, 8))`` caps the spatial dim so the
trunk's first Linear takes a fixed 32*8*8 = 2048 features.

测试 actor-critic 编码器不再构造 GB 级 Linear(3D 内存摆动的源头)。
"""

from __future__ import annotations

import torch

from src.train import ActorCritic


class TestActorCriticEncoderBounded:
    def test_trunk_linear_is_fixed_dim(self) -> None:
        model = ActorCritic(obs_shape=(256, 256, 3), num_actions=8)
        assert isinstance(model.trunk, torch.nn.Sequential)
        first = model.trunk[0]
        assert isinstance(first, torch.nn.Linear)
        # Must be the pooled dim, NOT 32*256*256 (~2M).
        assert first.in_features == 32 * 8 * 8  # 2048

    def test_encoder_has_downsample_layer(self) -> None:
        model = ActorCritic(obs_shape=(256, 256, 3), num_actions=8)
        has_pool = any(
            isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.encoder
        )
        assert has_pool

    def test_forward_256_runs_and_shapes(self) -> None:
        model = ActorCritic(obs_shape=(256, 256, 3), num_actions=8)
        obs = torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8)
        logits, value = model(obs)
        assert logits.shape == (2, 8)
        assert value.shape == (2,)

    def test_forward_64_runs_and_shapes(self) -> None:
        model = ActorCritic(obs_shape=(64, 64, 3), num_actions=8)
        obs = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
        logits, value = model(obs)
        assert logits.shape == (4, 8)
        assert value.shape == (4,)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
