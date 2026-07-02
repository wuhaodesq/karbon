"""Tests for :mod:`src.memory.bounded_replay`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.memory import (
    BoundedReplayBuffer,
    ColdShardTier,
    HotRingTier,
    ReplayBudget,
    Transition,
    WarmRingTier,
)


OBS_SHAPE = (4, 4, 3)


def _mk_transition(seed: int = 0) -> Transition:
    rng = np.random.default_rng(seed)
    return Transition(
        obs=rng.integers(0, 255, size=OBS_SHAPE, dtype=np.uint8),
        action=int(rng.integers(0, 7)),
        reward=float(rng.random()),
        next_obs=rng.integers(0, 255, size=OBS_SHAPE, dtype=np.uint8),
        done=bool(rng.integers(0, 2)),
        priority=float(rng.random() + 0.1),
    )


# =====================================================================
# HotRingTier
# =====================================================================


def test_hot_tier_capacity_enforced():
    tier = HotRingTier(capacity=8, obs_shape=OBS_SHAPE, device="cpu")
    assert tier.capacity == 8
    for i in range(20):  # more than capacity
        tier.add(_mk_transition(i))
    assert len(tier) == 8
    assert len(tier) <= tier.capacity  # Axiom 1


def test_hot_tier_returns_evicted_when_full():
    tier = HotRingTier(capacity=4, obs_shape=OBS_SHAPE, device="cpu")
    first = _mk_transition(0)
    assert tier.add(first) is None  # empty
    for i in range(1, 4):
        assert tier.add(_mk_transition(i)) is None
    # Now full — next add should evict something
    evicted = tier.add(_mk_transition(99))
    assert evicted is not None
    # Evicted should be the first one added
    np.testing.assert_array_equal(evicted.obs, first.obs)


def test_hot_tier_uniform_sample():
    tier = HotRingTier(capacity=16, obs_shape=OBS_SHAPE, device="cpu")
    for i in range(16):
        tier.add(_mk_transition(i))
    rng = np.random.default_rng(0)
    idx = tier.sample_indices(8, rng)
    batch = tier.gather(idx)
    assert batch["obs"].shape == (8, *OBS_SHAPE)
    assert batch["action"].shape == (8,)


def test_hot_tier_priority_update():
    tier = HotRingTier(capacity=4, obs_shape=OBS_SHAPE, device="cpu")
    for i in range(4):
        tier.add(_mk_transition(i))
    tier.update_priorities(np.array([0, 2]), np.array([5.0, 7.0]))
    assert float(tier.priority[0]) == pytest.approx(5.0)
    assert float(tier.priority[2]) == pytest.approx(7.0)


def test_hot_tier_state_roundtrip():
    tier = HotRingTier(capacity=8, obs_shape=OBS_SHAPE, device="cpu")
    for i in range(6):
        tier.add(_mk_transition(i))
    state = tier.state_dict()
    tier2 = HotRingTier(capacity=8, obs_shape=OBS_SHAPE, device="cpu")
    tier2.load_state_dict(state)
    assert len(tier2) == 6
    torch.testing.assert_close(tier.obs[:6], tier2.obs[:6])


# =====================================================================
# WarmRingTier
# =====================================================================


def test_warm_tier_capacity_enforced():
    tier = WarmRingTier(capacity=5)
    for i in range(12):
        tier.add(_mk_transition(i))
    assert len(tier) == 5
    assert len(tier) <= tier.capacity


def test_warm_tier_evicts_and_returns():
    tier = WarmRingTier(capacity=3)
    trs = [_mk_transition(i) for i in range(5)]
    evicted_at_full = tier.add(trs[0])
    assert evicted_at_full is None
    tier.add(trs[1])
    tier.add(trs[2])
    # Fourth entry should evict the first
    evicted = tier.add(trs[3])
    assert evicted is not None
    np.testing.assert_array_equal(evicted.obs, trs[0].obs)


# =====================================================================
# ColdShardTier
# =====================================================================


def test_cold_tier_shard_count_bounded(tmp_path):
    tier = ColdShardTier(
        archive_dir=tmp_path,
        max_shards=2,
        shard_size=4,
    )
    # Add 3 shards worth of transitions → total 12
    for i in range(3 * 4):
        tier.add(_mk_transition(i))
    tier.flush()
    shards = tier.known_shards()
    # max_shards=2 → oldest evicted
    assert len(shards) == 2, f"got shards={shards}"
    files = list(tmp_path.glob("shard_*.pkl"))
    assert len(files) == 2


def test_cold_tier_iter_returns_all(tmp_path):
    tier = ColdShardTier(archive_dir=tmp_path, max_shards=5, shard_size=3)
    added = [_mk_transition(i) for i in range(7)]
    for tr in added:
        tier.add(tr)
    tier.flush()
    fetched = list(tier.iter_all())
    assert len(fetched) == 7


def test_cold_tier_survives_restart(tmp_path):
    """Shards on disk should be re-discovered by a fresh instance."""
    t1 = ColdShardTier(archive_dir=tmp_path, max_shards=5, shard_size=2)
    for i in range(4):
        t1.add(_mk_transition(i))
    t1.flush()
    # New instance sees existing shards
    t2 = ColdShardTier(archive_dir=tmp_path, max_shards=5, shard_size=2)
    assert len(t2.known_shards()) == 2


# =====================================================================
# BoundedReplayBuffer (top-level orchestration)
# =====================================================================


def test_replay_total_capacity_bounded(tmp_path):
    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=4, warm_capacity=8, cold_max_shards=1, cold_shard_size=4),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    # Add far beyond capacity; verify neither tier overflows
    for i in range(100):
        buf.add(_mk_transition(i))
    buf.flush_cold()
    assert len(buf.hot) <= buf.hot.capacity
    assert len(buf.warm) <= buf.warm.capacity
    assert len(buf.cold.known_shards()) <= buf.cold._max_shards
    # Total bounded
    assert len(buf) <= buf.capacity


def test_replay_demotes_across_tiers(tmp_path):
    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=2, warm_capacity=2, cold_max_shards=2, cold_shard_size=2),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    # Fill hot, then warm, then some spill to cold
    for i in range(10):
        buf.add(_mk_transition(i))
    buf.flush_cold()
    assert len(buf.hot) == 2
    assert len(buf.warm) == 2
    # 10 total items, 2 in hot, 2 in warm, remaining 6 → cold (bounded to 2 shards × 2 = 4)
    assert len(buf.cold.known_shards()) <= 2


def test_replay_sample_shapes(tmp_path):
    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=8, warm_capacity=16, cold_max_shards=1, cold_shard_size=4),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    for i in range(30):
        buf.add(_mk_transition(i))
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, *OBS_SHAPE)
    assert batch["action"].shape == (8,)
    assert batch["reward"].shape == (8,)


def test_replay_prioritized_sample(tmp_path):
    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=32, warm_capacity=8, cold_max_shards=1, cold_shard_size=4),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    for i in range(32):
        buf.add(_mk_transition(i))
    batch, indices, weights = buf.sample_prioritized(6, alpha=0.6)
    assert batch["obs"].shape == (6, *OBS_SHAPE)
    assert indices.shape == (6,)
    assert weights.shape == (6,)
    assert weights.max() <= 1.0 + 1e-6


def test_replay_stats(tmp_path):
    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=4, warm_capacity=4, cold_max_shards=1, cold_shard_size=2),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    for i in range(6):
        buf.add(_mk_transition(i))
    s = buf.stats()
    assert s["hot"]["size"] <= 4
    assert s["warm"]["size"] <= 4
    assert s["total_capacity"] == buf.capacity


def test_replay_conforms_to_bounded_component(tmp_path):
    """BoundedReplayBuffer must satisfy HealthChecker.BoundedComponent protocol."""
    from src.monitoring.health_check import BoundedComponent

    buf = BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=2, warm_capacity=2, cold_max_shards=1, cold_shard_size=2),
        obs_shape=OBS_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )
    assert isinstance(buf, BoundedComponent)
