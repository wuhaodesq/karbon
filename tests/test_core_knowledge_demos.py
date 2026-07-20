"""Tests for core-knowledge demo generation + replay prefill (A#3 / P1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.core_knowledge_demos import (
    generate_all,
    gen_intuitive_physics,
    gen_number_sense,
    gen_object_permanence,
    seed_into,
)
from src.memory.bounded_replay import BoundedReplayBuffer, ReplayBudget


def _small_buffer() -> BoundedReplayBuffer:
    return BoundedReplayBuffer(
        budget=ReplayBudget(hot_capacity=64, warm_capacity=128, cold_max_shards=2, cold_shard_size=64),
        obs_shape=(64, 64, 3),
        device="cpu",
    )


def test_generate_all_returns_transitions():
    demos = generate_all(demos_per_prior=1)
    assert len(demos) > 0
    for tr in demos[:5]:
        assert tr.obs.dtype == np.uint8
        assert tr.next_obs.dtype == np.uint8
        assert tr.obs.shape == (64, 64, 3)


def test_each_prior_generator_works():
    assert len(gen_object_permanence(1)) > 0
    assert len(gen_intuitive_physics(1)) > 0
    assert len(gen_number_sense(1)) > 0


def test_prefill_inserts_with_demo_flag():
    buf = _small_buffer()
    demos = generate_all(demos_per_prior=1)
    n = buf.prefill(demos, demo_priority=4.0)
    assert n == len(demos)
    assert len(buf.hot) == min(n, buf.hot.capacity)
    # demo transitions are inserted with raised priority (4.0 > default 1.0),
    # so PER samples them more often — this is how P1 seeds the prior.
    assert float(buf.hot.priority[0].item()) == 4.0


def test_seed_into_helper():
    buf = _small_buffer()
    n = seed_into(buf, demos_per_prior=1)
    assert n > 0
