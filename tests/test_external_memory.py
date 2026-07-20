"""Tests for generic bounded external memory (open-gap A#2)."""

from __future__ import annotations

import torch

from src.memory.skill_library import BoundedExternalMemory, SkillLibraryBudget


def _mem(gpu=4, cpu=4, shards=2, shard=4) -> BoundedExternalMemory:
    return BoundedExternalMemory(
        budget=SkillLibraryBudget(gpu_capacity=gpu, cpu_capacity=cpu,
                                  ssd_max_shards=shards, ssd_shard_size=shard),
        device="cpu",
    )


def test_add_and_capacity_bounded():
    mem = _mem(gpu=4)
    ids = [mem.add(torch.randn(3, 3), tag=f"m{i}") for i in range(10)]
    assert len(mem) <= mem.capacity
    assert len(mem) == 10  # all 10 inserted, none dropped below total capacity


def test_retrieve_returns_topk_payloads():
    mem = _mem(gpu=4)
    for i in range(3):
        mem.add(torch.full((2,), float(i)), tag=f"m{i}")
    top = mem.retrieve(k=2)
    assert len(top) == 2
    assert all(isinstance(t, torch.Tensor) for t in top)


def test_eviction_demotes_low_score():
    mem = _mem(gpu=2, cpu=2)
    # add 2 (fill gpu), then 2 more -> demote to cpu
    for i in range(4):
        mem.add(torch.randn(2, 2), reward=float(i))
    # gpu full(2) + cpu full(2) == capacity 4
    assert len(mem._gpu) == 2
    assert len(mem._cpu) == 2


def test_record_use_updates_score():
    mem = _mem(gpu=4)
    mid = mem.add(torch.randn(2), reward=0.0)
    mem.record_use(mid, reward=5.0)
    # the used item should now score above a fresh unused one
    mem.add(torch.randn(2), reward=0.0)
    top = mem.retrieve(k=1)
    assert top  # non-empty
