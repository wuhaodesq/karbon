"""Tests for :mod:`src.memory.skill_library`."""

from __future__ import annotations

import time

import torch

from src.memory import (
    BoundedSkillLibrary,
    SkillEntry,
    SkillLibraryBudget,
    SkillWeights,
)


SKILL_SHAPE = (8, 4, 8)   # (d_out, rank, d_in)


def _make_lib(tmp_path, gpu_cap=2, cpu_cap=2, ssd_shards=2, ssd_shard_size=2, sim=0.99):
    return BoundedSkillLibrary(
        budget=SkillLibraryBudget(
            gpu_capacity=gpu_cap,
            cpu_capacity=cpu_cap,
            ssd_max_shards=ssd_shards,
            ssd_shard_size=ssd_shard_size,
            merge_similarity_threshold=sim,
        ),
        skill_shape=SKILL_SHAPE,
        device="cpu",
        archive_dir=tmp_path,
    )


# =====================================================================
# SkillWeights basics
# =====================================================================


def test_skill_weights_apply_shape():
    torch.manual_seed(0)
    A = torch.randn(8, 4)
    B = torch.randn(4, 8)
    w = SkillWeights(A=A, B=B)
    x = torch.randn(3, 8)
    y = w.apply(x)
    assert y.shape == (3, 8)
    assert w.num_params() == 8 * 4 + 4 * 8


def test_new_skill_generates_unique_ids(tmp_path):
    lib = _make_lib(tmp_path)
    a = lib.new_skill()
    b = lib.new_skill()
    assert a.id != b.id


# =====================================================================
# Bounded capacity
# =====================================================================


def test_gpu_tier_caps_at_gpu_capacity(tmp_path):
    lib = _make_lib(tmp_path, gpu_cap=3, cpu_cap=5, ssd_shards=2, ssd_shard_size=2, sim=0.9999)
    for _ in range(10):
        lib.add(lib.new_skill())
    stats = lib.stats()
    assert stats["gpu"]["size"] <= 3
    assert stats["cpu"]["size"] <= 5


def test_ssd_shard_count_bounded(tmp_path):
    lib = _make_lib(tmp_path, gpu_cap=1, cpu_cap=1, ssd_shards=1, ssd_shard_size=1, sim=0.9999)
    # Add so many the SSD tier has to trim
    for _ in range(20):
        lib.add(lib.new_skill())
    stats = lib.stats()
    # Only 1 shard of 1 skill max should remain on disk
    assert stats["ssd"]["shards"] <= 1


def test_total_len_le_capacity(tmp_path):
    lib = _make_lib(tmp_path)
    for _ in range(50):
        lib.add(lib.new_skill())
    assert len(lib) <= lib.capacity


def test_conforms_to_bounded_component(tmp_path):
    from src.monitoring.health_check import BoundedComponent
    lib = _make_lib(tmp_path)
    assert isinstance(lib, BoundedComponent)


# =====================================================================
# Eviction & scoring
# =====================================================================


def test_low_score_skill_evicted_first(tmp_path):
    """A skill with high usage + reward should stay on GPU tier when new skills push it."""
    lib = _make_lib(tmp_path, gpu_cap=2, cpu_cap=2, sim=0.9999)
    # Add a "star" skill and record heavy usage with high reward
    star = lib.new_skill(tag="star")
    lib.add(star)
    for _ in range(50):
        star.record_use(reward=1.0)

    # Add a fresh skill (no usage, low score)
    filler = lib.new_skill(tag="filler")
    lib.add(filler)

    # Now GPU is full. Adding another should evict `filler` (lowest score),
    # not `star`.
    lib.add(lib.new_skill(tag="new1"))

    top_tags = {e.tag for e in lib.top_k()}
    assert "star" in top_tags, f"star was evicted; top_k={top_tags}"


# =====================================================================
# Merging via similarity
# =====================================================================


def test_similar_skills_are_merged(tmp_path):
    """Two nearly-identical skills should merge into one."""
    lib = _make_lib(tmp_path, gpu_cap=4, cpu_cap=2, sim=0.9)
    torch.manual_seed(0)
    s1 = lib.new_skill()
    lib.add(s1)
    # Create s2 as a tiny perturbation of s1 → high cosine similarity
    s2 = SkillEntry(
        id=lib.new_skill().id,
        weights=SkillWeights(
            A=s1.weights.A + 1e-4 * torch.randn_like(s1.weights.A),
            B=s1.weights.B + 1e-4 * torch.randn_like(s1.weights.B),
        ),
    )
    result = lib.add(s2)
    assert result == "merged", f"expected merged, got {result}"
    assert len(lib.top_k()) == 1


def test_dissimilar_skills_not_merged(tmp_path):
    lib = _make_lib(tmp_path, gpu_cap=4, cpu_cap=2, sim=0.9)
    torch.manual_seed(0)
    a = lib.new_skill(scale=0.5)
    b = lib.new_skill(scale=0.5)
    lib.add(a)
    result = lib.add(b)
    assert result == "added", f"unexpected result {result}"
    assert len(lib.top_k()) == 2


# =====================================================================
# Skill entry metadata
# =====================================================================


def test_skill_record_use_updates_avg_reward():
    A = torch.zeros(8, 4)
    B = torch.zeros(4, 8)
    s = SkillEntry(id=0, weights=SkillWeights(A=A, B=B))
    assert s.avg_reward == 0.0
    for r in [1.0, 2.0, 3.0]:
        s.record_use(r)
    assert abs(s.avg_reward - 2.0) < 1e-6
    assert s.usage_count == 3


# =====================================================================
# Persistence
# =====================================================================


def test_state_dict_roundtrip(tmp_path):
    lib = _make_lib(tmp_path, gpu_cap=3, cpu_cap=2, sim=0.9999)
    for _ in range(6):
        lib.add(lib.new_skill(tag="s"))
    state = lib.state_dict()

    lib2 = _make_lib(tmp_path / "other", gpu_cap=3, cpu_cap=2, sim=0.9999)
    lib2.load_state_dict(state)

    assert len(lib2._gpu) == len(lib._gpu)
    assert len(lib2._cpu) == len(lib._cpu)
    # Same ids
    ids_before = sorted(e.id for e in lib.iter_all_in_memory())
    ids_after = sorted(e.id for e in lib2.iter_all_in_memory())
    assert ids_before == ids_after


def test_lookup_across_tiers(tmp_path):
    """With enough SSD capacity, every added skill remains retrievable."""
    lib = _make_lib(
        tmp_path, gpu_cap=1, cpu_cap=1,
        ssd_shards=8, ssd_shard_size=1,
        sim=0.9999,
    )
    ids = []
    for _ in range(5):
        s = lib.new_skill()
        lib.add(s)
        ids.append(s.id)
    for sid in ids:
        got = lib.get(sid)
        assert got is not None, f"missing skill id={sid}"
        assert got.id == sid


# =====================================================================
# Stats
# =====================================================================


def test_stats_shape(tmp_path):
    lib = _make_lib(tmp_path)
    for _ in range(3):
        lib.add(lib.new_skill())
    s = lib.stats()
    assert set(s.keys()) >= {"gpu", "cpu", "ssd", "total", "total_capacity", "next_id"}
