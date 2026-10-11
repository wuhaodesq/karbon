"""Tests for Stage 22 M3 — AnalogyEngine (bounded relation transfer).

M3 criterion: held-out analogies (novel pairs + novel sources) top-1
accuracy > random + 0.1.
"""

from __future__ import annotations

import torch

from src.models.analogy import AnalogyEngine


def _make_world(n_per=30, dim=16, seed=0, noise=0.05):
    """Entities in 3 relation chains: above (+z), older (+t), left (+x)."""
    g = torch.Generator().manual_seed(seed)
    R = {
        "above": torch.tensor([0.0, 1.0] + [0.0] * (dim - 2)),
        "older": torch.tensor([1.0, 0.0] + [0.0] * (dim - 2)),
        "left": torch.tensor([0.0, 0.0, 1.0] + [0.0] * (dim - 3)),
    }
    entities, pairs = {}, []
    for rname, rvec in R.items():
        for i in range(n_per):
            base = torch.randn(dim, generator=g)
            a = f"{rname}_a{i}"
            b = f"{rname}_b{i}"
            entities[a] = base + torch.randn(dim, generator=g) * noise
            entities[b] = base + rvec + torch.randn(dim, generator=g) * noise
            pairs.append((a, b))
    return entities, pairs


def _engine(entities, pairs):
    e = AnalogyEngine(dim=16, max_entities=512)
    for name, emb in entities.items():
        e.add_entity(name, emb)
    for a, b in pairs:
        e.add_mapping(a, b)
    return e


def test_analogy_transfer_held_out():
    """Novel pair example + novel source: top-1 vs random."""
    ents, pairs = _make_world(seed=1)
    eng = _engine(ents, pairs)
    # held-out evaluation: use each mapping as the demonstration, a DIFFERENT
    # source from the same relation family, and the true partner as target.
    hits, total = 0, 0
    for i, (a, b) in enumerate(pairs):
        # source: next pair's a in the same relation (a different triple)
        rname = a.split("_")[0]
        src_pairs = [(x, y) for x, y in pairs if x.startswith(rname)]
        if len(src_pairs) < 2:
            continue
        c, d_true = src_pairs[(i + 1) % len(src_pairs)]
        top = eng.analogize(a, b, c, top_k=1)
        if top and top[0][0] == d_true:
            hits += 1
        total += 1
    # random baseline: 1/(N-3) chance; with ~90 entities that is ~0.011
    rand = 1.0 / (eng.n_entities - 3)
    acc = hits / max(1, total)
    assert acc > rand + 0.1, f"acc={acc:.3f} rand={rand:.3f}"


def test_clean_offset_returns_exact_partner():
    g = torch.Generator().manual_seed(2)
    r = torch.tensor([0.0, 1.0] + [0.0] * 14)
    a = torch.randn(16, generator=g)
    ents = {"a": a, "b": a + r, "c": a + torch.randn(16, generator=g),
            "d": a + torch.randn(16, generator=g) + r}
    eng = AnalogyEngine(dim=16)
    for k, v in ents.items():
        eng.add_entity(k, v)
    top = eng.analogize("a", "b", "c", top_k=1)
    assert top and top[0][0] == "d"


def test_bounded_entities_and_mappings():
    e = AnalogyEngine(dim=8, max_entities=3, max_mappings=4)
    for i in range(5):
        e.add_entity(f"e{i}", torch.randn(8))
    assert e.n_entities == 3  # Axiom 1
    for i in range(6):
        e.add_mapping("e0", "e1")
    assert len(e._mappings) <= 4


def test_missing_entity_honest_empty():
    e = AnalogyEngine(dim=8)
    e.add_entity("a", torch.randn(8))
    assert e.analogize("a", "b", "c") == []


def test_state_dict_roundtrip():
    ents, pairs = _make_world(n_per=5, seed=3)
    eng = _engine(ents, pairs)
    sd = eng.state_dict()
    eng2 = AnalogyEngine(dim=16)
    eng2.load_state_dict(sd)
    assert eng2.n_entities == eng.n_entities
    top1 = eng.analogize(pairs[0][0], pairs[0][1], pairs[1][0], top_k=1)
    top2 = eng2.analogize(pairs[0][0], pairs[0][1], pairs[1][0], top_k=1)
    assert [t[0] for t in top1] == [t[0] for t in top2]
