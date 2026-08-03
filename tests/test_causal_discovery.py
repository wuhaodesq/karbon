"""Tests for CausalDiscovery — both the legacy WM-imagination mode and the
Stage-14 experience mode (real-trajectory intervention statistics)."""

from __future__ import annotations

import torch

from src.models.causal_discovery import CausalDiscovery, ExperienceTransition


def _seed(cd: CausalDiscovery, mags: list[tuple[int, float]], n: int = 200) -> None:
    """Fill the experience buffer with synthetic transitions.

    mags: list of (action, transition_magnitude) pairs. Every transition of
    action ``a`` has ``||s' - s|| = mag`` (plus small noise).
    """
    torch.manual_seed(0)
    for _ in range(n):
        a, mag = mags[0]
        s = torch.randn(8)
        s_next = s + torch.randn(8) * 0.01 * mag + mag * torch.randn(8) / 8
        cd.observe(s, a, s_next, 0.0, step=_)
        mags = mags[1:] + mags[:1]


def test_experience_mode_records_causal_edges() -> None:
    """An action with above-baseline transition magnitude gets a causal edge."""
    cd = CausalDiscovery(num_actions=3, min_intervention_effect=0.005,
                         mode="experience", buffer_capacity=1024)
    # action 0: big transitions (mag 1.0); actions 1-2: small (mag 0.1)
    _seed(cd, [(0, 1.0), (1, 0.1), (2, 0.1)], n=600)
    for i in range(20):  # repeated interventions let the EMA strength converge
        effects = cd.intervene_from_experience(step=1000 + i)

    assert "action_0_effect" in effects
    assert effects["action_0_effect"] > effects["action_1_effect"]
    assert len(cd) >= 1, "action_0 should have been recorded as a cause"

    causes = cd.query_why("world_state")
    assert any("action_0" in c for c in causes)


def test_experience_mode_requires_minimum_samples() -> None:
    cd = CausalDiscovery(num_actions=2, mode="experience", buffer_capacity=128)
    _seed(cd, [(0, 1.0), (1, 0.1)], n=20)  # below the 64-transition minimum
    effects = cd.intervene_from_experience(step=10)
    assert effects == {}


def test_experience_buffer_is_bounded() -> None:
    cd = CausalDiscovery(num_actions=2, mode="experience", buffer_capacity=32)
    _seed(cd, [(0, 1.0), (1, 0.1)], n=500)
    assert len(cd._buffer) <= 32  # BOUNDS-OK: capacity declared at construction


def test_observe_ignored_in_wm_mode() -> None:
    cd = CausalDiscovery(num_actions=2, mode="wm", buffer_capacity=16)
    _seed(cd, [(0, 1.0), (1, 0.1)], n=100)
    assert len(cd._buffer) == 0
    assert cd.intervene_from_experience(step=1) == {}


def test_legacy_wm_intervene_still_works() -> None:
    """The WM-imagination path must remain functional (3D world future use)."""
    cd = CausalDiscovery(num_actions=2, min_intervention_effect=0.001, mode="wm")

    class FakeWM:
        def __init__(self) -> None:
            self._i = 0

        def imagine_step(self, state, onehot):
            self._i += 1
            return (None, None)

        def decode(self, state):
            # deterministic decode depending on which call index we are at
            if self._i % 2 == 1:
                return torch.zeros(1, 4)
            return torch.ones(1, 4)

    class FakeState:
        pass

    effects = cd.intervene(FakeWM(), FakeState(), actual_action=0,
                           slot_states=torch.zeros(2, 4), step=5)
    assert "action_1_effect" in effects
    assert effects["action_1_effect"] > 0.0


def test_state_dict_roundtrip() -> None:
    cd = CausalDiscovery(num_actions=3, min_intervention_effect=0.005,
                         mode="experience", buffer_capacity=1024)
    _seed(cd, [(0, 1.0), (1, 0.1), (2, 0.1)], n=600)
    cd.intervene_from_experience(step=1000)

    cd2 = CausalDiscovery(num_actions=3, min_intervention_effect=0.005,
                          mode="experience", buffer_capacity=1024)
    cd2.load_state_dict(cd.state_dict())
    assert len(cd2) == len(cd)
    assert cd2.state_dict()["intervention_count"] == cd.state_dict()["intervention_count"]
    # restored graph answers the same query
    assert cd2.query_why("world_state") == cd.query_why("world_state")


def test_trim_graph_stays_bounded() -> None:
    cd = CausalDiscovery(num_actions=8, max_edges=4, min_intervention_effect=0.0,
                         mode="experience", buffer_capacity=4096)
    for i in range(100):
        # one strong + one weak transition per action id
        s = torch.randn(8)
        cd.observe(s, i % 8, s + torch.randn(8) * (1.0 if i % 2 == 0 else 0.0), 0.0, step=i)
    cd.intervene_from_experience(step=100)
    assert len(cd) <= 4, "graph must never exceed max_edges (Axiom 1)"


def test_experience_transition_detaches() -> None:
    """Stage-11 lesson: no autograd graph may be held in the buffer."""
    cd = CausalDiscovery(num_actions=2, mode="experience", buffer_capacity=16)
    s = torch.randn(8, requires_grad=True)
    cd.observe(s, 0, s * 2, 1.0, step=1)
    tr = cd._buffer[-1]
    assert isinstance(tr, ExperienceTransition)
    assert tr.state.requires_grad is False
    assert tr.next_state.requires_grad is False
