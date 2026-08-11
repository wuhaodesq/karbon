"""Tests for the Stage 19 NarrativeLoopController (记忆->叙事->策略调制闭环)."""

from __future__ import annotations

import pytest
import torch

from src.models.narrative_loop import NarrativeLoopController


class _FakeAuto:
    def __init__(self) -> None:
        self._events: list = []
        self._max = 100

    @property
    def capacity(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._events)

    def add_event(self, step, description, importance, episode_id, lesson=""):
        self._events.append({
            "step": step, "description": description, "importance": importance,
        })
        return self._events[-1]


class _FakeIdentity:
    def __init__(self) -> None:
        self._min_events = 1
        self._calls = 0

    def __call__(self, events):
        self._calls += 1
        return {
            "traits": {"openness": 0.8, "conscientiousness": 0.2,
                       "extraversion": 0.5, "agreeableness": 0.5,
                       "neuroticism": 0.1},
            "narrative": "I am someone who explores often.",
            "event_count": len(events),
        }


class _FakeSymbolBackend:
    def __init__(self) -> None:
        self._rules_db = [
            {"if": [("condition", (0,))], "then": ("action", (3,)),
             "confidence": 0.9},
            {"if": [("condition", (1,))], "then": ("action", (7,)),
             "confidence": 0.6},
        ]
        self._rule_count = 2
        self._total_queries = 0
        self._correct_predictions = 0
        self._inference_buffer: list = []

    def predict_action(self, if_preds):
        self._total_queries += 1
        action = if_preds[0][1][0]
        for rule in self._rules_db:
            if rule["if"][0][1][0] == action:
                # Real SymbolBackend.predict_action returns ("action", int)
                return type("R", (), {
                    "answers": [("action", rule["then"][1][0])],
                    "confidence": rule["confidence"],
                })()
        return type("R", (), {"answers": [], "confidence": 0.0})()

    def feedback(self, idx, correct):
        self._inference_buffer.append(correct)
        if correct:
            self._correct_predictions += 1


class _FakeThoughtLoop:
    def __init__(self) -> None:
        self._cached_lang_embedding = torch.zeros(128)
        self._has_active_thought = False
        self._think_calls = 0

    def maybe_think(self, hidden_state, episode_return=0.0, episode_done=False):
        self._think_calls += 1
        return None


def test_narrative_loop_stores_events():
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=_FakeAuto(),
                                 identity_narrative=_FakeIdentity(),
                                 symbol_backend=_FakeSymbolBackend())
    nl.episode_end_hook(step=1000, ep_ret=0.8, ep_id=1,
                        description="Completed task", lesson="Learn")
    assert len(nl.autobiographical._events) == 1


def test_narrative_loop_generates_narrative_periodically():
    auto = _FakeAuto()
    ident = _FakeIdentity()
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=auto,
                                 identity_narrative=ident,
                                 symbol_backend=_FakeSymbolBackend(),
                                 narrative_every_episodes=2)
    # episode 1: no narrative yet (episode_count=1, 1%2!=0)
    nl.episode_end_hook(100, 0.8, 1)
    assert nl._narrative_count == 0
    # episode 2: narrative triggers (2%2==0)
    nl.episode_end_hook(200, 0.8, 2)
    assert nl._narrative_count == 1
    assert nl.has_active_narrative
    assert "explores" in nl._last_narrative


def test_narrative_loop_symbol_bias():
    sb = _FakeSymbolBackend()
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=_FakeAuto(),
                                 identity_narrative=_FakeIdentity(),
                                 symbol_backend=sb,
                                 symbol_bias_weight=0.1)
    nl.episode_end_hook(100, 0.8, 1)
    bias = nl.get_symbol_bias()
    assert bias is not None
    assert bias.shape == (12,)
    # Best rule (conf 0.9) -> action 3, bias = 0.1 * 0.9 = 0.09
    assert abs(float(bias[3] - 0.09)) < 1e-6
    assert float(bias.sum()) > 0
    # Queries must have happened (kanren consumed, not count-only)
    assert sb._total_queries > 0


def test_narrative_loop_step_hook_delegates():
    tl = _FakeThoughtLoop()
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=_FakeAuto(),
                                 identity_narrative=_FakeIdentity(),
                                 symbol_backend=_FakeSymbolBackend(),
                                 thought_loop=tl)
    nl.step_hook(torch.zeros(128))
    assert tl._think_calls == 1


def test_narrative_loop_bounded():
    """Axiom 1: no unbounded growth in cached state."""
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=_FakeAuto(),
                                 identity_narrative=_FakeIdentity(),
                                 symbol_backend=_FakeSymbolBackend())
    for i in range(50):
        nl.episode_end_hook(i * 100, 0.8, i)
    assert len(nl._last_traits) == 5
    assert nl._symbol_bias.shape == (12,)
    assert nl.capacity == 1
    assert len(nl) == 1


def test_narrative_loop_graceful_degradation():
    """No components -> no crash, no bias."""
    nl = NarrativeLoopController(d_model=128, num_actions=12)
    nl.episode_end_hook(100, 0.8, 1)
    nl.step_hook(torch.zeros(128))
    assert nl.get_symbol_bias() is None
    assert not nl.has_active_narrative


def test_narrative_loop_state_dict_roundtrip():
    nl = NarrativeLoopController(d_model=128, num_actions=12,
                                 autobiographical=_FakeAuto(),
                                 identity_narrative=_FakeIdentity(),
                                 symbol_backend=_FakeSymbolBackend(),
                                 narrative_every_episodes=1)
    for i in range(3):
        nl.episode_end_hook(i * 100, 0.8, i)
    sd = nl.state_dict()
    assert sd["_narrative_count"] == 3
    nl2 = NarrativeLoopController(d_model=128, num_actions=12,
                                  autobiographical=_FakeAuto(),
                                  identity_narrative=_FakeIdentity(),
                                  symbol_backend=_FakeSymbolBackend())
    nl2.load_state_dict(sd)
    assert nl2._narrative_count == 3
    assert nl2._last_narrative == nl._last_narrative
    if nl.get_symbol_bias() is not None:
        assert torch.equal(nl2.get_symbol_bias(), nl.get_symbol_bias())
