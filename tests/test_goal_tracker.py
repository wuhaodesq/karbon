"""Tests for Stage 24 M2: GoalTracker (env-side achievement) + GoalSession."""
from src.models.goal_proposer import Goal, GoalProposer, GoalSession, GoalTracker


def _facts(**over):
    base = {"x": 0.0, "y": 0.0, "movement": 0.001, "success": False,
            "sibling_dist": 9.9, "danger": 0.9, "probe_active": False,
            "novelty": 0.0}
    base.update(over)
    return base


def _g(kind, target, u=0.5):
    return Goal(kind, target, u, "test")


def test_explore_achievement_on_new_zone():
    tr = GoalTracker()
    tr.start(_g("explore", "novel_zone"), None, (0.0, 0.0))
    assert tr.step(_facts())["chosen"] is False
    assert tr.step(_facts(x=5.0, y=5.0))["chosen"] is True


def test_practice_achievement_on_success():
    tr = GoalTracker()
    tr.start(_g("practice", "7"), None, (0.0, 0.0))
    assert tr.step(_facts(task_id="7"))["chosen"] is False
    assert tr.step(_facts(success=True, task_id="7"))["chosen"] is True


def test_approach_sibling_and_open_space():
    tr = GoalTracker()
    tr.start(_g("approach", "sibling"), None, (0.0, 0.0),
             _facts(sibling_dist=9.9))
    assert tr.step(_facts(sibling_dist=2.0))["chosen"] is False
    assert tr.step(_facts(sibling_dist=1.0))["chosen"] is True

    tr2 = GoalTracker()
    tr2.start(_g("approach", "open_space"), None, (0.0, 0.0),
              _facts(danger=0.9))
    assert tr2.step(_facts(danger=0.8))["chosen"] is False
    assert tr2.step(_facts(danger=0.1))["chosen"] is True


def test_ambient_proximity_is_not_achievement():
    tr = GoalTracker()
    tr.start(_g("approach", "sibling"), None, (0.0, 0.0),
             _facts(sibling_dist=1.0))  # already near at start
    assert tr.step(_facts(sibling_dist=0.5))["chosen"] is False

    tr2 = GoalTracker()
    tr2.start(_g("approach", "open_space"), None, (0.0, 0.0),
              _facts(danger=0.05))  # already safe at start
    assert tr2.step(_facts(danger=0.0))["chosen"] is False


def test_test_achievement_on_probe_edge():
    tr = GoalTracker()
    tr.start(_g("test", "hypothesis"), None, (0.0, 0.0))
    tr.step(_facts(probe_active=True))  # first True is already a rising edge
    tr2 = GoalTracker()
    tr2.start(_g("test", "hypothesis"), None, (0.0, 0.0))
    assert tr2.step(_facts(probe_active=False))["chosen"] is False
    assert tr2.step(_facts(probe_active=True))["chosen"] is True


def test_idle_achievement_on_stillness_window():
    tr = GoalTracker(ttl=400, idle_k=10)
    tr.start(_g("idle", "stillness"), None, (0.0, 0.0))
    hits = [tr.step(_facts(movement=0.001))["chosen"] for _ in range(9)]
    assert not any(hits)
    assert tr.step(_facts(movement=0.001))["chosen"] is True


def test_expiry_and_failure_feedback():
    gp = GoalProposer(seed=0)
    sess = GoalSession(gp, seed=0, ttl=3, inject=False)
    sess.tick(_facts(), due=True)  # start a window
    assert sess.tracker.active
    for _ in range(3):
        sess.tick(_facts(), due=False)
    sess.tick(_facts(), due=False)  # expiry recorded, new goal proposed
    assert sess._expired >= 1
    assert sess.telemetry()["n_proposals"] >= 1


def test_session_bonus_only_when_inject():
    s_off = GoalSession(GoalProposer(seed=0), seed=1, ttl=400, inject=False)
    s_off.tick(_facts(), due=True)
    s_off.tracker._chosen = _g("practice", "t", 0.5)
    s_off.tracker._hit = {"chosen": False, "random": False}
    r = s_off.tick(_facts(success=True, task_id="t"), due=False)
    assert r["chosen_achieved"] is True and r["bonus"] == 0.0

    s_on = GoalSession(GoalProposer(seed=0), seed=1, ttl=400, inject=True)
    s_on.tick(_facts(), due=True)
    s_on.tracker._chosen = _g("practice", "t", 0.5)
    s_on.tracker._hit = {"chosen": False, "random": False}
    r = s_on.tick(_facts(success=True, task_id="t"), due=False)
    assert r["chosen_achieved"] is True and 0.0 < r["bonus"] <= 0.2


def test_random_control_counted_without_bonus():
    gp = GoalProposer(seed=0)
    sess = GoalSession(gp, seed=2, ttl=400, inject=True)
    sess.tick(_facts(), due=True)
    # force both arms to the same achievable goal
    sess.tracker._chosen = _g("practice", "t", 0.5)
    sess.tracker._random = _g("practice", "t", 0.5)
    sess.tracker._hit = {"chosen": False, "random": False}
    r = sess.tick(_facts(success=True, task_id="t"), due=False)
    assert r["chosen_achieved"] and r["random_achieved"]
    assert sess.telemetry()["achieved_random"] == 1
    assert r["bonus"] <= 0.2


def test_session_state_roundtrip():
    gp = GoalProposer(seed=3)
    sess = GoalSession(gp, seed=3, ttl=10, inject=True)
    sess.tick(_facts(), due=True)
    st = sess.state_dict()
    sess2 = GoalSession(GoalProposer(seed=3), seed=3, ttl=10, inject=True)
    sess2.load_state_dict(st)
    assert sess2._n_proposals == sess._n_proposals
    assert sess2.telemetry()["n_proposals"] == sess.telemetry()["n_proposals"]


def test_drive_proxies_bounds_and_signs():
    sess = GoalSession(GoalProposer(seed=0), seed=0)
    d = sess.drive_proxies(_facts(success=True, sibling_dist=0.5, danger=0.0))
    assert set(d) == {"curiosity", "competence", "social", "safety", "rest"}
    assert all(0.2 <= v <= 1.0 for v in d.values())
    for _ in range(50):
        d = sess.drive_proxies(_facts(success=True))
    assert d["competence"] < 0.6  # repeated success satisfies competence
    far = sess.drive_proxies(_facts(sibling_dist=9.9))
    assert far["social"] <= 0.2



def test_regression_b18_award_resets_each_window():
    """B18: a second window must be able to register an achievement."""
    sess = GoalSession(GoalProposer(seed=0), seed=9, ttl=2, inject=True)
    sess.tick(_facts(), due=True)
    sess.tracker._chosen = _g("practice", "t", 0.5)
    sess.tracker._hit = {"chosen": False, "random": False}
    r1 = sess.tick(_facts(success=True, task_id="t"), due=False)
    assert r1["chosen_achieved"] and r1["bonus"] > 0.0
    # expire the window, start a new one, force the same achievable goal
    sess.tick(_facts(), due=False)
    sess.tick(_facts(), due=False)
    sess.tick(_facts(), due=True)
    sess.tracker._chosen = _g("practice", "t", 0.5)
    sess.tracker._hit = {"chosen": False, "random": False}
    r2 = sess.tick(_facts(success=True, task_id="t"), due=False)
    assert r2["chosen_achieved"] and r2["bonus"] > 0.0
    assert sess.telemetry()["achieved_chosen"] == 2




def test_practice_requires_matching_task():
    tr = GoalTracker()
    tr.start(_g("practice", "1"), None, (0.0, 0.0))
    assert tr.step(_facts(success=True, task_id="0"))["chosen"] is False
    assert tr.step(_facts(success=True, task_id="1"))["chosen"] is True


def test_explore_uses_one_meter_cells():
    tr = GoalTracker()
    tr.start(_g("explore", "novel_zone"), None, (0.0, 0.0))
    assert tr.step(_facts(x=0.9, y=0.9))["chosen"] is False  # same 1m cell
    assert tr.step(_facts(x=0.1, y=1.1))["chosen"] is True   # new cell

