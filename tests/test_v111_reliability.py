"""v1.1.1 click-reliability hotfix: one regression test per root cause found
with the owner's real-hand report, plus the new contracts (interaction log,
dispatch-synced feedback states, 1.5 s drag migration).

Synthetic hands (airdesk/sim/synthetic.py, airdesk/bench/clicks.py) - these
prove the algorithm, not a human hand; see README.md.
"""

import math
import time

import pytest
from conftest import Sim

from airdesk.bench import clicks as cb
from airdesk.config import Settings
from airdesk.core.clicklog import ClickLog
from airdesk.core.types import EngineStatus, Finger
from airdesk.sim.synthetic import Pose

IDX, MID, RING, LIT = Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE
RELAXED = {MID: 0.55, RING: 0.6, LIT: 0.6}


def manual(style="pinch_pointer", mode=None):
    s = Settings()
    s.finger.mode = "manual"
    s.gesture.click_style = style
    if mode:
        s.gesture.click_mode = mode         # "classic" = the v1.1.1 down/up lifecycle under test
    return s


def pinch(s, finger=IDX, curl=0.6, hold=4, open_to=0.0, rest=None, flex=0.0, gap=0.02):
    sc = cb.Scenario("t", finger=finger, rest=rest or {}, pointer_flex=flex, gap=gap)
    for k in range(4):
        s.step(cb._pose(sc, (k + 1) / 4, curl))
    s.step(cb._pose(sc, 1.0, curl), n=hold)
    for k in range(3):
        s.step(cb._pose(sc, 1 - (1 - open_to) * (k + 1) / 3, curl))
    s.step(cb._pose(sc, open_to, curl), n=8)


# ------------------------------------------------ root causes (reproduced)
def test_relaxed_hand_pinch_is_not_blocked_by_keyboard_gesture():
    """v1.1: four half-curled fingers near the thumb = 'five-fingertip
    gathering forming' -> every new pinch was silently blocked while the Aim
    Lock ring flashed. Reproduced at 45/60 missed right clicks."""
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    pinch(s, finger=MID, rest={RING: 0.6, LIT: 0.6}, flex=0.5)
    assert [c[0] for c in s.mouse.clicks()] == ["right"]
    s.mouse.events.clear()
    pinch(s, finger=IDX, rest=RELAXED)
    assert [c[0] for c in s.mouse.clicks()] == ["left"]


def test_small_relaxed_opening_releases_the_button():
    """v1.1 held the button until the contact re-opened past a fixed 0.36:
    a small, relaxed opening left it down (then a drag after the hold)."""
    s = Sim(manual(mode="classic"))
    s.settle()
    s.mouse.events.clear()
    pinch(s, open_to=0.68)                      # contact ~0.14 -> re-opens to ~0.33 only
    assert s.mouse.held == set()
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]


def test_fixed_finger_right_click_while_index_bends():
    """Manual / Lock finger: the pointing index bending in sympathy with a
    middle-finger pinch must not block or delay the right click."""
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    pinch(s, finger=MID, flex=0.65, rest={RING: 0.6, LIT: 0.6})
    assert [c[0] for c in s.mouse.clicks()] == ["right"]


def test_two_frame_quick_tap_is_one_click():
    """Found on real SendInput: a ~2-frame contact re-opened 1 ms before the
    Balanced 45 ms span completed and v1.1 discarded it (4 of 6 quick
    pinches and the double click did nothing)."""
    s = Sim(manual(mode="classic"))
    s.settle()
    s.mouse.events.clear()
    sc = cb.Scenario("tap", gap=0.02)
    for a in (0.4, 0.75, 1.0, 1.0, 0.35, 0.0):     # contact ~2 frames, crossing late in a frame
        s.step(cb._pose(sc, a, 0.65))
    s.step(Pose.pointing(IDX), n=6)
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]
    assert s.engine.clicks.latest().phase == "completed"


def test_one_frame_tracking_glitch_does_not_end_a_press():
    s = Sim(manual(mode="classic"))
    s.settle()
    s.mouse.events.clear()
    for k in range(4):
        s.step(Pose.closing(IDX, (k + 1) / 4, curl=0.7))
    s.step(Pose.closing(IDX, 1.0, curl=0.7), n=6)
    s.step(Pose.closing(IDX, 0.4, curl=0.7))     # one wild frame
    s.step(Pose.closing(IDX, 1.0, curl=0.7), n=6)
    assert s.mouse.of("button") == [("button", "left", True)]   # still one press, still held


def test_sticky_double_click_only_inside_double_click_time():
    """A second press snaps onto the previous click only inside Windows'
    double-click time from the previous button-down (v1.1 measured 0.5 s
    from the *release*, pulling later clicks onto a neighbouring control)."""
    s = Sim(manual())
    s.settle()
    e = s.engine
    cx, cy = e.pointer.cursor
    prev = (cx + 9.0, cy)
    e._last_down = (s.t - 0.25, prev)          # press confirms ~0.15 s later: 0.4 s < 0.5 s
    pinch(s)
    assert e.clicks.latest().pos == (round(prev[0]), round(prev[1]))
    s.settle(n=10)
    cx, cy = e.pointer.cursor
    prev = (cx + 9.0, cy)
    e._last_down = (s.t - 0.45, prev)          # ~0.6 s by the time it confirms: no snap
    pinch(s)
    assert math.dist(e.clicks.latest().pos, prev) > 5


def test_bench_realistic_scenarios_complete_every_click():
    rows = {r["scenario"]: r for r in cb.run_all(trials=6)}
    for name in ("relaxed hand", "contact gap 0.15", "partial release", "realistic mix",
                 "realistic mix, right click", "right click, index flexes", "Pinch Free left, index flexes",
                 "auto finger mode, relaxed"):
        r = rows[name]
        assert r["missed"] == r["stuck"] == r["duplicate"] == r["wrong_button"] == 0, r
        assert r["success"] >= 0.94, r


# ---------------------------------------------------- interaction log
class RefusingMouse:
    """MouseSink whose button-downs Windows refuses."""

    def __init__(self):
        self.calls = []
        self.last_error = "SendInput rejected the left button-down"

    def move(self, x, y):
        pass

    def button(self, b, down):
        self.calls.append((b, down))
        return False

    def click(self, b, n=1):
        return False

    def scroll(self, *a):
        pass

    def xbutton(self, w):
        pass

    def zoom(self, n):
        pass

    def release_all(self):
        pass

    def held_buttons(self):
        return ()

    def cursor_pos(self):
        return (960.0, 540.0)


def test_click_log_tracks_each_outcome():
    s = Sim(manual(mode="classic"))
    s.settle()
    pinch(s)
    it = s.engine.clicks.latest()
    assert it.phase == "completed" and it.down_ok and it.up_ok and it.button == "left"
    assert it.recognition_ms() is not None and 0 <= it.recognition_ms() < 150
    # tracking lost while pressed: cancelled with the reason, button released
    for k in range(4):
        s.step(Pose.closing(IDX, (k + 1) / 4, curl=0.7))
    s.step(Pose.closing(IDX, 1.0, curl=0.7), n=3)
    s.step(hand=False, n=12)
    it = s.engine.clicks.latest()
    assert it.phase == "cancelled" and "tracking lost" in it.reason and s.mouse.held == set()


def test_refused_button_down_is_rejected_and_never_followed_by_an_up():
    s = Sim(manual(mode="classic"))
    s.engine.mouse = m = RefusingMouse()
    s.settle()
    pinch(s)
    it = s.engine.clicks.latest()
    assert it.phase == "rejected" and "SendInput" in it.reason
    assert m.calls == [("left", True)]                # no button-up for a press that never went down
    assert s.engine.counters["clicks"] == 0


def test_feedback_states_follow_dispatch_outcomes():
    from airdesk.ui.feedback import feedback_state
    log = ClickLog()
    st = EngineStatus(armed=True, tracking=True, cursor=(10, 10))
    assert feedback_state(st, None, 0.0)[0] == ""
    st.aim_lock = (5, 5)
    assert feedback_state(st, None, 0.0)[:2] == ("aim", (5, 5))          # aim: nothing pressed
    it = log.begin("left", "click", (5, 5), 0.0)
    assert feedback_state(st, it, 0.0)[0] == "pending"                   # accepted, not submitted
    log.submitted(it, True, True)
    assert feedback_state(st, it, 0.0)[0] == "down"                      # Windows accepted the down
    log.submitted(it, False, True)
    st.aim_lock = None
    now = time.perf_counter()
    assert feedback_state(st, it, now)[0] == "done"
    assert feedback_state(st, it, now + 1.0)[0] == ""
    it2 = log.begin("right", "click", (7, 7), 1.0)
    log.submitted(it2, True, False, "blocked by Windows: the window under the cursor runs as administrator")
    kind, at, _, text = feedback_state(st, it2, time.perf_counter())
    assert kind == "reject" and at == (7, 7) and "admin" in text
    assert log.counters == {"accepted": 2, "down_submitted": 1, "up_submitted": 1, "completed": 1,
                            "rejected": 1, "cancelled": 0, "drags": 0, "aim_released": 0}


# ------------------------------------------------------- settings
@pytest.mark.parametrize("old_hold,expect", [(1.0, 1.2), (1.25, 1.25)])
def test_v11_settings_migrate_drag_hold(old_hold, expect):
    d = Settings().to_dict()
    d["version"] = 2
    d["gesture"]["hold_to_drag_s"] = old_hold
    d["gesture"]["cluster_engage"] = 0.32            # removed v1.1 keys are ignored
    s = Settings.from_dict(d)
    assert s.gesture.hold_to_drag_s == expect and s.version == 5
    assert Settings().gesture.hold_to_drag_s == 1.2


def test_v11_profile_keeps_new_drag_hold():
    s = Settings()
    prof = s.export_profile()
    prof["gesture"]["hold_to_drag_s"] = 1.0
    del prof["gesture"]["double_click_s"]           # what a v1.1 profile looks like
    s.apply_profile(prof)
    assert s.gesture.hold_to_drag_s == 1.2
