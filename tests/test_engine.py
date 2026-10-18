"""Interaction state machine driven by synthetic hands (no camera, no Win32)."""

import math

from conftest import Sim

from airdesk.config import Settings
from airdesk.core.types import Finger
from airdesk.sim.synthetic import Pose

IDX = Finger.INDEX


def pinch(finger=IDX, pointer=IDX):
    return Pose(extended={pointer}, pinch=finger)


def test_disarmed_emits_nothing():
    s = Sim(armed=False)
    s.step(n=30, dx=0.004)
    s.step(pinch(), n=5)
    s.settle()
    assert s.mouse.events == []
    assert s.engine.state.value == "DISARMED"


def test_pointer_moves_cursor_monotonically():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    s.step(n=30, dx=0.003)
    xs = [e[1] for e in s.mouse.of("move")]
    assert len(xs) > 10
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    assert xs[-1] - xs[0] > 100


def test_quick_pinch_is_one_complete_click_instant_default():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    s.step(pinch(), n=4)
    assert s.mouse.of("click") == [("click", "left", 1)]   # complete before the release
    s.settle(n=6)
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]
    assert s.mouse.of("button") == [] and s.mouse.drags() == []


def test_quick_pinch_is_one_down_up_click_classic():
    s = Sim(mode="classic")
    s.settle()
    s.mouse.events.clear()
    s.step(pinch(), n=4)
    s.settle(n=6)
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]
    assert s.mouse.of("click") == [] and s.mouse.drags() == []


def test_long_pinch_drags_and_releases_once():
    s = Sim(mode="classic")
    s.settle()
    s.mouse.events.clear()
    s.step(pinch(), n=48)                 # hold past hold_to_drag_s (1.2 s)
    s.step(pinch(), n=20, dx=0.004)       # move while held
    s.settle(n=6)
    buttons = s.mouse.of("button")
    assert buttons == [("button", "left", True), ("button", "left", False)]
    assert s.mouse.of("click") == [] and len(s.mouse.drags()) == 1
    assert s.mouse.held == set()
    moves_during_drag = [e for e in s.mouse.events if e[0] == "move"]
    assert moves_during_drag[-1][1] - moves_during_drag[0][1] > 50


def test_tracking_loss_during_drag_releases_button():
    s = Sim(mode="classic")
    s.settle()
    s.step(pinch(), n=15)
    assert s.mouse.held == {"left"}
    s.step(hand=False, n=12)              # > tracking_loss_grace_s
    assert s.mouse.held == set()
    assert s.engine.state.value == "TRACKING_LOST"
    s.settle()
    assert s.engine.state.value == "POINTER"


def test_secondary_pinch_right_clicks():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    s.step(pinch(Finger.MIDDLE), n=4)
    s.settle(n=6)
    assert [c[:2] for c in s.mouse.clicks()] == [("right", 1)]


def test_resting_pointing_hand_never_clicks():
    s = Sim()
    s.mouse.events.clear()
    for i in range(300):
        s.step(dx=0.002 * math.sin(i / 10), dy=0.002 * math.cos(i / 13))
    assert s.mouse.clicks() == [] and s.mouse.of("button") == []


def test_double_click_lands_on_same_pixel():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    for _ in range(2):
        s.step(pinch(), n=3)
        s.step(n=4, dx=0.0005)
    clicks = s.mouse.clicks()
    assert len(clicks) == 2
    assert clicks[0][2] == clicks[1][2]


def _point(f):
    """Pointing pose per finger. Since v1.1.3 the little finger points with
    the thumb out: little-only with the thumb tucked is Pinky Drag."""
    return Pose(extended={f, Finger.THUMB}) if f is Finger.LITTLE else Pose(extended={f})


def test_all_five_fingers_become_pointer_in_auto_mode():
    s = Sim()
    for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB):
        pose = _point(f)
        st = s.step(pose, n=20)
        assert st.finger == f.label, (f, st.finger)


def test_finger_switch_does_not_jump_cursor():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    for f in (Finger.MIDDLE, Finger.LITTLE, Finger.THUMB, Finger.RING, Finger.INDEX):
        s.step(_point(f), n=20)
    xs = [(e[1], e[2]) for e in s.mouse.of("move")]
    steps = [math.dist(a, b) for a, b in zip(xs, xs[1:])]
    assert s.engine.counters["finger_switches"] >= 5
    assert max(steps, default=0) < 40, max(steps)


def test_ambiguous_pose_keeps_current_finger():
    s = Sim()
    s.step(Pose.pointing(Finger.RING), n=20)
    st = s.step(Pose.open_palm(), n=30)
    assert st.finger == "Ring"


def test_manual_and_lock_modes():
    st = Settings()
    st.finger.mode = "manual"
    st.finger.finger = "little"
    s = Sim(st)
    assert s.step(Pose.pointing(IDX), n=20).finger == "Little"
    st2 = Settings()
    st2.finger.mode = "lock"
    st2.finger.finger = "middle"
    s2 = Sim(st2)
    assert s2.step(Pose.pointing(Finger.RING), n=40).finger == "Middle"


def test_scroll_pose_scrolls_in_hand_direction():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    s.step(Pose.two_finger(), n=8)
    s.step(Pose.two_finger(), n=25, dy=-0.006)   # hand up
    ups = [e for e in s.mouse.of("scroll") if e[2] > 0]
    downs = [e for e in s.mouse.of("scroll") if e[2] < 0]
    assert ups and not downs
    assert s.mouse.of("click") == []


def test_horizontal_scroll_axis_lock():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    s.step(Pose.two_finger(), n=8)
    s.step(Pose.two_finger(), n=25, dx=0.006)
    sc = s.mouse.of("scroll")
    assert sc and all(e[2] == 0 for e in sc) and any(e[1] != 0 for e in sc)


def test_clutch_freezes_cursor():
    s = Sim()
    s.settle()
    s.step(Pose.fist(), n=8)
    s.mouse.events.clear()
    s.step(Pose.fist(), n=20, dx=0.01)
    assert s.mouse.of("move") == []
    assert s.engine.state.value == "CLUTCH"


def test_pinch_over_keyboard_types_instead_of_clicking():
    st = Settings()
    st.keyboard.dual_hand = False     # v1.0-v1.1.2 single-pointer keyboard (dual-hand: test_v113_*)
    s = Sim(st)
    s.kb.visible = True
    s.kb.rect = (0, 0, 1920, 1080)
    s.settle()
    s.mouse.events.clear()
    s.step(pinch(), n=4)
    s.settle(n=6)
    assert s.kb.pressed == ["a"]
    assert s.mouse.of("click") == [] and s.mouse.of("button") == []


def test_dwell_typing_fires_once():
    st = Settings()
    st.keyboard.selection = "dwell"
    st.keyboard.dwell_ms = 400
    s = Sim(st)
    s.kb.visible = True
    s.kb.rect = (0, 0, 1920, 1080)
    s.step(n=60)
    assert s.kb.pressed == ["a"]


def test_emergency_stop_releases_and_disarms():
    s = Sim(mode="classic")
    s.settle()
    s.step(pinch(), n=15)
    assert s.mouse.held == {"left"}
    s.engine.emergency_stop()
    assert s.mouse.held == set()
    assert not s.engine.armed
    s.mouse.events.clear()
    s.step(n=20, dx=0.01)
    assert s.mouse.events == []


def test_precision_mode_reduces_gain():
    def travel(mode):
        st = Settings()
        st.pointer.mode = mode
        s = Sim(st)
        s.settle()
        s.mouse.events.clear()
        s.step(n=20, dx=0.003)
        xs = [e[1] for e in s.mouse.of("move")]
        return xs[-1] - xs[0]
    assert 0 < travel("precision") < travel("relative") * 0.6


def test_physical_mouse_yields_and_cancels_drag():
    s = Sim(mode="classic")
    s.settle()
    s.step(pinch(), n=15)
    assert s.mouse.held == {"left"}
    s.engine.notify_physical_mouse(100, 100, s.t)
    assert s.mouse.held == set()
    s.mouse.events.clear()
    s.step(n=10, dx=0.01)
    assert s.mouse.of("move") == []      # physical mouse wins for a moment


def test_noisy_finger_switches_never_click_or_drag():
    import numpy as np
    from airdesk.sim.synthetic import frame, observation
    s = Sim()
    rng = np.random.default_rng(11)
    order = [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB]
    for i in range(60):
        f = order[rng.integers(0, 5)]
        for _ in range(25):
            s.t += 1 / 30
            s.engine.process(frame(s.t, observation(_point(f), 0.5, 0.7, noise=0.0015, rng=rng)))
    assert s.mouse.of("click") == [] and s.mouse.of("button") == []
