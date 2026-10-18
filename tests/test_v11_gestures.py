"""v1.1 interaction contract (as amended in v1.1.1): fast pinch, 1.5 s drag,
Smart Aim Lock, click styles, V scrolling and the 🤟 keyboard toggle.

Synthetic hands at 30 fps (see airdesk/sim/synthetic.py). Pinches use the
realistic closing motion: the pinched finger - the pointer itself in Normal
style - flexes toward the thumb while the thumb travels to it.
"""

import math

import numpy as np
import pytest
from conftest import Sim

from airdesk.bench import aim
from airdesk.config import Settings
from airdesk.core.gestures import slot_contacts
from airdesk.core.types import Finger
from airdesk.sim.synthetic import Pose, frame, observation

IDX, MID, RING, LIT, THB = Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.LITTLE, Finger.THUMB


def pinch_once(s, finger=IDX, close=4, hold=4, curl=0.7, opening=3, dx=0.0, pointer=IDX):
    """Close over `close` frames, hold `hold` frames, open; return the aim point."""
    s.settle(pointer)
    aim_pt = s.engine.pointer.cursor
    for k in range(close):
        s.step(Pose.closing(finger, (k + 1) / close, pointer=pointer, curl=curl))
    s.step(Pose.closing(finger, 1.0, pointer=pointer, curl=curl), n=hold, dx=dx)
    for k in range(opening):
        s.step(Pose.closing(finger, 1 - (k + 1) / opening, pointer=pointer, curl=curl))
    s.settle(pointer, n=8)
    return aim_pt


def style(name):
    st = Settings()
    st.gesture.click_style = name
    return st


# ------------------------------------------------------------- A. fast pinch
@pytest.mark.parametrize("curl", [0.45, 0.7, 0.85])
def test_natural_index_pinch_clicks_at_aim_point(curl):
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    aim_pt = pinch_once(s, curl=curl, close=3)
    clicks = s.mouse.clicks()
    assert [c[:2] for c in clicks] == [("left", 1)]
    assert math.dist(clicks[0][2], aim_pt) <= 2.0


def test_fast_contact_and_short_release_still_click():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    pinch_once(s, close=2, hold=2, opening=1)        # ~0.13 s of contact
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]


def test_button_events_are_ordered_down_then_up_at_anchor():
    s = Sim(mode="classic")
    s.settle()
    s.mouse.events.clear()
    pinch_once(s)
    ev = [e for e in s.mouse.events if e[0] in ("button", "click")]
    assert ev == [("button", "left", True), ("button", "left", False)]
    i0, i1 = (i for i, e in enumerate(s.mouse.events) if e[0] == "button")
    between = {(e[1], e[2]) for e in s.mouse.events[i0:i1] if e[0] == "move"}
    assert len(between) <= 1 and s.mouse.held == set()


def test_presets_order_and_balanced_target():
    rows = {r["preset"]: r for r in aim.recognition_latency(trials=24)}
    assert all(r["missed"] == 0 and r["duplicate_presses"] == 0 for r in rows.values())
    f, b, st = (rows[k]["mean_ms"] for k in ("fast", "balanced", "stable"))
    assert f < b < st
    assert 45 <= f <= 65 and 45 <= b <= 70 and 95 <= st <= 125   # v1.1.2: deep-contact fast path (Balanced was ~80)


def test_low_confidence_landmarks_never_start_a_press():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    for k in range(8):
        o = observation(Pose.closing(IDX, min(1.0, (k + 1) / 3)), s.x, s.y)
        o.score = 0.3
        s.t += 1 / 30
        s.engine.process(frame(s.t, o))
    assert s.mouse.of("button") == []


def test_continuous_finger_switch_never_presses():
    """Index -> middle with blended in-between frames (the old pointer's tip
    passes the resting thumb): no phantom click in either style."""
    for name in ("pinch_pointer", "pinch_free"):
        s = Sim(style(name))
        rng = np.random.default_rng(3)
        s.settle()
        for a, b in ((IDX, MID), (MID, IDX), (IDX, RING), (RING, LIT), (LIT, IDX)) * 3:
            for k in range(12):
                c = min(1.0, k / 8)
                oa = observation(Pose(extended={a, Finger.THUMB}) if a is LIT else Pose.pointing(a),
                                 0.5, 0.75, noise=0.0012, rng=rng)
                ob = observation(Pose(extended={b, Finger.THUMB}) if b is LIT else Pose.pointing(b),
                                 0.5, 0.75, noise=0.0012, rng=rng)
                oa.image = oa.image * (1 - c) + ob.image * c
                s.t += 1 / 30
                s.engine.process(frame(s.t, oa))
            # (v1.1.3: the little finger points with the thumb out; little-only
            # with the thumb tucked is Pinky Drag, covered in test_v113_*.)
            s.step(Pose(extended={b, Finger.THUMB}) if b is LIT else Pose.pointing(b), n=15)
        assert s.mouse.of("button") == [] and s.mouse.of("click") == [], name


# ------------------------------------------------------------ B. click / drag
@pytest.mark.parametrize("hold_s", [0.05, 0.5, 1.0, 1.4])
def test_release_before_drag_threshold_is_a_click(hold_s):
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    pinch_once(s, close=3, hold=max(1, round(hold_s * 30) - 3))
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]
    assert s.mouse.drags() == [] and s.engine.counters["drags"] == 0


def test_hand_motion_before_threshold_does_not_drag():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    pinch_once(s, close=3, hold=15, dx=0.004)        # hand travels ~0.06 while pinched 0.6 s
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]
    assert s.mouse.drags() == []


def test_hold_to_drag_threshold_drags_with_single_button_down_and_no_jump():
    s = Sim(mode="classic")
    s.settle()
    s.mouse.events.clear()
    for k in range(3):
        s.step(Pose.closing(IDX, (k + 1) / 3))
    while s.engine.state.value != "PRESS_PENDING":
        s.step(Pose.closing(IDX, 1.0))
    t_accept = s.t
    while s.engine.state.value == "PRESS_PENDING":
        s.step(Pose.closing(IDX, 1.0))
    assert s.engine.state.value == "DRAGGING"
    assert 1.2 - 1e-6 <= s.t - t_accept < 1.2 + 1 / 30 + 1e-6   # exactly at the 1.2 s threshold
    n0 = len(s.mouse.events)
    s.step(Pose.closing(IDX, 1.0), n=20, dx=0.004)
    mv = [(e[1], e[2]) for e in s.mouse.events[n0:] if e[0] == "move"]
    steps = [math.dist(a, b) for a, b in zip(mv, mv[1:])]
    assert max(steps) < 40 and mv[-1][0] - mv[0][0] > 100
    s.settle(n=8)
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]
    assert s.mouse.held == set() and len(s.mouse.drags()) == 1


def test_thumb_pointer_long_press_is_deterministic():
    s = Sim()
    s.settle(THB, n=25)
    s.mouse.events.clear()
    thumb_press = Pose(extended=set(), thumb_press=True)
    s.step(thumb_press, n=4)
    s.settle(THB, n=8)
    assert [c[:2] for c in s.mouse.clicks()] == [("left", 1)]     # < 0.8 s = left click
    s.mouse.events.clear()
    s.step(thumb_press, n=33)                                    # ~1.1 s: 0.8 .. 1.5 s
    assert s.mouse.of("click") == [] and s.mouse.of("button") == []   # decided at the release
    s.settle(THB, n=8)
    assert s.mouse.of("click") == [("click", "right", 1)] and s.mouse.of("button") == []
    s.mouse.events.clear()
    s.step(thumb_press, n=50)                                    # >= 1.5 s = left drag
    assert s.engine.state.value == "DRAGGING" and s.mouse.held == {"left"}
    s.settle(THB, n=8)
    assert s.mouse.of("click") == [] and s.mouse.held == set()

    st = Settings()
    st.gesture.long_press_right_click_thumb = False
    s2 = Sim(st, mode="classic")
    s2.settle(THB, n=25)
    s2.mouse.events.clear()
    s2.step(thumb_press, n=50)
    assert s2.engine.state.value == "DRAGGING"


# ----------------------------------------------------------- C. aim lock
@pytest.mark.parametrize("name,finger", [("pinch_pointer", IDX), ("pinch_free", MID)])
def test_small_targets_are_hit_with_aim_lock(name, finger):
    rows = aim.small_targets(trials=12, styles=(name,), locks=(True,))
    for r in rows:
        assert r["hit"] == r["trials"] and r["wrong_target"] == 0, r


def test_aim_lock_beats_unlocked_pointer_in_normal_style():
    rows = {r["aim_lock"]: r for r in aim.aim_displacement(trials=20) if r["style"] == "pinch_pointer"}
    assert rows[True]["click_px"]["p95"] <= 3.0 and rows[True]["visual_px"]["p95"] <= 12.0
    assert rows[False]["visual_px"]["median"] > 50       # the v1.0 problem, reproduced


def test_pointing_motion_is_never_frozen_or_clicked():
    r = aim.false_locks(seconds=40)
    assert r["locked_fraction"] == 0.0 and r["presses"] == 0


def test_cancelled_pinch_resumes_pointing_without_jump():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    for a in (0.3, 0.55, 0.7, 0.7):                  # closes part-way ...
        s.step(Pose.closing(IDX, a))
    assert s.engine.gesture_label == "Aiming"
    for a in (0.5, 0.25, 0.0):                       # ... and opens again
        s.step(Pose.closing(IDX, a))
    s.step(n=6)
    assert s.engine.gesture_label in ("Point", "Pose change")
    mv = [(e[1], e[2]) for e in s.mouse.of("move")]
    n_before = len(mv)
    s.step(n=15, dx=0.003)
    mv = [(e[1], e[2]) for e in s.mouse.of("move")]
    steps = [math.dist(a, b) for a, b in zip(mv, mv[1:])]
    assert s.mouse.of("button") == [] and max(steps) < 40
    assert mv[-1][0] - mv[n_before - 1][0] > 60      # follows the hand again


def test_rapid_aim_then_pinch_clicks_where_motion_ended():
    r = aim.rapid_aim_then_pinch(trials=12)
    assert r["detected"] == 12 and r["click_px"]["max"] <= 6.0


def test_physical_mouse_cancels_aim_lock():
    s = Sim()
    s.settle()
    for a in (0.3, 0.6):
        s.step(Pose.closing(IDX, a))
    s.engine.notify_physical_mouse(300, 300, s.t)
    s.mouse.events.clear()
    s.step(Pose.closing(IDX, 1.0), n=6)
    assert s.mouse.of("button") == []                # yields while the physical mouse is in use


# ----------------------------------------------------- D. Normal / Pinch Free
def test_index_pointer_mappings_match_spec():
    assert slot_contacts(IDX, "pinch_pointer") == {
        "primary": "pinch:index", "secondary": "pinch:middle", "tertiary": "pinch:ring"}
    assert slot_contacts(IDX, "pinch_free") == {
        "primary": "pinch:middle", "secondary": "pinch:ring", "tertiary": "pinch:little"}
    assert slot_contacts(MID, "pinch_pointer")["primary"] == "pinch:middle"
    assert slot_contacts(MID, "pinch_free")["primary"] == "pinch:index"
    assert slot_contacts(THB, "pinch_free") == {"primary": "thumb_press"}


@pytest.mark.parametrize("name,finger,button", [
    ("pinch_pointer", IDX, "left"), ("pinch_pointer", MID, "right"), ("pinch_pointer", RING, "middle"),
    ("pinch_free", MID, "left"), ("pinch_free", RING, "right"), ("pinch_free", LIT, "middle"),
    ("pinch_free", IDX, None), ("pinch_pointer", LIT, None)])
def test_click_style_contact_results(name, finger, button):
    s = Sim(style(name))
    s.settle()
    s.mouse.events.clear()
    pinch_once(s, finger=finger, curl=0.6)
    got = [c[0] for c in s.mouse.clicks()]
    assert got == ([button] if button else []), (name, finger, got)


def test_style_switch_reaches_engine_and_cancels_pending_press():
    st = Settings()
    s = Sim(st, mode="classic")
    s.settle()
    s.step(Pose.closing(MID, 1.0), n=4)              # Normal: thumb+middle = right, held
    assert s.mouse.held == {"right"}
    st.gesture.click_style = "pinch_free"
    s.engine.configure(st)                           # what the UI does on change
    assert s.mouse.held == set() and s.engine.state.value == "POINTER"
    s.settle(n=10)
    s.mouse.events.clear()
    pinch_once(s, finger=MID, curl=0.6)
    assert [c[0] for c in s.mouse.clicks()] == ["left"]
    assert s.engine.process(frame(s.t + 0.03)).click_style == "pinch_free"


def test_style_persists_across_restart(tmp_path):
    st = Settings()
    st.gesture.click_style = "pinch_free"
    st.gesture.pinch_sensitivity = "fast"
    path = st.save(tmp_path / "settings.json")
    back = Settings.load(path)
    assert back.gesture.click_style == "pinch_free" and back.gesture.pinch_sensitivity == "fast"
    assert Settings().gesture.click_style == "pinch_pointer"     # default is Normal
    assert Settings().gesture.aim_lock is True


def test_v10_settings_migrate_to_owner_drag_hold():
    old = Settings().to_dict()
    old["version"] = 1
    old["gesture"]["hold_to_drag_s"] = 0.4
    old["keyboard"]["width"] = 1500
    s = Settings.from_dict(old)
    assert s.gesture.hold_to_drag_s == 1.2 and s.keyboard.width == 0 and s.version == 5


# ------------------------------------------------------------ E. V scrolling
def test_v_scroll_labels_wheel_and_smooth_exit():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    st = s.step(Pose.two_finger(), n=6)
    assert st.state == "SCROLLING" and st.gesture == "Scroll Ready"
    st = s.step(Pose.two_finger(), n=12, dy=0.006)   # hand down
    assert st.gesture == "Scrolling Down"
    assert s.mouse.of("move") == [] and any(e[2] < 0 for e in s.mouse.of("scroll"))
    st = s.step(Pose.two_finger(), n=25, dx=-0.0002, dy=-0.012)
    assert st.gesture == "Scrolling Up"
    cur = s.engine.pointer.cursor
    s.settle(n=10)
    mv = [(e[1], e[2]) for e in s.mouse.of("move")]
    assert s.mouse.of("button") == [] and s.mouse.of("click") == []
    assert all(math.dist(m, cur) < 30 for m in mv)   # resumes where it froze


def test_mostly_vertical_motion_never_scrolls_sideways():
    s = Sim()
    s.settle()
    s.step(Pose.two_finger(), n=6)
    s.mouse.events.clear()
    s.step(Pose.two_finger(), n=25, dx=0.0035, dy=0.005)
    sc = s.mouse.of("scroll")
    assert sc and all(e[1] == 0 for e in sc)


def test_scroll_never_starts_during_drag():
    s = Sim(mode="classic")
    s.settle()
    s.step(Pose.closing(IDX, 1.0), n=55)
    assert s.engine.state.value == "DRAGGING"
    s.step(Pose.two_finger(), n=3)
    assert s.mouse.of("scroll") == []


# --------------------------------------------------------- F. 🤟 keyboard toggle
def ily(s, frames=30, **kw):
    """Form 🤟 from a pointing hand (little finger and thumb come out), hold."""
    for k in range(3):
        s.step(Pose(extended={IDX}, curls={LIT: 1 - (k + 1) / 3}), **kw)
    return s.step(Pose.ily(), n=frames, **kw)


def relax(s):
    s.step(Pose.pointing(IDX), n=12)


def toggles(s):
    return s.commands.count("toggle_keyboard")


def test_ily_toggles_once_per_hold_with_progress():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    st = ily(s, frames=7)
    assert toggles(s) == 0 and 0.3 < st.progress < 1.0         # ~0.23 s: still holding
    st = ily(s, frames=60)
    assert toggles(s) == 1 and st.progress == 1.0              # held 2+ s: exactly one toggle
    relax(s)
    ily(s, frames=30)
    assert toggles(s) == 2                                     # released -> may toggle again
    relax(s)
    assert s.mouse.of("button") == [] and s.mouse.of("click") == []


def test_ily_hold_time_and_cooldown():
    s = Sim()
    s.settle()
    t0 = s.t
    while toggles(s) == 0:
        s.step(Pose.ily())
    assert 0.4 <= s.t - t0 <= 0.4 + 4 / 30 + 1e-6            # 2-frame debounce + 0.4 s hold (v1.1.2)
    s.step(Pose.pointing(IDX), n=4)                           # quick release ...
    s.step(Pose.ily(), n=8)
    assert toggles(s) == 1                                    # ... no second toggle before the hold


@pytest.mark.parametrize("kw", [{"mirror": True}, {"roll_deg": 35}, {"roll_deg": -35}])
def test_ily_works_for_either_hand_and_orientation(kw):
    s = Sim()
    s.settle()
    for _ in range(40):
        s.t += 1 / 30
        s.engine.process(frame(s.t, observation(Pose.ily(), s.x, s.y, **kw)))
    assert toggles(s) == 1


@pytest.mark.parametrize("pose", [Pose.fist(), Pose.two_finger(), Pose.closing(IDX, 1.0, curl=0.7),
                                  Pose.closing(MID, 1.0, curl=0.6), Pose.horns(), Pose.open_palm(),
                                  Pose.gather(1.0), Pose.ily(thumb=False)])
def test_other_poses_never_toggle_keyboard(pose):
    """Includes v1.1's five-fingertip gathering: no longer a keyboard gesture."""
    s = Sim()
    s.settle()
    s.step(pose, n=45)
    assert toggles(s) == 0


def test_keyboard_gesture_respects_disarm_and_drag():
    s = Sim(armed=False)
    s.settle()
    ily(s, frames=40)
    assert toggles(s) == 0
    s2 = Sim(mode="classic")
    s2.settle()
    s2.step(Pose.closing(IDX, 1.0), n=55)
    assert s2.engine.state.value == "DRAGGING"
    t_drop = None
    for _ in range(45):                    # thumb leaves the index for 🤟: the drag ends first
        st = s2.step(Pose.ily())
        if t_drop is None and st.state != "DRAGGING":
            t_drop = s2.t
            assert s2.mouse.held == set() and toggles(s2) == 0
        if toggles(s2):
            break
    assert t_drop is not None and toggles(s2) == 1 and s2.t - t_drop >= 0.4   # never while dragging


def test_keyboard_gesture_can_be_disabled():
    st = Settings()
    st.gesture.keyboard_gesture = False
    s = Sim(st)
    s.settle()
    ily(s, frames=40)
    assert toggles(s) == 0
