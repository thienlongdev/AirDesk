"""v1.1.2: Instant Auto-Click, Aim Lock recovery, settled contacts, Drag
Mode (1.2 s), 🤟 at 0.4 s, settings migration and feedback semantics.

Every scenario here is synthetic (generated landmarks); the owner's
physical-hand acceptance is the Benchmark Lab test, not these.
"""

import json
import time

import pytest
from conftest import Sim

from airdesk.bench import aim
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
        s.gesture.click_mode = mode
    return s


def downs(m):
    """Every button-down that reached the sink (a click batch counts once)."""
    return [e for e in m.events if e[0] == "click" or (e[0] == "button" and e[2])]


def close(s, finger=IDX, gap=0.02, curl=0.65, frames=4, rest=None, **kw):
    sc = cb.Scenario("t", finger=finger, gap=gap, rest=rest or {})
    for k in range(frames):
        s.step(cb._pose(sc, (k + 1) / frames, curl), **kw)
    return sc


def hold(s, sc, n, curl=0.65, **kw):
    for _ in range(n):
        s.step(cb._pose(sc, 1.0, curl), **kw)


def open_(s, sc, curl=0.65, frames=3):
    for k in range(frames):
        s.step(cb._pose(sc, 1 - (k + 1) / frames, curl))
    s.step(Pose.pointing(IDX), n=6)


# ------------------------------------------------------- A. instant click
def test_default_is_instant_auto_click_with_1_2_s_drag_and_0_4_s_keyboard():
    g = Settings().gesture
    assert g.click_mode == "instant" and g.hold_to_drag_s == 1.2 and g.keyboard_gesture_s == 0.4
    assert g.drag_one_shot is True and Settings().version == 5


def test_instant_click_completes_before_release_and_holding_never_repeats_or_drags():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = close(s)
    hold(s, sc, 2)
    assert s.mouse.of("click") == [("click", "left", 1)]       # complete while still pinched
    assert s.mouse.held == set()
    it = s.engine.clicks.latest()
    assert it.phase == "completed" and it.down_ok and it.up_ok and it.mode == "instant"
    hold(s, sc, 75, dx=0.003)                                  # 2.5 s more, hand moving
    assert len(downs(s.mouse)) == 1 and s.engine.counters["drags"] == 0
    assert s.mouse.of("button") == [] and s.engine.state.value == "PRESS_PENDING"
    open_(s, sc)
    assert len(downs(s.mouse)) == 1 and s.engine.state.value == "POINTER"
    sc = close(s)                                              # rearmed by the release
    hold(s, sc, 3)
    assert len(s.mouse.of("click")) == 2


def test_instant_click_recognition_and_submission_are_fast():
    s = Sim(manual())
    s.settle()
    sc = close(s)
    hold(s, sc, 3)
    it = s.engine.clicks.latest()
    assert it.recognition_ms() is not None and it.recognition_ms() < 100    # synthetic, 30 fps
    assert it.confirm_to_down_ms() is not None and it.confirm_to_down_ms() < 5.0


@pytest.mark.parametrize("style,finger,button", [
    ("pinch_pointer", IDX, "left"), ("pinch_pointer", MID, "right"), ("pinch_pointer", RING, "middle"),
    ("pinch_free", MID, "left"), ("pinch_free", RING, "right"), ("pinch_free", LIT, "middle"),
    ("pinch_free", IDX, None)])
def test_instant_mapping_is_the_click_style_mapping(style, finger, button):
    s = Sim(manual(style))
    s.settle()
    s.mouse.events.clear()
    sc = close(s, finger=finger, curl=0.6)
    hold(s, sc, 3, curl=0.6)
    open_(s, sc, curl=0.6)
    assert [e[1] for e in downs(s.mouse)] == ([button] if button else [])


def test_two_deliberate_pinches_make_a_double_click_on_one_pixel():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    for _ in range(2):
        sc = close(s, frames=3)
        hold(s, sc, 1)
        open_(s, sc, frames=2)
    clicks = s.mouse.clicks()
    assert len(clicks) == 2 and clicks[0][2] == clicks[1][2]
    assert s.engine.gesture_label.startswith(("Point", "Double click"))


def test_very_fast_double_pinch_167_ms_apart_is_two_clicks():
    """Found on real SendInput: a 0.2 s same-contact flicker guard swallowed
    the second pinch of a fast double click (5-frame cycle)."""
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = cb.Scenario("d", gap=0.02)
    for _ in range(2):
        for k in range(2):
            s.step(cb._pose(sc, (k + 1) / 2, 0.65))
        s.step(cb._pose(sc, 1.0, 0.65))
        for k in range(2):
            s.step(cb._pose(sc, 1 - (k + 1) / 2, 0.65))
    s.step(Pose.pointing(IDX), n=8)
    assert s.mouse.of("click") == [("click", "left", 1)] * 2


def test_ten_rapid_repeated_pinches_are_ten_clicks_no_duplicates():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    for _ in range(10):
        sc = close(s, frames=3)
        hold(s, sc, 2)
        for k in range(3):
            s.step(cb._pose(sc, 1 - (k + 1) / 3 * 0.8, 0.65))  # opens to 20 % only
        s.step(cb._pose(sc, 0.2, 0.65), n=3)
    assert len(downs(s.mouse)) == 10


def test_one_frame_glitch_while_held_never_clicks_twice():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = close(s)
    hold(s, sc, 8)
    s.step(cb._pose(sc, 0.4, 0.65))                            # one wild frame
    hold(s, sc, 8)
    open_(s, sc)
    assert len(downs(s.mouse)) == 1


def test_refused_instant_click_is_rejected_with_reason():
    class Refusing(type(Sim().mouse)):
        last_error = "SendInput rejected the left click"

        def click(self, button, count=1):
            self.events.append(("refused", button))
            return False
    s = Sim(manual())
    s.engine.mouse = m = Refusing()
    s.settle()
    sc = close(s)
    hold(s, sc, 3)
    it = s.engine.clicks.latest()
    assert it.phase == "rejected" and "SendInput" in it.reason and s.engine.counters["clicks"] == 0
    assert m.of("refused") == [("refused", "left")]


# ------------------------------------------ B. owner failure: settled contact
@pytest.mark.parametrize("gap,finger,button", [(0.05, IDX, "left"), (0.055, IDX, "left"),
                                                (0.05, MID, "right")])
def test_contact_settling_just_above_threshold_clicks(gap, finger, button):
    """The owner's 'Aim Lock, buttons: none' geometry: the fingertip
    landmarks stop at ~1.17-1.3x the engage threshold. v1.1.1: 0 clicks."""
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = close(s, finger=finger, gap=gap, curl=0.7, frames=5, rest=dict(RELAXED) if finger is IDX else {})
    hold(s, sc, 8, curl=0.7)
    assert [e[1] for e in downs(s.mouse)] == [button]
    assert s.engine.clicks.latest().evidence == "settled"
    hold(s, sc, 20, curl=0.7)                                  # a held, jittering plateau: still one
    open_(s, sc, curl=0.7)
    assert len(downs(s.mouse)) == 1


def test_contact_stopping_short_is_not_a_click_and_aim_lock_lets_go_fast():
    r = aim.aim_recovery(trials=12)
    assert r["aim_lock_armed"] == 12 and r["presses"] == 0
    assert r["release_after_stop_ms"]["max"] <= 450          # v1.1.1 held it up to 600 ms


def test_aim_lock_released_without_click_is_logged_with_reason():
    s = Sim(manual())
    s.settle()
    for k in range(5):
        s.step(Pose.closing(IDX, 0.5 * (k + 1) / 5, curl=0.6))
    s.step(Pose.closing(IDX, 0.5, curl=0.6), n=20)
    ev = s.engine.clicks.events()
    assert ev and ev[-1]["kind"] == "aim_released" and "contact" in ev[-1]["text"]
    assert ev[-1]["pair"] == "thumb+index" and s.engine.clicks.snapshot_counters()["aim_released"] >= 1
    assert downs(s.mouse) == [] and s.engine._aim is None


@pytest.mark.parametrize("curl", [0.55, 0.7, 0.85])
def test_relaxed_idle_hand_is_not_frozen_or_clicked(curl):
    """v1.1.1 froze this relaxed, noisy pointing hand in Aim Lock 47 % of the
    time (curl 0.85, noise 0.003)."""
    r = aim.relaxed_idle(seconds=20, noise=0.003, curl=curl)
    assert r["presses"] == 0 and r["locked_fraction"] < 0.05


def test_early_aim_candidate_during_pointing_never_clicks():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    for _ in range(6):                                         # thumb drifts toward the index and back
        for a in (0.2, 0.35, 0.45, 0.35, 0.2, 0.0):
            s.step(Pose.closing(IDX, a, curl=0.5), dx=0.002)
    assert downs(s.mouse) == []


def test_depth_noise_on_world_landmarks_still_clicks_once():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    import numpy as np
    rng = np.random.default_rng(4)
    for _ in range(3):
        sc = close(s, rest=dict(RELAXED), world=True, depth_noise=0.008, rng=rng)
        hold(s, sc, 6, world=True, depth_noise=0.008, rng=rng)
        for k in range(3):
            s.step(cb._pose(sc, 1 - (k + 1) / 3, 0.65), world=True, depth_noise=0.008, rng=rng)
        s.step(Pose.pointing(IDX), n=8, world=True, depth_noise=0.008, rng=rng)
    assert len(downs(s.mouse)) == 3


@pytest.mark.parametrize("scale", [0.6, 1.6])
def test_hand_distance_does_not_change_click_recognition(scale):
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = close(s, scale=scale)
    hold(s, sc, 3, scale=scale)
    assert [e[1] for e in downs(s.mouse)] == ["left"]


def test_low_tracking_confidence_never_clicks():
    s = Sim(manual())
    s.settle()
    s.mouse.events.clear()
    sc = close(s, score=0.35)
    hold(s, sc, 6, score=0.35)
    assert downs(s.mouse) == []


def test_camera_condition_hints_after_they_persist():
    s = Sim(manual())
    st = s.step(n=30, scale=0.3)                               # tiny hand: far from the camera
    assert "far" in st.hint
    st = s.step(n=2)
    assert st.hint == ""
    st = s.step(n=30, score=0.5)
    assert "light" in st.hint


# ------------------------------------------------------------ C. Drag Mode
def test_drag_mode_presses_once_at_1_2_s_with_no_click_before():
    s = Sim(manual())
    s.settle()
    s.engine.set_click_mode("drag")
    assert s.commands[-1] == "click_mode"
    s.mouse.events.clear()
    sc = close(s)
    t0 = None
    for _ in range(60):
        st = s.step(cb._pose(sc, 1.0, 0.65))
        if t0 is None and st.state == "PRESS_PENDING":
            t0 = s.t
        if s.mouse.held:
            break
        assert downs(s.mouse) == []                            # nothing before the threshold
    assert s.mouse.of("button") == [("button", "left", True)]
    assert 1.2 - 1e-6 <= s.t - t0 < 1.2 + 1 / 30 + 1e-6
    assert s.engine.state.value == "DRAGGING"
    hold(s, sc, 20, dx=0.004)
    open_(s, sc)
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]
    assert len(s.mouse.drags()) == 1 and s.mouse.of("click") == []
    assert s.engine.click_mode == "instant"                    # one-shot: back to Instant


def test_drag_mode_progress_is_reported_without_a_button():
    s = Sim(manual(mode="drag"))
    s.settle()
    sc = close(s)
    st = None
    for _ in range(15):
        st = s.step(cb._pose(sc, 1.0, 0.65))
    assert 0.2 < st.drag_progress < 1.0 and st.buttons == () and st.click_mode == "drag"


@pytest.mark.parametrize("frames", [3, 15, 33])
def test_drag_mode_release_before_threshold_does_nothing(frames):
    s = Sim(manual(mode="drag"))
    s.settle()
    s.mouse.events.clear()
    sc = close(s)
    hold(s, sc, frames)
    open_(s, sc)
    assert downs(s.mouse) == [] and s.mouse.of("button") == []
    it = s.engine.clicks.latest()
    assert it.phase == "cancelled" and "drag hold" in it.reason
    assert s.engine.click_mode == "drag"                       # still armed for the drag


def test_drag_mode_stays_on_when_one_shot_is_off():
    st = manual(mode="drag")
    st.gesture.drag_one_shot = False
    s = Sim(st)
    s.settle()
    sc = close(s)
    hold(s, sc, 40)
    open_(s, sc)
    assert s.engine.counters["drags"] == 1 and s.engine.click_mode == "drag"


def test_drag_mode_right_click_stays_instant():
    s = Sim(manual(mode="drag"))
    s.settle()
    s.mouse.events.clear()
    sc = close(s, finger=MID, curl=0.6)
    hold(s, sc, 2, curl=0.6)
    assert s.mouse.of("click") == [("click", "right", 1)]


def test_switching_mode_cancels_pending_drag_without_clicking():
    s = Sim(manual(mode="drag"))
    s.settle()
    sc = close(s)
    hold(s, sc, 10)
    s.engine.set_click_mode("instant")
    assert s.engine.state.value == "POINTER" and s.mouse.held == set()
    assert s.engine.clicks.latest().phase == "cancelled"
    hold(s, sc, 20)
    open_(s, sc)
    assert downs(s.mouse) == []                                # the switch never clicks


@pytest.mark.parametrize("how", ["tracking", "emergency", "physical"])
def test_drag_mode_drag_is_always_released(how):
    s = Sim(manual(mode="drag"))
    s.settle()
    sc = close(s)
    hold(s, sc, 40)
    assert s.mouse.held == {"left"}
    if how == "tracking":
        s.step(hand=False, n=12)
    elif how == "emergency":
        s.engine.emergency_stop()
    else:
        s.engine.notify_physical_mouse(100, 100, s.t)
    assert s.mouse.held == set()


def test_classic_mode_keeps_v111_click_on_release():
    s = Sim(manual(mode="classic"))
    s.settle()
    s.mouse.events.clear()
    sc = close(s)
    hold(s, sc, 3)
    assert s.mouse.of("button") == [("button", "left", True)]
    open_(s, sc)
    assert s.mouse.of("button") == [("button", "left", True), ("button", "left", False)]


def test_click_mode_survives_restart(tmp_path):
    st = Settings()
    st.gesture.click_mode = "classic"
    path = st.save(tmp_path / "settings.json")
    assert Settings.load(path).gesture.click_mode == "classic"


# ------------------------------------------------------- D. migration
@pytest.mark.parametrize("old,hold_s,expect", [(3, 1.5, 1.2), (3, 1.3, 1.3), (2, 1.0, 1.2), (1, 0.4, 1.2)])
def test_settings_migrate_to_1_2_s_drag_keeping_custom_values(old, hold_s, expect):
    d = Settings().to_dict()
    d["version"] = old
    d["gesture"]["hold_to_drag_s"] = hold_s
    d["gesture"]["keyboard_gesture_s"] = 0.7
    del d["gesture"]["click_mode"]
    s = Settings.from_dict(d)
    assert s.gesture.hold_to_drag_s == expect and s.gesture.keyboard_gesture_s == 0.4
    assert s.gesture.click_mode == "instant" and s.version == 5


def test_customised_keyboard_hold_is_kept_and_v111_file_is_backed_up(tmp_path):
    d = Settings().to_dict()
    d["version"] = 3
    d["gesture"]["keyboard_gesture_s"] = 0.9
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    s = Settings.load(path)
    assert s.gesture.keyboard_gesture_s == 0.9
    s.save(path)
    bak = tmp_path / "settings.v3.bak.json"
    assert bak.exists() and json.loads(bak.read_text(encoding="utf-8"))["version"] == 3


def test_v111_profile_keeps_current_drag_and_keyboard_hold():
    s = Settings()
    prof = s.export_profile()
    prof["gesture"]["hold_to_drag_s"] = 1.5
    prof["gesture"]["keyboard_gesture_s"] = 0.7
    del prof["gesture"]["click_mode"]
    s.apply_profile(prof)
    assert s.gesture.hold_to_drag_s == 1.2 and s.gesture.keyboard_gesture_s == 0.4


# ----------------------------------------------------- E. 🤟 and feedback
def test_ily_toggles_at_0_4_s_once_and_never_clicks():
    s = Sim()
    s.settle()
    s.mouse.events.clear()
    t0 = s.t
    while "toggle_keyboard" not in s.commands:
        s.step(Pose.ily())
        assert s.t - t0 < 1.0
    assert 0.4 <= s.t - t0 <= 0.4 + 4 / 30 + 1e-6
    s.step(Pose.ily(), n=40)
    assert s.commands.count("toggle_keyboard") == 1 and downs(s.mouse) == []


def test_feedback_never_shows_aim_as_a_click_and_shows_instant_check():
    from airdesk.ui.feedback import feedback_state
    log = ClickLog()
    st = EngineStatus(armed=True, tracking=True, cursor=(10, 10))
    st.aim_lock = (5, 5)
    assert feedback_state(st, None, 0.0)[0] == "aim"
    it = log.begin("left", "click", (5, 5), 0.0)
    log.submitted(it, True, True)
    log.submitted(it, False, True)                            # instant: both at once
    now = time.perf_counter()
    assert feedback_state(st, it, now)[0] == "done"           # the result wins over a new aim
    assert feedback_state(st, it, now + 1.0)[0] == "aim"
    st.aim_lock = None
    d = log.begin("left", "drag", (8, 8), 1.0)
    st.drag_progress = 0.5
    assert feedback_state(st, d, now)[:3] == ("dragwait", (8, 8), 0.5)
    log.end(d, "cancelled", "released before the 1.2 s drag hold (Drag Mode: no click)")
    kind, _, _, text = feedback_state(st, d, time.perf_counter())
    assert kind == "cancel" and text == "Released before drag"
    st.click_mode = "drag"
    assert feedback_state(st, d, time.perf_counter() + 5)[0] == "dragmode"
    st.click_mode, st.hint = "instant", "Hand too far - move closer"
    assert feedback_state(st, d, time.perf_counter() + 5)[:1] == ("hint",)


def test_status_exposes_contact_diagnostics():
    s = Sim(manual())
    s.settle()
    for k in range(3):
        st = s.step(Pose.closing(IDX, 0.6 * (k + 1) / 3, curl=0.6))
    assert st.contact["pair"] == "thumb+index" and st.contact["engage"] == 0.26
    assert st.contact["ratio"] > 1.0 and st.click_mode == "instant"
    if st.aim_lock:
        assert "thumb+index" in st.aim_reason


def test_gesture_log_export_has_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRDESK_HOME", str(tmp_path))
    from airdesk.ui.pages import export_gesture_log
    s = Sim(manual())
    s.settle()
    sc = close(s)
    hold(s, sc, 3)
    path = export_gesture_log(s.engine, s.s, "test")
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["interactions"][-1]["phase"] == "completed" and "gesture_events" in data
    assert "frame" not in json.dumps(data).lower().replace("frames", "")
